"""v2.0 / v2.1 source layout: metadata rewrite, path parsing, doctor audit."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_lancedb.legacy import (
    codebase_version,
    data_parquet_files,
    load_source_audit,
    normalize_codebase_version,
    parse_video_locator,
    write_v3_meta_from_v2,
)


def _write_episode_parquet(path, start, n, episode_index, width=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "index": pa.array(range(start, start + n), pa.int64()),
            "episode_index": pa.array([episode_index] * n, pa.int64()),
            "observation.state": pa.array(
                [[float(i), float(i)] for i in range(start, start + n)], pa.list_(pa.float32())
            ),
            "action": pa.array([[0.0, 1.0]] * n, pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, path)


def _v2_info(version, n_episodes, length, with_video=False):
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    if with_video:
        features["observation.images.cam"] = {"dtype": "video", "shape": [48, 64, 3]}
    return {
        "codebase_version": version,
        "fps": 10,
        "robot_type": "test",
        "total_episodes": n_episodes,
        "total_frames": n_episodes * length,
        "total_tasks": 1,
        "total_videos": n_episodes if with_video else 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            if with_video
            else None
        ),
        "features": features,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_v2_dataset(root, version="v2.1", n_episodes=2, length=5):
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = _v2_info(version, n_episodes, length, with_video=True)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": "pick cube"}])
    episodes = []
    stats_rows = []
    global_stats = {
        "observation.state": {"min": [0.0, 0.0], "max": [10.0, 10.0], "mean": [5.0, 5.0], "std": [1.0, 1.0]},
        "action": {"min": [0.0, 1.0], "max": [0.0, 1.0], "mean": [0.0, 1.0], "std": [0.0, 0.0]},
    }
    cursor = 0
    for ep in range(n_episodes):
        _write_episode_parquet(root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", cursor, length, ep)
        episodes.append({"episode_index": ep, "tasks": ["pick cube"], "length": length})
        stats_rows.append(
            {
                "episode_index": ep,
                "stats": {feat: {**vals, "count": [length]} for feat, vals in global_stats.items()},
            }
        )
        (root / "videos" / "chunk-000" / "observation.images.cam").mkdir(parents=True, exist_ok=True)
        (root / "videos" / "chunk-000" / "observation.images.cam" / f"episode_{ep:06d}.mp4").write_bytes(
            b"not-a-real-mp4"
        )
        cursor += length
    _write_jsonl(meta / "episodes.jsonl", episodes)
    if version == "v2.0":
        (meta / "stats.json").write_text(json.dumps(global_stats, indent=4))
    else:
        _write_jsonl(meta / "episodes_stats.jsonl", stats_rows)
        (meta / "stats.json").write_text(json.dumps(global_stats, indent=4))
    return root


def test_normalize_codebase_version():
    assert normalize_codebase_version("v2.1") == "v2.1"
    assert normalize_codebase_version("2.0") == "v2.0"
    assert normalize_codebase_version("v2.1.0") == "v2.1"
    assert normalize_codebase_version("v3.0") == "v3.0"
    assert normalize_codebase_version(None) == "v3.0"


def test_parse_video_locator_v3(tmp_path):
    videos = tmp_path / "videos"
    mp4 = videos / "observation.images.top" / "chunk-002" / "file-007.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"x")
    assert parse_video_locator(mp4, videos) == ("observation.images.top", 2, 7)


def test_parse_video_locator_v2(tmp_path):
    videos = tmp_path / "videos"
    mp4 = videos / "chunk-001" / "observation.images.cam" / "episode_001042.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"x")
    assert parse_video_locator(mp4, videos) == ("observation.images.cam", 1, 1042)


def test_parse_video_locator_v2_alt(tmp_path):
    videos = tmp_path / "videos"
    mp4 = videos / "observation.images.cam" / "chunk-000" / "episode_000003.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"x")
    assert parse_video_locator(mp4, videos) == ("observation.images.cam", 0, 3)


def test_data_parquet_files_v2_episode_order(tmp_path):
    make_v2_dataset(tmp_path, version="v2.1", n_episodes=3, length=4)
    files = data_parquet_files(tmp_path)
    assert [p.stem for p in files] == ["episode_000000", "episode_000001", "episode_000002"]


def test_write_v3_meta_from_v21(tmp_path):
    src = make_v2_dataset(tmp_path / "src", version="v2.1", n_episodes=2, length=5)
    out = tmp_path / "out" / "meta"
    assert write_v3_meta_from_v2(src, out) == "v2.1"

    info = json.loads((out / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert "total_chunks" not in info
    assert info["data_path"].endswith("file-{file_index:03d}.parquet")
    assert (out / "tasks.parquet").is_file()
    assert (out / "stats.json").is_file()
    episodes = pq.read_table(out / "episodes" / "chunk-000" / "file-000.parquet")
    assert episodes.num_rows == 2
    assert episodes.column("dataset_from_index").to_pylist() == [0, 5]
    assert episodes.column("dataset_to_index").to_pylist() == [5, 10]
    assert episodes.column("videos/observation.images.cam/from_timestamp").to_pylist() == [0.0, 0.0]
    assert episodes.column("videos/observation.images.cam/file_index").to_pylist() == [0, 1]
    assert episodes.column("videos/observation.images.cam/chunk_index").to_pylist() == [0, 0]


def test_write_v3_meta_from_v20(tmp_path):
    src = make_v2_dataset(tmp_path / "src", version="v2.0", n_episodes=1, length=3)
    out = tmp_path / "out" / "meta"
    assert write_v3_meta_from_v2(src, out) == "v2.0"
    info = json.loads((out / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    stats = json.loads((out / "stats.json").read_text())
    assert "observation.state" in stats
    assert codebase_version(src) == "v2.0"


def test_v2_source_audit_boundaries(tmp_path):
    src = make_v2_dataset(tmp_path, version="v2.0", n_episodes=2, length=4)
    audit = load_source_audit(src)
    assert audit.version == "v2.0"
    assert audit.starts == [0, 4]
    assert audit.ends == [4, 8]
    assert audit.total_frames == 8
    assert audit.video_keys == ["observation.images.cam"]
    assert len(audit.data_files) == 2
    assert all(p.exists() for p in audit.data_files)
