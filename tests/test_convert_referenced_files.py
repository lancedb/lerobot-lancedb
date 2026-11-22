"""The converter reads only files meta/episodes references; orphans are skipped with a warning."""

import json
from pathlib import Path

import av
import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_lancedb.convert import DamagedSourceError, convert

VIDEO_KEY = "observation.images.top"


def _write_data(path: Path, start: int, count: int, episode_index: int) -> None:
    """A LeRobot-style data parquet with monotonic absolute frame indices."""
    pq.write_table(
        pa.table(
            {
                "index": pa.array(range(start, start + count), pa.int64()),
                "episode_index": pa.array([episode_index] * count, pa.int64()),
                "task_index": pa.array([0] * count, pa.int64()),
                "observation.state": pa.array([[0.0, 0.0]] * count, pa.list_(pa.float32())),
            }
        ),
        path,
    )


def _write_video(path: Path, frames: int = 6) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            image = np.full((16, 16, 3), i * 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_dataset(root: Path, video_frames: int = 10) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    video_dir = root / "videos" / VIDEO_KEY / "chunk-000"
    video_dir.mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 20,
        "total_tasks": 0,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
            VIDEO_KEY: {"dtype": "video", "shape": [16, 16, 3], "names": ["height", "width", "channels"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    episodes = pa.table(
        {
            "episode_index": [0, 1],
            "length": [10, 10],
            "dataset_from_index": [0, 10],
            "dataset_to_index": [10, 20],
            "tasks": [["pick"], ["pick"]],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 1],
            f"videos/{VIDEO_KEY}/chunk_index": [0, 0],
            f"videos/{VIDEO_KEY}/file_index": [0, 1],
            f"videos/{VIDEO_KEY}/from_timestamp": [0.0, 0.0],
            f"videos/{VIDEO_KEY}/to_timestamp": [1.0, 1.0],
        }
    )
    pq.write_table(episodes, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    _write_data(root / "data" / "chunk-000" / "file-000.parquet", 0, 10, 0)
    _write_data(root / "data" / "chunk-000" / "file-001.parquet", 10, 10, 1)
    # Orphan shard: meta never references it, and its frame range sits inside the
    # referenced files. A filename-sorted read would hit it after file-001 and
    # fail the converter's order check, which is the DROID failure.
    _write_data(root / "data" / "chunk-000" / "file-002.parquet", 5, 10, 2)

    _write_video(video_dir / "file-000.mp4", video_frames)
    _write_video(video_dir / "file-001.mp4", video_frames)
    # Never referenced, so the converter never opens it as a video.
    (video_dir / "file-002.mp4").write_bytes(b"not an mp4")


def test_convert_reads_only_referenced_files_and_warns(tmp_path, capsys):
    root = tmp_path / "src"
    _write_dataset(root)

    convert(tmp_path / "out", root=root)

    printed = capsys.readouterr().out
    assert "skipping 1 orphan parquet file(s)" in printed
    assert "(10 rows)" in printed
    assert "skipping 1 orphan video file(s)" in printed

    db = lancedb.connect(str(tmp_path / "out"))
    frame_indices = db.open_table("frames").to_arrow().column("index").to_pylist()
    assert frame_indices == list(range(20))
    videos = db.open_table("videos").to_arrow()
    assert videos.column("file_index").to_pylist() == [0, 1]


def test_convert_refuses_missing_referenced_file(tmp_path):
    root = tmp_path / "src"
    _write_dataset(root)
    (root / "data" / "chunk-000" / "file-001.parquet").unlink()
    out = tmp_path / "out"

    # The default doctor gate reports the missing file; --allow-defects gets past
    # the gate but still cannot copy bytes that do not exist.
    with pytest.raises(DamagedSourceError, match="missing referenced parquet"):
        convert(out, root=root)
    assert not out.exists()

    with pytest.raises(FileNotFoundError, match="referenced by meta/episodes are missing"):
        convert(out, root=root, allow_defects=True)
    assert not out.exists()


def test_allow_defects_keeps_orphans_out_of_the_output(tmp_path, capsys):
    # The doctor gate blocks the short referenced videos, but allowing the defects
    # must not turn the unreferenced shard and video back into input.
    root = tmp_path / "src"
    _write_dataset(root, video_frames=6)
    out = tmp_path / "out"

    with pytest.raises(DamagedSourceError, match="short by 4"):
        convert(out, root=root)
    assert not out.exists()

    convert(out, root=root, allow_defects=True)

    captured = capsys.readouterr()
    assert "--allow-defects" in captured.err
    assert "short by 4" in captured.err
    assert "skipping 1 orphan parquet file(s)" in captured.out
    assert "skipping 1 orphan video file(s)" in captured.out

    db = lancedb.connect(str(out))
    assert db.open_table("frames").to_arrow().column("index").to_pylist() == list(range(20))
    assert db.open_table("videos").to_arrow().column("file_index").to_pylist() == [0, 1]
