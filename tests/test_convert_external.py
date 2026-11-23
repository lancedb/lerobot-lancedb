"""Zero-copy External Blob v2 conversion, source adapters, and LanceDBDataset roundtrip."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import av
import fsspec
import huggingface_hub
import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb._source import HubSource, parse_hf_dataset_uri
from lerobot_lancedb.convert import (
    FRAMES_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEOS_TABLE,
    DamagedSourceError,
    _resolve_video_mode,
    convert,
    main,
)

FPS = 10
VIDEO_KEY = "observation.images.top"


def _write_video(path: Path, frames: int, *, faststart: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"movflags": "faststart"} if faststart else {}
    with av.open(str(path), "w", options=options) as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "5", "preset": "ultrafast"}
        for i in range(frames):
            img = np.full((32, 32, 3), (i * 17 + 10) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_dataset(root: Path, *, frames_per_ep: int = 10, video_frames: int | None = None, with_orphans: bool = False) -> Path:
    total_frames = frames_per_ep * 2
    actual_video_frames = frames_per_ep if video_frames is None else video_frames

    meta = LeRobotDatasetMetadata.create(
        repo_id="local/ext-test",
        fps=FPS,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            VIDEO_KEY: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
        },
        root=root,
    )
    meta.save_episode_tasks(["reach"])
    for ep in range(2):
        start = ep * frames_per_ep
        meta.save_episode(
            episode_index=ep,
            episode_length=frames_per_ep,
            episode_tasks=["reach"],
            episode_stats={},
            episode_metadata={
                "data/chunk_index": 0,
                "data/file_index": ep,
                "dataset_from_index": start,
                "dataset_to_index": start + frames_per_ep,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": ep,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{VIDEO_KEY}/to_timestamp": frames_per_ep / FPS,
            },
        )
    meta.finalize()

    for ep in range(2):
        start = ep * frames_per_ep
        data_file = root / meta.get_data_file_path(ep)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "index": pa.array(range(start, start + frames_per_ep), pa.int64()),
                    "episode_index": pa.array([ep] * frames_per_ep, pa.int64()),
                    "task_index": pa.array([0] * frames_per_ep, pa.int64()),
                    "timestamp": pa.array([i / FPS for i in range(frames_per_ep)], pa.float32()),
                    "frame_index": pa.array(range(frames_per_ep), pa.int64()),
                    "observation.state": pa.array(
                        [[float(start + i), float(ep)] for i in range(frames_per_ep)],
                        pa.list_(pa.float32()),
                    ),
                }
            ),
            data_file,
        )
        _write_video(root / meta.get_video_file_path(ep, VIDEO_KEY), actual_video_frames, faststart=(ep == 0))

    if with_orphans:
        orphan_data = root / "data" / "chunk-000" / "file-999.parquet"
        pq.write_table(
            pa.table(
                {
                    "index": pa.array([5, 6], pa.int64()),
                    "episode_index": pa.array([9] * 2, pa.int64()),
                    "task_index": pa.array([0] * 2, pa.int64()),
                    "timestamp": pa.array([0.0, 0.1], pa.float32()),
                    "frame_index": pa.array([0, 1], pa.int64()),
                    "observation.state": pa.array([[0.0, 0.0], [0.0, 0.0]], pa.list_(pa.float32())),
                }
            ),
            orphan_data,
        )
        orphan_video = root / "videos" / VIDEO_KEY / "chunk-000" / "file-999.mp4"
        orphan_video.write_bytes(b"corrupt unreferenced orphan")

    assert json.loads((root / "meta" / "info.json").read_text())["total_frames"] == total_frames
    return root


def test_external_convert_roundtrip_pixel_exact_and_zero_copy(tmp_path: Path):
    # Include a space in the source path to verify file:// percent-encoding end-to-end.
    src = _make_dataset(tmp_path / "src dataset", frames_per_ep=10)
    out_copy = tmp_path / "out_copy"
    out_ext = tmp_path / "out_ext"

    convert(out_copy, root=src, video_mode="copy")
    # file:// root with default video_mode='auto' selects zero-copy external mode.
    convert(out_ext.as_uri(), root=src.as_uri())

    db_ext = lancedb.connect(str(out_ext))
    videos_tbl = db_ext.open_table(VIDEOS_TABLE)
    assert videos_tbl.schema.field(VIDEO_BLOB_COLUMN).metadata == {b"lancedb:logical_type": b"mp4"}

    descriptors = videos_tbl.to_arrow().column(VIDEO_BLOB_COLUMN).to_pylist()
    assert len(descriptors) == 2
    for ep, desc in enumerate(descriptors):
        mp4 = src / "videos" / VIDEO_KEY / "chunk-000" / f"file-{ep:03d}.mp4"
        assert desc["kind"] == 3  # External
        assert desc["blob_id"] == 1
        assert desc["position"] == 0
        assert desc["size"] == mp4.stat().st_size
        assert desc["blob_uri"] == f"{VIDEO_KEY}/chunk-000/file-{ep:03d}.mp4"

    ds_copy = LanceDBDataset(root=out_copy)
    ds_ext = LanceDBDataset(root=out_ext)
    assert len(ds_copy) == len(ds_ext) == 20
    for idx in (0, 4, 9, 10, 15, 19):
        item_c = ds_copy[idx]
        item_e = ds_ext[idx]
        assert torch.equal(item_c[VIDEO_KEY], item_e[VIDEO_KEY])
        assert torch.equal(item_c["observation.state"], item_e["observation.state"])


def test_external_convert_video_base_uri_override_without_network(tmp_path: Path):
    src = _make_dataset(tmp_path / "src", frames_per_ep=8)
    out = tmp_path / "out"
    remote_base = "s3://example-bucket/droid/videos"

    # Explicit video_base_uri implies external mode under default video_mode='auto'
    # and must not contact S3 during table creation.
    convert(out, root=src, video_base_uri=remote_base)

    db = lancedb.connect(str(out))
    descriptors = db.open_table(VIDEOS_TABLE).to_arrow().column(VIDEO_BLOB_COLUMN).to_pylist()
    assert [d["kind"] for d in descriptors] == [3, 3]
    assert [d["blob_id"] for d in descriptors] == [1, 1]
    assert [d["blob_uri"] for d in descriptors] == [
        f"{VIDEO_KEY}/chunk-000/file-000.mp4",
        f"{VIDEO_KEY}/chunk-000/file-001.mp4",
    ]


def test_external_convert_remote_fsspec_audits_defects_and_skips_orphans(tmp_path: Path, capsys):
    src = _make_dataset(tmp_path / "src", frames_per_ep=10, video_frames=6, with_orphans=True)

    # Mirror the dataset into an in-memory fsspec filesystem to exercise the remote
    # external path (meta/ + referenced data/ staging, range-read MP4 probing, remote orphan filter).
    mem_fs = fsspec.filesystem("memory")
    bucket_prefix = f"lerobot-ext-test-{tmp_path.name}"
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src).as_posix()
            mem_fs.pipe(f"{bucket_prefix}/{rel}", f.read_bytes())

    remote_root = f"memory://{bucket_prefix}"
    out = tmp_path / "out"

    with pytest.raises(DamagedSourceError, match="short by 4"):
        convert(out, root=remote_root)
    assert not out.exists()

    convert(out, root=remote_root, allow_defects=True)

    captured = capsys.readouterr()
    assert "--allow-defects" in captured.err
    assert "short by 4" in captured.err
    assert "skipping 1 orphan parquet file(s)" in captured.out
    assert "skipping 1 orphan video file(s)" in captured.out

    db = lancedb.connect(str(out))
    assert db.open_table(FRAMES_TABLE).count_rows() == 20
    assert db.open_table(VIDEOS_TABLE).count_rows() == 2


def test_external_convert_refuses_missing_referenced_video(tmp_path: Path):
    src = _make_dataset(tmp_path / "src", frames_per_ep=6)
    (src / "videos" / VIDEO_KEY / "chunk-000" / "file-001.mp4").unlink()
    out = tmp_path / "out"

    with pytest.raises(DamagedSourceError, match="missing referenced video"):
        convert(out, root=src, video_mode="external")
    assert not out.exists()

    with pytest.raises(FileNotFoundError, match="referenced by meta/episodes are missing"):
        convert(out, root=src, video_mode="external", allow_defects=True)
    assert not out.exists()


def test_uri_parsing_and_video_mode_resolution():
    assert parse_hf_dataset_uri("hf://datasets/lerobot/droid_1.0.1") == ("lerobot/droid_1.0.1", None, "")
    assert parse_hf_dataset_uri("hf://datasets/lerobot/droid_1.0.1@abc123/sub/dir") == (
        "lerobot/droid_1.0.1",
        "abc123",
        "sub/dir",
    )
    assert parse_hf_dataset_uri("hf://datasets/single@v1/sub") == ("single", "v1", "sub")
    with pytest.raises(ValueError, match="Only hf://datasets/"):
        parse_hf_dataset_uri("hf://models/foo/bar")

    assert _resolve_video_mode("auto", None, Path("/tmp/local")) == "copy"
    assert _resolve_video_mode("auto", "lerobot/pusht", Path("/tmp/local")) == "copy"
    assert _resolve_video_mode("auto", "lerobot/pusht", None) == "external"
    assert _resolve_video_mode("auto", None, "s3://bucket/ds") == "external"
    assert _resolve_video_mode("auto", None, "file:///tmp/local") == "external"
    assert _resolve_video_mode("auto", None, Path("/tmp/local"), video_base_uri="s3://b/v") == "external"

    with pytest.raises(ValueError, match="requires a local --root"):
        _resolve_video_mode("copy", None, "s3://bucket/ds")
    with pytest.raises(ValueError, match="video_base_uri is only valid"):
        _resolve_video_mode("copy", None, Path("/tmp/local"), video_base_uri="s3://b/v")


def test_hub_source_pins_commit_sha_and_handles_http_ranges(monkeypatch):
    sha = "a" * 40

    class _FakeHfApi:
        endpoint = "https://hub.example.test"

        def __init__(self, token=None):
            self.token = token

        def dataset_info(self, repo_id, revision="main"):
            assert repo_id == "lerobot/pusht"
            assert revision == "main"
            return SimpleNamespace(sha=sha)

    payload = b"0123456789abcdef"
    calls: list[tuple[str, str]] = []
    responses = [
        SimpleNamespace(status_code=429, headers={"Retry-After": "0"}, content=b"", raise_for_status=lambda: None),
        SimpleNamespace(status_code=206, headers={}, content=payload[4:10], raise_for_status=lambda: None),
        SimpleNamespace(status_code=200, headers={}, content=payload, raise_for_status=lambda: None),
    ]

    class _FakeSession:
        def __init__(self):
            self.headers = {}

        def mount(self, *_args):
            return None

        def get(self, url, headers=None, **_kwargs):
            calls.append((url, (headers or {}).get("Range", "")))
            if responses:
                return responses.pop(0)
            resp = requests.Response()
            resp.status_code = 404
            resp.url = url
            return resp

        def close(self):
            return None

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(requests, "Session", _FakeSession)

    with HubSource("lerobot/pusht", token="hf_test_token") as src:
        assert src.commit_sha == sha
        assert src.default_video_base_uri(["videos/cam/chunk-000/file-000.mp4"]) == (
            f"hf://datasets/lerobot/pusht@{sha}/videos"
        )
        # 429 retry followed by 206 Partial Content
        assert src.read_range("videos/cam/file 0.mp4", 4, 6) == b"456789"
        assert calls[0][0] == f"https://hub.example.test/datasets/lerobot/pusht/resolve/{sha}/videos/cam/file%200.mp4"
        assert calls[0][1] == "bytes=4-9"
        # 200 OK full payload is sliced to the requested range
        assert src.read_range("videos/cam/file-000.mp4", 2, 4) == b"2345"
        # Non-transient 404 fails immediately without retrying
        calls_before_404 = len(calls)
        with pytest.raises(requests.HTTPError):
            src.read_range("videos/cam/missing.mp4", 0, 8)
        assert len(calls) == calls_before_404 + 1


def test_cli_external_mode_and_invalid_storage_option(tmp_path: Path, monkeypatch):
    src = _make_dataset(tmp_path / "src", frames_per_ep=6)
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["lerobot-lance-convert", "--root", str(src), "--out", str(out), "--storage-option", "NO_EQUALS"],
    )
    with pytest.raises(SystemExit) as bad_opt:
        main()
    assert bad_opt.value.code == 2

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-lance-convert",
            "--root",
            str(src),
            "--out",
            str(out),
            "--video-mode",
            "external",
            "--video-workers",
            "2",
        ],
    )
    main()

    descriptors = lancedb.connect(str(out)).open_table(VIDEOS_TABLE).to_arrow().column(VIDEO_BLOB_COLUMN).to_pylist()
    assert len(descriptors) == 2
    assert all(d["kind"] == 3 and d["blob_id"] == 1 for d in descriptors)
