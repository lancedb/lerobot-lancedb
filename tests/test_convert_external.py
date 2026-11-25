"""Zero-copy External Blob v2 conversion, source adapters, and LanceDBDataset roundtrip."""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from shutil import copytree
from threading import Thread
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit
from xml.etree.ElementTree import Element, SubElement, tostring

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
from lerobot.datasets import dataset_metadata as upstream_meta
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchcodec.decoders import VideoDecoder

from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb._source import (
    FsspecSource,
    HubSource,
    LocalSource,
    S3Source,
    _validate_content_range,
    parse_hf_dataset_uri,
)
from lerobot_lancedb.convert import (
    _MANAGED_OUTPUT_ENTRIES,
    FRAMES_TABLE,
    META_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEOS_TABLE,
    DamagedSourceError,
    _commit_local_output,
    _resolve_video_mode,
    convert,
    main,
)
from lerobot_lancedb.mp4_index import probe_local_mp4

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


def _make_dataset(
    root: Path,
    *,
    frames_per_ep: int = 10,
    video_frames: int | None = None,
    with_orphans: bool = False,
    video_key: str = VIDEO_KEY,
) -> Path:
    total_frames = frames_per_ep * 2
    actual_video_frames = frames_per_ep if video_frames is None else video_frames

    meta = LeRobotDatasetMetadata.create(
        repo_id="local/ext-test",
        fps=FPS,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            video_key: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
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
                f"videos/{video_key}/chunk_index": 0,
                f"videos/{video_key}/file_index": ep,
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": frames_per_ep / FPS,
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
        _write_video(root / meta.get_video_file_path(ep, video_key), actual_video_frames, faststart=(ep == 0))

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
        orphan_video = root / "videos" / video_key / "chunk-000" / "file-999.mp4"
        orphan_video.write_bytes(b"corrupt unreferenced orphan")

    assert json.loads((root / "meta" / "info.json").read_text())["total_frames"] == total_frames
    return root


@contextmanager
def _s3_loopback_server(files: dict[str, bytes], bucket: str = "example-bucket"):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return None

        def _reply(self, data: bytes, status: int = 200, headers: dict[str, str] | None = None):
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", '"test-etag"')
            self.send_header("Last-Modified", "Sun, 06 Nov 1994 08:49:37 GMT")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(data)

        def _serve(self):
            url = urlsplit(self.path)
            request_path = unquote(url.path)
            if not request_path.startswith(f"/{bucket}"):
                self._reply(b"", 404)
                return
            query = parse_qs(url.query)
            if query.get("list-type") == ["2"]:
                prefix = query.get("prefix", [""])[0]
                root_el = Element("ListBucketResult")
                SubElement(root_el, "IsTruncated").text = "false"
                for key, data in sorted(files.items()):
                    if key.startswith(prefix):
                        child = SubElement(root_el, "Contents")
                        SubElement(child, "Key").text = key
                        SubElement(child, "Size").text = str(len(data))
                self._reply(tostring(root_el))
                return
            key = request_path.removeprefix(f"/{bucket}/")
            if key not in files:
                self._reply(b"", 404)
                return
            data = files[key]
            rng = self.headers.get("Range")
            if rng:
                start_s, _, end_s = rng.removeprefix("bytes=").partition("-")
                lo = int(start_s)
                hi = min(int(end_s), len(data) - 1) if end_s else len(data) - 1
                self._reply(data[lo : hi + 1], 206, {"Content-Range": f"bytes {lo}-{hi}/{len(data)}"})
            else:
                self._reply(data)

        do_HEAD = do_GET = _serve

    with ThreadingHTTPServer(("127.0.0.1", 0), _Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def test_external_convert_roundtrip_pixel_exact_and_zero_copy(tmp_path: Path, monkeypatch):
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
        assert desc["blob_id"] == 0
        assert desc["position"] == 0
        assert desc["size"] == mp4.stat().st_size
        assert desc["blob_uri"] == mp4.as_uri()

    ds_copy = LanceDBDataset(root=out_copy)
    ds_ext = LanceDBDataset(root=out_ext)
    upstream = LeRobotDataset(repo_id="local/ext-test", root=out_ext, return_uint8=True)
    managed = lancedb.connect(out_copy).open_table(VIDEOS_TABLE)
    for table in (managed, videos_tbl):
        descriptor = table.take_row_ids([0]).select(["file_size", VIDEO_BLOB_COLUMN]).to_arrow()
        assert pa.types.is_struct(descriptor.column(VIDEO_BLOB_COLUMN).type)
        assert descriptor.nbytes < descriptor["file_size"][0].as_py()
    paths = sorted((src / "videos").rglob("*.mp4"))
    reference = [VideoDecoder(str(path), seek_mode="approximate") for path in paths]
    assert len(ds_copy) == len(ds_ext) == len(upstream) == 20
    for idx in (0, 4, 9, 10, 15, 19):
        item_c = ds_copy[idx]
        native = upstream[idx]
        with monkeypatch.context() as patch:
            def unexpected(*args, **kwargs):
                pytest.fail("A local external RGB read used blob I/O")

            patch.setattr(lancedb.table.LanceTable, "fetch_blob_files", unexpected)
            patch.setattr(lancedb.table.LanceTable, "fetch_blob_ranges", unexpected)
            item_e = ds_ext[idx]
        frame = reference[idx // 10].get_frames_at(indices=[idx % 10])
        assert torch.equal(native[VIDEO_KEY], frame.data[0])
        assert torch.equal(native["timestamp"], frame.pts_seconds[0].float())
        assert torch.equal(item_e["timestamp"], native["timestamp"])
        assert torch.equal(item_e[VIDEO_KEY], frame.data[0] / 255.0)
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


def test_hub_source_pins_commit_sha_and_validates_http_ranges(monkeypatch):
    sha = "a" * 40

    class _FakeHfApi:
        endpoint = "https://hub.example.test"

        def __init__(self, token=None):
            self.token = token

        def dataset_info(self, repo_id, revision="main"):
            assert repo_id == "lerobot/pusht"
            assert revision == "main"
            return SimpleNamespace(sha=sha)

    class _FakeResp:
        def __init__(self, status_code: int, headers: dict[str, str], body: bytes, url: str = ""):
            self.status_code = status_code
            self.headers = headers
            self._body = body
            self.url = url
            self.body_consumed = False
            self.raw_closed = False

            class _Raw:
                def close(_raw_self):
                    self.raw_closed = True

            self.raw = _Raw()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def iter_content(self, chunk_size: int = 65536):
            self.body_consumed = True
            for i in range(0, len(self._body), chunk_size):
                yield self._body[i : i + chunk_size]

        def raise_for_status(self):
            if self.status_code >= 400:
                resp = requests.Response()
                resp.status_code = self.status_code
                resp.url = self.url
                resp.raise_for_status()

    payload = b"0123456789abcdef"
    calls: list[tuple[str, str, bool]] = []
    resp_200 = _FakeResp(200, {"Content-Length": str(len(payload))}, payload)
    responses = [
        _FakeResp(429, {"Retry-After": "0"}, b""),
        _FakeResp(206, {"Content-Range": f"bytes 4-9/{len(payload)}"}, payload[4:10]),
        _FakeResp(206, {}, payload[2:6]),  # missing Content-Range
        _FakeResp(206, {"Content-Range": f"bytes 0-3/{len(payload)}"}, payload[0:4]),  # mismatched offset
        resp_200,
    ]

    class _FakeSession:
        def __init__(self):
            self.headers = {}

        def mount(self, *_args):
            return None

        def get(self, url, headers=None, stream=False, **_kwargs):
            calls.append((url, (headers or {}).get("Range", ""), stream))
            if responses:
                r = responses.pop(0)
                r.url = url
                return r
            return _FakeResp(404, {}, b"", url=url)

        def close(self):
            return None

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(requests, "Session", _FakeSession)

    with HubSource("lerobot/pusht", token="hf_test_token") as src:
        assert src.commit_sha == sha
        assert src.default_video_base_uri(["videos/cam/chunk-000/file-000.mp4"]) == (
            f"hf://datasets/lerobot/pusht@{sha}/videos"
        )
        # 429 retry followed by valid 206 Partial Content (stream=True).
        assert src.read_range("videos/cam/file 0.mp4", 4, 6) == b"456789"
        assert calls[0][0] == f"https://hub.example.test/datasets/lerobot/pusht/resolve/{sha}/videos/cam/file%200.mp4"
        assert calls[0][1] == "bytes=4-9"
        assert calls[0][2] is True
        # Missing or mismatched Content-Range on 206 fails loudly.
        with pytest.raises(RuntimeError, match="Missing Content-Range"):
            src.read_range("videos/cam/file-000.mp4", 2, 4)
        with pytest.raises(RuntimeError, match="Unexpected Content-Range"):
            src.read_range("videos/cam/file-000.mp4", 2, 4)
        # 200 OK fails immediately without reading body and closes the raw stream.
        with pytest.raises(RuntimeError, match="HTTP 200 instead of 206"):
            src.read_range("videos/cam/file-000.mp4", 2, 4)
        assert not resp_200.body_consumed
        assert resp_200.raw_closed
        # Non-transient 404 fails immediately without retrying.
        calls_before_404 = len(calls)
        with pytest.raises(requests.HTTPError):
            src.read_range("videos/cam/missing.mp4", 0, 8)
        assert len(calls) == calls_before_404 + 1


def test_remote_sources_reject_path_traversal_before_io(tmp_path: Path):
    staging = tmp_path / "staging"
    staging.mkdir()
    escape_target = tmp_path / "escape.txt"

    # 1. FsspecSource rejects malicious meta/ keys containing '..' before writing anything.
    mem_fs = fsspec.filesystem("memory")
    bucket_prefix = f"lerobot-traversal-{tmp_path.name}"
    mem_fs.pipe(f"{bucket_prefix}/meta/../escape.txt", b"pwned")
    src_fs = FsspecSource(f"memory://{bucket_prefix}")
    with pytest.raises(ValueError, match="Unsafe relative path"):
        src_fs.materialize_meta(staging)
    assert not escape_target.exists()
    assert list(staging.iterdir()) == []

    # 2. S3Source rejects malicious S3 object keys containing '..' before any local I/O.
    files = {"ds/meta/../escape.txt": b"pwned", "ds/meta/info.json": b"{}"}
    with _s3_loopback_server(files) as endpoint:
        s3_opts = {"aws_endpoint": endpoint, "aws_region": "us-east-1", "aws_skip_signature": "true"}
        with S3Source("s3://example-bucket/ds", storage_options=s3_opts) as src_s3:
            with pytest.raises(ValueError, match="Unsafe relative path"):
                src_s3.materialize_meta(staging)
    assert not escape_target.exists()
    assert list(staging.iterdir()) == []


def test_local_overwrite_replaces_meta_and_preserves_existing_on_failure(tmp_path: Path):
    src_a = _make_dataset(tmp_path / "src_a", frames_per_ep=10)
    src_b = _make_dataset(tmp_path / "src_b", frames_per_ep=6)
    (src_a / "meta" / "stale_from_a.txt").write_text("stale")
    out = tmp_path / "out"

    convert(out, root=src_a, video_mode="external")
    assert len(LanceDBDataset(root=out)) == 20

    # Re-converting a smaller dataset to the same local path refreshes both disk meta/ and meta.lance.
    convert(out, root=src_b, video_mode="external")
    assert not (out / "meta" / "stale_from_a.txt").exists()
    assert json.loads((out / "meta" / "info.json").read_text())["total_frames"] == 12
    db = lancedb.connect(str(out))
    meta_tbl = db.open_table(META_TABLE).to_arrow()
    meta_paths = meta_tbl["path"].to_pylist()
    assert "stale_from_a.txt" not in meta_paths
    assert json.loads(meta_tbl["data"][meta_paths.index("info.json")].as_py())["total_frames"] == 12
    assert db.open_table(FRAMES_TABLE).count_rows() == 12
    ds_b = LanceDBDataset(root=out)
    assert len(ds_b) == 12
    assert ds_b[0][VIDEO_KEY].shape == (3, 32, 32)
    assert ds_b[11][VIDEO_KEY].shape == (3, 32, 32)

    # Failed re-conversion with rejected video_base_uri leaves existing output completely intact.
    for bad_base in ("file:///nonexistent/#fragment", "invalid_relative_base"):
        with pytest.raises(ValueError):
            convert(out, root=src_b, video_base_uri=bad_base)
        assert lancedb.connect(str(out)).open_table(VIDEOS_TABLE).count_rows() == 2
        assert LanceDBDataset(root=out)[0][VIDEO_KEY].shape == (3, 32, 32)


def test_truncated_faststart_mp4_rejected_by_external_convert(tmp_path: Path):
    src = _make_dataset(tmp_path / "src", frames_per_ep=10)
    damaged = src / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4"
    orig = probe_local_mp4(damaged)
    damaged.write_bytes(damaged.read_bytes()[: int(orig.kf_positions[-1]) - 1])

    out = tmp_path / "out"
    with pytest.raises(DamagedSourceError, match="truncated MP4"):
        convert(out, root=src, video_mode="external")
    assert not out.exists()


def test_revision_pinned_in_copy_and_external_never_falls_back_to_hub(tmp_path: Path, monkeypatch):
    selected = _make_dataset(tmp_path / "selected", frames_per_ep=6)
    sha = "b" * 40
    calls: list[tuple[str, str]] = []

    class _FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def dataset_info(self, repo_id, revision="main"):
            calls.append(("dataset_info", f"{repo_id}@{revision}"))
            return SimpleNamespace(sha=sha)

    def _fake_snapshot_download(*, repo_id, repo_type, revision, token):
        assert repo_type == "dataset"
        calls.append(("snapshot_download", f"{repo_id}@{revision}"))
        return str(selected)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

    # 1. copy mode with explicit --revision resolves commit SHA and converts strictly from that snapshot.
    convert(tmp_path / "managed_out", repo_id="test/fixture", video_mode="copy", revision="v3.0")
    assert calls == [
        ("dataset_info", "test/fixture@v3.0"),
        ("snapshot_download", f"test/fixture@{sha}"),
    ]

    # 2. copy mode with omitted --revision preserves LeRobotDataset default (v3.0 + local cache).
    ds_calls: list[tuple[str, dict]] = []

    def _fake_lerobot_dataset(repo_id, **kwargs):
        ds_calls.append((repo_id, kwargs))
        return SimpleNamespace(root=selected)

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", _fake_lerobot_dataset)
    convert(tmp_path / "managed_default_out", repo_id="test/fixture", video_mode="copy")
    assert ds_calls == [("test/fixture", {})]

    # 2b. LeRobot 0.6.0 compatibility: explicit token without --revision works when LeRobotDataset
    # has no token= parameter (reusing $HF_LEROBOT_HOME cache if present, else downloading with token).
    class _LegacyLeRobotDataset:
        def __init__(self, repo_id):  # no token parameter (LeRobot 0.6.0)
            raise AssertionError("Should not call 0.6.0 LeRobotDataset with explicit token")

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", _LegacyLeRobotDataset)
    fake_home = tmp_path / "lerobot_home"
    copytree(selected, fake_home / "test/cached")
    monkeypatch.setattr(upstream_meta, "HF_LEROBOT_HOME", fake_home)
    calls.clear()
    convert(
        tmp_path / "managed_06_cached",
        repo_id="test/cached",
        video_mode="copy",
        source_storage_options={"token": "hf_06"},
    )
    assert calls == []
    convert(
        tmp_path / "managed_06_remote",
        repo_id="test/uncached",
        video_mode="copy",
        source_storage_options={"token": "hf_06"},
    )
    assert calls == [
        ("dataset_info", f"test/uncached@{upstream_meta.CODEBASE_VERSION}"),
        ("snapshot_download", f"test/uncached@{sha}"),
    ]

    # 3. Passing --revision with a non-hf:// --root fails loudly instead of being ignored.
    with pytest.raises(ValueError, match="--revision is only supported"):
        convert(tmp_path / "bad_rev_out", root=selected, video_mode="copy", revision=sha)

    # 4. external mode fails loudly when staged meta/ is incomplete instead of pulling another revision.
    (selected / "meta" / "tasks.parquet").unlink()

    class _StagedSource(LocalSource):
        def materialize_meta(self, staging_dir: Path) -> Path:
            copytree(self.root / "meta", staging_dir / "meta")
            return staging_dir

    def _forbid_hub(*_args, **_kwargs):
        raise AssertionError("External conversion must never fall back to Hub metadata download")

    monkeypatch.setattr("lerobot_lancedb.convert.open_source", lambda **_kw: _StagedSource(selected))
    monkeypatch.setattr(upstream_meta, "get_safe_version", _forbid_hub)
    monkeypatch.setattr(upstream_meta, "snapshot_download", _forbid_hub)

    with pytest.raises(FileNotFoundError, match="tasks.parquet"):
        convert(tmp_path / "ext_out", repo_id="test/fixture", revision="requested-sha")


def test_tail_clipped_elst_rejected_by_external_convert(tmp_path: Path):
    src = _make_dataset(tmp_path / "src", frames_per_ep=10)
    video = src / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4"
    buf = bytearray(video.read_bytes())
    elst_idx = buf.index(b"elst")
    struct.pack_into(">I", buf, elst_idx + 12, 400)
    video.write_bytes(bytes(buf))

    out = tmp_path / "out"
    with pytest.raises(DamagedSourceError, match="edit list truncates trailing samples"):
        convert(out, root=src, video_mode="external")
    assert not out.exists()


def test_remote_output_refuses_in_place_overwrite(tmp_path: Path, monkeypatch):
    src = _make_dataset(tmp_path / "src", frames_per_ep=6)

    class _ExistingRemoteDb:
        def list_tables(self, page_token=None):
            return SimpleNamespace(tables=[FRAMES_TABLE, VIDEOS_TABLE, META_TABLE], page_token=None)

    monkeypatch.setattr("lerobot_lancedb.convert._connect_output", lambda *_a, **_kw: _ExistingRemoteDb())
    for mode in ("external", "copy"):
        with pytest.raises(FileExistsError, match="already contains table"):
            convert("s3://example-bucket/already-published", root=src, video_mode=mode)


def test_local_publish_serializes_writers_and_preserves_backup_on_rollback_error(tmp_path: Path, monkeypatch):
    out = tmp_path / "out"
    build_a = tmp_path / "build_a"
    build_b = tmp_path / "build_b"
    for root in (out, build_a, build_b):
        for name in _MANAGED_OUTPUT_ENTRIES:
            (root / name).mkdir(parents=True, exist_ok=True)
            (root / name / "owner").write_text(root.name)

    # 1. Two concurrent publications serialize cleanly (even across distinct TMPDIRs)
    # and leave a coherent single-owner output.
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_commit_local_output, build_a, out)
        f2 = pool.submit(_commit_local_output, build_b, out)
        f1.result(10)
        f2.result(10)
    owners = {(out / name / "owner").read_text() for name in _MANAGED_OUTPUT_ENTRIES}
    assert len(owners) == 1 and owners <= {"build_a", "build_b"}
    assert (tmp_path / ".out.publish.lock").exists()

    # 2. If swap fails AND rollback also fails, the backup directory is preserved rather than deleted.
    build_c = tmp_path / "build_c"
    for name in _MANAGED_OUTPUT_ENTRIES:
        (build_c / name).mkdir(parents=True, exist_ok=True)
        (build_c / name / "owner").write_text("build_c")

    real_rename = Path.rename

    def _flaky_rename(self: Path, target: Path):
        if self == build_c / f"{FRAMES_TABLE}.lance":
            raise OSError("simulated publish failure")
        if self.name == "meta" and self.parent.name.startswith(".out.old-"):
            raise OSError("simulated rollback failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _flaky_rename)
    with pytest.raises(OSError, match="simulated publish failure"):
        _commit_local_output(build_c, out)
    backups = list(tmp_path.glob(".out.old-*"))
    assert len(backups) == 1
    assert (backups[0] / "meta" / "owner").exists()


def test_range_validation_and_fsspec_core_compat(tmp_path: Path, monkeypatch):
    # 1. Content-Range rejects last-byte >= total-length.
    with pytest.raises(RuntimeError, match="last byte 7 >= total length 7"):
        _validate_content_range("bytes 4-7/7", offset=4, max_length=4, label="clip.mp4")

    # 2. S3Source requires Content-Range and exact body length matching Content-Range.
    for content_range, body_bytes, match in [
        (None, b"1234", "Missing Content-Range"),
        ("bytes 4-6/20", b"1234", "exceeded Content-Range"),
        ("bytes 4-7/20", b"12", "Truncated S3 Range"),
    ]:
        body = BytesIO(body_bytes)
        resp: dict[str, object] = {"Body": body, "ResponseMetadata": {"HTTPStatusCode": 206}}
        if content_range is not None:
            resp["ContentRange"] = content_range
        fake_s3 = S3Source.__new__(S3Source)
        fake_s3._client = SimpleNamespace(get_object=lambda _r=resp, **_kw: _r, close=lambda: None)
        fake_s3.prefix = ""
        fake_s3.bucket = "test-bucket"
        with pytest.raises(RuntimeError, match=match):
            fake_s3.read_range("clip.mp4", 4, 4)
        assert body.closed

    # 3. FsspecSource uses fsspec.core.url_to_fs (compatible with fsspec==2024.2.0 / gcsfs==2024.2.0).
    monkeypatch.delattr(fsspec, "url_to_fs", raising=False)
    mem_fs = fsspec.filesystem("memory")
    prefix = f"lerobot-fsspec-compat-{tmp_path.name}"
    mem_fs.pipe(f"{prefix}/meta/info.json", b"{}")
    src = FsspecSource(f"memory://{prefix}")
    assert "meta/info.json" in src.list_files()

    # 4. GCS streaming range reader validates 206 + Content-Range and aborts on 200 without reading body.
    loop = asyncio.new_event_loop()
    loop_thread = Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:

        class _FakeAioResp:
            def __init__(self, status: int, headers: dict[str, str], body: bytes):
                self.status = status
                self.headers = headers
                self._body = body
                self.closed = False
                self.read_started = False

                class _Content:
                    async def iter_chunked(_c_self, chunk_size: int):
                        self.read_started = True
                        for i in range(0, len(self._body), chunk_size):
                            yield self._body[i : i + chunk_size]

                self.content = _Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def close(self):
                self.closed = True

            def raise_for_status(self):
                if self.status >= 400:
                    raise RuntimeError(f"HTTP {self.status}")

        resp_206 = _FakeAioResp(206, {"Content-Range": "bytes 4-9/16"}, b"456789")
        resp_200 = _FakeAioResp(200, {"Content-Length": "2097152"}, b"X" * 64)
        gcs_resps = [resp_206, resp_200]

        class _FakeGcsFs:
            requests_timeout = 10.0

            def __init__(self):
                self.loop = loop
                self.session = SimpleNamespace(get=lambda *_a, **_kw: gcs_resps.pop(0))

            async def _set_session(self):
                return self.session

            def url(self, path: str) -> str:
                return f"https://gcs.example.test/{path}"

            def _get_headers(self, headers: dict[str, str]) -> dict[str, str]:
                return dict(headers)

            def _get_params(self, kwargs: dict[str, str]) -> dict[str, str]:
                return dict(kwargs)

        fake_gcs = FsspecSource.__new__(FsspecSource)
        fake_gcs._fs = _FakeGcsFs()
        fake_gcs._root_path = "bucket/ds"
        fake_gcs.root_uri = "gs://bucket/ds"
        fake_gcs._is_gcs = True

        assert fake_gcs.read_range("videos/clip.mp4", 4, 6) == b"456789"
        with pytest.raises(RuntimeError, match="HTTP 200 instead of 206"):
            fake_gcs.read_range("videos/clip.mp4", 0, 64)
        assert resp_200.closed and not resp_200.read_started
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()


def test_s3_loopback_roundtrip_storage_options_and_special_keys(tmp_path: Path, monkeypatch):
    files: dict[str, bytes] = {}
    sources: dict[str, tuple[Path, str]] = {}
    cases = [
        ("dataset space", VIDEO_KEY),
        ("dataset_hash", f"{VIDEO_KEY}#left"),
        ("dataset_query", f"{VIDEO_KEY}?query"),
        ("dataset_pct", f"{VIDEO_KEY}%23"),
        ("dataset_utf8", f"{VIDEO_KEY}é"),
    ]
    for prefix, vkey in cases:
        src = _make_dataset(tmp_path / f"src_{prefix}", frames_per_ep=6, video_key=vkey)
        sources[prefix] = (src, vkey)
        for f in src.rglob("*"):
            if f.is_file():
                files[f"{prefix}/{f.relative_to(src).as_posix()}"] = f.read_bytes()

    with _s3_loopback_server(files) as source_ep, _s3_loopback_server({}) as wrong_ep:
        # Point default env AWS_ENDPOINT at an empty server so reads can ONLY succeed
        # if LanceDBDataset propagates storage_options to videos.lance.
        monkeypatch.setenv("AWS_ENDPOINT", wrong_ep)
        monkeypatch.setenv("AWS_ALLOW_HTTP", "true")
        monkeypatch.setenv("AWS_SKIP_SIGNATURE", "true")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy")

        source_opts = {
            "aws_endpoint": source_ep,
            "aws_region": "us-east-1",
            "aws_skip_signature": "true",
        }
        reader_global = source_opts | {"aws_allow_http": "true"}
        reader_scoped = {f"base_1.{k}": v for k, v in reader_global.items()}

        # Verify percent-encoded root URI (dataset%20space) resolves to raw key prefix 'dataset space/'.
        out_enc = tmp_path / "out_encoded_space"
        convert(out_enc, root="s3://example-bucket/dataset%20space", source_storage_options=source_opts)
        ds_global = LanceDBDataset(root=out_enc, storage_options=reader_global)
        ds_scoped = LanceDBDataset(root=out_enc, storage_options=reader_scoped)
        assert ds_global[0][VIDEO_KEY].shape == (3, 32, 32)
        assert torch.equal(ds_global[0][VIDEO_KEY], ds_scoped[0][VIDEO_KEY])

        # Verify all special video key characters round-trip pixel-identically over S3.
        for prefix, vkey in cases:
            out = tmp_path / f"out_{prefix}"
            convert(out, root=f"s3://example-bucket/{prefix}", source_storage_options=source_opts)
            ds_s3 = LanceDBDataset(root=out, storage_options=reader_scoped)
            # Compare against local external conversion of the exact same source dataset.
            local_src, _ = sources[prefix]
            out_local = tmp_path / f"local_{prefix}"
            convert(out_local, root=local_src, video_mode="external")
            ds_local = LanceDBDataset(root=out_local)
            for idx in (0, 5, 6, 11):
                assert torch.equal(ds_s3[idx][vkey], ds_local[idx][vkey])


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
    assert all(d["kind"] == 3 and d["blob_id"] == 0 for d in descriptors)
