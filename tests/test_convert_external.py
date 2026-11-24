"""Zero-copy External Blob v2 conversion, source adapters, and LanceDBDataset roundtrip."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb._source import FsspecSource, HubSource, LocalSource, S3Source, parse_hf_dataset_uri
from lerobot_lancedb.convert import (
    FRAMES_TABLE,
    META_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEOS_TABLE,
    DamagedSourceError,
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


def test_revision_forwarded_in_copy_and_external_never_falls_back_to_hub(tmp_path: Path, monkeypatch):
    selected = _make_dataset(tmp_path / "selected", frames_per_ep=6)

    # 1. copy mode forwards revision to LeRobotDataset.
    recorded: list[tuple[str, dict]] = []

    def _fake_lerobot_dataset(repo_id, **kwargs):
        recorded.append((repo_id, kwargs))
        return SimpleNamespace(root=selected)

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", _fake_lerobot_dataset)
    convert(tmp_path / "managed_out", repo_id="test/fixture", video_mode="copy", revision="requested-sha")
    assert recorded == [("test/fixture", {"revision": "requested-sha"})]

    # 2. external mode fails loudly when staged meta/ is incomplete instead of pulling another revision.
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
    assert all(d["kind"] == 3 and d["blob_id"] == 1 for d in descriptors)
