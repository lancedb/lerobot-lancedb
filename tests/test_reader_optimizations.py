"""Cold External Blob reads and batch decoding contracts."""

import importlib
import json
import pickle
import subprocess
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

import av
import lancedb
import numpy as np
import pytest
import torch
from test_convert_external import FPS, VIDEO_KEY, _make_dataset, _s3_loopback_server
from torchcodec.decoders import VideoDecoder

from lerobot_lancedb import reader
from lerobot_lancedb.convert import convert


@pytest.fixture(scope="module")
def indexed_videos(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("indexed-videos")
    frame_count = 900
    src = _make_dataset(tmp / "src", frames_per_ep=frame_count)
    paths = sorted((src / "videos").rglob("*.mp4"))
    # Noise keeps the target GOP beyond the open probe. Disable scene cuts so
    # both encoders honor the specified GOP length on identical input frames.
    for path, gop in zip(paths, (4, 60), strict=True):
        rng = np.random.default_rng(27)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=FPS)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            stream.options = {"g": str(gop), "sc_threshold": "0", "crf": "12", "preset": "ultrafast"}
            for _ in range(frame_count):
                rgb = rng.integers(256, size=(32, 32, 3), dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    output = tmp / "converted"
    convert(output, root=src, video_base_uri="s3://example-bucket/videos")
    files = {p.relative_to(src).as_posix(): p.read_bytes() for p in paths}
    with _s3_loopback_server(files) as endpoint:
        options = {
            "aws_endpoint": endpoint,
            "aws_allow_http": "true",
            "aws_skip_signature": "true",
            "aws_region": "us-east-1",
        }
        yield output, paths, options, frame_count


@pytest.mark.parametrize("episode", [0, 1], ids=["short-gop", "long-gop"])
def test_cold_external_read(indexed_videos, episode, monkeypatch):
    output, paths, options, frame_count = indexed_videos
    indices = [500, 501, 500]
    reference = VideoDecoder(str(paths[episode]), seek_mode="approximate").get_frames_at(indices=indices)
    rounds = []
    responses = []
    fetch = lancedb.table.LanceTable.fetch_blob_ranges
    send_header = BaseHTTPRequestHandler.send_header

    def count_fetch(table, column, ranges):
        rounds.append(list(ranges))
        return fetch(table, column, ranges)

    def count_response(handler, header, value):
        if handler.command == "GET" and header == "Content-Length" and handler.path.endswith(".mp4"):
            responses.append((handler.headers.get("Range"), int(value)))
        return send_header(handler, header, value)

    monkeypatch.setattr(lancedb.table.LanceTable, "fetch_blob_ranges", count_fetch)
    monkeypatch.setattr(BaseHTTPRequestHandler, "send_header", count_response)

    def read_cold():
        rounds.clear()
        responses.clear()
        ds = reader.LanceDBDataset(root=output, storage_options=options, return_uint8=True, use_mini_mp4=False)
        items = ds.__getitems__([episode * frame_count + i for i in indices])
        for pos, item in enumerate(items):
            assert torch.equal(item[VIDEO_KEY], reference.data[pos])
            assert torch.equal(item["timestamp"], reference.pts_seconds[pos].float())
        _, source = ds._decoder_cache.get((VIDEO_KEY, 0, episode))
        assert source.fallback_bytes == 0
        assert responses and all(byte_range for byte_range, _ in responses)
        return len(rounds), sum(size for _, size in responses)

    rounds_used, bytes_used = read_cold()
    print(f"episode={episode} rounds={rounds_used} bytes={bytes_used}")
    assert rounds_used == 1
    if episode == 0:
        monkeypatch.setattr(reader, "_range_slack", lambda *args: 64 * 1024)
        rounds_fixed, bytes_fixed = read_cold()
        print(f"fixed slack bytes={bytes_fixed}")
        assert rounds_fixed == 1
        assert bytes_used < bytes_fixed


def test_cold_multi_file_mini_ranges(indexed_videos, monkeypatch):
    output, paths, options, count = indexed_videos
    responses = []
    handles = []
    send_header = BaseHTTPRequestHandler.send_header
    fetch_handles = lancedb.table.LanceTable.fetch_blob_files

    def record(handler, header, value):
        if handler.command == "GET" and header == "Content-Length" and handler.path.endswith(".mp4"):
            start, end = handler.headers["Range"].removeprefix("bytes=").split("-")
            responses.append((handler.path, int(start), int(end), int(value)))
        return send_header(handler, header, value)

    def open_files(table, column, ids):
        handles.extend(ids)
        return fetch_handles(table, column, ids)

    monkeypatch.setattr(BaseHTTPRequestHandler, "send_header", record)
    monkeypatch.setattr(lancedb.table.LanceTable, "fetch_blob_files", open_files)
    indices = [count + 501, 500, count + 501]
    baseline = reader.LanceDBDataset(
        root=output, storage_options=options, use_mini_mp4=False, return_uint8=True
    ).__getitems__(indices)
    baseline_gets, baseline_bytes = len(responses), sum(size for *_, size in responses)
    assert len(handles) == 2
    responses.clear()
    handles.clear()
    ds = reader.LanceDBDataset(
        root=output, storage_options=options, return_uint8=True, columns=[VIDEO_KEY, "timestamp"]
    )
    results = ds.__getitems__(indices)
    assert len(responses) == 2 < baseline_gets
    assert not handles and not ds._mini_ineligible
    bytes_used = sum(size for *_, size in responses)
    assert bytes_used < baseline_bytes
    for name, start, end, _ in responses:
        path = next(p for p in paths if name.endswith(p.name))
        meta = reader.build_video_byte_index(path)
        assert end < meta["moov_offset"] or start >= meta["moov_offset"] + meta["moov_size"]
    print(f"Cold files: {baseline_gets} -> 2 GETs, {baseline_bytes} -> {bytes_used} bytes")
    responses.clear()
    repeat = ds.__getitems__(indices)
    assert not responses and not handles
    for first, second, expected in zip(results, repeat, baseline, strict=True):
        for key in first:
            assert torch.equal(first[key], expected[key]) and torch.equal(first[key], second[key])
    restored = pickle.loads(pickle.dumps(ds))
    assert restored._decoder_cache._total_bytes == restored._index_cache._total_bytes == 0


def test_mini_pts_mismatch_reopens_only_failed_file(indexed_videos, monkeypatch):
    root, paths, options, count = indexed_videos
    ds = reader.LanceDBDataset(root=root, storage_options=options, return_uint8=True)
    real_decoder = VideoDecoder
    opened = []

    def open_decoder(source, **kwargs):
        decoder = real_decoder(source, **kwargs)
        if isinstance(source, bytes):
            # Only one mini-container has one frame. Corrupt its returned PTS
            # while keeping its pixels, so the same-batch retry is observable.
            if decoder.metadata.num_frames == 1:
                get_frames = decoder.get_frames_at

                def wrong_pts(**kwargs):
                    batch = get_frames(**kwargs)
                    batch.pts_seconds.add_(1.0)
                    return batch

                decoder.get_frames_at = wrong_pts
        else:
            opened.append(source)
        return decoder

    monkeypatch.setattr("torchcodec.decoders.VideoDecoder", open_decoder)
    items = ds.__getitems__([500, count + 501])
    for item, path, frame in zip(items, paths, [500, 501], strict=True):
        expected = real_decoder(path).get_frames_at([frame])
        assert torch.equal(item[VIDEO_KEY], expected.data[0])
        assert torch.equal(item["timestamp"], expected.pts_seconds[0].float())
    assert ds._mini_ineligible == {(VIDEO_KEY, 0, 0)}
    assert len(opened) == 1 and isinstance(opened[0], reader._SparseBlobSource)


def test_open_gop_falls_back_to_sparse_decoder(tmp_path):
    src = _make_dataset(tmp_path / "source", frames_per_ep=40)
    path = sorted((src / "videos").rglob("*.mp4"))[0]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=32x32:rate={FPS}",
            "-frames:v",
            "40",
            "-c:v",
            "libx264",
            "-g",
            "8",
            "-bf",
            "3",
            "-x264-params",
            "open-gop=1:b-adapt=0:scenecut=0",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
    )
    out = tmp_path / "managed"
    convert(out, root=src, video_mode="copy")
    key = (VIDEO_KEY, 0, 0)
    ds = reader.LanceDBDataset(root=out, return_uint8=True)
    index = reader.ParsedSampleIndex.parse(
        lancedb.connect(out)
        .open_table("videos")
        .take_row_ids([0])
        .select(["sample_index"])
        .to_arrow()[0][0]
        .as_py()
    )
    sync = int(index.sync[1])
    ordinal = index.order.tolist().index(sync)
    with pytest.raises(ValueError, match="leading pictures"):
        index.spans([ordinal])
    start, end = index.byte_span(sync, sync)
    with pytest.raises(ValueError, match="IDR"):
        index._check_idr(path.read_bytes()[start:end])
    ids = [ordinal, ordinal + 2, ordinal, 1]
    reference = VideoDecoder(path).get_frames_at(indices=ids)
    for pos, result in enumerate(ds.__getitems__(ids)):
        assert torch.equal(result[VIDEO_KEY], reference.data[pos])
        assert torch.equal(result["timestamp"], reference.pts_seconds[pos].float())
    assert key in ds._mini_ineligible and key in ds._decoder_cache


@pytest.mark.parametrize("batch_size", [2, 64])
def test_rgb_deduplicates_and_scatters_in_sample_order(indexed_videos, monkeypatch, batch_size):
    output, paths, options, _ = indexed_videos
    ds = reader.LanceDBDataset(
        root=output, storage_options=options, delta_timestamps={VIDEO_KEY: [-0.1, 0.0, 0.1]}
    )
    expected_indices = [[1, 2, 3], [0, 0, 1], [1, 2, 3]]
    decoder = VideoDecoder(str(paths[0]), seek_mode="approximate")
    expected = [decoder.get_frames_at(indices=indices).data / 255.0 for indices in expected_indices]
    calls = []
    get_frames = VideoDecoder.get_frames_at

    def count_decode(self, *, indices):
        calls.append(indices)
        return get_frames(self, indices=indices)

    monkeypatch.setattr(reader, "_RGB_DECODE_MAX_FRAMES", batch_size)
    monkeypatch.setattr(VideoDecoder, "get_frames_at", count_decode)
    items = ds.__getitems__([2, 0, 2])
    assert calls == ([[0, 1], [2, 3]] if batch_size == 2 else [[0, 1, 2, 3]])
    for item, frames in zip(items, expected, strict=True):
        assert torch.equal(item[VIDEO_KEY], frames)


def test_depth_projection_and_deltas_in_spawned_worker(tmp_path):
    path = _make_dataset(tmp_path / "source")
    info_file = path / "meta/info.json"
    info = json.loads(info_file.read_text())
    feature = info["features"][VIDEO_KEY]
    feature.update(shape=[32, 32, 1], info={"is_depth_map": True, "depth_unit": "mm"})
    info_file.write_text(json.dumps(info))
    for clip in (path / "videos").rglob("*.mp4"):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=32x32:rate={FPS}",
                "-frames:v",
                "10",
                "-pix_fmt",
                "gray12le",
                "-c:v",
                "libx265",
                "-threads",
                "1",
                "-x265-params",
                "lossless=1:pools=none:frame-threads=1:log-level=error",
                str(clip),
            ],
            check=True,
        )
    out = tmp_path / "managed"
    convert(out, root=path, video_mode="copy")
    options = dict(root=out, columns=[VIDEO_KEY], delta_timestamps={VIDEO_KEY: [-0.1, 0.0]})
    ds = reader.LanceDBDataset(**options)
    expected = reader.LanceDBDataset(**options, use_mini_mp4=False).__getitems__([0, 1])
    ds[0]
    assert ds._depth_keys == [VIDEO_KEY] and ds._index_cache._total_bytes == 0
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=[[0, 1]], num_workers=1, multiprocessing_context="spawn"
    )
    (batch,) = loader
    assert batch[VIDEO_KEY].shape == (2, 2, 1, 32, 32)
    assert set(batch) == {VIDEO_KEY, VIDEO_KEY + "_is_pad"}
    for key in batch:
        assert torch.equal(batch[key], torch.stack([row[key] for row in expected]))


def test_depth_merges_adjacent_windows_within_budget(monkeypatch):
    ds = reader.LanceDBDataset.__new__(reader.LanceDBDataset)
    ds.meta = SimpleNamespace(fps=10, features={"depth": {"shape": [1, 32, 32]}})
    ds.tolerance_s = 1e-4
    monkeypatch.setattr(reader, "_DEPTH_DECODE_MAX_SPAN_FRAMES", 4)
    calls = []

    def decode(_source, timestamps, _file_key):
        calls.append(timestamps)
        return torch.tensor(timestamps).reshape(-1, 1, 1, 1)

    monkeypatch.setattr(ds, "_decode_depth_frames", decode)
    requests = [(0, [0.2, 0.1, 0.2]), (1, [0.0, 0.1]), (2, [0.3]), (3, [0.4]), (4, [4.0])]
    decoded = dict(ds._decode_depth_file_requests(None, requests, ("depth", 0, 0)))
    assert calls == [[0.0, 0.1, 0.2, 0.3], [0.4], [4.0]]
    for sample_idx, timestamps in requests:
        assert torch.equal(decoded[sample_idx], torch.tensor(timestamps).reshape(-1, 1, 1, 1).squeeze(0))


def test_episode_subset_mapping(indexed_videos):
    output, _, options, n = indexed_videos
    ds = reader.LanceDBDataset(root=output, storage_options=options, episodes=[1, 1])
    mapping = ds.absolute_to_relative_idx
    assert len(ds) == 2 * n and len(mapping) == n
    assert list(mapping) == list(range(n, 2 * n))
    assert mapping[n] == n and mapping[2 * n - 1] == 2 * n - 1
    assert [item["index"].item() for item in ds.__getitems__([0, n, -1])] == [n, n, 2 * n - 1]
    with pytest.raises(KeyError):
        _ = mapping[n - 1]
    with pytest.raises(IndexError):
        _ = ds[2 * n]
    # Worker serialization stays proportional to intervals even for billion-frame episodes.
    huge = type(mapping)(np.array([0, 3_000_000_000]), np.array([1_000_000_000, 4_000_000_000]))
    assert huge[3_000_000_001] == 1_000_000_001
    assert huge.rel_to_abs(-1) == 3_999_999_999
    assert len(pickle.dumps(huge)) < 1024


SHA = "ABCDEF0123456789" * 2 + "ABCDEF01"
HF_URI = "hf://datasets/test/robot"


@pytest.mark.parametrize(
    ("uri", "revision", "options", "expected"),
    [
        (HF_URI + "@" + SHA + "/dir", None, {}, "true"),
        (HF_URI, SHA, {}, "true"),
        (HF_URI, None, {"revision": SHA}, "true"),
        (HF_URI, SHA, {"hf_enable_resolve_cache": "false"}, "false"),
        (HF_URI, SHA, {"revision": "main"}, None),
        (HF_URI + "@main", None, {}, None),
        (HF_URI + "/subdir@" + SHA, None, {}, None),
        (HF_URI, SHA[:12], {}, None),
        ("s3://bucket/dataset", SHA, {}, None),
    ],
)
def test_resolve_cache_only_for_immutable_hub_revisions(uri, revision, options, expected):
    original = options.copy()
    actual = reader._storage_options(uri, options, revision, token=False)
    assert actual.get("hf_enable_resolve_cache") == expected
    assert options == original
    assert "token" not in actual


def test_remote_metadata_cache_and_per_table_routing(indexed_videos, tmp_path, monkeypatch):
    output, _, blob_options, _ = indexed_videos
    local_db = lancedb.connect(output)
    opens = []
    connections = []

    def open_table(name, *, storage_options):
        opens.append((name, storage_options.copy()))
        return local_db.open_table(name, storage_options=storage_options)

    def connect(uri, *, storage_options):
        assert uri == HF_URI
        connections.append(storage_options.copy())
        return SimpleNamespace(open_table=open_table)

    monkeypatch.setattr(reader, "HF_LEROBOT_HOME", tmp_path / "cache")
    monkeypatch.setattr(reader.lancedb, "connect", connect)
    scoped = {f"base_1.{key}": value for key, value in blob_options.items()}
    options = {**scoped, "aws_endpoint": "https://a.invalid"}
    ds = reader.LanceDBDataset(root=HF_URI, storage_options=options, revision=SHA, token="explicit")
    assert ds[0][VIDEO_KEY].shape == (3, 32, 32)
    expected = {**options, "token": "explicit", "revision": SHA, "hf_enable_resolve_cache": "true"}
    assert connections == [expected, expected]
    assert opens == [(table, expected) for table in ("meta", "frames", "videos")]
    assert "token" not in options
    _, rotated = reader.resolve_lance_root(None, HF_URI, options, SHA, "rotated")
    _, new_endpoint = reader.resolve_lance_root(
        None, HF_URI, options | {"aws_endpoint": "https://b.invalid"}, SHA
    )
    _, new_revision = reader.resolve_lance_root(None, HF_URI, options, "main")
    assert rotated == ds.root
    assert len({ds.root, new_endpoint, new_revision}) == 3


def test_repo_id_routes_hub_credentials_and_revision(indexed_videos, monkeypatch):
    output, _, _, _ = indexed_videos
    downloads = []

    def snapshot(repo_id, **kwargs):
        downloads.append((repo_id, kwargs["token"], kwargs["revision"]))
        return str(output)

    monkeypatch.setattr(reader.huggingface_hub, "snapshot_download", snapshot)
    ds = reader.LanceDBDataset(repo_id="test/robot", storage_options={"token": "supplied", "revision": SHA})
    assert downloads == [("test/robot", "supplied", SHA)]
    assert ds.meta.revision == SHA and ds._storage_options["hf_enable_resolve_cache"] == "true"


@pytest.mark.parametrize(
    "options", [{}, {"revision": "main"}, {"hf_revision": "v3.0"}, {"hf_enable_resolve_cache": "false"}]
)
def test_mutable_ref_uses_resolved_snapshot(indexed_videos, tmp_path, monkeypatch, options):
    output, _, _, _ = indexed_videos
    snapshot = tmp_path / "snapshots" / SHA
    snapshot.mkdir(parents=True)
    (snapshot / "meta").symlink_to(output / "meta")
    monkeypatch.setattr(reader.huggingface_hub, "snapshot_download", lambda *a, **kw: str(snapshot))
    ds = reader.LanceDBDataset(repo_id="test/robot", revision="main", storage_options=options, token=False)
    assert ds._storage_options["revision"] == SHA and "hf_revision" not in ds._storage_options
    assert ds._storage_options["hf_enable_resolve_cache"] == options.get("hf_enable_resolve_cache", "true")


def test_only_whole_local_objects_qualify(tmp_path):
    path = tmp_path / "clip with space.mp4"
    path.write_bytes(b"1234")
    descriptor = dict(kind=3, blob_id=0, position=0, size=4, blob_uri=path.as_uri())
    assert (
        reader._local_video_path(descriptor, 4)
        == reader._local_video_path(descriptor | {"size": 0}, 4)
        == path
    )
    for override in [
        {"position": 1},
        {"size": 3},
        {"size": 5},
        {"blob_id": 1},
        {"kind": 1},
        {"blob_uri": "videos/old.mp4", "blob_id": 1},
        {"blob_uri": "s3://bucket/clip.mp4"},
        {"blob_uri": "file://server/clip.mp4"},
        {"blob_uri": tmp_path.as_uri()},
    ]:
        assert reader._local_video_path(descriptor | override, 4) is None
    assert reader._local_video_path({"kind": 3}, 4) is None
    assert reader._local_video_path(None, 4) is None
    path.write_bytes(b"short")
    assert reader._local_video_path(descriptor, 4) is None
    path.unlink()
    assert reader._local_video_path(descriptor, 4) is None


def test_external_hf_unprefixed_cache_option():
    key = "base_1." + "enable_resolve_cache"
    options = {key: "false"}
    reader._apply_external_hf_options(options, {"1": {"uri": HF_URI + "@" + SHA}}, False)
    assert options[key] == "false" and "base_1.hf_enable_resolve_cache" not in options


@pytest.mark.parametrize("legacy", [True, False], ids=["nine-columns", "null-index"])
def test_relative_local_base_uses_sparse_decoder(tmp_path, monkeypatch, legacy):
    source = _make_dataset(tmp_path / "source")
    output = tmp_path / "old-layout"
    monkeypatch.setattr(
        "lerobot_lancedb.convert._external_table_options",
        lambda uri, *_: {"initial_bases": [lancedb.BasePath(path=uri, name="source_videos", id=1)]},
    )
    convert(output, root=source, video_mode="external", write_sample_index=False)
    table = lancedb.connect(output).open_table("videos")
    if legacy:
        table.drop_columns(["sample_index"])
    else:
        assert table.search().select(["sample_index"]).to_arrow()[0].null_count == 2
    expected = VideoDecoder(str(sorted((source / "videos").rglob("*.mp4"))[0])).get_frames_at(indices=[1])
    ds = reader.LanceDBDataset(root=output, return_uint8=True)
    assert torch.equal(ds[1][VIDEO_KEY], expected.data[0])
    assert isinstance(ds._decoder_cache.get((VIDEO_KEY, 0, 0))[1], reader._SparseBlobSource)


def test_direct_open_failure_falls_back_to_blob(tmp_path, monkeypatch):
    src = _make_dataset(tmp_path / "source")
    output = tmp_path / "out"
    convert(output, root=src, video_mode="external")
    path = sorted((src / "videos").rglob("*.mp4"))[0]
    expected = VideoDecoder(str(path)).get_frames_at(indices=[3])
    attempts = []

    def open_decoder(source, **kwargs):
        attempts.append(source)
        if isinstance(source, str):
            raise OSError("native opener failed")
        return VideoDecoder(source, **kwargs)

    monkeypatch.setattr("torchcodec.decoders.VideoDecoder", open_decoder)
    ds = reader.LanceDBDataset(root=output, return_uint8=True)
    item = ds[3]
    assert attempts[0] == str(path) and isinstance(attempts[1], reader._SparseBlobSource)
    assert torch.equal(item[VIDEO_KEY], expected.data[0])
    assert torch.equal(item["timestamp"], expected.pts_seconds[0].float())
    # Depth uses PyAV over a sparse source even for an eligible whole file.
    ds = reader.LanceDBDataset(root=output)
    ds._depth_keys = [VIDEO_KEY]
    ds._ensure_open()
    key = (VIDEO_KEY, 0, 0)
    decoder, source = ds._prepare_files([key])[key]
    assert decoder is None and isinstance(source, reader._SparseBlobSource)


@pytest.mark.parametrize("cache_option", [None, "base_1.hf_enable_resolve_cache", "hf_enable_resolve_cache"])
def test_hybrid_hub_pinned_range_reads(tmp_path, monkeypatch, cache_option):
    src = _make_dataset(tmp_path / "source")
    output = tmp_path / "hybrid"
    base_uri = f"{HF_URI}@{SHA}/videos"
    convert(output, root=src, video_base_uri=base_uri)
    meta = reader.LeRobotDatasetMetadata("test/robot", root=output)
    assert meta.total_frames == 20
    assert json.loads((output / "meta/info.json").read_text())["lance_external_bases"] == {
        "1": {"uri": base_uri, "hf_revision": SHA}
    }
    paths = sorted((src / "videos").rglob("*.mp4"))
    files = {path.relative_to(src).as_posix(): path.read_bytes() for path in paths}
    expected = VideoDecoder(str(paths[0])).get_frames_at(indices=[4])
    requests = []
    send_response = BaseHTTPRequestHandler.send_response

    def record(handler, *args):
        if handler.command == "GET":
            requests.append(
                (handler.path, handler.headers.get("Range"), handler.headers.get("Authorization"))
            )
        return send_response(handler, *args)

    monkeypatch.setattr(BaseHTTPRequestHandler, "send_response", record)
    prefix = f"datasets/test/robot/resolve/{SHA}"
    with _s3_loopback_server(files, bucket=prefix) as endpoint:
        # The LanceDB 0.42.0b26 HF provider takes its transport endpoint from the environment.
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        options = {"hf_endpoint": endpoint}
        if cache_option:
            options[cache_option] = "false"
        ds = reader.LanceDBDataset(root=output, storage_options=options, token="fixture", return_uint8=True)
        assert ds._storage_options["base_1.hf_enable_resolve_cache"] == ("false" if cache_option else "true")
        assert ds._storage_options["base_1.hf_endpoint"] == endpoint
        assert ds._storage_options["base_1.revision"] == SHA
        item = ds[4]
    assert torch.equal(item[VIDEO_KEY], expected.data[0])
    assert torch.equal(item["timestamp"], expected.pts_seconds[0].float())
    assert requests and all(
        path.startswith(f"/{prefix}/videos/") and rng and auth == "Bearer fixture"
        for path, rng, auth in requests
    )


def test_projection_skips_unused_cameras_and_tabular_columns(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    source = importlib.import_module("bench_workloads").synthetic(tmp_path / "src", "droid-short-gop")
    output = tmp_path / "out"
    convert(output, root=source, video_mode="external")
    cam = "observation.images.cam1"
    delta = {cam: [-1 / 15, 0.0], "action": [-1 / 15, 0.0]}
    indices = [2, 0, 2]
    ref = reader.LanceDBDataset(root=output, delta_timestamps=delta).__getitems__(indices)
    # Removing unused videos catches any accidental open, including the sparse fallback.
    for path in (source / "videos").rglob("*.mp4"):
        if cam not in path.parts:
            path.unlink()
    ds = reader.LanceDBDataset(root=output, columns=[cam, "action"], delta_timestamps=delta)
    assert set(ds.features) == {cam, "action"}
    assert set(ds._fetch_columns) == {"action", "timestamp"}
    for item, full in zip(ds.__getitems__(indices), ref, strict=True):
        assert set(item) == {cam, "action", cam + "_is_pad", "action_is_pad"}
        assert all(torch.equal(item[k], full[k]) for k in item)

    (output / "videos.lance").rename(output / "hidden-videos")
    for columns in (["action"], ["task", "action"], ["task"]):
        ds = reader.LanceDBDataset(root=output, columns=columns)
        item = ds[0]
        assert set(item) == set(columns)
        if "task" in columns:
            assert item["task"] == ref[1]["task"]
        assert ds._videos_table is ds._decode_pool is ds._prefetch_pool is None
    for columns, deltas, error in [
        ([], {}, "non-empty"),
        (["typo"], {}, "Unknown"),
        (["task"], delta, "missing"),
    ]:
        with pytest.raises(ValueError, match=error):
            reader.LanceDBDataset(root=output, columns=columns, delta_timestamps=deltas)
    info_file = output / "meta/info.json"
    info = json.loads(info_file.read_text())
    info["features"]["still"] = {"dtype": "image", "shape": [32, 32, 3], "names": None}
    info_file.write_text(json.dumps(info))
    assert set(reader.LanceDBDataset(root=output, columns=["task"])[0]) == {"task"}
    with pytest.raises(NotImplementedError, match="Image-backed"):
        reader.LanceDBDataset(root=output, columns=["still"])
