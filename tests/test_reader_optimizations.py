"""Cold External Blob reads and batch decoding contracts."""

import pickle
from http.server import BaseHTTPRequestHandler
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
        ds = reader.LanceDBDataset(root=output, storage_options=options, return_uint8=True)
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
