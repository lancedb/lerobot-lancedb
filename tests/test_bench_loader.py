"""Check observable input, replay, transport and A/B contracts of the loader benchmark."""

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import av
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("bench_loader")


def test_proxy_ranges_and_accounting(tmp_path, bench):
    keys = {
        "frames.lance/data/f": "frames",
        "videos.lance/data/v": "videos_descriptor",
        "videos/c.mp4": "video_payload",
        "data/f.parquet": "parquet",
        "meta/info.json": "other",
    }
    for key in keys:
        path = tmp_path / key
        path.parent.mkdir(parents=True)
        path.write_bytes(b"0123456789")
    with bench.range_server(tmp_path, 20) as (endpoint, traffic):
        root = f"{endpoint}/{bench.BUCKET}"
        start = time.monotonic()
        with urlopen(Request(f"{root}/videos/c.mp4", method="HEAD")) as reply:
            assert reply.headers["Content-Length"] == "10" and not reply.read()
        assert time.monotonic() - start >= 0.020
        for key in keys:
            with urlopen(Request(f"{root}/{key}", headers={"Range": "bytes=2-4"})) as reply:
                assert reply.status == 206 and reply.read() == b"234"
                assert reply.headers["Content-Range"] == "bytes 2-4/10"
        for value, expected in [("bytes=-2", b"89"), ("bytes=7-", b"789")]:
            with urlopen(Request(f"{root}/videos/c.mp4", headers={"Range": value})) as reply:
                assert reply.read() == expected
        with pytest.raises(HTTPError) as error:
            urlopen(Request(f"{root}/videos/c.mp4", headers={"Range": "bytes=10-"}))
        assert error.value.code == 416
        counts = traffic.snapshot()
        assert counts["total"] == {"gets": 8, "bytes": 20}
        for kind in keys.values():
            assert counts[kind] == (
                {"gets": 4, "bytes": 8} if kind == "video_payload" else {"gets": 1, "bytes": 3}
            )


@pytest.mark.parametrize(
    "workload,shape,gop", [("droid-short-gop", (320, 180, 3), 2), ("long-gop-720p", (1280, 720, 2), 30)]
)
def test_synthetic_clips(tmp_path, bench, workload, shape, gop):
    source = bench.source_path(workload, tmp_path, None, None)
    meta = bench.LeRobotDatasetMetadata(f"local/{workload}", root=source)
    assert meta.total_episodes >= 2 and len(meta.video_keys) == shape[2]
    for key in meta.video_keys:
        for ep in range(meta.total_episodes):
            with av.open(str(source / meta.get_video_file_path(ep, key))) as container:
                stream = container.streams.video[0]
                assert (stream.width, stream.height) == shape[:2]
                indices = [i for i, pkt in enumerate(container.demux(stream)) if pkt.is_keyframe]
                assert len(indices) >= 2 and indices == list(range(0, stream.frames, gop))


def test_loaders_and_compare(tmp_path, bench, monkeypatch, capsys):
    output = tmp_path / "report.json"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "bench_loader.py"),
            "--workload",
            "droid-short-gop",
            "--warmup-steps",
            "1",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--rtt-ms",
            "1",
            "--episode-pool",
            "--lookahead-batches",
            "2",
            "--verify-correctness",
            "--output-json",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text())
    parquet, local, hybrid, control, lookahead, streaming, replay, pool = report["rows"]
    assert lookahead["anchor_batches"] == control["anchor_batches"] == hybrid["anchor_batches"]
    assert pool["anchor_batches"] == replay["anchor_batches"]
    assert parquet["anchor_batches"] == local["anchor_batches"] == hybrid["anchor_batches"]
    assert streaming["anchor_batches"] == replay["anchor_batches"]
    assert parquet["traffic"]["total"]["gets"] == local["traffic"]["total"]["gets"] == 0
    assert all(row["traffic"]["video_payload"]["bytes"] > 0 for row in (hybrid, streaming, replay))
    assert streaming["traffic"]["parquet"]["bytes"] > 0
    assert all(row["measured_samples"] == 2 and row["steady_samples_per_s"] > 0 for row in report["rows"])
    candidate = tmp_path / "candidate.json"
    parquet["steady_samples_per_s"] += 12
    candidate.write_text(json.dumps(report))
    bench.main(["--compare", str(output), str(candidate)])
    assert "+12.00" in capsys.readouterr().out
    parquet["anchor_batches"][0].reverse()
    candidate.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Unmatched data or anchors"):
        bench.main(["--compare", str(output), str(candidate)])
