"""Range-Read MP4 byte-index parity against PyAV and container guardrails."""

from __future__ import annotations

import struct
from pathlib import Path

import av
import numpy as np
import pytest

from lerobot_lancedb.mp4_index import DEFAULT_HEADER_PROBE_BYTES, probe_local_mp4, probe_mp4_byte_index
from lerobot_lancedb.reader import build_video_byte_index


def _encode_mp4(path: Path, frames: int = 24, *, faststart: bool = False, bframes: int = 0) -> None:
    container_options = {"movflags": "faststart"} if faststart else {}
    with av.open(str(path), "w", options=container_options) as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "6", "bf": str(bframes), "preset": "ultrafast"}
        for i in range(frames):
            img = np.full((32, 32, 3), (i * 11) % 255, dtype=np.uint8)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


@pytest.mark.parametrize(
    ("name", "faststart", "bframes", "probe_bytes", "expected_gets"),
    [
        # Small probe forces the 2-GET tail-moov jump over mdat.
        ("tail_moov_jump", False, 0, 64, 2),
        # Default 1 MiB probe reads small faststart MP4s in 1 GET.
        ("faststart_single_get", True, 0, DEFAULT_HEADER_PROBE_BYTES, 1),
        # Small probe on faststart exercises incremental prefix expansion.
        ("faststart_expand", True, 0, 64, 2),
        # Open-GOP / B-frame stream exercises ctts + presentation-order sort + edit list.
        ("open_gop_bframes", True, 2, DEFAULT_HEADER_PROBE_BYTES, 1),
    ],
)
def test_probe_matches_pyav_across_layouts(
    tmp_path: Path,
    name: str,
    faststart: bool,
    bframes: int,
    probe_bytes: int,
    expected_gets: int,
):
    mp4 = tmp_path / f"{name}.mp4"
    _encode_mp4(mp4, frames=24, faststart=faststart, bframes=bframes)

    expected = build_video_byte_index(mp4)
    probed = probe_local_mp4(mp4, header_probe_bytes=probe_bytes)

    assert probed.to_index_dict() == expected
    assert probed.sample_count == 24
    assert probed.faststart is faststart
    assert probed.range_gets == expected_gets


def test_probe_rejects_fragmented_and_multi_mdat_and_handles_mdat_size_zero(tmp_path: Path):
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"isom\x00\x00\x02\x00"

    # 1. Fragmented MP4 (top-level moof) is rejected loudly.
    moof = struct.pack(">I4s", 8, b"moof")
    frag = ftyp + moof
    with pytest.raises(ValueError, match="fragmented MP4"):
        probe_mp4_byte_index(
            "frag.mp4",
            lambda _p, off, n: frag[off : off + n],
            file_size=len(frag),
        )

    # 2. Fragmented MP4 signalled via mvex inside moov is rejected even if moof is past probe window.
    mvex = struct.pack(">I4s", 8, b"mvex")
    moov_with_mvex = struct.pack(">I4s", 8 + len(mvex), b"moov") + mvex
    mdat = struct.pack(">I4s", 12, b"mdat") + b"\x00\x00\x00\x00"
    frag_mvex = ftyp + moov_with_mvex + mdat
    with pytest.raises(ValueError, match="mvex"):
        probe_mp4_byte_index(
            "frag_mvex.mp4",
            lambda _p, off, n: frag_mvex[off : off + n],
            file_size=len(frag_mvex),
        )

    # 3. Multiple mdat boxes are rejected loudly.
    multi = ftyp + mdat + mdat
    with pytest.raises(ValueError, match="multiple mdat"):
        probe_mp4_byte_index(
            "multi.mp4",
            lambda _p, off, n: multi[off : off + n],
            file_size=len(multi),
        )

    # 4. Faststart MP4 whose mdat has size==0 (extends to EOF) parses cleanly.
    mp4 = tmp_path / "faststart.mp4"
    _encode_mp4(mp4, frames=12, faststart=True)
    raw = bytearray(mp4.read_bytes())
    mdat_pos = raw.index(b"mdat") - 4
    struct.pack_into(">I", raw, mdat_pos, 0)
    raw_bytes = bytes(raw)

    probed = probe_mp4_byte_index(
        "mdat_zero.mp4",
        lambda _p, off, n: raw_bytes[off : off + n],
        file_size=len(raw_bytes),
    )
    assert probed.to_index_dict() == build_video_byte_index(mp4)
    assert probed.sample_count == 12
