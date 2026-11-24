"""Range-Read MP4 byte-index parity against PyAV and container guardrails."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from lerobot_lancedb.mp4_index import (
    DEFAULT_HEADER_PROBE_BYTES,
    DEFAULT_MAX_PROBE_BYTES,
    _parse_stz2,
    probe_local_mp4,
    probe_mp4_byte_index,
)
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


def test_probe_rejects_truncated_faststart_and_trailing_mdat(tmp_path: Path):
    mp4 = tmp_path / "faststart.mp4"
    _encode_mp4(mp4, frames=12, faststart=True)
    raw = mp4.read_bytes()
    orig = probe_local_mp4(mp4)

    # 1. Truncated mid-mdat (explicit mdat size exceeds file_size).
    trunc_len = int(orig.kf_positions[-1]) - 1
    trunc = raw[:trunc_len]
    with pytest.raises(ValueError, match="truncated MP4"):
        probe_mp4_byte_index(
            "trunc.mp4",
            lambda _p, off, n: trunc[off : off + n],
            file_size=len(trunc),
        )

    # 2. Truncated mid-mdat when mdat has size==0 (detected via sample byte bounds).
    trunc_zero = bytearray(trunc)
    mdat_pos = trunc_zero.index(b"mdat") - 4
    struct.pack_into(">I", trunc_zero, mdat_pos, 0)
    trunc_zero_bytes = bytes(trunc_zero)
    with pytest.raises(ValueError, match="truncated MP4"):
        probe_mp4_byte_index(
            "trunc_zero.mp4",
            lambda _p, off, n: trunc_zero_bytes[off : off + n],
            file_size=len(trunc_zero_bytes),
        )

    # 3. Trailing second mdat past initial header probe window is still rejected.
    with_trailing_mdat = raw + struct.pack(">I4s", 8, b"mdat")
    with pytest.raises(ValueError, match="multiple mdat"):
        probe_mp4_byte_index(
            "trailing_mdat.mp4",
            lambda _p, off, n: with_trailing_mdat[off : off + n],
            file_size=len(with_trailing_mdat),
            header_probe_bytes=min(1024, len(raw) // 2),
        )

    # 4. Trailing second mdat after a >64 MiB free box is detected by jumping box headers
    # without reading the padding payload.
    free_size = DEFAULT_MAX_PROBE_BYTES + 8
    free_hdr = struct.pack(">I4s", free_size, b"free")
    tail_mdat = struct.pack(">I4s", 8, b"mdat")
    virtual_size = len(raw) + free_size + len(tail_mdat)
    total_read = 0

    def _virtual_read(_p: str, off: int, length: int) -> bytes:
        nonlocal total_read
        end = min(off + length, virtual_size)
        out = bytearray()
        pos = off
        while pos < end:
            if pos < len(raw):
                take = min(end, len(raw)) - pos
                out.extend(raw[pos : pos + take])
            elif pos < len(raw) + len(free_hdr):
                start_in = pos - len(raw)
                take = min(end - pos, len(free_hdr) - start_in)
                out.extend(free_hdr[start_in : start_in + take])
            elif pos < len(raw) + free_size:
                take = min(end, len(raw) + free_size) - pos
                out.extend(b"\x00" * take)
            else:
                start_in = pos - (len(raw) + free_size)
                take = min(end - pos, len(tail_mdat) - start_in)
                out.extend(tail_mdat[start_in : start_in + take])
            pos += take
        total_read += len(out)
        return bytes(out)

    with pytest.raises(ValueError, match="multiple mdat"):
        probe_mp4_byte_index("padded_tail.mp4", _virtual_read, file_size=virtual_size)
    assert total_read == DEFAULT_HEADER_PROBE_BYTES + len(tail_mdat)


def test_probe_supports_stz2_and_elst_bounds_parity_with_pyav(tmp_path: Path):
    # 1. Rewrite stsz -> stz2 (16-bit compact sample sizes) and verify parity with PyAV.
    orig = tmp_path / "stsz.mp4"
    compact = tmp_path / "stz2.mp4"
    _encode_mp4(orig, frames=24, faststart=False)
    raw = orig.read_bytes()
    idx = raw.index(b"stsz") - 4
    box_size = struct.unpack_from(">I", raw, idx)[0]
    fixed_size, count = struct.unpack_from(">II", raw, idx + 12)
    assert fixed_size == 0 and box_size == 20 + count * 4
    lengths = struct.unpack_from(f">{count}I", raw, idx + 20)
    stz2_box = struct.pack(">I4sI3xBI", 20 + count * 2, b"stz2", 0, 16, count)
    stz2_box += struct.pack(f">{count}H", *lengths)
    pad = box_size - len(stz2_box)
    assert pad >= 8
    stz2_box += struct.pack(">I4s", pad, b"free") + bytes(pad - 8)
    compact.write_bytes(raw[:idx] + stz2_box + raw[idx + box_size :])

    assert probe_local_mp4(compact).to_index_dict() == build_video_byte_index(compact)
    # Also verify 4-bit and 8-bit stz2 decoding.
    assert _parse_stz2(struct.pack(">I3xBI", 0, 8, 3) + bytes([5, 12, 250])).tolist() == [5, 12, 250]
    assert _parse_stz2(struct.pack(">I3xBI", 0, 4, 3) + bytes([0x3A, 0x70])).tolist() == [3, 10, 7]

    # 2. Non-frame-aligned leading elst media_time snaps to the first displayed sample
    # with raw_pts >= media_time and reports exact presented frame count, matching PyAV.
    bframe_mp4 = tmp_path / "bframes.mp4"
    trimmed_mp4 = tmp_path / "trimmed_elst.mp4"
    _encode_mp4(bframe_mp4, frames=24, faststart=True, bframes=2)
    buf = bytearray(bframe_mp4.read_bytes())
    elst_pos = buf.index(b"elst")
    version = buf[elst_pos + 4]
    assert version == 0
    seg_dur, media_time, rate_int, rate_frac = struct.unpack_from(">Iihh", buf, elst_pos + 12)
    struct.pack_into(">Iihh", buf, elst_pos + 12, seg_dur, media_time + 3379, rate_int, rate_frac)
    trimmed_mp4.write_bytes(bytes(buf))

    probed_trimmed = probe_local_mp4(trimmed_mp4)
    assert probed_trimmed.to_index_dict() == build_video_byte_index(trimmed_mp4)
    with av.open(str(trimmed_mp4)) as container:
        decoded_trimmed = sum(1 for _ in container.decode(video=0))
    assert probed_trimmed.sample_count == decoded_trimmed < 24

    # 3. Tail-truncating elst (segment_duration shorter than encoded samples) is rejected loudly.
    clipped_mp4 = tmp_path / "clipped_elst.mp4"
    buf_clip = bytearray(bframe_mp4.read_bytes())
    struct.pack_into(">Iihh", buf_clip, elst_pos + 12, 400, media_time, rate_int, rate_frac)
    clipped_mp4.write_bytes(bytes(buf_clip))
    with pytest.raises(ValueError, match="edit list truncates trailing samples"):
        probe_local_mp4(clipped_mp4)

    # 4. Zero-duration elst (segment_duration == 0, which FFmpeg decodes as 0 frames) is rejected.
    zero_mp4 = tmp_path / "zero_elst.mp4"
    buf_zero = bytearray(bframe_mp4.read_bytes())
    struct.pack_into(">Iihh", buf_zero, elst_pos + 12, 0, media_time, rate_int, rate_frac)
    zero_mp4.write_bytes(bytes(buf_zero))
    with pytest.raises(ValueError, match="non-positive segment_duration=0"):
        probe_local_mp4(zero_mp4)

    # 5. FFmpeg av_rescale rounding boundary (1917 ms * 12288 / 1000 -> 23556 ticks;
    # media_time 2044 + 23556 == 25600 == final sample PTS -> PyAV decodes 23 of 24 frames).
    boundary_mp4 = tmp_path / "boundary_elst.mp4"
    buf_boundary = bytearray(bframe_mp4.read_bytes())
    struct.pack_into(">Iihh", buf_boundary, elst_pos + 12, 1917, 2044, rate_int, rate_frac)
    boundary_mp4.write_bytes(bytes(buf_boundary))
    with pytest.raises(ValueError, match="23 of 24 samples presented"):
        probe_local_mp4(boundary_mp4)

    # 6. Open-GOP trimmed MP4 presenting leading B-frame(s) before first sync sample PTS
    # (whose decodability cannot be proven from container metadata alone) is rejected.
    ctts_pos = bframe_mp4.read_bytes().index(b"ctts")
    buf_opengop = bytearray(bframe_mp4.read_bytes())
    # Give sample 0 (the first sync sample) a larger composition offset than sample 1 so
    # sample 1 presents earlier than sample 0 while falling inside elst [media_time, edit_end).
    c0_count, c0_off = struct.unpack_from(">II", buf_opengop, ctts_pos + 12)
    assert c0_count >= 1
    struct.pack_into(">II", buf_opengop, ctts_pos + 12, 1, c0_off + 2048)
    opengop_mp4 = tmp_path / "opengop_leading.mp4"
    opengop_mp4.write_bytes(bytes(buf_opengop))
    with pytest.raises(ValueError, match="open-GOP leading sample"):
        probe_local_mp4(opengop_mp4)

    if shutil.which("ffmpeg"):
        full_og = tmp_path / "full_opengop.mp4"
        trimmed_og = tmp_path / "trimmed_opengop.mp4"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=10", "-frames:v", "36",
                "-c:v", "libx264", "-g", "6", "-bf", "2",
                "-x264-params", "open-gop=1:b-adapt=0:scenecut=0",
                "-threads", "1", "-movflags", "faststart", str(full_og),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", "0.45", "-i", str(full_og), "-c", "copy", "-movflags", "faststart",
                str(trimmed_og),
            ],
            check=True,
        )
        with pytest.raises(ValueError, match="open-GOP leading sample"):
            probe_local_mp4(trimmed_og)


def test_probe_tail_moov_jumps_large_free_padding_before_and_after_moov(tmp_path: Path):
    orig = tmp_path / "tail.mp4"
    _encode_mp4(orig, frames=24, faststart=False)
    expected = build_video_byte_index(orig)
    raw = orig.read_bytes()
    moov_offset = expected["moov_offset"]
    mdat_offset = raw.index(b"mdat") - 4
    prefix = bytearray(raw[:moov_offset])
    body_pad = 16384
    mdat_size = struct.unpack_from(">I", prefix, mdat_offset)[0]
    struct.pack_into(">I", prefix, mdat_offset, mdat_size + body_pad)
    moov = raw[moov_offset:]
    free_size = DEFAULT_MAX_PROBE_BYTES + 4096
    free_hdr = struct.pack(">I4s", free_size, b"free")

    for placement in ("before", "after"):
        segments = (
            [(bytes(prefix), len(prefix) + body_pad), (free_hdr, free_size), (moov, len(moov))]
            if placement == "before"
            else [(bytes(prefix), len(prefix) + body_pad), (moov, len(moov)), (free_hdr, free_size)]
        )
        total_size = sum(seg_len for _, seg_len in segments)
        read_sizes: list[int] = []

        def _read(
            _p: str,
            off: int,
            length: int,
            segs=segments,
            size=total_size,
            sizes=read_sizes,
        ) -> bytes:
            sizes.append(length)
            end = min(off + length, size)
            out = bytearray()
            pos = off
            cursor = 0
            for head, seg_len in segs:
                seg_end = cursor + seg_len
                if pos < seg_end and pos < end:
                    rel = pos - cursor
                    take = min(end, seg_end) - pos
                    if rel < len(head):
                        head_take = min(take, len(head) - rel)
                        out.extend(head[rel : rel + head_take])
                        take -= head_take
                    if take > 0:
                        out.extend(b"\x00" * take)
                    pos = min(end, seg_end)
                cursor = seg_end
            return bytes(out)

        probed = probe_mp4_byte_index(f"free_{placement}.mp4", _read, file_size=total_size, header_probe_bytes=64)
        assert probed.sample_count == 24
        assert probed.kf_indices.tolist() == expected["kf_indices"]
        assert probed.kf_positions.tolist() == expected["kf_positions"]
        assert probed.moov_size == len(moov)
        assert probed.bytes_read < 64 * 1024

        if placement == "after":
            # Verify first tail read respects max_probe_bytes even with default 1 MiB header_probe_bytes.
            read_sizes.clear()
            probed_bounded = probe_mp4_byte_index(
                "free_after_bounded.mp4", _read, file_size=total_size, max_probe_bytes=4096
            )
            assert probed_bounded.sample_count == 24
            assert all(sz <= 4096 for sz in read_sizes)
