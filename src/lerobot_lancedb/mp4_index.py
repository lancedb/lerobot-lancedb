#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Range-Read MP4 byte-index extractor for zero-copy External Blob v2 conversion.

Parses ISO-BMFF ``moov`` sample tables directly from minimal byte-range reads
(typically 1 Range GET for faststart MP4s, 2 for tail-``moov`` MP4s) without
downloading full video payloads or demuxing packets with PyAV.

The extracted 5-column byte index (``file_size``, ``moov_offset``, ``moov_size``,
``kf_indices``, ``kf_positions``) matches PyAV's ``build_video_byte_index()``
across faststart, tail-``moov``, and Open-GOP / B-frame streams, while rejecting
unsupported container layouts (fragmented MP4, multiple ``mdat`` boxes, corrupt
sample-table counts) before any Lance table is written.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_HEADER_PROBE_BYTES = 1024 * 1024  # 1 MiB covers most faststart moov boxes in 1 GET
DEFAULT_MAX_PROBE_BYTES = 64 * 1024 * 1024
_MAX_CONSTANT_SAMPLES = 50_000_000
_TAIL_HEADER_WINDOW_BYTES = 4096
_MAX_TAIL_BOXES = 1024


@dataclass(frozen=True)
class _Box:
    """One parsed ISO-BMFF box header in absolute file coordinates."""

    type: bytes
    start: int
    header_size: int
    end: int

    @property
    def payload_start(self) -> int:
        return self.start + self.header_size

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ProbedMp4:
    """Byte-index and container metadata extracted from one MP4 via range reads."""

    file_size: int
    moov_offset: int
    moov_size: int
    kf_indices: NDArray[np.int64]
    kf_positions: NDArray[np.int64]
    sample_count: int
    fps: float
    faststart: bool
    range_gets: int
    bytes_read: int

    def to_index_dict(self) -> dict[str, Any]:
        """Return the 5-column byte index expected by ``videos.lance``."""
        return {
            "file_size": self.file_size,
            "moov_offset": self.moov_offset,
            "moov_size": self.moov_size,
            "kf_indices": self.kf_indices.tolist(),
            "kf_positions": self.kf_positions.tolist(),
        }


def _iter_boxes(
    data: bytes,
    start: int,
    end: int,
    *,
    absolute_base: int = 0,
    file_size: int | None = None,
    allow_truncated: bool = False,
) -> list[_Box]:
    """Walk sequential ISO-BMFF boxes within ``data[start:end]``."""
    boxes: list[_Box] = []
    pos = start
    while pos + 8 <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]
        header_size = 8
        if size == 1:
            if pos + 16 > end:
                break
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header_size = 16
        elif size == 0:
            size = (file_size - (absolute_base + pos)) if file_size is not None else (end - pos)
        if size < header_size:
            raise ValueError(f"Malformed MP4 box {box_type!r} at offset {absolute_base + pos}: size={size}")
        box_end_rel = pos + size
        if box_end_rel > end and not allow_truncated:
            break
        boxes.append(_Box(box_type, absolute_base + pos, header_size, absolute_base + box_end_rel))
        pos = box_end_rel
    return boxes


def _find_child(data: bytes, box_type: bytes) -> _Box:
    for box in _iter_boxes(data, 0, len(data)):
        if box.type == box_type:
            return box
    raise ValueError(f"Missing MP4 box {box_type.decode('latin1')}")


def _find_optional_child(data: bytes, box_type: bytes) -> _Box | None:
    for box in _iter_boxes(data, 0, len(data)):
        if box.type == box_type:
            return box
    return None


def _find_child_payload(parent: bytes, box_type: bytes) -> bytes:
    box = _find_child(parent, box_type)
    return parent[box.payload_start : box.end]


def _find_descendant(data: bytes, path: list[bytes]) -> bytes:
    current = data
    for box_type in path:
        current = _find_child_payload(current, box_type)
    return current


def _find_video_trak(moov: bytes) -> bytes:
    for trak in _iter_boxes(moov, 0, len(moov)):
        if trak.type != b"trak":
            continue
        payload = moov[trak.payload_start : trak.end]
        mdia = _find_optional_child(payload, b"mdia")
        if mdia is None:
            continue
        mdia_payload = payload[mdia.payload_start : mdia.end]
        hdlr_box = _find_optional_child(mdia_payload, b"hdlr")
        if hdlr_box is None:
            continue
        hdlr = mdia_payload[hdlr_box.payload_start : hdlr_box.end]
        if len(hdlr) >= 12 and hdlr[8:12] == b"vide":
            return payload
    raise ValueError("No video track found in MP4 moov box")


def _parse_timescale_and_duration(payload: bytes, box_name: str) -> tuple[int, int]:
    if not payload:
        raise ValueError(f"Empty {box_name} box")
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            raise ValueError(f"Truncated v1 {box_name} box")
        timescale, duration = struct.unpack_from(">IQ", payload, 20)
    elif version == 0:
        if len(payload) < 20:
            raise ValueError(f"Truncated v0 {box_name} box")
        timescale, duration = struct.unpack_from(">II", payload, 12)
    else:
        raise ValueError(f"Unsupported {box_name} version {version}")
    if timescale <= 0:
        raise ValueError(f"Invalid {box_name} timescale {timescale}")
    return timescale, duration


def _parse_stts(payload: bytes, sample_count: int) -> NDArray[np.int64]:
    if len(payload) < 8:
        raise ValueError("Truncated stts box")
    count = struct.unpack_from(">I", payload, 4)[0]
    if 8 + count * 8 > len(payload):
        raise ValueError(f"Malformed stts: entry count {count} exceeds box length {len(payload)}")
    if count == 0:
        if sample_count != 0:
            raise ValueError(f"Empty stts for {sample_count} samples")
        return np.empty(0, dtype=np.int64)
    arr = np.frombuffer(payload, dtype=">u4", count=count * 2, offset=8).reshape(count, 2)
    counts = arr[:, 0].astype(np.int64)
    deltas = arr[:, 1].astype(np.int64)
    total = int(counts.sum())
    if total != sample_count:
        raise ValueError(f"stts describes {total} samples, stsz describes {sample_count}")
    if count == 1:
        return np.full(sample_count, int(deltas[0]), dtype=np.int64)
    return np.repeat(deltas, counts)


def _parse_ctts(stbl: bytes, sample_count: int) -> NDArray[np.int64]:
    box = _find_optional_child(stbl, b"ctts")
    if box is None:
        return np.zeros(sample_count, dtype=np.int64)
    payload = stbl[box.payload_start : box.end]
    if len(payload) < 8:
        raise ValueError("Truncated ctts box")
    version = payload[0]
    if version not in (0, 1):
        raise ValueError(f"Unsupported ctts version {version}")
    count = struct.unpack_from(">I", payload, 4)[0]
    if 8 + count * 8 > len(payload):
        raise ValueError(f"Malformed ctts: entry count {count} exceeds box length {len(payload)}")
    dt = np.dtype([("count", ">u4"), ("offset", ">i4" if version == 1 else ">u4")])
    arr = np.frombuffer(payload, dtype=dt, count=count, offset=8)
    counts = arr["count"].astype(np.int64)
    offsets = arr["offset"].astype(np.int64)
    total = int(counts.sum())
    if total != sample_count:
        raise ValueError(f"ctts describes {total} samples, stsz describes {sample_count}")
    return np.repeat(offsets, counts)


def _parse_edit_list(trak: bytes) -> tuple[int, int, int] | None:
    """Return ``(empty_duration, segment_duration, media_time)`` for a single rate-1 edit, or None."""
    edts = _find_optional_child(trak, b"edts")
    if edts is None:
        return None
    elst = _find_optional_child(trak[edts.payload_start : edts.end], b"elst")
    if elst is None:
        return None
    payload = trak[edts.payload_start + elst.payload_start : edts.payload_start + elst.end]
    if len(payload) < 8:
        raise ValueError("Truncated elst box")
    version = payload[0]
    if version not in (0, 1):
        raise ValueError(f"Unsupported elst version {version}")
    count = struct.unpack_from(">I", payload, 4)[0]
    if count == 0:
        return None
    entry_size = 20 if version == 1 else 12
    if 8 + count * entry_size > len(payload):
        raise ValueError(f"Malformed elst: count {count} exceeds box length {len(payload)}")
    entries = [
        struct.unpack_from(">Qqhh" if version == 1 else ">Iihh", payload, 8 + idx * entry_size)
        for idx in range(count)
    ]
    empty_duration = 0
    if len(entries) == 2 and entries[0][1] == -1 and entries[0][2:] == (1, 0):
        empty_duration = entries.pop(0)[0]
    if len(entries) != 1 or entries[0][1] < 0 or entries[0][2:] != (1, 0):
        raise ValueError(f"Unsupported MP4 edit list (entries={entries!r}): expected one rate-one media edit")
    return empty_duration, entries[0][0], entries[0][1]


def _parse_stsz(payload: bytes) -> NDArray[np.int64]:
    if len(payload) < 12:
        raise ValueError("Truncated stsz box")
    sample_size, sample_count = struct.unpack_from(">II", payload, 4)
    if sample_size:
        if sample_count > _MAX_CONSTANT_SAMPLES:
            raise ValueError(f"Implausible constant-size stsz sample_count={sample_count}")
        return np.full(sample_count, sample_size, dtype=np.int64)
    required = 12 + sample_count * 4
    if required > len(payload):
        raise ValueError(
            f"Malformed stsz: sample_count={sample_count} requires {required} bytes, box has {len(payload)}"
        )
    return np.frombuffer(payload, dtype=">u4", count=sample_count, offset=12).astype(np.int64)


def _parse_stz2(payload: bytes) -> NDArray[np.int64]:
    if len(payload) < 12:
        raise ValueError("Truncated stz2 box")
    field_size = payload[7]
    sample_count = struct.unpack_from(">I", payload, 8)[0]
    if field_size == 16:
        required = 12 + sample_count * 2
        if required > len(payload):
            raise ValueError(
                f"Malformed stz2: sample_count={sample_count} requires {required} bytes, box has {len(payload)}"
            )
        return np.frombuffer(payload, dtype=">u2", count=sample_count, offset=12).astype(np.int64)
    if field_size == 8:
        required = 12 + sample_count
        if required > len(payload):
            raise ValueError(
                f"Malformed stz2: sample_count={sample_count} requires {required} bytes, box has {len(payload)}"
            )
        return np.frombuffer(payload, dtype=np.uint8, count=sample_count, offset=12).astype(np.int64)
    if field_size == 4:
        nbytes = (sample_count + 1) // 2
        required = 12 + nbytes
        if required > len(payload):
            raise ValueError(
                f"Malformed stz2: sample_count={sample_count} requires {required} bytes, box has {len(payload)}"
            )
        raw = np.frombuffer(payload, dtype=np.uint8, count=nbytes, offset=12)
        out = np.empty(nbytes * 2, dtype=np.int64)
        out[0::2] = (raw >> 4) & 0x0F
        out[1::2] = raw & 0x0F
        return out[:sample_count]
    raise ValueError(f"Unsupported stz2 field_size={field_size}")


def _parse_sample_sizes(stbl: bytes) -> NDArray[np.int64]:
    stsz = _find_optional_child(stbl, b"stsz")
    if stsz is not None:
        return _parse_stsz(stbl[stsz.payload_start : stsz.end])
    stz2 = _find_optional_child(stbl, b"stz2")
    if stz2 is not None:
        return _parse_stz2(stbl[stz2.payload_start : stz2.end])
    raise ValueError("Missing MP4 box stsz/stz2")


def _parse_stsc(payload: bytes) -> list[tuple[int, int, int]]:
    if len(payload) < 8:
        raise ValueError("Truncated stsc box")
    count = struct.unpack_from(">I", payload, 4)[0]
    if 8 + count * 12 > len(payload):
        raise ValueError(f"Malformed stsc: count={count} exceeds box length {len(payload)}")
    arr = np.frombuffer(payload, dtype=">u4", count=count * 3, offset=8).reshape(count, 3)
    return [(int(r[0]), int(r[1]), int(r[2])) for r in arr]


def _parse_chunk_offsets(stbl: bytes) -> NDArray[np.int64]:
    co64 = _find_optional_child(stbl, b"co64")
    if co64 is not None:
        payload = stbl[co64.payload_start : co64.end]
        if len(payload) < 8:
            raise ValueError("Truncated co64 box")
        count = struct.unpack_from(">I", payload, 4)[0]
        if 8 + count * 8 > len(payload):
            raise ValueError(f"Malformed co64: count={count} exceeds box length {len(payload)}")
        raw = np.frombuffer(payload, dtype=">u8", count=count, offset=8)
        if raw.size and int(raw.max()) > np.iinfo(np.int64).max:
            raise ValueError("Malformed co64: chunk offset exceeds int64 range")
        return raw.astype(np.int64)
    stco = _find_optional_child(stbl, b"stco")
    if stco is None:
        raise ValueError("Missing stco/co64 chunk offsets")
    payload = stbl[stco.payload_start : stco.end]
    if len(payload) < 8:
        raise ValueError("Truncated stco box")
    count = struct.unpack_from(">I", payload, 4)[0]
    if 8 + count * 4 > len(payload):
        raise ValueError(f"Malformed stco: count={count} exceeds box length {len(payload)}")
    return np.frombuffer(payload, dtype=">u4", count=count, offset=8).astype(np.int64)


def _parse_stss(stbl: bytes, sample_count: int) -> NDArray[np.int64]:
    box = _find_optional_child(stbl, b"stss")
    if box is None:
        return np.arange(sample_count, dtype=np.int64)
    payload = stbl[box.payload_start : box.end]
    if len(payload) < 8:
        raise ValueError("Truncated stss box")
    count = struct.unpack_from(">I", payload, 4)[0]
    if 8 + count * 4 > len(payload):
        raise ValueError(f"Malformed stss: count={count} exceeds box length {len(payload)}")
    raw = np.frombuffer(payload, dtype=">u4", count=count, offset=8).astype(np.int64) - 1
    if len(raw) and (int(raw.min()) < 0 or int(raw.max()) >= sample_count):
        raise ValueError(f"stss keyframe index out of bounds [0, {sample_count})")
    return raw


def _compute_sample_offsets(
    stsc: list[tuple[int, int, int]],
    chunk_offsets: NDArray[np.int64],
    sample_sizes: NDArray[np.int64],
) -> NDArray[np.int64]:
    n_samples = len(sample_sizes)
    n_chunks = len(chunk_offsets)
    if n_samples == 0:
        return np.empty(0, dtype=np.int64)
    if not stsc or n_chunks == 0:
        raise ValueError("Empty stsc or chunk offset table for non-empty video track")

    samples_per_chunk = np.zeros(n_chunks, dtype=np.int64)
    for idx, (first_chunk, count, _desc) in enumerate(stsc):
        start_chunk = first_chunk - 1
        end_chunk = (stsc[idx + 1][0] - 1) if idx + 1 < len(stsc) else n_chunks
        if start_chunk < 0 or end_chunk > n_chunks or start_chunk > end_chunk:
            raise ValueError(f"Invalid stsc chunk range [{start_chunk}, {end_chunk}) for {n_chunks} chunks")
        samples_per_chunk[start_chunk:end_chunk] = count

    total_mapped = int(samples_per_chunk.sum())
    if total_mapped < n_samples:
        raise ValueError(f"stsc maps {total_mapped} samples, fewer than stsz {n_samples}")
    if total_mapped > n_samples:
        # Rare trailing empty/partial chunk entry: clip cleanly in chunk order
        offsets = np.empty(n_samples, dtype=np.int64)
        sample_idx = 0
        for chunk_idx in range(n_chunks):
            pos = int(chunk_offsets[chunk_idx])
            take = min(int(samples_per_chunk[chunk_idx]), n_samples - sample_idx)
            for _ in range(take):
                offsets[sample_idx] = pos
                pos += int(sample_sizes[sample_idx])
                sample_idx += 1
            if sample_idx >= n_samples:
                break
        return offsets

    chunk_starts = np.repeat(chunk_offsets, samples_per_chunk)
    prefix_sizes = np.empty(n_samples, dtype=np.int64)
    prefix_sizes[0] = 0
    if n_samples > 1:
        np.cumsum(sample_sizes[:-1], out=prefix_sizes[1:])
    chunk_first_sample = np.zeros(n_chunks, dtype=np.int64)
    if n_chunks > 1:
        np.cumsum(samples_per_chunk[:-1], out=chunk_first_sample[1:])
    nonempty = samples_per_chunk > 0
    chunk_base_prefix = np.repeat(prefix_sizes[chunk_first_sample[nonempty]], samples_per_chunk[nonempty])
    return chunk_starts + (prefix_sizes - chunk_base_prefix)


def _parse_moov_keyframes(
    moov_payload: bytes,
    *,
    path: str,
    mdat_payload_start: int,
    mdat_end: int,
    file_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int, float]:
    """Parse keyframe indices/positions, sample count, and average fps from a ``moov`` payload."""
    if _find_optional_child(moov_payload, b"mvex") is not None:
        raise ValueError(f"{path}: fragmented MP4 (mvex box in moov) is not supported")

    movie_timescale, _ = _parse_timescale_and_duration(_find_descendant(moov_payload, [b"mvhd"]), "mvhd")
    trak_payload = _find_video_trak(moov_payload)
    media_timescale, _ = _parse_timescale_and_duration(
        _find_descendant(trak_payload, [b"mdia", b"mdhd"]), "mdhd"
    )
    stbl = _find_descendant(trak_payload, [b"mdia", b"minf", b"stbl"])

    sample_sizes = _parse_sample_sizes(stbl)
    n_samples = len(sample_sizes)
    if n_samples == 0:
        raise ValueError(f"{path}: video track has 0 samples")

    sample_durations = _parse_stts(_find_child_payload(stbl, b"stts"), n_samples)
    total_duration = int(sample_durations.sum())
    if total_duration <= 0:
        raise ValueError(f"{path}: non-positive total sample duration {total_duration}")
    avg_fps = (n_samples * float(media_timescale)) / float(total_duration)

    stsc = _parse_stsc(_find_child_payload(stbl, b"stsc"))
    chunk_offsets = _parse_chunk_offsets(stbl)
    sync_samples = _parse_stss(stbl, n_samples)
    if len(sync_samples) == 0:
        raise ValueError(f"{path}: no sync samples (empty kf_positions)")

    sample_offsets = _compute_sample_offsets(stsc, chunk_offsets, sample_sizes)
    min_offset = int(sample_offsets.min())
    if min_offset < 0 or bool((sample_offsets > file_size - sample_sizes).any()):
        max_end = int((sample_offsets + sample_sizes).max())
        raise ValueError(
            f"{path}: truncated MP4 (sample byte range [{min_offset}, {max_end}) exceeds file_size={file_size})"
        )
    if min_offset < mdat_payload_start or bool((sample_offsets > mdat_end - sample_sizes).any()):
        max_end = int((sample_offsets + sample_sizes).max())
        raise ValueError(
            f"{path}: sample byte range [{min_offset}, {max_end}) falls outside "
            f"mdat [{mdat_payload_start}, {mdat_end})"
        )

    sample_pts_units = np.empty(n_samples, dtype=np.int64)
    sample_pts_units[0] = 0
    if n_samples > 1:
        np.cumsum(sample_durations[:-1], out=sample_pts_units[1:])
    composition_offsets = _parse_ctts(stbl, n_samples)
    raw_pts_units = sample_pts_units + composition_offsets

    first_sync = int(sync_samples[0])
    first_sync_pts = int(raw_pts_units[first_sync])

    edit = _parse_edit_list(trak_payload)
    if edit is not None:
        empty_duration, segment_duration, media_time = edit
        if segment_duration <= 0:
            raise ValueError(
                f"{path}: MP4 edit list has non-positive segment_duration={segment_duration} "
                "(presents 0 samples)"
            )
        # Match FFmpeg mov_fix_index (libavformat/mov.c): av_rescale uses AV_ROUND_NEAR_INF.
        edit_duration_media = (segment_duration * media_timescale + movie_timescale // 2) // movie_timescale
        if edit_duration_media <= 0:
            raise ValueError(
                f"{path}: MP4 edit list has 0-tick media duration (presents 0 samples)"
            )
        edit_end_media = media_time + edit_duration_media
        max_raw_pts = int(raw_pts_units.max())
        if max_raw_pts >= edit_end_media:
            presented = int(((raw_pts_units >= media_time) & (raw_pts_units < edit_end_media)).sum())
            raise ValueError(
                f"{path}: MP4 edit list truncates trailing samples "
                f"({presented} of {n_samples} samples presented within edit duration); "
                "unsupported in external mode"
            )
        if first_sync > 0:
            pre_sync_in_edit = int((raw_pts_units[:first_sync] >= media_time).sum())
            if pre_sync_in_edit > 0:
                raise ValueError(
                    f"{path}: MP4 edit list presents {pre_sync_in_edit} sample(s) before first sync "
                    f"sample (index {first_sync}); unsupported in external mode"
                )
        leading_in_edit = int(
            ((raw_pts_units[first_sync:] >= media_time) & (raw_pts_units[first_sync:] < first_sync_pts)).sum()
        )
        if leading_in_edit > 0:
            raise ValueError(
                f"{path}: MP4 edit list presents {leading_in_edit} open-GOP leading sample(s) "
                f"before first sync sample PTS ({first_sync_pts}); container metadata cannot confirm "
                "decodability — re-encode or trim at a closed keyframe"
            )
        presented_count = int((raw_pts_units[first_sync:] >= media_time).sum())
        if presented_count == 0:
            raise ValueError(f"{path}: MP4 edit list presents 0 samples")
        # Match FFmpeg's mov_fix_index: snap media_time to the first sample with
        # raw_pts >= media_time so stream-copy trimmed files align with PyAV PTS.
        ge = raw_pts_units[first_sync:][raw_pts_units[first_sync:] >= media_time]
        snapped_media_time = int(ge.min())
        sample_pts = (raw_pts_units - snapped_media_time).astype(np.float64) / float(media_timescale)
        sample_pts += float(empty_duration) / float(movie_timescale)
    else:
        if first_sync > 0:
            raise ValueError(
                f"{path}: MP4 video track starts with {first_sync} non-sync sample(s) before first "
                "keyframe; unsupported in external mode"
            )
        leading_before_sync = int((raw_pts_units < first_sync_pts).sum())
        if leading_before_sync > 0:
            raise ValueError(
                f"{path}: MP4 starts with {leading_before_sync} open-GOP leading sample(s) before "
                f"first sync sample PTS ({first_sync_pts}); container metadata cannot confirm decodability"
            )
        presented_count = n_samples
        sample_pts = raw_pts_units.astype(np.float64) / float(media_timescale)

    # Sort by presentation-order frame index to match PyAV's `packet.pts * time_base * fps`
    # even on Open-GOP / B-frame streams where decode order differs from presentation order.
    kf_idx = np.rint(sample_pts[sync_samples] * avg_fps).astype(np.int64)
    kf_pos = sample_offsets[sync_samples].astype(np.int64, copy=False)
    order = np.lexsort((kf_pos, kf_idx))
    return kf_idx[order], kf_pos[order], presented_count, avg_fps


def probe_mp4_byte_index(
    path: str,
    read_range: Callable[[str, int, int], bytes],
    *,
    file_size: int,
    header_probe_bytes: int = DEFAULT_HEADER_PROBE_BYTES,
    max_probe_bytes: int = DEFAULT_MAX_PROBE_BYTES,
) -> ProbedMp4:
    """Extract the 5-column byte index and frame count from one MP4 using range reads.

    Parameters
    ----------
    path:
        Identifier passed through to ``read_range(path, offset, length)`` and error messages.
    read_range:
        Callable returning up to ``length`` bytes starting at ``offset``.
    file_size:
        Known total byte size of the MP4 file (e.g. from object-store listing or ``stat``).
    header_probe_bytes:
        Initial prefix size read from offset 0 (default 1 MiB).
    max_probe_bytes:
        Upper bound on total prefix or tail bytes read when locating ``moov``.
    """
    if file_size <= 0:
        raise ValueError(f"{path}: invalid file_size={file_size}")
    if max_probe_bytes < 16:
        raise ValueError(f"{path}: max_probe_bytes must be >= 16, got {max_probe_bytes}")

    range_gets = 0
    bytes_read = 0

    def tracked_read(offset: int, length: int) -> bytes:
        nonlocal range_gets, bytes_read
        if length <= 0:
            return b""
        chunk = read_range(path, offset, length)
        range_gets += 1
        bytes_read += len(chunk)
        return chunk

    def scan_trailing_boxes(start_pos: int) -> None:
        """Jump box-by-box from ``start_pos`` to EOF, rejecting trailing moof/mdat/moov."""
        pos = start_pos
        box_count = 0
        tail_bytes_read = 0
        while pos < file_size:
            remaining_budget = max_probe_bytes - tail_bytes_read
            if remaining_budget <= 0:
                raise ValueError(
                    f"{path}: trailing box scan exceeded max_probe_bytes={max_probe_bytes}"
                )
            win_len = min(_TAIL_HEADER_WINDOW_BYTES, remaining_budget, file_size - pos)
            buf = tracked_read(pos, win_len)
            tail_bytes_read += len(buf)
            if not buf:
                raise ValueError(f"{path}: empty range read while scanning trailing boxes at offset {pos}")
            boxes = _iter_boxes(
                buf, 0, len(buf), absolute_base=pos, file_size=file_size, allow_truncated=True
            )
            if not boxes:
                raise ValueError(f"{path}: truncated trailing MP4 box header at offset {pos}")
            for box in boxes:
                box_count += 1
                if box_count > _MAX_TAIL_BOXES:
                    raise ValueError(
                        f"{path}: too many trailing top-level boxes after mdat (>{_MAX_TAIL_BOXES})"
                    )
                if box.type == b"moof":
                    raise ValueError(f"{path}: fragmented MP4 (tail moof box) is not supported")
                if box.type == b"mdat":
                    raise ValueError(f"{path}: multiple mdat boxes (tail mdat) are not supported")
                if box.type == b"moov":
                    raise ValueError(f"{path}: multiple moov boxes (tail moov) are not supported")
                if box.end > file_size:
                    raise ValueError(
                        f"{path}: truncated trailing MP4 box {box.type!r} "
                        f"(ends at {box.end}, file_size={file_size})"
                    )
            next_pos = boxes[-1].end
            if next_pos <= pos:
                raise ValueError(f"{path}: non-advancing trailing MP4 box at offset {pos}")
            pos = next_pos

    probe_limit = min(max_probe_bytes, file_size)
    data = tracked_read(0, min(header_probe_bytes, probe_limit))

    while True:
        top = _iter_boxes(data, 0, len(data), absolute_base=0, file_size=file_size, allow_truncated=True)

        if any(b.type == b"moof" for b in top):
            raise ValueError(f"{path}: fragmented MP4 (top-level moof box) is not supported")

        mdat_boxes = [b for b in top if b.type == b"mdat"]
        if len(mdat_boxes) > 1:
            raise ValueError(f"{path}: multiple mdat boxes ({len(mdat_boxes)}) are not supported")

        moov_boxes = [b for b in top if b.type == b"moov"]
        if len(moov_boxes) > 1:
            raise ValueError(f"{path}: multiple moov boxes ({len(moov_boxes)}) are not supported")

        moov_box = moov_boxes[0] if moov_boxes else None
        mdat_box = mdat_boxes[0] if mdat_boxes else None
        has_moov = moov_box is not None and moov_box.end <= len(data)
        has_mdat = mdat_box is not None

        if has_moov and has_mdat:
            assert moov_box is not None and mdat_box is not None
            if mdat_box.end > file_size:
                raise ValueError(
                    f"{path}: truncated MP4 (mdat ends at {mdat_box.end}, file_size={file_size})"
                )
            if top[-1].end > file_size:
                raise ValueError(
                    f"{path}: truncated trailing MP4 box {top[-1].type!r} "
                    f"(ends at {top[-1].end}, file_size={file_size})"
                )
            if top[-1].end < file_size:
                scan_trailing_boxes(top[-1].end)
            moov_payload = data[moov_box.payload_start : moov_box.end]
            kf_indices, kf_positions, sample_count, avg_fps = _parse_moov_keyframes(
                moov_payload,
                path=path,
                mdat_payload_start=mdat_box.payload_start,
                mdat_end=mdat_box.end,
                file_size=file_size,
            )
            return ProbedMp4(
                file_size=file_size,
                moov_offset=moov_box.start,
                moov_size=moov_box.size,
                kf_indices=kf_indices,
                kf_positions=kf_positions,
                sample_count=sample_count,
                fps=avg_fps,
                faststart=moov_box.start < mdat_box.start,
                range_gets=range_gets,
                bytes_read=bytes_read,
            )

        if has_mdat and moov_box is None:
            assert mdat_box is not None
            if mdat_box.end > file_size:
                raise ValueError(
                    f"{path}: truncated MP4 (mdat ends at {mdat_box.end}, file_size={file_size})"
                )
            if mdat_box.end == file_size:
                raise ValueError(f"{path}: mdat reaches EOF ({mdat_box.end}/{file_size}) with no moov box")
            if top[-1].end > file_size:
                raise ValueError(
                    f"{path}: truncated trailing MP4 box {top[-1].type!r} "
                    f"(ends at {top[-1].end}, file_size={file_size})"
                )

            pos = top[-1].end
            box_count = 0
            tail_bytes_read = 0
            tail_moov: _Box | None = None
            moov_payload: bytes | None = None
            first_tail_read = True

            while pos < file_size:
                remaining_budget = max_probe_bytes - tail_bytes_read
                if remaining_budget <= 0:
                    raise ValueError(
                        f"{path}: tail probe exceeded max_probe_bytes={max_probe_bytes}"
                    )
                if first_tail_read and tail_moov is None:
                    win_len = min(
                        max(header_probe_bytes, _TAIL_HEADER_WINDOW_BYTES),
                        remaining_budget,
                        file_size - pos,
                    )
                else:
                    win_len = min(_TAIL_HEADER_WINDOW_BYTES, remaining_budget, file_size - pos)
                first_tail_read = False

                buf = tracked_read(pos, win_len)
                tail_bytes_read += len(buf)
                if not buf:
                    raise ValueError(f"{path}: empty range read while scanning tail at offset {pos}")
                boxes = _iter_boxes(
                    buf, 0, len(buf), absolute_base=pos, file_size=file_size, allow_truncated=True
                )
                if not boxes:
                    raise ValueError(f"{path}: truncated trailing MP4 box header at offset {pos}")

                for box in boxes:
                    box_count += 1
                    if box_count > _MAX_TAIL_BOXES:
                        raise ValueError(
                            f"{path}: too many trailing top-level boxes after mdat (>{_MAX_TAIL_BOXES})"
                        )
                    if box.type == b"moof":
                        raise ValueError(f"{path}: fragmented MP4 (tail moof box) is not supported")
                    if box.type == b"mdat":
                        raise ValueError(f"{path}: multiple mdat boxes (tail mdat) are not supported")
                    if box.end > file_size:
                        raise ValueError(
                            f"{path}: truncated trailing MP4 box {box.type!r} "
                            f"(ends at {box.end}, file_size={file_size})"
                        )
                    if box.type == b"moov":
                        if tail_moov is not None:
                            raise ValueError(f"{path}: multiple moov boxes (tail moov) are not supported")
                        if box.size > max_probe_bytes:
                            raise ValueError(
                                f"{path}: tail moov ({box.size} bytes) exceeds max_probe_bytes={max_probe_bytes}"
                            )
                        tail_moov = box
                        buf_end = pos + len(buf)
                        if box.end <= buf_end:
                            moov_payload = buf[box.payload_start - pos : box.end - pos]
                        else:
                            have = buf[box.payload_start - pos :] if box.payload_start < buf_end else b""
                            need_from = max(box.payload_start, buf_end)
                            need_len = box.end - need_from
                            if tail_bytes_read + need_len > max_probe_bytes:
                                raise ValueError(
                                    f"{path}: tail moov read ({tail_bytes_read + need_len} bytes) "
                                    f"exceeds max_probe_bytes={max_probe_bytes}"
                                )
                            rest = tracked_read(need_from, need_len)
                            tail_bytes_read += len(rest)
                            if len(rest) != need_len:
                                raise ValueError(
                                    f"{path}: truncated tail moov read ({len(rest)}/{need_len} bytes)"
                                )
                            moov_payload = have + rest

                next_pos = boxes[-1].end
                if next_pos <= pos:
                    raise ValueError(f"{path}: non-advancing trailing MP4 box at offset {pos}")
                pos = next_pos

            if tail_moov is None or moov_payload is None:
                raise ValueError(f"{path}: expected 1 tail moov box, found 0")

            kf_indices, kf_positions, sample_count, avg_fps = _parse_moov_keyframes(
                moov_payload,
                path=path,
                mdat_payload_start=mdat_box.payload_start,
                mdat_end=mdat_box.end,
                file_size=file_size,
            )
            return ProbedMp4(
                file_size=file_size,
                moov_offset=tail_moov.start,
                moov_size=tail_moov.size,
                kf_indices=kf_indices,
                kf_positions=kf_positions,
                sample_count=sample_count,
                fps=avg_fps,
                faststart=False,
                range_gets=range_gets,
                bytes_read=bytes_read,
            )

        if len(data) >= probe_limit:
            missing = []
            if not has_mdat:
                missing.append("mdat")
            if not has_moov:
                missing.append("moov")
            raise ValueError(
                f"{path}: could not find complete {'/'.join(missing)} within first {len(data)} bytes"
            )

        last_box = top[-1] if top else None
        truncated = last_box is not None and last_box.end > len(data)
        target = last_box.end + 16 if truncated and last_box is not None else 2 * len(data)
        read_len = min(target, probe_limit) - len(data)
        chunk = tracked_read(len(data), read_len)
        if not chunk:
            raise ValueError(f"{path}: empty range read at offset {len(data)}")
        data += chunk


def probe_local_mp4(
    path: Path,
    *,
    header_probe_bytes: int = DEFAULT_HEADER_PROBE_BYTES,
    max_probe_bytes: int = DEFAULT_MAX_PROBE_BYTES,
) -> ProbedMp4:
    """Convenience wrapper to probe a local MP4 file via ``seek``/``read``."""
    file_size = path.stat().st_size
    with open(path, "rb") as fh:

        def _read(_path: str, offset: int, length: int) -> bytes:
            fh.seek(offset)
            return fh.read(length)

        return probe_mp4_byte_index(
            str(path),
            _read,
            file_size=file_size,
            header_probe_bytes=header_probe_bytes,
            max_probe_bytes=max_probe_bytes,
        )
