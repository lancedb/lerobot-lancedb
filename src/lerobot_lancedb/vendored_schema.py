#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Schema contract between this converter and lerobot's ``LanceDBDataset`` loader.

TEMPORARY VENDORED COPY of the schema contract from
``lerobot.datasets.lancedb_dataset`` (table/column names, ``to_lance_column``,
``build_video_byte_index`` and its ``_find_moov`` helper). Switch to
``from lerobot.datasets.lancedb_dataset import ...`` and delete this module
once the upstream lerobot PR merges — released lerobot does not ship the
loader yet, so importing it here would break every user on PyPI lerobot.

Do not edit the vendored definitions independently: the loader reads exactly
what these write, byte for byte and name for name.
"""

from __future__ import annotations

from pathlib import Path

import av

FRAMES_TABLE = "frames"
VIDEOS_TABLE = "videos"
META_TABLE = "meta"
VIDEO_BLOB_COLUMN = "video_bytes"
# Byte-index columns on the videos table, written at conversion time by
# ``build_video_byte_index``. They map a frame window to the byte ranges its
# decode needs, so a whole batch's video bytes travel in one
# ``fetch_blob_ranges`` call. The keyframe columns work for any
# container/codec and assume constant frame rate (upstream's reader assumes
# the same); the moov columns are mp4-specific (store 0/0 for other
# containers).
VIDEO_INDEX_COLUMNS = ("file_size", "moov_offset", "moov_size", "kf_indices", "kf_positions")


def to_lance_column(key: str) -> str:
    """Map a LeRobot feature key to its Lance column name."""
    return key.replace(".", "_")


def _find_moov(read_at, file_size: int) -> tuple[int, int]:
    """Locate the mp4 ``moov`` box by walking top-level box headers."""
    offset = 0
    while offset < file_size:
        header = read_at(offset, 16)
        box_size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]
        if box_size == 1:
            box_size = int.from_bytes(header[8:16], "big")
        elif box_size == 0:
            box_size = file_size - offset
        if box_type == b"moov":
            return offset, box_size
        offset += box_size
    raise ValueError("no moov box found")


def build_video_byte_index(path: str | Path) -> dict:
    """Compute the byte-index columns for one video file.

    Converters store the returned dict alongside the video's blob row; the
    remote reader uses it to translate frame windows into byte ranges (from
    the keyframe preceding the window to the next keyframe after it).

    Works for any container/codec pyav can demux (keyframe flags and packet
    byte offsets are universal). Frame indices are derived as
    ``round(pts * average_rate)``, i.e. constant frame rate is assumed —
    matching the upstream reader's timestamp-to-index conversion. The moov
    fields are meaningful for mp4 only; see ``VIDEO_INDEX_COLUMNS``.
    """
    path = Path(path)
    file_size = path.stat().st_size
    kf_entries = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for packet in container.demux(stream):
            if packet.pts is None or not packet.is_keyframe or packet.pos is None:
                continue
            kf_entries.append((round(float(packet.pts * packet.time_base) * fps), packet.pos))
    kf_entries.sort()
    with open(path, "rb") as f:

        def read_at(offset: int, length: int) -> bytes:
            f.seek(offset)
            return f.read(length)

        moov_offset, moov_size = _find_moov(read_at, file_size)
    return {
        "file_size": file_size,
        "moov_offset": moov_offset,
        "moov_size": moov_size,
        "kf_indices": [index for index, _ in kf_entries],
        "kf_positions": [position for _, position in kf_entries],
    }
