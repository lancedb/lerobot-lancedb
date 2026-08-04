#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Temporary decode shim for the vendored reader (`reader.py`).

Released lerobot's ``decode_video_frames_pyav`` does two things the reader's depth
path needs and that only landed in the loader PR: it accepts a seekable file-like
object (instead of ``str()``-ing the path), and it compares timestamps in float64
(float32 quantizes ~0.25 ms at hour-scale timestamps and trips ``tolerance_s`` on
long aggregated videos). This vendors that fixed function. RGB datasets never call
it (the reader only decodes depth through pyav), so this matters only for depth.

Delete this module once the loader lands upstream and ``reader.py`` imports
``decode_video_frames_pyav`` from ``lerobot.datasets.video_utils`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import torch

try:
    from lerobot.datasets.video_utils import FrameTimestampError
except ImportError:  # older lerobot: define our own (same base as upstream)

    class FrameTimestampError(ValueError):
        pass


logger = logging.getLogger(__name__)


def decode_video_frames_pyav(
    video_path,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:
    """PyAV frame decode by timestamp. Vendored from lerobot's video_utils with the
    loader-PR fixes (file-like input + float64 tolerance). See the module docstring."""
    if isinstance(video_path, (str, Path)):
        video_path = str(video_path)
    # else: a seekable file-like object is handed to av.open unchanged.

    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(
            round(first_ts / stream.time_base) - 1,
            backward=True,
            any_frame=False,
            stream=stream,
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logger.info(f"frame loaded at timestamp={current_ts:.4f}")
            if is_depth:
                arr = frame.to_ndarray(format="gray12le")  # (H, W) uint12
                loaded_frames.append(torch.from_numpy(arr).unsqueeze(0).contiguous())
            else:
                arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                loaded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

    if not loaded_frames:
        raise FrameTimestampError(
            f"No frames could be decoded from {video_path} in the timestamp range [{first_ts}, {last_ts}]."
        )

    # float64: float32 quantizes ~0.25 ms at hour-scale timestamps, tripping
    # tolerance_s spuriously on long aggregated video files.
    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)

    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance "
            f"({min_[~is_within_tol]} > {tolerance_s=}). The closest loadable frame is too far in "
            f"time.\nqueried: {query_ts}\nloaded: {loaded_ts_t}\nvideo: {video_path}\nbackend: pyav"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )

    if return_uint8 or is_depth:
        return closest_frames
    return closest_frames.type(torch.float32) / 255
