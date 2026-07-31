#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Companion tooling for lerobot's native Lance dataset support.

This package ships two tools:

- ``lerobot-lance-convert`` (:mod:`lerobot_lancedb.convert`) — converts a
  LeRobot v3.0 dataset to the three-table Lance layout read by lerobot's
  ``LanceDBDataset``.
- ``lerobot-lance-doctor`` (:mod:`lerobot_lancedb.doctor`) — audits an
  upstream-format dataset for silent defects before you convert or train.

The LOADER is not here anymore: it lives in lerobot core as
``lerobot.datasets.lancedb_dataset.LanceDBDataset``. Versions of this
package before 0.3.0 shipped the old standalone loader classes; pin
``lerobot-lancedb<0.3`` if you still need them.
"""

from __future__ import annotations

from .vendored_schema import (
    FRAMES_TABLE,
    META_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEO_INDEX_COLUMNS,
    VIDEOS_TABLE,
    build_video_byte_index,
    to_lance_column,
)

__version__ = "0.3.0"

__all__ = [
    "FRAMES_TABLE",
    "META_TABLE",
    "VIDEOS_TABLE",
    "VIDEO_BLOB_COLUMN",
    "VIDEO_INDEX_COLUMNS",
    "build_video_byte_index",
    "to_lance_column",
    "__version__",
]
