#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Companion tooling for lerobot's native Lance dataset support.

This package ships:

- ``lerobot-lance-convert`` (:mod:`lerobot_lancedb.convert`) — converts a
  LeRobot v2.0, v2.1, or v3.0 dataset to the three-table Lance layout.
- ``lerobot-lance-doctor`` (:mod:`lerobot_lancedb.doctor`) — audits an
  upstream-format dataset for silent defects before you convert or train.
- :class:`LanceDBDataset` — the map-style training loader for the Lance layout.

``LanceDBDataset`` is a **temporarily vendored** copy of lerobot's loader (the
open reader PR), so ``pip install lerobot-lancedb`` works end to end against
released lerobot. Once the loader lands in lerobot core, this re-exports
``lerobot.datasets.lancedb_dataset.LanceDBDataset`` and the vendored copy is
deleted. Import it the same way either way::

    from lerobot_lancedb import LanceDBDataset
    ds = LanceDBDataset(root="s3://bucket/pusht-lance")
"""

from __future__ import annotations

from .reader import (
    FRAMES_TABLE,
    META_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEO_INDEX_COLUMNS,
    VIDEOS_TABLE,
    LanceDBDataset,
    build_video_byte_index,
    is_lance_dataset,
    lance_mp_context,
    to_lance_column,
)

__version__ = "0.3.0"

__all__ = [
    "LanceDBDataset",
    "is_lance_dataset",
    "lance_mp_context",
    "FRAMES_TABLE",
    "META_TABLE",
    "VIDEOS_TABLE",
    "VIDEO_BLOB_COLUMN",
    "VIDEO_INDEX_COLUMNS",
    "build_video_byte_index",
    "to_lance_column",
    "__version__",
]
