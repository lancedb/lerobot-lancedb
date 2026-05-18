#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""End-to-end tests that load reference datasets from the HuggingFace Hub.

These exercise the ``repo_id=`` code path against the published
``lance-format/pusht-lerobot-lancedb`` and ``lance-format/pusht-lerobot-lancedb-video``
datasets — the first publicly-loadable lerobot-lancedb-format Lance
datasets on the Hub.

Network-dependent. Skipped automatically if ``LEROBOT_LANCEDB_SKIP_HUB_TESTS=1``
or if the Hub is unreachable.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("datasets")
pytest.importorskip("huggingface_hub")


pytestmark = pytest.mark.skipif(
    os.environ.get("LEROBOT_LANCEDB_SKIP_HUB_TESTS") == "1",
    reason="LEROBOT_LANCEDB_SKIP_HUB_TESTS=1 set — skipping network-dependent Hub tests.",
)


FRAME_REPO = "lance-format/pusht-lerobot-lancedb"
VIDEO_REPO = "lance-format/pusht-lerobot-lancedb-video"


def test_frame_dataset_loads_from_hub():
    """``repo_id=`` against a published frame-layout dataset round-trips."""
    from lerobot_lancedb import LeRobotLanceDataset

    ds = LeRobotLanceDataset(repo_id=FRAME_REPO)
    assert len(ds) == 25_650  # known pusht frame count
    s = ds[0]
    assert s["observation.image"].shape == (3, 96, 96)
    assert s["observation.state"].shape == (2,)
    # Reads at a different offset must also work (catches "first row only" bugs).
    s2 = ds[5000]
    assert s2["observation.image"].shape == (3, 96, 96)
    assert int(s2["frame_index"]) >= 0


def test_video_dataset_loads_from_hub():
    """``repo_id=`` against a published video-blob dataset round-trips.

    Also exercises the local materialization fallback for ``_videos.lance``
    (lance's pyarrow API has no ``hf://`` provider for blob fetches).
    """
    from lerobot_lancedb import LeRobotLanceVideoDataset

    ds = LeRobotLanceVideoDataset(repo_id=VIDEO_REPO)
    assert len(ds) == 25_650
    s = ds[0]
    assert s["observation.image"].shape == (3, 96, 96)
    s2 = ds[5000]
    assert s2["observation.image"].shape == (3, 96, 96)


def test_init_rejects_non_lance_hub_repo():
    """Pointing at a regular parquet+mp4 dataset must fail at init, not later.

    Regression test for the silent-init-success bug: prior to the
    table-existence probe, ``LeRobotLanceDataset(repo_id='lerobot/pusht')``
    succeeded (because the ``meta/`` sidecar shape matches) and only blew
    up on ``__getitem__`` with an opaque 200-line lance traceback.
    """
    from lerobot_lancedb import LeRobotLanceDataset, LeRobotLanceVideoDataset

    with pytest.raises(FileNotFoundError, match="No lance table"):
        LeRobotLanceDataset(repo_id="lerobot/pusht")

    with pytest.raises(FileNotFoundError, match="Missing lance table"):
        LeRobotLanceVideoDataset(repo_id="lerobot/pusht")
