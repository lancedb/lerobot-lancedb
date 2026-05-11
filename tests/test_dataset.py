#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""End-to-end round-trip + spawn-worker tests for the Lance backend."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest
import torch

pytest.importorskip("lancedb", reason="lancedb is required")
pytest.importorskip("datasets", reason="datasets (lerobot[dataset]) is required")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_lancedb import LeRobotLanceDataset, convert_to_lance, make_lerobot_dataset
from lerobot_lancedb.auto import _detect_backend


_DEFAULT_FPS = 10
_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (4,), "names": ["s0", "s1", "s2", "s3"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
    # Use ``image`` to avoid the FFmpeg codec round-trip in tests; image and
    # video columns both become JPEG ``binary`` in Lance.
    "observation.image": {
        "dtype": "image",
        "shape": (3, 32, 48),
        "names": ["channels", "height", "width"],
    },
}


def _make_frame(rng: np.random.Generator) -> dict:
    img = rng.integers(0, 256, size=(3, 32, 48), dtype=np.uint8)
    return {
        "observation.state": torch.tensor(rng.standard_normal(4), dtype=torch.float32),
        "action": torch.tensor(rng.standard_normal(2), dtype=torch.float32),
        "observation.image": img,
        "task": "test_task",
    }


@pytest.fixture(scope="module")
def parquet_dataset(tmp_path_factory):
    """Build a tiny parquet+image LeRobotDataset on disk."""
    # LeRobotDatasetMetadata.create insists on a non-existent dir.
    root = tmp_path_factory.mktemp("src_ds_holder") / "ds"
    ds = LeRobotDataset.create(
        repo_id="lance_test/tiny",
        fps=_DEFAULT_FPS,
        features=_FEATURES,
        root=root,
        use_videos=False,
    )
    rng = np.random.default_rng(42)
    ep_lens = [5, 7]
    for ep in range(2):
        for _ in range(ep_lens[ep]):
            ds.add_frame(_make_frame(rng))
        ds.save_episode()
    ds.finalize()
    return root, sum(ep_lens), ep_lens


@pytest.fixture(scope="module")
def lance_dataset_dir(parquet_dataset, tmp_path_factory):
    src_root, _, _ = parquet_dataset
    out = tmp_path_factory.mktemp("lance_out")
    convert_to_lance(
        repo_id="lance_test/tiny",
        output=out,
        src_root=src_root,
        revision=None,
        overwrite=True,
        progress=False,
    )
    return out


# ── subclass identity ────────────────────────────────────────────────


def test_is_lerobot_dataset_subclass(lance_dataset_dir):
    """isinstance(ds, LeRobotDataset) must hold so trainers accept us."""
    ds = LeRobotLanceDataset(root=lance_dataset_dir, return_uint8=True)
    assert isinstance(ds, LeRobotDataset)


# ── round-trip / structure ───────────────────────────────────────────


def test_convert_creates_lance_table_and_meta(lance_dataset_dir):
    assert (lance_dataset_dir / "tiny.lance").is_dir()
    assert (lance_dataset_dir / "meta" / "info.json").is_file()
    assert (lance_dataset_dir / "meta" / "stats.json").is_file()
    assert (lance_dataset_dir / "meta" / "tasks.parquet").is_file()


def _open_src(src_root):
    """Open the upstream parquet LeRobotDataset, with ``return_uint8`` if supported."""
    import inspect

    kwargs: dict = {"repo_id": "lance_test/tiny", "root": src_root}
    if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
        kwargs["return_uint8"] = True
    return LeRobotDataset(**kwargs)


def test_round_trip_single_item_matches_source(parquet_dataset, lance_dataset_dir):
    src_root, total_frames, _ = parquet_dataset
    src = _open_src(src_root)
    lance_ds = LeRobotLanceDataset(root=lance_dataset_dir, return_uint8=True)

    assert len(lance_ds) == total_frames == len(src)
    assert lance_ds.fps == src.fps
    assert set(lance_ds.features) == set(src.features)

    for idx in (0, 1, total_frames - 1):
        a = src[idx]
        b = lance_ds[idx]
        assert int(a["episode_index"]) == int(b["episode_index"])
        assert int(a["index"]) == int(b["index"])
        assert int(a["frame_index"]) == int(b["frame_index"])
        torch.testing.assert_close(a["observation.state"], b["observation.state"])
        torch.testing.assert_close(a["action"], b["action"])
        a_img = a["observation.image"].to(torch.float32)
        b_img = b["observation.image"].to(torch.float32)
        assert a_img.shape == b_img.shape
        # JPEG is lossy; bound the mean abs delta.
        assert (a_img - b_img).abs().mean().item() < 16.0


def test_batched_getitems_matches_per_item(lance_dataset_dir):
    lance_ds = LeRobotLanceDataset(root=lance_dataset_dir, return_uint8=True)
    indices = [0, 2, 5]
    batched = lance_ds.__getitems__(indices)
    individual = [lance_ds[i] for i in indices]
    for b, s in zip(batched, individual, strict=True):
        torch.testing.assert_close(b["observation.state"], s["observation.state"])
        torch.testing.assert_close(b["action"], s["action"])
        assert int(b["index"]) == int(s["index"])


# ── delta_timestamps ────────────────────────────────────────────────


def test_delta_timestamps_window_and_padding(lance_dataset_dir):
    """3-step action window must be (3, A) with correct pad mask."""
    lance_ds = LeRobotLanceDataset(
        root=lance_dataset_dir,
        delta_timestamps={"action": [-0.1, 0.0, 0.1]},  # 1/_DEFAULT_FPS = 0.1s
        return_uint8=True,
    )
    item = lance_ds[0]
    assert item["action"].shape == (3, 2)
    assert item["action_is_pad"].tolist() == [True, False, False]
    mid = lance_ds[2]
    assert mid["action_is_pad"].tolist() == [False, False, False]


# ── episode sampler integration ─────────────────────────────────────


def test_episodes_kwarg_stored_for_sampler(lance_dataset_dir):
    """Lance reader stores episodes; the sampler does the actual filtering."""
    lance_ds = LeRobotLanceDataset(root=lance_dataset_dir, episodes=[1], return_uint8=True)
    assert lance_ds.episodes == [1]
    # Length is unfiltered — the sampler does the actual filtering.
    assert len(lance_ds) == 12  # 5 + 7


# ── pickling / spawn workers ─────────────────────────────────────────


def test_pickle_round_trip_drops_native_handles(lance_dataset_dir):
    import pickle

    ds = LeRobotLanceDataset(root=lance_dataset_dir, return_uint8=True)
    _ = ds[0]
    blob = pickle.dumps(ds)
    restored = pickle.loads(blob)
    assert restored._table is None
    assert restored._db is None
    assert restored._perm is None
    item = restored[0]
    assert "observation.state" in item


def test_spawn_worker_smoke(lance_dataset_dir):
    """A 2-worker DataLoader should iterate the dataset without crashing."""
    ds = LeRobotLanceDataset(root=lance_dataset_dir, return_uint8=True)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        shuffle=False,
        multiprocessing_context=mp.get_context("spawn"),
    )
    seen = 0
    for batch in loader:
        seen += int(batch["index"].numel())
        if seen >= len(ds):
            break
    assert seen >= 1


# ── auto helper ─────────────────────────────────────────────────────


def test_auto_detect_backend_rules(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _detect_backend(None, empty) == "parquet"

    lance_dir = tmp_path / "with_lance"
    (lance_dir / "pusht.lance").mkdir(parents=True)
    assert _detect_backend(None, lance_dir) == "lance"

    assert _detect_backend("me/pusht_lance", None) == "lance"
    assert _detect_backend("lerobot/pusht", None) == "parquet"
    assert _detect_backend("s3://bucket/foo.lance", None) == "lance"
    assert _detect_backend("s3://bucket/foo", None) == "parquet"


def test_make_lerobot_dataset_routes_to_lance(lance_dataset_dir):
    ds = make_lerobot_dataset(root=lance_dataset_dir, return_uint8=True)
    assert isinstance(ds, LeRobotLanceDataset)
    assert isinstance(ds, LeRobotDataset)
    assert len(ds) > 0


def test_make_lerobot_dataset_forces_lance(lance_dataset_dir):
    ds = make_lerobot_dataset(backend="lance", root=lance_dataset_dir, return_uint8=True)
    assert isinstance(ds, LeRobotLanceDataset)
