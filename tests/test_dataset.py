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


def _h264_encoder_kwargs() -> dict:
    """Encoder kwargs for LeRobotDataset.create across lerobot versions.

    h264 instead of the libsvtav1 default: faster and always present in CI ffmpeg.
    """
    import inspect

    params = inspect.signature(LeRobotDataset.create).parameters
    if "vcodec" in params:
        return {"vcodec": "h264"}
    from lerobot.configs.video import RGBEncoderConfig

    return {"rgb_encoder": RGBEncoderConfig(vcodec="h264")}


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


def test_lance_dataset_owns_write_api_delegation():
    """Write API shape matches LeRobot without inheriting its writer internals."""
    assert LeRobotLanceDataset.add_frame is not LeRobotDataset.add_frame
    assert LeRobotLanceDataset.save_episode is not LeRobotDataset.save_episode
    assert LeRobotLanceDataset.clear_episode_buffer is not LeRobotDataset.clear_episode_buffer
    assert LeRobotLanceDataset.has_pending_frames is not LeRobotDataset.has_pending_frames


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
    # Force CPU decode so we can compare bytewise against the parquet+mp4 source,
    # which always returns CPU tensors. (With ``decode_device="auto"`` the Lance
    # path produces CUDA tensors on a GPU box and the comparison crashes on the
    # device mismatch.)
    lance_ds = LeRobotLanceDataset(root=lance_dataset_dir, return_uint8=True, decode_device="cpu")

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


# ── direct Lance writer ──────────────────────────────────────────────


def test_direct_writer_add_frame_round_trip(tmp_path):
    root = tmp_path / "direct_lance"
    ds = LeRobotLanceDataset.create(
        repo_id="lance_test/direct",
        fps=_DEFAULT_FPS,
        features=_FEATURES,
        root=root,
        use_videos=False,
    )
    rng = np.random.default_rng(123)
    frames = [_make_frame(rng) for _ in range(4)]
    for frame in frames:
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()

    assert (root / "direct.lance").is_dir()
    assert (root / "meta" / "info.json").is_file()
    assert (root / "meta" / "stats.json").is_file()
    assert (root / "meta" / "tasks.parquet").is_file()

    read = LeRobotLanceDataset(root=root, return_uint8=True, decode_device="cpu")
    assert len(read) == 4
    assert read.fps == _DEFAULT_FPS
    assert set(read.features) == set(ds.features)

    item = read[0]
    torch.testing.assert_close(item["observation.state"], frames[0]["observation.state"])
    torch.testing.assert_close(item["action"], frames[0]["action"])
    assert int(item["episode_index"]) == 0
    assert int(item["frame_index"]) == 0
    assert int(item["index"]) == 0
    assert item["task"] == "test_task"
    assert item["observation.image"].shape == (3, 32, 48)
    assert item["observation.image"].dtype == torch.float32


def test_direct_writer_save_episode_batch(tmp_path):
    root = tmp_path / "direct_batch_lance"
    ds = LeRobotLanceDataset.create(
        repo_id="lance_test/direct_batch",
        fps=_DEFAULT_FPS,
        features=_FEATURES,
        root=root,
        use_videos=False,
        data_files_size_in_mb=64,
        video_files_size_in_mb=128,
    )
    assert ds.meta.data_files_size_in_mb == 64
    assert ds.meta.video_files_size_in_mb == 128
    rng = np.random.default_rng(321)
    states = rng.standard_normal((3, 4)).astype(np.float32)
    actions = rng.standard_normal((3, 2)).astype(np.float32)
    images = rng.integers(0, 256, size=(3, 3, 32, 48), dtype=np.uint8)
    ds.save_episode(
        {
            "observation.state": states,
            "action": actions,
            "observation.image": images,
            "task": "batch_task",
        }
    )
    ds.finalize()

    read = LeRobotLanceDataset(root=root, return_uint8=True, decode_device="cpu")
    assert len(read) == 3
    item = read[2]
    torch.testing.assert_close(item["observation.state"], torch.from_numpy(states[2]))
    torch.testing.assert_close(item["action"], torch.from_numpy(actions[2]))
    assert int(item["frame_index"]) == 2
    assert item["task"] == "batch_task"


def test_direct_writer_streams_multiple_episodes_to_one_fragment(tmp_path):
    import lancedb

    root = tmp_path / "direct_multi_episode_lance"
    ds = LeRobotLanceDataset.create(
        repo_id="lance_test/direct_multi_episode",
        fps=_DEFAULT_FPS,
        features=_FEATURES,
        root=root,
        use_videos=False,
    )
    rng = np.random.default_rng(456)

    for _ in range(3):
        for _ in range(2):
            ds.add_frame(_make_frame(rng))
        ds.save_episode()
    ds.finalize()

    db = lancedb.connect(str(root))
    table = db.open_table("direct_multi_episode")
    lance_ds = table.to_lance()
    assert lance_ds.count_rows() == 6
    assert len(lance_ds.get_fragments()) == 1

    read = LeRobotLanceDataset(root=root, return_uint8=True, decode_device="cpu")
    assert len(read) == 6
    assert int(read[0]["episode_index"]) == 0
    assert int(read[2]["episode_index"]) == 1
    assert int(read[5]["episode_index"]) == 2
    assert int(read[5]["index"]) == 5


def test_direct_writer_video_dtype_uses_frames_layout(tmp_path):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["s0", "s1", "s2", "s3"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
        "observation.image": {
            "dtype": "video",
            "shape": (3, 32, 48),
            "names": ["channels", "height", "width"],
        },
    }
    root = tmp_path / "direct_video_lance"
    ds = LeRobotLanceDataset.create(
        repo_id="lance_test/direct_video",
        fps=_DEFAULT_FPS,
        features=features,
        root=root,
        use_videos=True,
    )
    rng = np.random.default_rng(55)
    for _ in range(2):
        ds.add_frame(
            {
                "observation.state": rng.standard_normal(4).astype(np.float32),
                "action": rng.standard_normal(2).astype(np.float32),
                "observation.image": rng.integers(0, 256, size=(3, 32, 48), dtype=np.uint8),
                "task": "video_task",
            }
        )
    ds.save_episode()
    ds.finalize()

    read = LeRobotLanceDataset(root=root, return_uint8=True, decode_device="cpu")
    item = read[0]
    assert read.meta.video_keys == ["observation.image"]
    assert (root / "direct_video.lance").is_dir()
    assert not (root / "direct_video_videos.lance").exists()
    assert item["observation.image"].shape == (3, 32, 48)
    assert item["observation.image"].dtype == torch.uint8


# ── GPU JPEG decode (skipped if no CUDA) ──────────────────────────────


def test_decode_on_gpu_returns_cuda_tensors(lance_dataset_dir):
    """If CUDA is available, ``decode_device='cuda'`` returns CUDA tensors.

    The actual NVJPEG decode is also significantly faster (~10× over
    libjpeg-turbo) on a typical GPU, but we don't benchmark here — this
    is just an API smoke-test.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    ds = LeRobotLanceDataset(
        root=lance_dataset_dir, return_uint8=True, decode_device="cuda"
    )
    item = ds[0]
    assert item["observation.image"].device.type == "cuda"


# ── multi-camera (dtype=image) ───────────────────────────────────────


_MULTICAM_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (4,), "names": ["s0", "s1", "s2", "s3"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
    "observation.images.cam_left": {
        "dtype": "image",
        "shape": (3, 24, 32),
        "names": ["channels", "height", "width"],
    },
    "observation.images.cam_right": {
        "dtype": "image",
        "shape": (3, 24, 32),
        "names": ["channels", "height", "width"],
    },
}


@pytest.fixture(scope="module")
def multicam_parquet_dataset(tmp_path_factory):
    """A 2-camera parquet+image dataset (no mp4 codec)."""
    root = tmp_path_factory.mktemp("multicam_holder") / "ds"
    ds = LeRobotDataset.create(
        repo_id="lance_test/multicam",
        fps=_DEFAULT_FPS,
        features=_MULTICAM_FEATURES,
        root=root,
        use_videos=False,
    )
    rng = np.random.default_rng(7)
    for _ in range(2):  # two episodes of 4 frames each
        for _ in range(4):
            ds.add_frame(
                {
                    "observation.state": torch.tensor(
                        rng.standard_normal(4), dtype=torch.float32
                    ),
                    "action": torch.tensor(rng.standard_normal(2), dtype=torch.float32),
                    "observation.images.cam_left": rng.integers(
                        0, 256, size=(3, 24, 32), dtype=np.uint8
                    ),
                    "observation.images.cam_right": rng.integers(
                        0, 256, size=(3, 24, 32), dtype=np.uint8
                    ),
                    "task": "multicam_task",
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture(scope="module")
def multicam_lance_dir(multicam_parquet_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("multicam_lance_out")
    convert_to_lance(
        repo_id="lance_test/multicam",
        output=out,
        src_root=multicam_parquet_dataset,
        overwrite=True,
        progress=False,
    )
    return out


def test_multicam_two_image_keys_decoded(multicam_lance_dir):
    """Both camera keys must come back as (C, H, W) tensors per frame."""
    ds = LeRobotLanceDataset(root=multicam_lance_dir, return_uint8=True)
    item = ds[0]
    assert "observation.images.cam_left" in item
    assert "observation.images.cam_right" in item
    # dtype="image" features always come back as float32 [0, 1] regardless of
    # return_uint8 — that's what upstream LeRobotDataset does too.
    for cam in ("observation.images.cam_left", "observation.images.cam_right"):
        assert item[cam].shape == (3, 24, 32)
        assert item[cam].dtype == torch.float32


def test_multicam_batched_decode_is_one_call_per_key(multicam_lance_dir):
    """Batched getitems must produce one tensor per (sample, camera).

    The internal blob_layout is keyed per-camera and decoded in one batched
    torchvision call per camera; this test catches regressions where camera
    keys get mixed up or one decode call leaks blobs from another camera.
    """
    ds = LeRobotLanceDataset(root=multicam_lance_dir, return_uint8=True)
    batch = ds.__getitems__([0, 1, 2, 3])
    for s in range(4):
        assert batch[s]["observation.images.cam_left"].shape == (3, 24, 32)
        assert batch[s]["observation.images.cam_right"].shape == (3, 24, 32)
    # Different cameras must produce different pixels (JPEG of random arrays).
    assert not torch.equal(
        batch[0]["observation.images.cam_left"],
        batch[0]["observation.images.cam_right"],
    )


# ── subtasks ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def subtasks_lance_dir(parquet_dataset, tmp_path_factory):
    """Reuse the standard parquet fixture; add a subtasks sidecar manually.

    Skipped under lerobot>=0.6, which removed ``meta.subtasks``.

    Subtasks aren't writable via LeRobotDataset.add_frame in 0.5.x — they
    live as a parquet sidecar (``meta/subtasks.parquet``) that the converter
    copies verbatim. We synthesise one with a single subtask row + a
    subtask_index column inside the parquet via direct file edits.
    """
    import shutil

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_root, _, _ = parquet_dataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if not hasattr(LeRobotDatasetMetadata, "subtasks"):
        pytest.skip("installed lerobot does not expose meta.subtasks (removed in 0.6)")
    holder = tmp_path_factory.mktemp("subtasks_holder")
    src_copy = holder / "ds"
    shutil.copytree(src_root, src_copy)

    # 1) Add a subtasks.parquet sidecar (index name "subtask").
    subtasks_df = pd.DataFrame({"subtask_index": [0, 1]}, index=["sub_a", "sub_b"])
    subtasks_df.index.name = "subtask"
    subtasks_df.to_parquet(src_copy / "meta" / "subtasks.parquet")

    # 2) Inject a subtask_index column into every data parquet (round-robin
    #    between subtasks 0 and 1 so the test can distinguish them).
    data_files = sorted((src_copy / "data").rglob("*.parquet"))
    assert data_files, "expected at least one data parquet"
    for fp in data_files:
        tbl = pq.read_table(fp)
        n = tbl.num_rows
        sub_idx = pa.array([i % 2 for i in range(n)], type=pa.int32())
        tbl = tbl.append_column("subtask_index", sub_idx)
        pq.write_table(tbl, fp)

    # 3) Register the new column in info.json so meta.features knows about it.
    import json

    info_path = src_copy / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["subtask_index"] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    info_path.write_text(json.dumps(info))

    # 4) Convert.
    out = tmp_path_factory.mktemp("subtasks_lance_out")
    convert_to_lance(
        repo_id="lance_test/tiny",
        output=out,
        src_root=src_copy,
        overwrite=True,
        progress=False,
    )
    return out


def test_subtask_index_round_trips(subtasks_lance_dir):
    """subtask_index column + subtask string lookup both flow through."""
    ds = LeRobotLanceDataset(root=subtasks_lance_dir, return_uint8=True)
    item0 = ds[0]
    item1 = ds[1]

    # Numeric column round-trips.
    assert int(item0["subtask_index"]) == 0
    assert int(item1["subtask_index"]) == 1

    # String resolution via meta.subtasks.
    assert item0.get("subtask") == "sub_a"
    assert item1.get("subtask") == "sub_b"


# ── video features (dtype="video") — gated on ffmpeg ─────────────────


@pytest.fixture(scope="module")
def video_parquet_dataset(tmp_path_factory):
    """A 1-camera video-backed dataset; needs ffmpeg to encode mp4 chunks.

    Skipped automatically when ffmpeg / torchcodec aren't available.
    """
    pytest.importorskip("torchcodec", reason="torchcodec is required for video features")
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not on PATH")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["s0", "s1", "s2", "s3"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
        # dtype="video" → mp4 storage on the source side; the converter
        # decodes and re-encodes per-frame as JPEG into the Lance table.
        "observation.image": {
            "dtype": "video",
            "shape": (3, 32, 48),
            "names": ["channels", "height", "width"],
        },
    }

    root = tmp_path_factory.mktemp("video_src_holder") / "ds"
    # Use h264; libsvtav1 (the LeRobot default) is slow + can be missing.
    ds = LeRobotDataset.create(
        repo_id="lance_test/video",
        fps=_DEFAULT_FPS,
        features=features,
        root=root,
        use_videos=True,
        # lerobot<0.6 takes vcodec=...; >=0.6 takes rgb_encoder=RGBEncoderConfig(...)
        **_h264_encoder_kwargs(),
    )
    rng = np.random.default_rng(11)
    for _ in range(2):
        for _ in range(4):
            ds.add_frame(
                {
                    "observation.state": torch.tensor(
                        rng.standard_normal(4), dtype=torch.float32
                    ),
                    "action": torch.tensor(rng.standard_normal(2), dtype=torch.float32),
                    "observation.image": rng.integers(
                        0, 256, size=(3, 32, 48), dtype=np.uint8
                    ),
                    "task": "video_task",
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture(scope="module")
def video_lance_dir(video_parquet_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("video_lance_out")
    convert_to_lance(
        repo_id="lance_test/video",
        output=out,
        src_root=video_parquet_dataset,
        overwrite=True,
        progress=False,
    )
    return out


def test_video_round_trip_shape_and_normalization(video_parquet_dataset, video_lance_dir):
    """A dtype=video feature must convert to JPEG-binary in Lance and come
    back as a (C, H, W) tensor. Pixels won't match byte-for-byte (mp4 → frame
    → JPEG → frame is a triple-lossy path), but shape, dtype and range must
    be right and the image must not be all-zeros.
    """
    ds = LeRobotLanceDataset(root=video_lance_dir, return_uint8=True)
    item = ds[0]
    img = item["observation.image"]
    assert img.shape == (3, 32, 48)
    # dtype=video honors return_uint8 (unlike dtype=image).
    assert img.dtype == torch.uint8
    # Decoded image isn't degenerate.
    assert int(img.max()) - int(img.min()) > 5


def test_video_return_uint8_false_yields_float32(video_lance_dir):
    """With return_uint8=False, video frames come back as float32 in [0, 1]."""
    ds = LeRobotLanceDataset(root=video_lance_dir, return_uint8=False)
    img = ds[0]["observation.image"]
    assert img.dtype == torch.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
