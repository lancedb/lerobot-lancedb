#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Worked examples: converting real LeRobot datasets to Lance.

Each section below was actually run against the dataset named in the heading
— the schemas and sample rows are the literal output, not made up. All
datasets are public on the Hugging Face Hub under ``lerobot/*``.

To reproduce, install the package and run the shell commands. The Python
functions at the bottom of this file do the same thing programmatically.

::

    pip install lerobot-lancedb

What the converter does:

1. Resolves the source dataset (download from the Hub if not cached).
2. Walks episodes, reads the parquet frame rows and (when present) decodes
   the mp4 video chunks via the upstream ``decode_video_frames`` path.
3. Re-encodes every image / video frame as a JPEG blob and writes one
   Lance row per frame, schema-aligned across all episodes.
4. Copies ``meta/`` (info.json, stats.json, tasks.parquet, episodes/, and
   subtasks.parquet when present) verbatim into the output directory so
   downstream LeRobot code can read it back unchanged.

Output directory layout (same for every dataset)::

    <output>/
      <table>.lance/         ← the Lance table (read by LeRobotLanceDataset)
        data/...             ← column-paged storage; one fragment per write batch
        _versions/...        ← Lance transaction log
        _transactions/...
      meta/                  ← verbatim copy of the source meta/
        info.json
        stats.json
        tasks.parquet
        subtasks.parquet     (only when the source has subtasks)
        episodes/chunk-NNN/file-NNN.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from lerobot.utils.utils import init_logging

from lerobot_lancedb import LeRobotLanceDataset, benchmark_throughput, convert_to_lance


# ──────────────────────────────────────────────────────────────────────
# 1. Single-camera, video-backed — ``lerobot/pusht``
# ──────────────────────────────────────────────────────────────────────
#
# Smallest of the four. The single observation lives in a multi-episode
# mp4 chunk; the converter decodes those frames and re-encodes them as
# per-row JPEG blobs.
#
# Shell:
#     lerobot-convert-to-lance \
#         --repo-id=lerobot/pusht \
#         --output=./outputs/datasets/pusht_lance \
#         --overwrite
#
# Source layout (HF Hub):
#     data/chunk-000/file-000.parquet           ← tabular columns
#     videos/observation.image/chunk-000/file-000.mp4
#     meta/{info.json, stats.json, tasks.parquet, episodes/}
#
# Lance schema (./outputs/datasets/pusht_lance/pusht.lance):
#     episode_index:       int32
#     frame_index:         int32
#     index:               int64                  (global frame id)
#     timestamp:           float
#     task_index:          int32
#     observation_image:   binary                  ← JPEG-encoded
#     observation_state:   fixed_size_list<float>[2]
#     action:              fixed_size_list<float>[2]
#     next_reward:         fixed_size_list<float>[1]
#     next_done:           fixed_size_list<float>[1]
#     next_success:        fixed_size_list<float>[1]
#
# Sample rows (first 3 frames of episode 0):
#     ep=0 f=0 ts=0.000 img=<2383 B>  state=[222.00, 97.00]  action=[233.0, 71.0]  reward=[0.19]
#     ep=0 f=1 ts=0.100 img=<2436 B>  state=[225.25, 89.31]  action=[229.0, 83.0]  reward=[0.19]
#     ep=0 f=2 ts=0.200 img=<2383 B>  state=[227.59, 84.53]  action=[229.0, 86.0]  reward=[0.19]
#
# Stats — 206 episodes, 25 650 frames; conversion took ~14 s; lance table 60 MB.


# ──────────────────────────────────────────────────────────────────────
# 2. Single-camera, image-in-parquet (no video codec) — ``lerobot/pusht_image``
# ──────────────────────────────────────────────────────────────────────
#
# Same dataset as #1 but the source stores images as PIL bytes inside the
# parquet rows — there's no videos/ directory at all. Lance schema is
# identical to #1 because both upstream variants converge to the same
# row-per-frame layout.
#
# Shell:
#     lerobot-convert-to-lance \
#         --repo-id=lerobot/pusht_image \
#         --output=./outputs/datasets/pusht_image_lance \
#         --overwrite
#
# Source layout:
#     data/chunk-000/file-000.parquet     ← observation.image is a parquet column
#     meta/...
#     (no videos/ dir)
#
# Lance schema: identical to #1.
#
# Sample rows (first 3 frames of episode 0):
#     ep=0 f=0 ts=0.000 img=<2406 B>  state=[222.00, 97.00]  action=[233.0, 71.0]
#     ep=0 f=1 ts=0.100 img=<2433 B>  state=[225.25, 89.31]  action=[229.0, 83.0]
#     ep=0 f=2 ts=0.200 img=<2396 B>  state=[227.59, 84.53]  action=[229.0, 86.0]
#
# Stats — 206 episodes, 25 650 frames; conversion ~14 s; lance table 60 MB.


# ──────────────────────────────────────────────────────────────────────
# 3. Subtask labels — ``lerobot/pusht-subtask``
# ──────────────────────────────────────────────────────────────────────
#
# Same scene as pusht, plus a ``subtask_index`` column and a
# ``meta/subtasks.parquet`` sidecar that maps indices → human-readable
# names ("phase 1" / "phase 2" / "phase 3" for this dataset). The
# converter preserves both. ``LeRobotLanceDataset.__getitem__`` resolves
# the integer to a ``subtask`` string via ``meta.subtasks.iloc[idx]``.
#
# Shell:
#     lerobot-convert-to-lance \
#         --repo-id=lerobot/pusht-subtask \
#         --output=./outputs/datasets/pusht-subtask_lance \
#         --overwrite
#
# Lance schema (additions vs #1 marked with ←):
#     episode_index:           int32
#     frame_index:             int32
#     index:                   int64
#     timestamp:               float
#     task_index:              int32
#     subtask_index:           int32                       ←
#     observation_image:       binary
#     observation_state:       fixed_size_list<float>[2]
#     action:                  fixed_size_list<float>[2]
#     next_reward:             fixed_size_list<float>[1]
#     next_done:               fixed_size_list<float>[1]
#     next_success:            fixed_size_list<float>[1]
#     task_index_high_level:   fixed_size_list<float>[1]   ← user-defined feature
#
# meta/subtasks.parquet (sidecar; copied verbatim from the source):
#                  subtask_index
#     subtask
#     phase 1                  0
#     phase 2                  1
#     phase 3                  2
#
# 1 523 of 25 650 frames are labeled (subtask_index >= 0); the rest are
# ``-1`` ("no subtask").
#
# Loading back:
#     ds = LeRobotLanceDataset(root="./outputs/datasets/pusht-subtask_lance")
#     item = ds[0]
#     item["subtask_index"]   # tensor(-1)
#     item.get("subtask")     # None unless subtask_index >= 0
#
# Stats — 206 episodes, 25 650 frames; conversion ~12 s.


# ──────────────────────────────────────────────────────────────────────
# 4. Multi-camera, real-world bimanual — ``lerobot/aloha_static_cups_open``
# ──────────────────────────────────────────────────────────────────────
#
# Real ALOHA bimanual teleoperation data: 4 cameras (overhead high + low,
# plus a wrist camera on each arm), 14-DOF state/action (7 joints per arm),
# 50 Hz. This is the main "multi-camera" example.
#
# Shell:
#     lerobot-convert-to-lance \
#         --repo-id=lerobot/aloha_static_cups_open \
#         --output=./outputs/datasets/aloha_static_cups_open_lance \
#         --overwrite
#
# Source layout:
#     data/chunk-000/file-000.parquet
#     videos/observation.images.cam_high/chunk-000/file-000.mp4
#     videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4
#     videos/observation.images.cam_low/chunk-000/file-000.mp4
#     videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4
#     meta/...
#
# Lance schema (note the four binary image columns; ``.`` in feature
# names becomes ``_`` because Lance reserves dot as a struct-path
# separator):
#     episode_index:                        int32
#     frame_index:                          int32
#     index:                                int64
#     timestamp:                            float
#     task_index:                           int32
#     observation_images_cam_high:          binary     ← was observation.images.cam_high
#     observation_images_cam_left_wrist:    binary     ←
#     observation_images_cam_low:           binary     ←
#     observation_images_cam_right_wrist:   binary     ←
#     observation_state:                    fixed_size_list<float>[14]
#     action:                               fixed_size_list<float>[14]
#     next_done:                            fixed_size_list<float>[1]
#
# Sample rows (first 2 frames of episode 0; image columns shown as
# JPEG-blob byte length per camera):
#     ep=0 f=0 ts=0.000
#       cam_high=<42 205 B> cam_left_wrist=<48 456 B>
#       cam_low=<39 893 B>  cam_right_wrist=<48 909 B>
#       state=[-0.00, -0.96, ...]  action=[-0.01, -0.95, ...]
#     ep=0 f=1 ts=0.020
#       cam_high=<42 222 B> cam_left_wrist=<48 717 B>
#       cam_low=<39 893 B>  cam_right_wrist=<48 906 B>
#
# At read time, all four cameras are JPEG-decoded in batched calls per
# key — one ``torchvision.io.decode_jpeg`` invocation per camera per
# batch, not 4N decodes. ``LeRobotLanceDataset.__getitem__`` returns:
#     item["observation.images.cam_high"]        # (3, 480, 640) tensor
#     item["observation.images.cam_left_wrist"]  # (3, 480, 640) tensor
#     ...                                         (note: caller-facing keys
#                                                  keep the dotted form)
#
# Stats — 50 episodes, 20 000 frames; conversion ~13 min on a single CPU
# core (JPEG re-encoding is the bottleneck — 80 000 image encodes total).
# Source mp4 chunks total ~486 MB; lance output is 3.6 GB because every
# frame is now an independent JPEG (no inter-frame compression). The
# size-up-front is intentional: it's what makes random access fast.


# ──────────────────────────────────────────────────────────────────────
# Throughput comparison: parquet+mp4 vs Lance (measured)
# ──────────────────────────────────────────────────────────────────────
#
# Run with ``python examples/conversion.py --benchmark``. Numbers below
# are steady-state batches/sec after a 5-batch warmup, batch=32, on an
# M-series Mac with local SSD.
#
# Important — we only benchmark on the **larger** multi-camera datasets
# below. The pusht-family datasets (50 MB total) live entirely in the OS
# file cache after a couple of epochs, so both backends read from RAM
# and the comparison degenerates into "who has less per-batch Python
# overhead". That's not a useful measurement for choosing a backend; the
# question that matters is "what happens on real data that doesn't fit
# in RAM and forces actual I/O + decode work". The aloha datasets answer
# that.
#
# Two independent 4-camera × 480×640 ALOHA datasets — using both lets you
# spot run-to-run variance vs. structural effects. Lance is consistently
# faster on every condition. Speedup ratios shown vs. parquet at the same
# num_workers.
#
# lerobot/aloha_static_cups_open  (50 eps, 20 000 frames, 3.6 GB Lance):
#   condition                          backend     bps   frames/s  speedup
#   ───────────────────────────────────────────────────────────────────────
#   shuffled, nw=0                     parquet    2.43         78
#                                      lance      5.80        186   2.39x
#   shuffled, nw=4                     parquet    3.58        115
#                                      lance     13.90        445   3.88x
#   shuffled + delta_timestamps, nw=0  parquet    0.98         31
#                                      lance      1.75         56   1.78x
#   shuffled + delta_timestamps, nw=4  parquet    1.56         50
#                                      lance      2.53         81   1.62x
#
# lerobot/aloha_static_ziploc_slide  (56 eps, 16 800 frames, 3.2 GB Lance):
#   condition                          backend     bps   frames/s  speedup
#   ───────────────────────────────────────────────────────────────────────
#   shuffled, nw=0                     parquet    2.56         82
#                                      lance      6.05        193   2.36x
#   shuffled, nw=4                     parquet    5.99        192
#                                      lance     16.01        512   2.67x
#   shuffled + delta_timestamps, nw=0  parquet    1.24         40
#                                      lance      2.26         72   1.82x
#   shuffled + delta_timestamps, nw=4  parquet    1.92         62
#                                      lance      3.36        108   1.75x
#
# Why "only" 1.6-1.8× under realistic delta_timestamps?
#
# Per-batch profile on aloha_static_cups_open (4 cams × 480×640, bs=32):
#
#   lance fetch (Permutation.take):    7 ms  (~ 2%)
#   bytes → pylist (4 cameras):        5 ms  (~ 2%)
#   JPEG decode (libjpeg-turbo CPU): 272 ms  (~96%)
#
# 96% of Lance's batch time is CPU JPEG decode. Both backends spend most
# of their time decoding 256 × 480×640 frames per batch, and we can't
# beat raw libjpeg-turbo on the same CPU. The lance fetch is essentially
# free; the speedup over parquet+mp4 comes from avoiding torchcodec's
# per-window seek, which is significant but bounded.
#
# Where the gap widens (much larger speedups):
#   * GPU NVJPEG — set ``decode_device='cuda'`` to decode JPEGs on the GPU.
#     ~10× faster than CPU libjpeg-turbo. Extrapolating the profile above:
#     272 ms → 27 ms means total goes 284 → 39 ms = ~7× the current Lance
#     throughput, and decoded tensors land on the GPU directly (no H2D
#     copy in the training loop). The parquet+mp4 path could in theory
#     use NVDEC for the same effect but it's much harder to set up, and
#     torchcodec's CUDA support is patchy across codecs.
#   * Cloud reads (S3 / GCS / HF Buckets) — parquet+mp4 pays a network
#     round-trip per chunk fetch and per torchcodec seek. Lance reads the
#     specific byte-ranges it needs in one go. Expect 10-20× on cold
#     cloud reads.
#   * Larger batches / no delta_timestamps — torchcodec's seek amortization
#     advantage disappears when the read pattern is one independent frame
#     per item.


# ──────────────────────────────────────────────────────────────────────
# Programmatic conversion (equivalent to the shell commands above)
# ──────────────────────────────────────────────────────────────────────
#
# Useful in a script / notebook where you want to embed conversion in a
# pipeline, push to the Hub right after, etc.


DATASETS = [
    {"repo_id": "lerobot/pusht", "output": "outputs/datasets/pusht_lance"},
    {"repo_id": "lerobot/pusht_image", "output": "outputs/datasets/pusht_image_lance"},
    {"repo_id": "lerobot/pusht-subtask", "output": "outputs/datasets/pusht-subtask_lance"},
    {
        "repo_id": "lerobot/aloha_static_cups_open",
        "output": "outputs/datasets/aloha_static_cups_open_lance",
    },
    {
        "repo_id": "lerobot/aloha_static_ziploc_slide",
        "output": "outputs/datasets/aloha_static_ziploc_slide_lance",
    },
]


# Only the larger real-world datasets are benchmarked — see the comment
# block above for why we exclude the pusht-family.
#
# Both entries are 4-camera × 480×640 ALOHA bimanual datasets — the
# canonical "real training data" shape. Using two independent datasets
# of the same shape lets readers spot run-to-run variance vs structural
# effects.
_ALOHA_DELTAS = {
    # Observe (t-1, t) at 50 Hz; predict 16-step action chunk.
    "observation.images.cam_high": [-0.02, 0.0],
    "observation.images.cam_left_wrist": [-0.02, 0.0],
    "observation.images.cam_low": [-0.02, 0.0],
    "observation.images.cam_right_wrist": [-0.02, 0.0],
    "observation.state": [-0.02, 0.0],
    "action": [i / 50 for i in range(-1, 15)],
}


BENCHMARK_DATASETS = [
    {
        "repo_id": "lerobot/aloha_static_cups_open",
        "output": "outputs/datasets/aloha_static_cups_open_lance",
        "delta_timestamps": _ALOHA_DELTAS,
        "num_batches": 25,
        "warmup": 5,
    },
    {
        "repo_id": "lerobot/aloha_static_ziploc_slide",
        "output": "outputs/datasets/aloha_static_ziploc_slide_lance",
        "delta_timestamps": _ALOHA_DELTAS,
        "num_batches": 25,
        "warmup": 5,
    },
]


def convert_all(
    overwrite: bool = False,
    push_to_hub_owner: str | None = None,
) -> None:
    """Run every conversion in ``DATASETS``; optionally upload.

    Args:
        overwrite: Replace existing Lance tables.
        push_to_hub_owner: When set (e.g. ``"me"``), upload each converted
            dataset to ``hf://datasets/{owner}/{name}_lance`` after conversion.
            Requires ``HF_TOKEN`` or ``huggingface-cli login``.
    """
    for entry in DATASETS:
        repo_id = entry["repo_id"]
        output = Path(entry["output"])
        name = repo_id.split("/")[-1]

        push_to_hub = None
        if push_to_hub_owner is not None:
            push_to_hub = f"{push_to_hub_owner}/{name}_lance"

        t0 = time.perf_counter()
        logging.info("Converting %s → %s", repo_id, output)
        convert_to_lance(
            repo_id=repo_id,
            output=output,
            overwrite=overwrite,
            push_to_hub=push_to_hub,
            progress=False,
        )
        logging.info("  done in %.1fs", time.perf_counter() - t0)


def benchmark_all(
    batch_size: int = 32,
    num_workers: tuple[int, ...] = (0, 4),
) -> None:
    """Run the throughput benchmark on each entry in ``BENCHMARK_DATASETS``.

    Each entry is benchmarked twice — once without ``delta_timestamps``
    (raw frame-level random access) and once with the dataset's typical
    training-time delta window. The pair is what makes the comparison
    interpretable: the first row is "how fast can each backend serve
    independent frames"; the second is "how fast under a realistic
    training read pattern".

    Args:
        batch_size: DataLoader batch size.
        num_workers: Iterable of ``num_workers`` values to benchmark.
    """
    for entry in BENCHMARK_DATASETS:
        repo_id = entry["repo_id"]
        output = entry["output"]
        deltas = entry.get("delta_timestamps")
        num_batches = entry.get("num_batches", 30)
        warmup = entry.get("warmup", 5)

        if not Path(output).exists():
            logging.warning(
                "Skipping %s — %s not found. Run convert_all() first.",
                repo_id,
                output,
            )
            continue

        # Run A: no delta_timestamps.
        print(f"\n=== {repo_id} — shuffled, no delta_timestamps ===")
        benchmark_throughput(
            repo_id=repo_id,
            lance_root=output,
            batch_size=batch_size,
            num_workers=num_workers,
            num_batches=num_batches,
            warmup=warmup,
        )

        # Run B: with delta_timestamps (realistic training).
        if deltas:
            print(f"\n=== {repo_id} — shuffled + delta_timestamps ===")
            benchmark_throughput(
                repo_id=repo_id,
                lance_root=output,
                batch_size=batch_size,
                num_workers=num_workers,
                num_batches=num_batches,
                warmup=warmup,
                delta_timestamps=deltas,
            )


def smoke_load(root: str | Path) -> None:
    """Open a converted dataset and pull one frame to verify it works."""
    ds = LeRobotLanceDataset(root=root)
    logging.info("Loaded %s", ds)
    item = ds[0]
    logging.info("First item keys: %s", sorted(item.keys()))


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--push-to-hub-owner",
        type=str,
        default=None,
        help="Optional HF Hub user/org. Each dataset will be uploaded to "
        "hf://datasets/{owner}/{name}_lance.",
    )
    parser.add_argument(
        "--smoke-load",
        action="store_true",
        help="After converting, open each output to verify readability.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="After converting, run throughput comparisons against the upstream "
        "parquet+mp4 reader. Only the larger real-world datasets in "
        "BENCHMARK_DATASETS are measured — see the module docstring for why.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion (use existing output dirs) and go straight to "
        "smoke-load / benchmark.",
    )
    parser.add_argument(
        "--bench-batch-size",
        type=int,
        default=32,
        help="DataLoader batch size used by the throughput benchmark.",
    )
    parser.add_argument(
        "--bench-num-workers",
        type=int,
        nargs="+",
        default=[0, 4],
        help="DataLoader num_workers values to benchmark.",
    )
    args = parser.parse_args()

    if not args.skip_convert:
        convert_all(
            overwrite=args.overwrite,
            push_to_hub_owner=args.push_to_hub_owner,
        )

    if args.smoke_load:
        for entry in DATASETS:
            smoke_load(entry["output"])

    if args.benchmark:
        benchmark_all(
            batch_size=args.bench_batch_size,
            num_workers=tuple(args.bench_num_workers),
        )


if __name__ == "__main__":
    main()
