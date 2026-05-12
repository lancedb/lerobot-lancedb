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
# Run with ``python examples/conversion.py --benchmark`` to reproduce. Each
# row reports steady-state batches/sec after a 5-10 batch warmup, batch
# size 32, on an M-series Mac with local SSD. We measure ``num_workers=0``
# (single-process baseline) and ``num_workers=4`` (typical training config).
#
# dataset                        backend   nw      bps   frames/s   speedup
# ─────────────────────────────────────────────────────────────────────────
# pusht                          parquet    0    76.20      2438
# pusht                          parquet    4   292.98      9375
# pusht                          lance      0   218.53      6993    2.87x
# pusht                          lance      4   512.18     16390    1.75x
# pusht_image                    parquet    0   159.28      5097
# pusht_image                    parquet    4   466.88     14940
# pusht_image                    lance      0   192.86      6172    1.21x
# pusht_image                    lance      4   269.05      8609    0.58x   ⚠
# pusht-subtask                  parquet    0    81.45      2606
# pusht-subtask                  parquet    4   278.16      8901
# pusht-subtask                  lance      0   188.77      6041    2.32x
# pusht-subtask                  lance      4   471.08     15075    1.69x
# aloha_static_cups_open         parquet    0     2.42        78
# aloha_static_cups_open         parquet    4     4.41       141
# aloha_static_cups_open         lance      0     6.16       197    2.54x
# aloha_static_cups_open         lance      4    16.70       534    3.79x
#
# When Lance helps:
#   * Video-backed datasets — the parquet+mp4 path pays per-batch torchcodec
#     seek + decode; Lance just JPEG-decodes binary blobs (faster, GIL-free).
#   * Multi-camera datasets — the gap widens with the number of cameras,
#     and the bigger the frames the more dramatic the win (aloha @ nw=4 is
#     a textbook example: 3.79× faster).
#   * Cloud reads (S3 / HF Buckets / GCS) — not measured here, but
#     theoretically the largest gain because Lance avoids decoder fetch
#     round-trips entirely.
#
# When it doesn't (be honest about this):
#   * ``pusht_image`` at high ``num_workers``: parquet+pillow is already
#     plenty fast for 96×96 image-in-parquet frames, and at nw=4 the
#     bottleneck moves off-disk; Lance ends up *slower* here (0.58×).
#     Use Lance for the video / multi-cam case; for tiny image-in-parquet
#     datasets the upstream reader is fine.


# ──────────────────────────────────────────────────────────────────────
# Programmatic conversion (equivalent to the shell commands above)
# ──────────────────────────────────────────────────────────────────────
#
# Useful in a script / notebook where you want to embed conversion in a
# pipeline, push to the Hub right after, etc.


# Each entry has per-dataset benchmark knobs because aloha's parquet+mp4
# path is ~30x slower than the pusht variants — we need a smaller
# ``num_batches`` to keep the benchmark from taking forever.
DATASETS = [
    {
        "repo_id": "lerobot/pusht",
        "output": "outputs/datasets/pusht_lance",
        "bench": {"num_batches": 100, "warmup": 10},
    },
    {
        "repo_id": "lerobot/pusht_image",
        "output": "outputs/datasets/pusht_image_lance",
        "bench": {"num_batches": 100, "warmup": 10},
    },
    {
        "repo_id": "lerobot/pusht-subtask",
        "output": "outputs/datasets/pusht-subtask_lance",
        "bench": {"num_batches": 100, "warmup": 10},
    },
    {
        "repo_id": "lerobot/aloha_static_cups_open",
        "output": "outputs/datasets/aloha_static_cups_open_lance",
        "bench": {"num_batches": 30, "warmup": 5},
    },
]


def convert_all(
    overwrite: bool = False,
    push_to_hub_owner: str | None = None,
    benchmark: bool = False,
    batch_size: int = 32,
    num_workers: tuple[int, ...] = (0, 4),
) -> None:
    """Run every conversion in ``DATASETS``; optionally upload + benchmark.

    Args:
        overwrite: Replace existing Lance tables.
        push_to_hub_owner: When set (e.g. ``"me"``), upload each converted
            dataset to ``hf://datasets/{owner}/{name}_lance`` after conversion.
            Requires ``HF_TOKEN`` or ``huggingface-cli login``.
        benchmark: After each conversion, run a quick throughput comparison
            (parquet+mp4 vs Lance) and print the result table.
        batch_size, num_workers: DataLoader knobs forwarded to the benchmark.
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

        if benchmark:
            bench_kwargs = entry.get("bench", {})
            print(f"\n=== Throughput: {repo_id} ===")
            benchmark_throughput(
                repo_id=repo_id,
                lance_root=output,
                batch_size=batch_size,
                num_workers=num_workers,
                **bench_kwargs,
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
        help="After each conversion, print a throughput comparison "
        "(parquet+mp4 vs Lance) for that dataset.",
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

    convert_all(
        overwrite=args.overwrite,
        push_to_hub_owner=args.push_to_hub_owner,
        benchmark=args.benchmark,
        batch_size=args.bench_batch_size,
        num_workers=tuple(args.bench_num_workers),
    )

    if args.smoke_load:
        for entry in DATASETS:
            smoke_load(entry["output"])


if __name__ == "__main__":
    main()
