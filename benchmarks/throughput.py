#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DataLoader throughput benchmark: parquet+mp4 vs Lance.

Example::

    lerobot-convert-to-lance --repo-id=lerobot/pusht \\
        --output=./outputs/pusht_lance --overwrite

    python benchmarks/throughput.py \\
        --repo-id=lerobot/pusht \\
        --lance-root=./outputs/pusht_lance \\
        --batch-size=64 --num-workers 0 4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_lancedb import LeRobotLanceDataset


def _run(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    warmup: int,
) -> tuple[float, float]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    seen = 0
    t0 = None
    total_t0 = time.perf_counter()
    for i, _batch in enumerate(loader):
        if i == warmup:
            t0 = time.perf_counter()
        seen += 1
        if seen >= num_batches:
            break
    elapsed_steady = time.perf_counter() - (t0 if t0 is not None else total_t0)
    steady_batches = max(0, seen - warmup)
    bps = steady_batches / elapsed_steady if elapsed_steady > 0 else float("nan")
    return time.perf_counter() - total_t0, bps


def _print_table(rows: list[dict]) -> None:
    headers = ["backend", "num_workers", "total_s", "steady_bps", "frames_per_s"]
    widths = [max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths, strict=True)))


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--src-root", type=str, default=None)
    parser.add_argument("--lance-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        choices=["parquet", "lance"],
        default=["parquet", "lance"],
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for nw in args.num_workers:
        for backend in args.backends:
            if backend == "parquet":
                import inspect

                kwargs: dict = {"root": args.src_root}
                if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
                    kwargs["return_uint8"] = True
                ds: torch.utils.data.Dataset = LeRobotDataset(args.repo_id, **kwargs)
            else:
                ds = LeRobotLanceDataset(root=Path(args.lance_root), return_uint8=True)

            logging.info("Benchmarking backend=%s num_workers=%d", backend, nw)
            total, bps = _run(
                ds,
                batch_size=args.batch_size,
                num_workers=nw,
                num_batches=args.num_batches,
                warmup=args.warmup,
            )
            rows.append(
                {
                    "backend": backend,
                    "num_workers": nw,
                    "total_s": f"{total:.2f}",
                    "steady_bps": f"{bps:.2f}",
                    "frames_per_s": f"{bps * args.batch_size:.0f}",
                }
            )

    print()
    _print_table(rows)


if __name__ == "__main__":
    main()
