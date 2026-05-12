#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DataLoader throughput benchmark: parquet+mp4 vs Lance.

Standalone CLI wrapping :func:`lerobot_lancedb.benchmark_throughput`.

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

from lerobot.utils.utils import init_logging

from lerobot_lancedb import benchmark_throughput


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

    benchmark_throughput(
        repo_id=args.repo_id,
        lance_root=args.lance_root,
        src_root=args.src_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        warmup=args.warmup,
        backends=args.backends,
    )


if __name__ == "__main__":
    main()
