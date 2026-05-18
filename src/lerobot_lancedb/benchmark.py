#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared DataLoader throughput benchmark for parquet+mp4 vs Lance.

Imported by ``examples/conversion.py`` and ``benchmarks/throughput.py`` so
both surfaces use the same measurement code. Lance is significantly faster
on video-heavy and multi-camera datasets and roughly even (or slightly
slower at high worker counts) on small image-in-parquet datasets — see
the README for full numbers.

Note: this module forces ``spawn`` mode multiprocessing because the Lance
worker-spawn safety code does. If your script is launched via ``-c`` /
``<stdin>`` / a REPL, spawn workers can't re-import the main module — run
this as a real script file.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .dataset import LeRobotLanceDataset

logger = logging.getLogger(__name__)


def _open_parquet(
    repo_id: str,
    root: str | Path | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
) -> LeRobotDataset:
    """Open the upstream parquet+mp4 dataset, with ``return_uint8`` if supported."""
    kwargs: dict[str, Any] = {"root": root}
    if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
        kwargs["return_uint8"] = True
    if delta_timestamps:
        kwargs["delta_timestamps"] = delta_timestamps
    return LeRobotDataset(repo_id, **kwargs)


def _measure(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    warmup: int,
) -> tuple[float, float]:
    """Return ``(total_seconds, steady_state_batches_per_second)``."""
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
    t0: float | None = None
    total_start = time.perf_counter()
    for i, _batch in enumerate(loader):
        if i == warmup:
            t0 = time.perf_counter()
        seen += 1
        if seen >= num_batches:
            break
    elapsed = time.perf_counter() - (t0 if t0 is not None else total_start)
    steady = max(0, seen - warmup)
    bps = steady / elapsed if elapsed > 0 else float("nan")
    return time.perf_counter() - total_start, bps


def benchmark_throughput(
    repo_id: str,
    lance_root: str | Path,
    *,
    src_root: str | Path | None = None,
    batch_size: int = 32,
    num_workers: Iterable[int] = (0, 4),
    num_batches: int = 100,
    warmup: int = 10,
    backends: Iterable[str] = ("parquet", "lance"),
    delta_timestamps: dict[str, list[float]] | None = None,
    decode_device: str | None = "auto",
    print_results: bool = True,
) -> list[dict]:
    """Run a throughput comparison and (optionally) print it.

    Args:
        repo_id: Source dataset repo id (used to open the parquet+mp4 path).
        lance_root: Local dir produced by :func:`convert_to_lance`.
        src_root: Optional override for the parquet+mp4 source root. Defaults
            to the standard ``$HF_LEROBOT_HOME/{repo_id}`` cache.
        batch_size, num_workers, num_batches, warmup: Standard DataLoader knobs.
            ``warmup`` batches are timed-out before steady-state measurement
            starts.
        backends: Which backends to measure (``"parquet"``, ``"lance"``, or both).
        print_results: Print a tidy table to stdout. Always returns the rows.

    Returns:
        One dict per (backend, num_workers) row:
        ``{"backend", "num_workers", "total_s", "steady_bps", "frames_per_s"}``.
    """
    rows: list[dict] = []
    for nw in num_workers:
        for backend in backends:
            if backend == "parquet":
                ds: torch.utils.data.Dataset = _open_parquet(
                    repo_id, root=src_root, delta_timestamps=delta_timestamps
                )
            elif backend == "lance":
                ds = LeRobotLanceDataset(
                    root=Path(lance_root),
                    return_uint8=True,
                    delta_timestamps=delta_timestamps,
                    decode_device=decode_device,
                )
            else:
                raise ValueError(f"unknown backend {backend!r}")

            logger.info("Benchmarking %s backend=%s num_workers=%d", repo_id, backend, nw)
            total, bps = _measure(
                ds,
                batch_size=batch_size,
                num_workers=nw,
                num_batches=num_batches,
                warmup=warmup,
            )
            rows.append(
                {
                    "backend": backend,
                    "num_workers": nw,
                    "total_s": total,
                    "steady_bps": bps,
                    "frames_per_s": bps * batch_size,
                }
            )

    if print_results:
        print_throughput_table(rows, batch_size=batch_size)
    return rows


def print_throughput_table(rows: list[dict], *, batch_size: int) -> None:
    """Render benchmark rows as a tidy table on stdout."""
    headers = ["backend", "num_workers", "total_s", "steady_bps", "frames_per_s", "speedup_vs_parquet"]
    # Compute speedup vs parquet at the same num_workers.
    parquet_bps = {
        (r["backend"], r["num_workers"]): r["steady_bps"] for r in rows if r["backend"] == "parquet"
    }
    augmented = []
    for r in rows:
        sp = ""
        if r["backend"] != "parquet":
            ref = parquet_bps.get(("parquet", r["num_workers"]))
            if ref:
                sp = f"{r['steady_bps'] / ref:.2f}x"
        augmented.append(
            {
                "backend": r["backend"],
                "num_workers": str(r["num_workers"]),
                "total_s": f"{r['total_s']:.2f}",
                "steady_bps": f"{r['steady_bps']:.2f}",
                "frames_per_s": f"{r['frames_per_s']:.0f}",
                "speedup_vs_parquet": sp,
            }
        )
    widths = [max(len(h), max((len(r[h]) for r in augmented), default=0)) for h in headers]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in augmented:
        print("  ".join(r[h].ljust(w) for h, w in zip(headers, widths, strict=True)))


__all__ = ["benchmark_throughput", "print_throughput_table"]
