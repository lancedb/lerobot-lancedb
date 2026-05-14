#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Cross-format benchmark for lerobot-lancedb.

Measures, for each (dataset × storage format) combination:

* on-disk size,
* pixel-level diff vs upstream (the canonical reference each format is
  trying to preserve),
* dataloader throughput at fixed batch and num_workers.

Formats:

* ``jpeg-95`` — current default (JPEG-95, 4:2:0 chroma)
* ``jpeg-100-444`` — near-lossless JPEG (q=100, no chroma subsampling)
* ``png`` — bit-exact, lossless (CPU decode only)
* ``video`` — Lance blob v2 with the original mp4 bytes stored verbatim;
  on-the-fly torchcodec decode

Conversions run on demand and are cached under ``--out-root``. Re-running
the script reuses what's already there.

Run:
    python examples/benchmark_formats.py \\
        --repos lerobot/pusht lerobot/pusht_image lerobot/aloha_static_cups_open \\
        --batch-size 32 --num-workers 4 --num-batches 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

from lerobot_lancedb import LeRobotLanceDataset, LeRobotLanceVideoDataset
from lerobot_lancedb.writer import convert_to_lance, convert_to_lance_video


log = logging.getLogger("benchmark_formats")


FORMATS = ("jpeg-95", "jpeg-100-444", "png", "video")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--repos",
        nargs="+",
        default=["lerobot/pusht", "lerobot/pusht_image", "lerobot/aloha_static_cups_open"],
    )
    p.add_argument("--out-root", type=Path, default=Path("outputs/datasets"))
    p.add_argument("--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--delta-timestamps", action="store_true",
                   help="Use a realistic-training-shape delta_timestamps spec "
                        "(8 frames per sample on the action key) to surface "
                        "the random-access read pattern. Default off (single-frame access).")
    p.add_argument("--n-pixel-samples", type=int, default=16, help="Frames to sample for pixel diff.")
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--skip-pixel-diff", action="store_true")
    p.add_argument("--results-json", type=Path, default=Path("outputs/benchmark_formats.json"))
    return p.parse_args()


def _format_dir(out_root: Path, repo_id: str, fmt: str) -> Path:
    # Match the naming used by examples/conversion.py: drop the org prefix.
    base = repo_id.split("/")[-1]
    suffix = {
        "jpeg-95": "_lance",
        "jpeg-100-444": "_lance_j100",
        "png": "_lance_lossless",
        "video": "_lance_video",
    }[fmt]
    return out_root / f"{base}{suffix}"


def _is_image_dataset(repo_id: str) -> bool:
    meta = LeRobotDatasetMetadata(repo_id)
    return bool(meta.image_keys) and not bool(meta.video_keys)


def ensure_dataset(repo_id: str, fmt: str, out_root: Path) -> Path | None:
    target = _format_dir(out_root, repo_id, fmt)
    if fmt == "video" and _is_image_dataset(repo_id):
        log.info("[%s] format=video N/A for image-stored dataset; skipping", repo_id)
        return None
    ready = (
        target.exists()
        and (target / "meta" / "info.json").exists()
        and any(target.glob("*.lance"))
    )
    if ready:
        log.info("[%s] format=%s reusing %s", repo_id, fmt, target)
        return target
    log.info("[%s] format=%s converting → %s", repo_id, fmt, target)
    t0 = time.perf_counter()
    if fmt == "jpeg-95":
        convert_to_lance(repo_id=repo_id, output=target, overwrite=True, progress=False)
    elif fmt == "jpeg-100-444":
        convert_to_lance(
            repo_id=repo_id, output=target, jpeg_quality=100,
            chroma_subsampling=0, overwrite=True, progress=False,
        )
    elif fmt == "png":
        convert_to_lance(
            repo_id=repo_id, output=target, lossless=True, overwrite=True, progress=False,
        )
    elif fmt == "video":
        convert_to_lance_video(repo_id=repo_id, output=target, overwrite=True, progress=False)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    log.info("[%s] format=%s convert wall=%.1fs", repo_id, fmt, time.perf_counter() - t0)
    return target


def _du_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def open_dataset(repo_id: str, fmt: str, root: Path, *, delta_timestamps=None) -> torch.utils.data.Dataset:
    if fmt == "video":
        return LeRobotLanceVideoDataset(
            root=root, repo_id=repo_id, delta_timestamps=delta_timestamps, return_uint8=True,
        )
    return LeRobotLanceDataset(
        root=root, repo_id=repo_id, delta_timestamps=delta_timestamps, return_uint8=True,
        decode_device="cpu",
    )


def _delta_timestamps_for(repo_id: str) -> dict[str, list[float]]:
    """Realistic-training delta pattern: 8 frames per sample for each video/action
    key (matches the shape lerobot's DiffusionPolicy/ACT use during training)."""
    meta = LeRobotDatasetMetadata(repo_id)
    fps = meta.fps
    deltas = [t / fps for t in range(-1, 7)]  # 8 frames
    out: dict[str, list[float]] = {}
    for k in meta.camera_keys:
        out[k] = deltas
    if "action" in meta.features:
        out["action"] = deltas
    if "observation.state" in meta.features:
        out["observation.state"] = deltas
    return out


def pixel_diff_against_upstream(repo_id: str, fmt: str, root: Path, n_samples: int) -> dict[str, float]:
    """Return mean/max/visible-frac across N random frames, averaged across camera keys."""
    up = LeRobotDataset(repo_id=repo_id)
    ds = open_dataset(repo_id, fmt, root)
    rng = np.random.default_rng(0)
    indices = rng.choice(min(len(up), len(ds)), size=min(n_samples, len(up)), replace=False)

    cams = list(up.meta.camera_keys)
    means, maxes, vis = [], [], []
    for idx in indices:
        a_batch = up[int(idx)]
        b_batch = ds[int(idx)]
        for cam in cams:
            a = a_batch[cam]
            b = b_batch[cam]
            if a.dtype == torch.uint8: a = a.float() / 255
            if b.dtype == torch.uint8: b = b.float() / 255
            d = (a - b).abs()
            means.append(float(d.mean().item()))
            maxes.append(float(d.max().item()))
            vis.append(float((d > 2 / 255).float().mean().item()))
    return {
        "mean_abs_diff": float(np.mean(means)),
        "max_abs_diff": float(np.max(maxes)),
        "visible_frac": float(np.mean(vis)),
    }


def throughput(
    repo_id: str,
    fmt: str,
    root: Path,
    *,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    warmup: int,
    delta_timestamps: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    ds = open_dataset(repo_id, fmt, root, delta_timestamps=delta_timestamps)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        drop_last=True, persistent_workers=num_workers > 0,
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
    return {
        "batches_per_s": bps,
        "frames_per_s": bps * batch_size,
        "total_s": time.perf_counter() - total_start,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    results: list[dict[str, Any]] = []

    # Also include upstream as a row (for reference throughput / size).
    for repo_id in args.repos:
        meta = LeRobotDatasetMetadata(repo_id)
        is_image = _is_image_dataset(repo_id)
        log.info("=== %s (image-stored=%s) ===", repo_id, is_image)

        dts = _delta_timestamps_for(repo_id) if args.delta_timestamps else None

        # Upstream row
        try:
            up_size = _du_bytes(meta.root)
        except Exception:
            up_size = -1
        upstream_row: dict[str, Any] = {
            "repo_id": repo_id,
            "format": "upstream",
            "is_image_dataset": is_image,
            "size_bytes": up_size,
        }
        if not args.skip_throughput:
            try:
                up_ds = LeRobotDataset(repo_id=repo_id, delta_timestamps=dts)
                tp = throughput_with_ds(
                    up_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    num_batches=args.num_batches, warmup=args.warmup,
                )
                upstream_row.update(tp)
            except Exception as e:
                upstream_row["error"] = repr(e)
        upstream_row["mean_abs_diff"] = 0.0
        upstream_row["max_abs_diff"] = 0.0
        upstream_row["visible_frac"] = 0.0
        results.append(upstream_row)

        for fmt in args.formats:
            row: dict[str, Any] = {"repo_id": repo_id, "format": fmt, "is_image_dataset": is_image}
            try:
                root = ensure_dataset(repo_id, fmt, args.out_root)
                if root is None:
                    row["skipped"] = "N/A for image-stored dataset"
                    results.append(row)
                    continue
                row["size_bytes"] = _du_bytes(root)
                if not args.skip_pixel_diff:
                    row.update(pixel_diff_against_upstream(repo_id, fmt, root, args.n_pixel_samples))
                if not args.skip_throughput:
                    row.update(throughput(
                        repo_id, fmt, root,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        num_batches=args.num_batches, warmup=args.warmup,
                        delta_timestamps=dts,
                    ))
            except Exception as e:
                row["error"] = repr(e)
                log.exception("Failed [%s, %s]", repo_id, fmt)
            results.append(row)

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(results, indent=2))
    print_summary(results)


def throughput_with_ds(ds, *, batch_size, num_workers, num_batches, warmup) -> dict[str, float]:
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        drop_last=True, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    seen = 0
    t0 = None
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
    return {
        "batches_per_s": bps,
        "frames_per_s": bps * batch_size,
        "total_s": time.perf_counter() - total_start,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    repos = sorted({r["repo_id"] for r in results})
    for repo in repos:
        rows = [r for r in results if r["repo_id"] == repo]
        print()
        print(f"=== {repo} ===")
        headers = ["format", "size_MB", "frames/s", "mean_abs", "max_abs", "visible_%"]
        widths = [14, 10, 10, 10, 10, 11]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for r in rows:
            size_mb = r.get("size_bytes", -1) / 1024 / 1024 if r.get("size_bytes", -1) >= 0 else float("nan")
            fps = r.get("frames_per_s", float("nan"))
            note = r.get("skipped") or r.get("error") or ""
            cells = [
                r["format"],
                f"{size_mb:.1f}" if size_mb == size_mb else "n/a",
                f"{fps:.1f}" if fps == fps else "n/a",
                f"{r.get('mean_abs_diff', float('nan')):.5f}" if "mean_abs_diff" in r else "—",
                f"{r.get('max_abs_diff', float('nan')):.4f}" if "max_abs_diff" in r else "—",
                f"{100*r.get('visible_frac', float('nan')):.2f}" if "visible_frac" in r else "—",
            ]
            print("  ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + (f"  {note}" if note else ""))


if __name__ == "__main__":
    main()
