#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""lerobot-lance-doctor: audit a LeRobot v2.0 / v2.1 / v3.0 dataset for silent defects.

Every large public dataset we converted shipped at least one of these:
  droid     44% orphan parquet rows + orphan videos (works by filename-sort luck)
  berkeley  aggregated videos missing tail frames vs what meta implies
  agibot    zero-byte videos inside whole episodes

Checks (all read-only, no decode: container metadata only):
  1 META        metadata loads at all
  2 BOUNDARIES  episode ranges tile [0, total_frames) exactly
  3 DATA        every referenced parquet exists; total rows == total_frames;
                orphan parquet files under data/
  4 VIDEOS      every referenced video exists and is non-empty; orphan videos
  5 SUPPLY      per video file: frames the container declares (moov) covers
                the frames meta implies episodes need (catches truncation)

Usage: lerobot-lance-doctor --root /path/to/dataset [--repo-id name]
Exit code: number of failed checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

FAIL = "\033[91mFAIL\033[0m"
OK = "\033[92m ok \033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    import av
    import pyarrow.parquet as pq

    from .legacy import load_source_audit

    failures = 0

    def report(name: str, problems: list[str]) -> None:
        nonlocal failures
        if problems:
            failures += 1
            print(f"[{FAIL}] {name}")
            for p in problems[:10]:
                print(f"         {p}")
            if len(problems) > 10:
                print(f"         ... and {len(problems) - 10} more")
        else:
            print(f"[{OK}] {name}")

    # 1 META
    try:
        src = load_source_audit(args.root, repo_id=args.repo_id)
    except Exception as err:
        print(f"[{FAIL}] META: metadata failed to load: {err}")
        return 1
    print(
        f"[{OK}] META: {src.version}, {src.total_episodes} episodes, "
        f"{src.total_frames} frames, fps {src.fps}"
    )

    starts, ends = src.starts, src.ends

    # 2 BOUNDARIES
    problems = []
    if starts and starts[0] != 0:
        problems.append(f"first episode starts at {starts[0]}, not 0")
    if ends and ends[-1] != src.total_frames:
        problems.append(f"last episode ends at {ends[-1]}, meta declares {src.total_frames}")
    problems += [
        f"gap/overlap between episode {i} (ends {ends[i]}) and {i + 1} (starts {starts[i + 1]})"
        for i in range(len(starts) - 1)
        if starts[i + 1] != ends[i]
    ]
    report("BOUNDARIES: episode ranges tile [0, total_frames)", problems)

    # 3 DATA
    referenced_data = set(src.data_files)
    problems = [
        f"missing referenced parquet: {p.relative_to(args.root) if p.is_relative_to(args.root) else p}"
        for p in sorted(referenced_data)
        if not p.exists()
    ]
    rows = sum(pq.read_metadata(p).num_rows for p in referenced_data if p.exists())
    if rows != src.total_frames:
        problems.append(f"referenced parquets hold {rows} rows, meta declares {src.total_frames}")
    on_disk = set((args.root / "data").rglob("*.parquet")) if (args.root / "data").is_dir() else set()
    orphans = sorted(on_disk - referenced_data)
    if orphans:
        orphan_rows = sum(pq.read_metadata(p).num_rows for p in orphans)
        problems.append(
            f"{len(orphans)} orphan parquet file(s) never referenced by meta "
            f"({orphan_rows} rows; loaders that glob directories may read these)"
        )
    report("DATA: parquet files referenced, accounted, no orphans", problems)

    # 4 + 5 VIDEOS
    problems = []
    supply_problems = []
    for path, implied_end in sorted(src.video_end_ts.items()):
        rel = path.relative_to(args.root) if path.is_relative_to(args.root) else path
        if not path.exists():
            problems.append(f"missing referenced video: {rel}")
            continue
        if path.stat().st_size == 0:
            problems.append(f"zero-byte video: {rel}")
            continue
        implied_frames = round(implied_end * src.fps)
        try:
            with av.open(str(path)) as container:
                declared = container.streams.video[0].frames
        except Exception as err:
            problems.append(f"unreadable video {rel}: {type(err).__name__}")
            continue
        if declared and declared < implied_frames:
            supply_problems.append(
                f"{rel}: container declares {declared} frames, episodes imply {implied_frames} "
                f"(short by {implied_frames - declared})"
            )
    on_disk_videos = set((args.root / "videos").rglob("*.mp4")) if (args.root / "videos").is_dir() else set()
    orphan_videos = sorted(on_disk_videos - set(src.video_end_ts))
    if orphan_videos:
        problems.append(f"{len(orphan_videos)} orphan video file(s) never referenced by meta")
    report("VIDEOS: referenced, non-empty, readable, no orphans", problems)
    report("SUPPLY: every video holds the frames meta implies", supply_problems)

    print(f"\n{failures} of 5 checks failed" if failures else "\nall 5 checks passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
