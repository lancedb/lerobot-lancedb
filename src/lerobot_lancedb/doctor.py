#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""lerobot-lance-doctor: audit a LeRobot v3.0 dataset (upstream format) for silent defects.

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

``audit()`` returns the checks as data so ``lerobot-lance-convert`` can run them
before it copies anything; the CLI below only renders them.

Usage: lerobot-lance-doctor --root /path/to/dataset [--repo-id name]
Exit code: number of failed checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

FAIL = "\033[91mFAIL\033[0m"
OK = "\033[92m ok \033[0m"


@dataclass(frozen=True)
class Problem:
    """One defect a check found."""

    message: str
    #: Whether the defect is in data an episode references, so any copy of this
    #: dataset carries it. False only for problems confined to files no episode
    #: references: leaving those files out leaves the referenced data sound, so
    #: they do not block a conversion.
    blocks_conversion: bool


@dataclass(frozen=True)
class Check:
    name: str
    problems: list[Problem]

    @property
    def failed(self) -> bool:
        return bool(self.problems)


@dataclass(frozen=True)
class Audit:
    meta: LeRobotDatasetMetadata
    checks: list[Check]

    @property
    def failed_checks(self) -> int:
        return sum(check.failed for check in self.checks)

    @property
    def blocking_problems(self) -> list[Problem]:
        """Defects a copy of the dataset cannot drop, because episodes read them."""
        return [p for check in self.checks for p in check.problems if p.blocks_conversion]


def _defect(message: str) -> Problem:
    """A problem in data episodes reference: a copy of the dataset carries it."""
    return Problem(message, blocks_conversion=True)


def audit(root: Path, repo_id: str | None = None) -> Audit:
    """Run every check against a dataset on local disk.

    Metadata load errors propagate: no check can say anything about the data
    before ``LeRobotDatasetMetadata`` has read ``meta/``.
    """
    import av
    import pyarrow.parquet as pq
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    # 1 META: loaded here instead of checked, because a failure means there is
    # nothing left to inspect. main() turns that into the reported failure.
    meta = LeRobotDatasetMetadata(repo_id or str(root), root=root)

    # 2 BOUNDARIES
    starts = [int(x) for x in meta.episodes["dataset_from_index"]]
    ends = [int(x) for x in meta.episodes["dataset_to_index"]]
    problems = []
    if starts and starts[0] != 0:
        problems.append(_defect(f"first episode starts at {starts[0]}, not 0"))
    if ends and ends[-1] != meta.total_frames:
        problems.append(_defect(f"last episode ends at {ends[-1]}, meta declares {meta.total_frames}"))
    problems += [
        _defect(f"gap/overlap between episode {i} (ends {ends[i]}) and {i + 1} (starts {starts[i + 1]})")
        for i in range(len(starts) - 1)
        if starts[i + 1] != ends[i]
    ]
    checks = [Check("BOUNDARIES: episode ranges tile [0, total_frames)", problems)]

    # 3 DATA
    referenced_data = {root / meta.get_data_file_path(ep) for ep in range(meta.total_episodes)}
    problems = [
        _defect(f"missing referenced parquet: {p.relative_to(root)}")
        for p in sorted(referenced_data)
        if not p.exists()
    ]
    rows = sum(pq.read_metadata(p).num_rows for p in referenced_data if p.exists())
    if rows != meta.total_frames:
        problems.append(_defect(f"referenced parquets hold {rows} rows, meta declares {meta.total_frames}"))
    on_disk = set((root / "data").rglob("*.parquet"))
    orphans = sorted(on_disk - referenced_data)
    if orphans:
        orphan_rows = sum(pq.read_metadata(p).num_rows for p in orphans)
        problems.append(
            Problem(
                f"{len(orphans)} orphan parquet file(s) never referenced by meta "
                f"({orphan_rows} rows; loaders that glob directories may read these)",
                blocks_conversion=False,
            )
        )
    checks.append(Check("DATA: parquet files referenced, accounted, no orphans", problems))

    # 4 + 5 VIDEOS
    problems = []
    supply_problems = []
    referenced_videos: dict[Path, float] = {}  # path -> max implied end timestamp
    columns = meta.episodes.column_names
    for key in meta.video_keys:
        to_ts = (
            meta.episodes[f"videos/{key}/to_timestamp"] if f"videos/{key}/to_timestamp" in columns else None
        )
        from_ts = meta.episodes[f"videos/{key}/from_timestamp"]
        for ep in range(meta.total_episodes):
            path = root / meta.get_video_file_path(ep, key)
            end = (
                float(to_ts[ep])
                if to_ts is not None
                else float(from_ts[ep]) + (ends[ep] - starts[ep]) / meta.fps
            )
            referenced_videos[path] = max(referenced_videos.get(path, 0.0), end)
    for path, implied_end in sorted(referenced_videos.items()):
        rel = path.relative_to(root)
        if not path.exists():
            problems.append(_defect(f"missing referenced video: {rel}"))
            continue
        if path.stat().st_size == 0:
            problems.append(_defect(f"zero-byte video: {rel}"))
            continue
        implied_frames = round(implied_end * meta.fps)
        try:
            with av.open(str(path)) as container:
                declared = container.streams.video[0].frames
        except Exception as err:
            problems.append(_defect(f"unreadable video {rel}: {type(err).__name__}"))
            continue
        if declared and declared < implied_frames:
            supply_problems.append(
                _defect(
                    f"{rel}: container declares {declared} frames, episodes imply {implied_frames} "
                    f"(short by {implied_frames - declared})"
                )
            )
    on_disk_videos = set((root / "videos").rglob("*.mp4")) if (root / "videos").is_dir() else set()
    orphan_videos = sorted(on_disk_videos - set(referenced_videos))
    if orphan_videos:
        problems.append(
            Problem(f"{len(orphan_videos)} orphan video file(s) never referenced by meta", blocks_conversion=False)
        )
    checks.append(Check("VIDEOS: referenced, non-empty, readable, no orphans", problems))
    checks.append(Check("SUPPLY: every video holds the frames meta implies", supply_problems))

    return Audit(meta=meta, checks=checks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    try:
        result = audit(args.root, args.repo_id)
    except Exception as err:
        print(f"[{FAIL}] META: metadata failed to load: {err}")
        return 1
    meta = result.meta
    print(f"[{OK}] META: {meta.total_episodes} episodes, {meta.total_frames} frames, fps {meta.fps}")

    for check in result.checks:
        if not check.failed:
            print(f"[{OK}] {check.name}")
            continue
        print(f"[{FAIL}] {check.name}")
        for problem in check.problems[:10]:
            print(f"         {problem.message}")
        if len(check.problems) > 10:
            print(f"         ... and {len(check.problems) - 10} more")

    failures = result.failed_checks
    # META is reported above rather than living in `checks`: when it fails there
    # is nothing else to check, so it has no list of problems to render.
    total = len(result.checks) + 1
    print(f"\n{failures} of {total} checks failed" if failures else f"\nall {total} checks passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
