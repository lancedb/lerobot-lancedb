#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backend auto-detection for :func:`make_lerobot_dataset`.

Exists so user code (notebooks, examples, third-party scripts) can write
``make_lerobot_dataset("./pusht_lance")`` or
``make_lerobot_dataset("me/pusht_lance")`` and get the right reader for free,
without learning two class names.

Rules, cheapest first:

1. Explicit ``backend="lance"`` / ``"parquet"`` wins.
2. ``root`` argument:
   * Points at a directory ending in ``.lance`` → Lance.
   * Points at a directory containing a ``*.lance/`` subdir → Lance.
   * Anything else → parquet+mp4 (the default).
3. Positional locator:
   * URI ending in ``.lance`` → Lance.
   * Existing local path with a ``*.lance/`` inside → Lance.
   * HF Hub repo id ending in ``_lance`` (convention) → Lance.
   * Otherwise → parquet+mp4.

The Hub-repo suffix convention (``*_lance``) is the only "guess" — users who
publish converted datasets under another name should pass ``backend="lance"``
explicitly. We don't probe the Hub for a ``*.lance`` listing because that
would add a network round-trip to ``__init__``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _looks_like_lance_path(p: Path) -> bool:
    if p.suffix == ".lance":
        return True
    if p.is_dir():
        return any(p.glob("*.lance"))
    return False


def _detect_backend(
    locator: str | Path | None,
    root: str | Path | None,
) -> str:
    """Return ``"lance"`` or ``"parquet"`` based on locator/root heuristics."""
    if root is not None:
        if _looks_like_lance_path(Path(root)):
            return "lance"
    if locator is not None:
        if isinstance(locator, str) and "://" in locator:
            return "lance" if locator.rstrip("/").lower().endswith(".lance") else "parquet"
        if _looks_like_lance_path(Path(locator)):
            return "lance"
        if isinstance(locator, str) and locator.endswith("_lance"):
            return "lance"
    return "parquet"


def make_lerobot_dataset(
    repo_id_or_root: str | Path | None = None,
    *,
    root: str | Path | None = None,
    backend: str | None = None,
    **kwargs: Any,
):
    """Construct a LeRobot dataset, picking the storage backend automatically.

    Returns either :class:`lerobot.datasets.LeRobotDataset` (parquet+mp4) or
    :class:`lerobot_lancedb.LeRobotLanceDataset` — the latter is a subclass
    of the former, so ``isinstance(returned, LeRobotDataset)`` always holds.

    Args:
        repo_id_or_root: First positional locator. Accepts a HF Hub repo id
            (``'lerobot/pusht'``), a local path (``'./pusht_lance'`` or
            ``'./pusht.lance'``), or a cloud URI
            (``'s3://bucket/pusht.lance'``).
        root: Optional explicit local root, same semantics as
            :class:`LeRobotDataset.__init__`. Takes precedence over
            ``repo_id_or_root`` for parquet+mp4 datasets in a non-standard
            location.
        backend: Force a backend (``"parquet"`` or ``"lance"``). ``None`` =
            auto-detect.
        **kwargs: Forwarded to the chosen reader. Parquet-only kwargs
            (``video_backend``, ``download_videos``, ``force_cache_sync``)
            are silently dropped when routing to the Lance backend.

    Examples::

        ds = make_lerobot_dataset("lerobot/pusht")             # parquet+mp4
        ds = make_lerobot_dataset(root="./pusht_lance")        # Lance (auto)
        ds = make_lerobot_dataset("me/pusht_lance")            # Lance (auto via _lance suffix)
        ds = make_lerobot_dataset("s3://bucket/pusht.lance")   # Lance (cloud)
        ds = make_lerobot_dataset("custom", backend="lance", root="./somewhere")
    """
    chosen = backend or _detect_backend(repo_id_or_root, root)
    if chosen not in ("parquet", "lance"):
        raise ValueError(f"backend must be 'parquet' or 'lance', got {chosen!r}")

    if chosen == "lance":
        # Local import — keeps the user-visible import surface focused and
        # lets parquet-only callers avoid importing lance if they never need it.
        from .dataset import LeRobotLanceDataset  # noqa: PLC0415

        lance_kwargs = _split_lance_kwargs(repo_id_or_root, root, kwargs)
        return LeRobotLanceDataset(**lance_kwargs)

    if repo_id_or_root is None:
        raise TypeError(
            "make_lerobot_dataset requires a repo id or root path as the first argument."
        )
    return LeRobotDataset(str(repo_id_or_root), root=root, **kwargs)


def _split_lance_kwargs(
    locator: str | Path | None,
    root: str | Path | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # ``video_backend`` etc. only make sense for the parquet+mp4 reader;
    # silently drop them when routing to Lance so the same caller code
    # works against either backend. ``decode_device`` is Lance-only and
    # passes through unchanged.
    drop = {"video_backend", "download_videos", "force_cache_sync"}
    for k, v in kwargs.items():
        if k in drop:
            continue
        out[k] = v

    if root is not None:
        out["root"] = root

    if locator is None:
        return out

    if isinstance(locator, str) and "://" in locator:
        out["uri"] = locator
        return out

    lp = Path(locator)
    if _looks_like_lance_path(lp):
        out["root"] = str(lp)
        return out

    out["repo_id"] = str(locator)
    return out


__all__ = ["make_lerobot_dataset"]
