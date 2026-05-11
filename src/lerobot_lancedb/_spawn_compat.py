#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Multiprocessing-mode helpers used by the Lance backend.

LanceDB maintains an internal tokio runtime in the parent process. The runtime
does not survive ``fork()`` cleanly: a worker spawned via fork can hit a torn
heap inherited from a parent thread that was mid-allocation, producing glibc
heap-corruption crashes (``malloc(): invalid size``, ``free(): invalid
pointer``). The crashes are deterministic on cloud reads (S3, HF Buckets) at
``num_workers >= 8`` and flaky at lower counts. Forcing ``spawn`` on Linux
avoids the issue at the cost of ~7s extra import time per worker, paid once.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys


_SPAWN_FORCED = False


def force_spawn_for_lance() -> None:
    """Idempotently switch multiprocessing to ``spawn`` on Linux.

    Called from :class:`LeRobotLanceDataset.__init__`. No-op when:

    * Running on macOS (already defaults to spawn).
    * The current process has already chosen a non-fork start method.
    """
    global _SPAWN_FORCED
    if _SPAWN_FORCED:
        return
    _SPAWN_FORCED = True

    if sys.platform != "linux":
        return

    current = mp.get_start_method(allow_none=True)
    if current not in (None, "fork"):
        return

    try:
        mp.set_start_method("spawn", force=True)
        logging.info(
            "lerobot-lancedb: multiprocessing start method set to 'spawn' "
            "(was %s) — workaround for lancedb fork-unsafety.",
            current or "default (fork)",
        )
    except RuntimeError as exc:
        logging.warning(
            "lerobot-lancedb could not switch multiprocessing to 'spawn' "
            "(%s); DataLoader workers may crash.",
            exc,
        )

    try:
        import torch

        torch.multiprocessing.set_sharing_strategy("file_system")
    except (ImportError, RuntimeError) as exc:
        logging.warning(
            "lerobot-lancedb could not switch torch sharing strategy to "
            "'file_system' (%s); workers may misbehave on /dev/shm IPC.",
            exc,
        )


__all__ = ["force_spawn_for_lance"]
