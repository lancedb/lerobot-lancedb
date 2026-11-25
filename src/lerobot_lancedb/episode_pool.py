"""Exactly-once iterable sampling from a bounded pool of matched episodes."""

from __future__ import annotations

import os
from collections import Counter, defaultdict, deque
from copy import copy
from itertools import islice
from operator import index

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .reader import LanceDBDataset, lance_mp_context


def _shards(episodes, counts, size):
    shards = [[] for _ in range(size)]
    loads = [0] * size
    for episode in sorted(episodes, key=lambda ep: (-counts[ep], ep)):
        dest = min(range(size), key=loads.__getitem__)
        shards[dest].append(episode)
        loads[dest] += counts[episode]
    return shards


def _distributed(rank, world_size):
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        default_rank, default_size = dist.get_rank(), dist.get_world_size()
    else:
        default_rank, default_size = os.getenv("RANK", 0), os.getenv("WORLD_SIZE", 1)
    rank = int(default_rank if rank is None else rank)
    size = int(default_size if world_size is None else world_size)
    if not 0 <= rank < size:
        raise ValueError(f"Invalid rank {rank} for world_size {size}")
    return rank, size


def _fallback_pool(counts, size, seed, epoch, strategy):
    # Same RNG draws as LeRobot 0.7's ExactCoveragePool for installations without it.
    rng = np.random.default_rng([seed, epoch])
    pending = deque(rng.permutation(sorted(counts)).tolist())
    resident = {}
    turn = []
    while resident or pending:
        while pending and len(resident) < size:
            ep = pending.popleft()
            resident[ep] = rng.permutation(counts[ep]).tolist()
        if strategy == "round_robin":
            if not turn:
                turn = list(resident)
                rng.shuffle(turn)
            ep = turn.pop()
        else:
            target = rng.integers(sum(map(len, resident.values())))
            for ep in resident:
                target -= len(resident[ep])
                if target < 0:
                    break
        frame = resident[ep].pop()
        if not resident[ep]:
            del resident[ep]
        yield ep, frame


def _worker_cursor(totals, offset, batch_size):
    """Replay only batch lengths, including short batches and exhausted workers."""
    consumed = [0] * len(totals)
    worker = 0
    while offset:
        size = min(batch_size, totals[worker] - consumed[worker])
        if size > offset:
            raise ValueError("Multi-worker resume requires an offset at a delivered batch boundary")
        consumed[worker] += size
        offset -= size
        worker = (worker + 1) % len(totals)
    return consumed, worker


class _FrameReuse:
    """Raw frames with another known consumer in this emission window."""

    def __init__(self, counts, max_frames, max_bytes):
        self.counts = counts
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.frames = {}
        self.nbytes = 0

    def consume(self, key, value):
        self.counts[key] -= 1
        if self.counts[key] == 0:
            if key in self.frames:
                self.nbytes -= self.frames.pop(key).nbytes
        elif (
            key not in self.frames
            and len(self.frames) < self.max_frames
            and self.nbytes + value.nbytes <= self.max_bytes
        ):
            self.frames[key] = value.clone(memory_format=torch.contiguous_format)
            self.nbytes += value.nbytes

    def clear(self):
        self.frames.clear()
        self.nbytes = 0


def _fill_videos(reader, prepared, requests, cache, item):
    missing = {
        file: [ts for ts in dict.fromkeys(times) if (file, ts) not in cache.frames]
        for file, times in requests.items()
    }
    to_decode = {file: [(0, times)] for file, times in missing.items() if times}
    decoded = (
        reader._decode_requests(to_decode, {k: prepared[k] for k in to_decode}, 1)[0]
        if to_decode
        else {}
    )
    for file, times in requests.items():
        seen = {}
        if missing[file]:
            values = decoded[file[0]]
            frames = values.unsqueeze(0) if len(missing[file]) == 1 else values
            seen.update(zip(missing[file], frames, strict=True))
        for ts in set(times) - seen.keys():
            seen[ts] = cache.frames[file, ts]
        item[file[0]] = torch.stack([seen[ts] for ts in times]).squeeze(0)
        for ts, frame in seen.items():
            cache.consume((file, ts), frame)


def _emit_window(reader, indices, max_frames, max_bytes):
    plans = reader._plan_batch(indices)
    rows, positions = reader._batch_rows(plans)
    windows = reader._plan_file_windows(plans)
    prepared = reader._prepare_files(sorted(windows), windows) if windows else {}
    columns = reader._fetch_rows(rows)
    by_anchor = [dict() for _ in plans]
    counts = Counter()
    if windows:
        for file, requests in reader._build_video_requests(plans, columns["timestamp"], positions).items():
            for anchor, times in requests:
                # Exact source timestamps distinguish queries with different PTS tolerance outcomes.
                by_anchor[anchor][file] = times
                counts.update((file, ts) for ts in set(times))
    cache = _FrameReuse(counts, max_frames, max_bytes)
    try:
        for plan, requests in zip(plans, by_anchor, strict=True):
            item = reader._build_item(plan, columns, positions)
            _fill_videos(reader, prepared, requests, cache, item)
            if reader.image_transforms is not None:
                reader._transform_item(item)
            yield item
    finally:
        cache.clear()


class LanceDBEpisodePoolDataset(IterableDataset):
    """Rank- and worker-sharded episode sampling with bounded video window coalescing."""

    def __init__(
        self,
        repo_id=None,
        root=None,
        *,
        reader=None,
        where=None,
        tasks=None,
        episodes=None,
        episode_pool_size=None,
        buffer_size=32,
        seed=42,
        shuffle=True,
        order="shuffled",
        sampling_strategy="remaining",
        batch_size=1,
        rank=None,
        world_size=None,
        prefetch_anchors=8,
        max_decoded_reuse_frames=64,
        max_decoded_reuse_bytes=128 << 20,
        **reader_options,
    ):
        super().__init__()
        if reader is None:
            reader = LanceDBDataset(repo_id, root, episodes=episodes, **reader_options)
            episodes = None
        elif reader_options or repo_id is not None or root is not None:
            raise ValueError("Set reader options on the supplied LanceDBDataset")
        self.reader = reader
        self.rank, self.world_size = _distributed(rank, world_size)
        self.pool_size = min(buffer_size, 32) if episode_pool_size is None else episode_pool_size
        if (
            min(batch_size, prefetch_anchors, self.pool_size) < 1
            or min(max_decoded_reuse_frames, max_decoded_reuse_bytes) < 0
        ):
            raise ValueError("Invalid batch, pool, prefetch or frame reuse limit")
        if order not in ("sequential", "shuffled") or sampling_strategy not in ("remaining", "round_robin"):
            raise ValueError("Unknown order or sampling strategy")
        self.seed, self.shuffle, self.order, self.strategy = seed, shuffle, order, sampling_strategy
        self.prefetch_anchors = prefetch_anchors
        self.max_frames, self.max_bytes = max_decoded_reuse_frames, max_decoded_reuse_bytes
        if where is not None or tasks is not None:
            selection = reader.select_anchors(where=where, tasks=tasks, episodes=episodes)
            self.anchors = defaultdict(list)
            for ep, rel in zip(selection.episode_indices, selection.indices, strict=True):
                self.anchors[int(ep)].append(int(rel))
        else:
            selected = range(reader.meta.total_episodes) if reader.episodes is None else reader.episodes
            if episodes is not None:
                selected = set(selected).intersection(episodes)
            self.anchors = {}
            for ep in selected:
                start, end = reader._episode_bounds(ep)
                if start < end:
                    rel = start if reader._subset_index is None else reader._subset_index[start]
                    self.anchors[ep] = range(rel, rel + end - start)
        self.counts = {ep: len(anchors) for ep, anchors in self.anchors.items()}
        self.episodes = _shards(self.anchors, self.counts, self.world_size)[self.rank]
        self.batch_size = batch_size
        self.set_epoch(0)

    def __len__(self):
        return sum(self.counts[ep] for ep in self.episodes)

    def set_epoch(self, epoch):
        self.epoch, self.offset, self._resume = index(epoch), 0, 0

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset, "batch_size": self.batch_size}

    def load_state_dict(self, state):
        epoch, offset, batch_size = (index(state[k]) for k in ("epoch", "offset", "batch_size"))
        if epoch < 0 or not 0 <= offset <= len(self) or batch_size < 1:
            raise ValueError("Invalid episode-pool checkpoint")
        self.epoch, self.offset, self._resume, self.batch_size = epoch, offset, offset, batch_size

    def loader(self, *, num_workers=0, **options):
        """Track batches delivered to the parent for exact multi-worker state_dict()."""
        if num_workers:
            options.setdefault("multiprocessing_context", lance_mp_context())
        loader = DataLoader(self, batch_size=self.batch_size, num_workers=num_workers, **options)
        delivered = self._resume
        iterator = iter(loader)
        try:
            for batch in iterator:
                delivered += len(next(iter(batch.values())))
                self.offset = delivered
                yield batch
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            self._resume = 0

    def _plan(self, episodes):
        if self.order == "sequential":
            for ep in sorted(episodes):
                yield from self.anchors[ep]
            return
        counts = {ep: self.counts[ep] for ep in episodes}
        epoch = self.epoch if self.shuffle else 0
        try:
            from lerobot.streaming.episode_pool import ExactCoveragePool  # noqa: PLC0415
        except ImportError:
            planner = _fallback_pool(counts, self.pool_size, self.seed, epoch, self.strategy)
        else:
            planner = ExactCoveragePool(
                list(counts.items()),
                self.pool_size,
                seed=self.seed,
                epoch=epoch,
                sampling_strategy=self.strategy,
            )
        for ep, frame in planner:
            yield self.anchors[ep][frame]

    def __iter__(self):
        worker = get_worker_info()
        num_workers, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        if num_workers > 1 and len(self.episodes) < num_workers:
            raise ValueError(f"Rank owns {len(self.episodes)} episodes, fewer than its {num_workers} workers")
        shards = _shards(self.episodes, self.counts, num_workers)
        offset = self._resume
        self._resume = 0
        self.offset = offset
        if num_workers > 1:
            totals = [sum(self.counts[ep] for ep in shard) for shard in shards]
            consumed, rotation = _worker_cursor(totals, offset, self.batch_size)
            worker_id = (rotation + worker_id) % num_workers
            offset = consumed[worker_id]
        stream = islice(self._plan(shards[worker_id]), offset, None)
        reader = copy(self.reader)
        try:
            while chunk := list(islice(stream, self.prefetch_anchors)):
                reader._ensure_open()
                for item in _emit_window(reader, chunk, self.max_frames, self.max_bytes):
                    self.offset += 1
                    yield item
        finally:
            for pool in (reader._prefetch_pool, reader._decode_pool):
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
