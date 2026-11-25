"""Ordered DataLoader batches with optional same-worker compressed range prefetch."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from itertools import islice
from multiprocessing.util import Finalize

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from .reader import VIDEO_BLOB_COLUMN, lance_mp_context


class RangeBuffer:
    """Single foreground owner. A future owns its reservation until consumed or closed."""

    def __init__(self, budget):
        self.budget = budget
        self.reserved = 0
        self.pending = {}
        self.groups = {}
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="lance-lookahead")

    def _release(self, key):
        future, _ = self.pending.pop(key)
        remaining, size = self.groups[future]
        if remaining == 1:
            del self.groups[future]
            self.reserved -= size
        else:
            self.groups[future] = remaining - 1, size

    def submit(self, table, ranges):
        for key, (future, _) in list(self.pending.items()):
            row, start, end = key
            if future.done() and not any(r == row and a < end and start < b for r, a, b in ranges):
                self._release(key)
        accepted = []
        size = 0
        for row, start, end in ranges:
            # Adjacent batches can request overlapping slices of the same GOP.
            covered = sorted((a, b) for r, a, b in self.pending if r == row and a < end and start < b)
            for a, b in [*covered, (end, end)]:
                if start < a and a - start <= self.budget - self.reserved - size:
                    accepted.append((row, start, a))
                    size += a - start
                start = max(start, b)
        if accepted:
            requests = [(row, a, b - a) for row, a, b in accepted]
            self.reserved += size
            future = self.executor.submit(table.fetch_blob_ranges, VIDEO_BLOB_COLUMN, requests)
            self.groups[future] = len(accepted), size
            self.pending.update((key, (future, i)) for i, key in enumerate(accepted))

    def consume(self, row, source, ranges):
        for key in list(self.pending):
            owner, start, end = key
            if owner != row or not any(a < end and start < b for a, b in ranges):
                continue
            future, i = self.pending[key]
            try:
                # Lance returns one contiguous Arrow array. Retain the whole
                # reservation until its last member leaves the buffer.
                data = future.result()[i].as_py()
                if len(data) == end - start:
                    source.add(start, data)
            except Exception:
                pass  # the normal foreground range plan retries every missing byte
            finally:
                self._release(key)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.groups.clear()
        self.reserved = 0


class _HintBatches:
    def __init__(self, batches, stride, depth):
        self.batches, self.stride, self.depth = batches, stride, depth

    def __iter__(self):
        stream = (tuple(batch) for batch in self.batches)
        queue = deque(islice(stream, self.depth * self.stride + 1))
        while queue:
            current = queue.popleft()
            yield [(current, tuple(islice(queue, self.stride - 1, None, self.stride)))]
            following = next(stream, None)
            if following is not None:
                queue.append(following)

    def __len__(self):
        return len(self.batches)


class _WorkerReader:
    def __init__(self, reader, budget):
        self.reader, self.budget = copy(reader), budget
        self.finalizer = None

    def __getitems__(self, ticket):
        indices, hints = ticket[0]
        if self.finalizer is None:
            self.finalizer = Finalize(self, self.close, exitpriority=10)
        # The parent consumes futures only inside the matching foreground read.
        if self.reader._lookahead_buffer is None and self.budget and hints:
            self.reader._lookahead_buffer = RangeBuffer(self.budget)
        self.reader._lookahead_hints = hints if self.reader._lookahead_buffer is not None else ()
        return self.reader.__getitems__(indices)

    def close(self):
        if self.reader._lookahead_buffer is not None:
            self.reader._lookahead_buffer.close()
            self.reader._lookahead_buffer = None
        for pool in (self.reader._prefetch_pool, self.reader._decode_pool):
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        if self.finalizer is not None:
            self.finalizer.cancel()


@contextmanager
def lookahead_loader(
    dataset,
    *,
    batch_size=1,
    shuffle=False,
    batch_sampler=None,
    num_workers=0,
    lookahead_batches=2,
    prefetch_bytes=64 << 20,
    generator=None,
    drop_last=False,
    **kwargs,
):
    """Yield one ordered DataLoader iterator and release its workers on context exit."""
    if min(lookahead_batches, prefetch_bytes, num_workers) < 0 or not kwargs.get("in_order", True):
        raise ValueError("Lookahead requires nonnegative limits and in_order=True")
    if batch_sampler is None:
        sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
    elif shuffle or drop_last:
        raise ValueError("batch_sampler specifies its own order and batch boundaries")
    if num_workers:
        kwargs.setdefault("multiprocessing_context", lance_mp_context())
    wrapper = _WorkerReader(dataset, prefetch_bytes)
    loader = DataLoader(
        wrapper,
        batch_sampler=_HintBatches(batch_sampler, max(1, num_workers), lookahead_batches),
        generator=generator,
        num_workers=num_workers,
        **kwargs,
    )
    iterator = iter(loader)
    try:
        yield iterator
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        wrapper.close()
