"""Ordered DataLoader batches with optional same-worker compressed range prefetch."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="lance-lookahead")

    def submit(self, table, ranges):
        # Keep reservations from older hints only while they can still be consumed.
        for key, future in list(self.pending.items()):
            row, start, end = key
            if future.done() and not any(r == row and a < end and start < b for r, a, b in ranges):
                del self.pending[key]
                self.reserved -= end - start
        accepted = {}
        for row, start, end in ranges:
            # Adjacent batches can ask for different slices of the same GOP.
            covered = sorted((a, b) for r, a, b in self.pending if r == row and a < end and start < b)
            for a, b in [*covered, (end, end)]:
                if start < a and a - start <= self.budget - self.reserved:
                    key = row, start, a
                    self.reserved += a - start
                    accepted[key] = self.pending[key] = Future()
                start = max(start, b)
        if accepted:
            self.executor.submit(self._read, table, accepted)

    @staticmethod
    def _read(table, accepted):
        try:
            requests = [(row, a, b - a) for row, a, b in accepted]
            blobs = table.fetch_blob_ranges(VIDEO_BLOB_COLUMN, requests)
            for (_, a, b), future, blob in zip(accepted, accepted.values(), blobs, strict=True):
                if len(data := blob.as_py()) != b - a:
                    raise ValueError("Truncated lookahead range")
                future.set_result(data)
        except Exception as error:
            for future in accepted.values():
                if not future.done():
                    future.set_exception(error)

    def consume(self, row, source, ranges):
        for key in list(self.pending):
            owner, start, end = key
            if owner != row or not any(a < end and start < b for a, b in ranges):
                continue
            future = self.pending.pop(key)
            try:
                data = future.result()
            except Exception:
                continue  # foreground range planning retries any missing bytes
            else:
                source.add(start, data)
            finally:
                self.reserved -= end - start

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
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
            self.reader._lookahead_buffer = RangeBuffer(self.budget)
            self.finalizer = Finalize(self, self.close, exitpriority=10)
        self.reader._lookahead_hints = hints
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
