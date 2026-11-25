import threading
import time

import lancedb
import pyarrow as pa
import pytest
import torch
from test_convert_external import VIDEO_KEY, _make_dataset
from torch.utils.data import DataLoader

from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb.convert import convert
from lerobot_lancedb.lookahead import RangeBuffer
from lerobot_lancedb.reader import _SparseBlobSource


def augment(frame):
    return frame.float() + torch.rand(())


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("lookahead")
    convert(root / "lance", root=_make_dataset(root / "source"))
    return root / "lance"


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("explicit", [True, False])
def test_order_transform_and_shutdown(dataset_root, workers, explicit):
    batches = [[19, 0, 4], [1], [18, 18], [11, 6]]

    def run(depth):
        ds = LanceDBDataset(root=dataset_root, image_transforms=augment, return_uint8=True)
        torch.manual_seed(17)
        options = dict(num_workers=workers, generator=torch.Generator().manual_seed(42))
        if workers:
            options.update(multiprocessing_context="spawn", prefetch_factor=1)
        if explicit:
            options.update(batch_sampler=batches)
        else:
            options.update(batch_size=2, shuffle=True)
        if depth is None:
            return list(DataLoader(ds, **options))[:3]
        with ds.lookahead_loader(lookahead_batches=depth, **options) as it:
            children = getattr(it, "_workers", [])
            observed = [next(it) for _ in range(3)]
        assert all(not p.is_alive() for p in children)
        assert not any(t.name.startswith("lance-lookahead") for t in threading.enumerate())
        return observed

    expected = run(None)
    for depth in (0, 2):
        for a, b in zip(expected, run(depth), strict=True):
            assert torch.equal(a["index"], b["index"]) and torch.equal(a[VIDEO_KEY], b[VIDEO_KEY])


def test_range_io_overlaps_current_decode(dataset_root, monkeypatch):
    started, decoding, finished = threading.Event(), threading.Event(), threading.Event()
    fetch = lancedb.table.LanceTable.fetch_blob_ranges
    decode = LanceDBDataset._decode_mini_file
    background = []

    def fetch_ranges(table, column, requests):
        if threading.current_thread().name.startswith("lance-lookahead"):
            background.extend(requests)
            started.set()
            assert decoding.wait(3)
            time.sleep(0.020)
            result = fetch(table, column, requests)
            finished.set()
            return result
        return fetch(table, column, requests)

    def decode_mini(reader, *args):
        decoding.set()
        assert started.wait(3) and finished.wait(3)
        return decode(reader, *args)

    monkeypatch.setattr(lancedb.table.LanceTable, "fetch_blob_ranges", fetch_ranges)
    monkeypatch.setattr(LanceDBDataset, "_decode_mini_file", decode_mini)
    ds = LanceDBDataset(root=dataset_root)
    with ds.lookahead_loader(batch_sampler=[[0], [10]], lookahead_batches=1) as it:
        assert [b["index"].item() for b in it] == [0, 10]
    assert finished.is_set() and background


def test_partial_vector_read_retains_whole_arrow_allocation():
    class Table:
        def fetch_blob_ranges(self, column, requests):
            return pa.array([bytes(n) for _, _, n in requests])

    buffer = RangeBuffer(8)
    table = Table()
    try:
        buffer.submit(table, [(1, 0, 4), (2, 0, 4)])
        buffer.consume(1, _SparseBlobSource(4, None), [(0, 4)])
        buffer.submit(table, [(2, 0, 4), (3, 0, 4)])
        assert buffer.reserved == 8 and (3, 0, 4) not in buffer.pending
        buffer.consume(2, _SparseBlobSource(4, None), [(0, 4)])
        assert buffer.reserved == 0
    finally:
        buffer.close()


@pytest.mark.parametrize("failed", [True, False])
def test_in_flight_reservation_and_recovery(failed):
    started, finish = threading.Event(), threading.Event()
    calls = []

    class Table:
        def fetch_blob_ranges(self, column, requests):
            calls.extend(requests)
            started.set()
            assert finish.wait(5)
            if failed:
                raise OSError("range unavailable")
            return pa.array([b"test"])

    buffer = RangeBuffer(4)
    source = _SparseBlobSource(16, None)
    try:
        buffer.submit(Table(), [(2, 0, 4), (2, 4, 8)])
        assert started.wait(5) and buffer.reserved == 4
        buffer.submit(Table(), [(2, 0, 4), (2, 4, 6)])
        finish.set()
        buffer.consume(2, source, [(0, 4)])
        assert source.covers(0, 4) == (not failed)
        assert calls == [(2, 0, 4)] and buffer.reserved == 0
    finally:
        finish.set()
        buffer.close()
