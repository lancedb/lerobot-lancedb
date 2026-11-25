import threading

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
