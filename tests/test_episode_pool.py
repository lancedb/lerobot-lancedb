from collections import Counter
from contextlib import closing

import pytest
import torch
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from test_convert_external import FPS, VIDEO_KEY, _make_dataset

from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb.convert import convert
from lerobot_lancedb.episode_pool import _fallback_pool, _FrameReuse, _shards


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    root = tmp_path_factory.mktemp("episode-pool")
    source = _make_dataset(root / "native")
    convert(root / "lance", root=source, video_mode="copy")
    return root / "lance", source


def assert_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if torch.is_tensor(a[key]):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key]


def test_shards_and_window_parity(roots, monkeypatch):
    root, source = roots
    options = dict(
        repo_id="local/ext-test",
        return_uint8=True,
        delta_timestamps={
            VIDEO_KEY: [-1 / FPS, 0, 1 / FPS],
            "observation.state": [-1 / FPS, 0, 1 / FPS],
        },
    )
    reader = LanceDBDataset(root=root, **options)
    reference = StreamingLeRobotDataset(root=source, episode_pool_size=2, repeat=False, **options)
    with closing(iter(reference)) as stream, closing(iter(reader.episode_pool(episode_pool_size=2))) as pool:
        for actual, expected in zip(pool, stream, strict=True):
            assert_equal(actual, expected)
    reads = []
    fetch = LanceDBDataset._fetch_rows

    def track(self, rows):
        reads.extend(rows)
        return fetch(self, rows)

    monkeypatch.setattr(LanceDBDataset, "_fetch_rows", track)
    pool = reader.episode_pool(where="frame_index IN (0, 5, 9)", tasks=["reach"], order="sequential")
    pool.load_state_dict({"epoch": 3, "offset": 4, "batch_size": 1})
    samples = list(pool)
    assert [int(row["index"]) for row in samples] == [15, 19]
    assert reads == [14, 15, 16, 18, 19]
    for sample in samples:
        assert_equal(sample, reader[int(sample["index"])])
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert reader.episode_pool().episodes == [1] and reader.episode_pool(rank=0).episodes == [0]
    assert _shards(range(5), [10, 1, 2, 11, 5], 2) == [[3, 2, 1], [0, 4]]
    with pytest.raises(ValueError, match="fewer than"):
        next(reader.episode_pool().loader(num_workers=2))


class _TrackingReader(LanceDBDataset):
    def _fetch_rows(self, rows):
        with self.log.open("a") as stream:
            print(*rows, file=stream)
        return super()._fetch_rows(rows)


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_before_short_batch(roots, workers, tmp_path):
    root, _ = roots
    reader = _TrackingReader(root=root, columns=["index"])
    reader.log = tmp_path / "reads"

    def pool():
        return reader.episode_pool(batch_size=3)

    def batches(ds):
        return [row["index"].tolist() for row in ds.loader(num_workers=workers)]

    full = batches(pool())
    assert sorted(i for row in full for i in row) == list(range(20)) and len(full[-1]) < 3
    for cut in [1, len(full) - 1]:
        ds = pool()
        with closing(ds.loader(num_workers=workers)) as iterator:
            for _ in range(cut):
                next(iterator)
            state = ds.state_dict()
        assert state == {"epoch": 0, "offset": sum(map(len, full[:cut])), "batch_size": 3}
        resumed = pool()
        resumed.load_state_dict(state)
        reader.log.write_text("")
        assert batches(resumed) == full[cut:]
        assert sorted(map(int, reader.log.read_text().split())) == sorted(
            i for row in full[cut:] for i in row
        )
        resumed.set_epoch(2)
        assert resumed.state_dict()["offset"] == 0


def test_cache_contains_only_frames_with_another_pending_consumer(roots, monkeypatch):
    root, _ = roots
    caches = []
    consume = _FrameReuse.consume

    def observe(self, key, value):
        if not caches or self is not caches[-1]:
            caches.append(self)
        consume(self, key, value)

    monkeypatch.setattr(_FrameReuse, "consume", observe)
    reader = LanceDBDataset(root=root, delta_timestamps={VIDEO_KEY: [0, 1 / FPS]})
    ds = reader.episode_pool(episodes=[0], where="frame_index IN (3, 4, 5)", order="sequential")
    with closing(iter(ds)) as it:
        for expected in (4, 5, None):
            next(it)
            assert [i for file, i in caches[-1].frames] == ([] if expected is None else [expected])
    with closing(iter(ds)) as it:
        next(it)
        assert caches[-1].nbytes > 0
    assert not caches[-1].frames and caches[-1].nbytes == 0


def test_hit_only_storage_and_caps():
    batch = torch.arange(160).view(10, 16)
    cache = _FrameReuse(Counter(a=2, b=2, c=1), 1, batch[0].nbytes)
    cache.consume("a", batch[0])
    cache.consume("b", batch[1])
    cache.consume("c", batch[2])
    assert list(cache.frames) == ["a"]
    assert cache.frames["a"].untyped_storage().nbytes() == cache.nbytes == batch[0].nbytes
    assert cache.frames["a"].data_ptr() != batch.data_ptr()
    cache.consume("a", batch[0])
    assert cache.nbytes == 0 and not cache.frames
    cache = _FrameReuse(Counter(a=2), 2, batch[0].nbytes - 1)
    cache.consume("a", batch[0])
    assert not cache.frames


@pytest.mark.parametrize("strategy", ["remaining", "round_robin"])
def test_legacy_planner_matches_upstream(strategy):
    pool = pytest.importorskip("lerobot.streaming.episode_pool").ExactCoveragePool
    counts = {0: 2, 1: 7, 3: 1, 4: 5}
    assert list(_fallback_pool(counts, 2, 13, 1, strategy)) == list(
        pool(list(counts.items()), 2, seed=13, epoch=1, sampling_strategy=strategy)
    )
