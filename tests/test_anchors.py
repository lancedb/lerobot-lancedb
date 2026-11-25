from pathlib import Path

import lancedb
import pyarrow as pa
import pytest
import torch
from test_convert_external import FPS, VIDEO_KEY, _make_dataset

from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb.convert import convert


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("anchors")
    src = _make_dataset(root / "native")
    convert(root / "converted", root=src, scalar_indexes=False)
    return root / "converted"


def test_multifragment_sampling_and_projection(converted: Path, monkeypatch):
    db = lancedb.connect(converted)
    table = db.open_table("frames")
    assert table.list_indices() == []
    rows = table.to_arrow()
    rows = rows.set_column(
        rows.schema.get_field_index("task_index"), "task_index", pa.array([0] * 10 + [1] * 6 + [0] * 4)
    )
    table = db.create_table("frames", rows.slice(0, 10), mode="overwrite")
    table.add(rows.slice(10))
    assert table.search().where("episode_index = 1").with_row_id(True).to_arrow()["_rowid"][0].as_py() != 10
    all_episodes = LanceDBDataset(root=converted, columns=["index"])
    mixed = all_episodes.select_anchors(where="episode_index=0 OR frame_index<5")
    assert mixed.sample_weights(episode_weights={0: 40, 1: 10}).tolist() == [4] * 10 + [2] * 5
    assert mixed.sample_weights(task_weights={"reach": 20, 1: 25}).tolist() == [2] * 10 + [5] * 5
    reader = LanceDBDataset(
        root=converted,
        episodes=[1],
        columns=[VIDEO_KEY, "index", "observation.state"],
        delta_timestamps={
            VIDEO_KEY: [-1 / FPS, 0, 1 / FPS],
            "observation.state": [-1 / FPS, 0, 1 / FPS],
        },
    )
    selection = reader.select_anchors(
        episodes=[0, 1], tasks=[0, 1], where="frame_index IN (0, 8, 9)", cameras=[]
    )
    assert selection.indices.tolist() == [0, 8, 9]
    assert len(reader.select_anchors(tasks=[])) == 0
    assert len(reader.select_anchors(episodes=[0])) == 0
    assert selection.sample_weights(episode_weights={1: 6}).tolist() == [2, 2, 2]
    assert selection.sample_weights(task_weights={"reach": 6}).tolist() == [0, 3, 3]
    assert selection.sample_weights(task_weights={0: 6}).tolist() == [0, 3, 3]
    sampler = selection.weighted_sampler(
        4, anchor_weights=[0, 1, 0], generator=torch.Generator().manual_seed(7)
    )
    assert list(sampler) == [1] * 4
    reads = []
    fetch = selection.dataset._fetch_rows

    def track(indices):
        reads.append(indices)
        return fetch(indices)

    def unexpected(*args, **kwargs):
        pytest.fail("Projected-out video payload was fetched")

    monkeypatch.setattr(selection.dataset, "_fetch_rows", track)
    monkeypatch.setattr(lancedb.table.LanceTable, "fetch_blob_ranges", unexpected)
    monkeypatch.setattr(lancedb.table.LanceTable, "fetch_blob_files", unexpected)
    batch = next(iter(torch.utils.data.DataLoader(selection, sampler=sampler, batch_size=4)))
    assert reads == [[17, 18, 19]]
    assert batch["index"].tolist() == [18] * 4 and VIDEO_KEY not in batch
    assert selection[0]["observation.state_is_pad"].tolist() == [True, False, False]
