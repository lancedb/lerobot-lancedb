"""The streaming frames builder: order-preserving, one commit, memory-bounded."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_lancedb.convert import _frames_reader


def _write(path, start, n, width=4):
    """A LeRobot-style data parquet: monotonic `index`, a vector feature, a scalar."""
    t = pa.table(
        {
            "index": pa.array(range(start, start + n), pa.int64()),
            "observation.state": pa.array(
                [[float(i)] * width for i in range(start, start + n)], pa.list_(pa.float32())
            ),
            "episode_index": pa.array([0] * n, pa.int64()),
        }
    )
    pq.write_table(t, path)


def test_frames_reader_streams_in_order(tmp_path):
    f0, f1 = tmp_path / "file-000.parquet", tmp_path / "file-001.parquet"
    _write(f0, 0, 10)
    _write(f1, 10, 15)

    tbl = _frames_reader([f0, f1]).read_all()

    # all rows, in index order (row N == frame N), across both files
    assert tbl.num_rows == 25
    assert tbl.column("index").to_pylist() == list(range(25))
    # dotted key renamed, vector feature became a fixed-size list
    assert "observation_state" in tbl.schema.names
    assert "observation.state" not in tbl.schema.names
    field = tbl.schema.field("observation_state")
    assert pa.types.is_fixed_size_list(field.type) and field.type.list_size == 4


def test_frames_reader_v2_episode_filenames(tmp_path):
    from lerobot_lancedb.legacy import data_parquet_files

    d0 = tmp_path / "data" / "chunk-000"
    d0.mkdir(parents=True)
    _write(d0 / "episode_000000.parquet", 0, 10)
    _write(d0 / "episode_000001.parquet", 10, 5)

    tbl = _frames_reader(data_parquet_files(tmp_path)).read_all()
    assert tbl.num_rows == 15
    assert tbl.column("index").to_pylist() == list(range(15))


def test_frames_reader_rejects_unsorted(tmp_path):
    f0, f1 = tmp_path / "file-000.parquet", tmp_path / "file-001.parquet"
    _write(f0, 0, 10)
    _write(f1, 10, 15)
    # feeding files out of index order must fail loudly, not silently mis-order
    with pytest.raises(ValueError, match="not index-sorted"):
        _frames_reader([f1, f0]).read_all()
