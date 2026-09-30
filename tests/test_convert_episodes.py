"""The episodes table: meta/episodes parquet minus stats, one row per episode, lance-safe names."""

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_lancedb.convert import _episodes_table


def _write_episodes(meta_dir, start, n):
    cam = "videos/observation.images.top"
    rows = {
        "episode_index": list(range(start, start + n)),
        "length": [10] * n,
        "dataset_from_index": [10 * i for i in range(start, start + n)],
        "dataset_to_index": [10 * (i + 1) for i in range(start, start + n)],
        "tasks": [["pick"]] * n,
        f"{cam}/chunk_index": [0] * n,
        f"{cam}/file_index": [i // 2 for i in range(start, start + n)],
        f"{cam}/from_timestamp": [float(i % 2) for i in range(start, start + n)],
        f"{cam}/to_timestamp": [float(i % 2) + 1.0 for i in range(start, start + n)],
        "stats/observation.state/mean": [[0.5]] * n,
    }
    out = meta_dir / "episodes" / "chunk-000"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out / f"file-{start:03d}.parquet")


def test_episodes_table_drops_stats_and_renames(tmp_path):
    _write_episodes(tmp_path, 0, 3)
    _write_episodes(tmp_path, 3, 2)

    tbl = _episodes_table(tmp_path)

    assert tbl.column("episode_index").to_pylist() == [0, 1, 2, 3, 4]
    assert not any(name.startswith("stats/") for name in tbl.schema.names)
    assert "videos/observation_images_top/file_index" in tbl.schema.names
    assert not any("." in name for name in tbl.schema.names)
    assert tbl.column("videos/observation_images_top/file_index").to_pylist() == [0, 0, 1, 1, 2]
    assert tbl.column("dataset_from_index").to_pylist() == [0, 10, 20, 30, 40]
