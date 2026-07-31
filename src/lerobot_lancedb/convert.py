#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert a LeRobot v3.0 dataset to the Lance layout read by lerobot's LanceDBDataset.

Layout produced:
    <out>/
      meta/         # copied verbatim from the source dataset
      frames.lance  # one row per frame, tabular features (dots -> underscores)
      videos.lance  # one row per source mp4, bytes verbatim in a blob v2 column
      meta.lance    # one row per meta/ file (path, bytes) — metadata transport for remote roots

Usage:
    lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance
    lerobot-lance-convert --root /path/to/local/dataset --out ./pusht-lance
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.language import LANGUAGE_COLUMNS

# Temporary vendored copy of the schema contract; switch to
# `from lerobot.datasets.lancedb_dataset import ...` once the upstream
# lerobot PR merges.
from .vendored_schema import (
    FRAMES_TABLE,
    META_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEOS_TABLE,
    build_video_byte_index,
    to_lance_column,
)

VIDEO_BATCH_BYTES = 512 * 1024 * 1024


def _storage_type(dtype: pa.DataType) -> pa.DataType:
    """Replace Arrow extension types (e.g. the JSON type inside language
    tool_calls) with their storage type, recursively. Lance rejects extension
    types it doesn't know; the JSON type's storage is plain string, which is
    also what upstream lerobot falls back to on older pyarrow."""
    if isinstance(dtype, pa.BaseExtensionType):
        return _storage_type(dtype.storage_type)
    if pa.types.is_list(dtype):
        return pa.list_(_storage_type(dtype.value_type))
    if pa.types.is_large_list(dtype):
        return pa.large_list(_storage_type(dtype.value_type))
    if pa.types.is_struct(dtype):
        return pa.struct([field.with_type(_storage_type(field.type)) for field in dtype])
    return dtype


def convert(out_dir: Path, repo_id: str | None = None, root: Path | None = None) -> None:
    """Convert one LeRobot v3.0 dataset to the three-table Lance layout.

    Pass ``root`` to convert a dataset already on disk (nothing is
    downloaded), or ``repo_id`` alone to download from the Hub first
    (into ``$HF_LEROBOT_HOME``, like any other lerobot tool).
    """
    if root is not None:
        src_root = Path(root)
        if not (src_root / "meta").is_dir():
            raise FileNotFoundError(f"{src_root} does not look like a LeRobot dataset (no meta/ dir)")
    elif repo_id is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id)  # downloads to $HF_LEROBOT_HOME if needed
        src_root = Path(ds.root)
    else:
        raise ValueError("pass --repo-id or --root")

    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "meta").exists():
        shutil.copytree(src_root / "meta", out_dir / "meta")

    db = lancedb.connect(str(out_dir))

    print("building meta table ...")
    meta_schema = pa.schema([pa.field("path", pa.string()), pa.field("data", pa.large_binary())])
    meta_files = sorted(f for f in (src_root / "meta").rglob("*") if f.is_file())
    db.create_table(
        META_TABLE,
        pa.Table.from_pylist(
            [{"path": str(f.relative_to(src_root / "meta")), "data": f.read_bytes()} for f in meta_files],
            schema=meta_schema,
        ),
        mode="overwrite",
    )
    print(f"  {len(meta_files)} files")

    print("building frames table ...")
    files = sorted((src_root / "data").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(f) for f in files]).sort_by("index")
    fields, arrays = [], []
    for field in table.schema:
        column = table.column(field.name).combine_chunks()
        if field.name in LANGUAGE_COLUMNS:
            # Variable-length list<struct> message rows (lerobot#3467): keep the
            # nested layout, only stripping extension types lance can't store.
            target = _storage_type(column.type)
            if target != column.type:
                column = column.cast(target)
        elif pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            column = pa.FixedSizeListArray.from_arrays(column.flatten(), len(column[0]))
        arrays.append(column)
        fields.append(pa.field(to_lance_column(field.name), column.type))
    db.create_table(FRAMES_TABLE, pa.Table.from_arrays(arrays, schema=pa.schema(fields)), mode="overwrite")
    print(f"  {table.num_rows} rows")

    video_files = sorted((src_root / "videos").rglob("*.mp4")) if (src_root / "videos").is_dir() else []
    print(f"building videos table from {len(video_files)} files ...")
    schema = pa.schema(
        [
            pa.field("video_key", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("file_index", pa.int64()),
            pa.field("file_size", pa.int64()),
            pa.field("moov_offset", pa.int64()),
            pa.field("moov_size", pa.int64()),
            pa.field("kf_indices", pa.list_(pa.int64())),
            pa.field("kf_positions", pa.list_(pa.int64())),
            lancedb.blob(VIDEO_BLOB_COLUMN),
        ]
    )
    videos_table = db.create_table(VIDEOS_TABLE, schema=schema, mode="overwrite")
    pending, pending_bytes = [], 0
    for mp4 in video_files:
        data = mp4.read_bytes()
        pending.append(
            {
                "video_key": mp4.parts[-3],
                "chunk_index": int(mp4.parent.name.split("-")[1]),
                "file_index": int(mp4.stem.split("-")[1]),
                **build_video_byte_index(mp4),
                VIDEO_BLOB_COLUMN: data,
            }
        )
        pending_bytes += len(data)
        if pending_bytes >= VIDEO_BATCH_BYTES:
            videos_table.add(pending)
            pending, pending_bytes = [], 0
    if pending:
        videos_table.add(pending)
    print(f"  {videos_table.count_rows()} rows")

    # Scalar indexes: not used by the training loader (it reads by _rowid),
    # but they make user-side SQL fast (episode/task filtering, exploration).
    frames_table = db.open_table(FRAMES_TABLE)
    frames_table.create_scalar_index("episode_index")
    frames_table.create_scalar_index("task_index", index_type="BITMAP")
    if video_files:
        videos_table.create_scalar_index("video_key", index_type="BITMAP")

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"done: {out_dir} ({total / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id", default=None, help="Hub dataset id, e.g. lerobot/pusht (downloads if not cached)"
    )
    parser.add_argument(
        "--root", default=None, type=Path, help="local LeRobot dataset dir (nothing is downloaded)"
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory for the Lance layout")
    args = parser.parse_args()
    if args.repo_id is None and args.root is None:
        parser.error("pass --repo-id and/or --root")
    convert(args.out, repo_id=args.repo_id, root=args.root)


if __name__ == "__main__":
    main()
