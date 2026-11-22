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
      meta/         # copied from the source dataset + storage_format stamped in info.json
      frames.lance  # one row per frame, tabular features (dots -> underscores)
      videos.lance  # one row per source mp4, bytes verbatim in a blob v2 column
      meta.lance    # one row per meta/ file (path, bytes) — metadata transport for remote roots

Usage:
    lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance
    lerobot-lance-convert --root /path/to/local/dataset --out ./pusht-lance
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.language import LANGUAGE_COLUMNS

from .doctor import audit

# Schema contract comes from the (temporarily vendored) reader, so the writer and
# reader can't disagree. Once the loader lands upstream, reader.py is deleted and
# this imports from `lerobot.datasets.lancedb_dataset` instead.
from .reader import (
    FRAMES_TABLE,
    META_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEOS_TABLE,
    build_video_byte_index,
    to_lance_column,
)

VIDEO_BATCH_BYTES = 512 * 1024 * 1024


class DamagedSourceError(RuntimeError):
    """The source dataset is damaged where its own metadata references it, or the
    audit that looks for that damage could not run.

    The converter copies parquet rows and mp4 files byte for byte, so a defect the
    doctor can see becomes a defect of the output; the resulting dataset converts
    fine and then fails inside the video decoder at read time.
    """


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


def _frames_reader(files: list[Path]) -> pa.RecordBatchReader:
    """Stream the frames table from LeRobot ``data/`` parquet files, in index order.

    Yields ~one record batch at a time (bounded memory) instead of concatenating
    every file into one in-memory table, and applies the Lance column transforms
    per batch: dotted keys -> underscores, fixed-width vector features (list) ->
    fixed-size list, language columns (list<struct>) keep their nested layout with
    extension types stripped.

    Row order is load-bearing for the loader (row N == absolute frame N), so the
    ``index`` column is verified to stay monotonically non-decreasing across the
    whole stream; a dataset whose ``data/`` files are not index-sorted fails loudly
    here rather than silently producing a mis-ordered table. (LeRobot v3.0 writes
    ``data/`` in ascending index order, so this never triggers for valid datasets.)
    """
    schema0 = pq.read_schema(files[0])
    out_fields, plan = [], []
    for field in schema0:
        if field.name in LANGUAGE_COLUMNS:
            target = _storage_type(field.type)
            plan.append((field.name, "lang", target))
            out_fields.append(pa.field(to_lance_column(field.name), target))
        elif pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            width = len(pq.read_table(files[0], columns=[field.name]).column(field.name)[0])
            out_type = pa.list_(field.type.value_type, width)
            plan.append((field.name, "fsl", width))
            out_fields.append(pa.field(to_lance_column(field.name), out_type))
        else:
            plan.append((field.name, "asis", None))
            out_fields.append(pa.field(to_lance_column(field.name), field.type))
    out_schema = pa.schema(out_fields)

    def batches():
        last = None
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches():
                if batch.num_rows == 0:
                    continue
                idx = batch.column("index")
                diffs = pc.subtract(idx.slice(1), idx.slice(0, len(idx) - 1)) if len(idx) > 1 else None
                lo, hi = idx[0].as_py(), idx[-1].as_py()
                if (last is not None and lo <= last) or (diffs is not None and pc.min(diffs).as_py() < 0):
                    raise ValueError(
                        f"frames are not index-sorted at {f} (index {lo}..{hi} after {last}); "
                        "the streaming converter requires index-ordered data/ files"
                    )
                last = hi
                cols = []
                for name, kind, arg in plan:
                    col = batch.column(name)
                    if kind == "lang" and col.type != arg:
                        col = col.cast(arg)
                    elif kind == "fsl":
                        col = pa.FixedSizeListArray.from_arrays(col.flatten(), arg)
                    cols.append(col)
                yield pa.RecordBatch.from_arrays(cols, schema=out_schema)

    return pa.RecordBatchReader.from_batches(out_schema, batches())


def _unique(paths: Iterable[Path]) -> list[Path]:
    """De-duplicate paths while keeping first-seen order."""
    return list(dict.fromkeys(paths))


def _referenced_files(meta: LeRobotDatasetMetadata, root: Path) -> tuple[list[Path], list[Path]]:
    """The data and video files ``meta/episodes`` references, in reading order.

    This is the same reference model as the doctor's DATA and VIDEOS checks: the
    files the metadata says the dataset contains. Public datasets can carry files
    the metadata never points at (DROID 1.0.1 ships 70 unreferenced parquet
    shards, out of index order); those are not part of the dataset, so the
    converter must not read them. Data files keep episode order rather than path
    order, so the frames reader still sees monotonically increasing indices.
    """
    episodes = range(meta.total_episodes)
    data_files = _unique(root / meta.get_data_file_path(ep) for ep in episodes)
    video_files = _unique(
        root / meta.get_video_file_path(ep, key) for key in meta.video_keys for ep in episodes
    )
    return data_files, video_files


def _check_referenced_files_exist(root: Path, data_files: list[Path], video_files: list[Path]) -> None:
    """Fail before writing anything when a referenced file is missing.

    Selecting by metadata means a missing file can no longer hide behind the
    old directory glob; the source dataset is incomplete and must be fixed.
    """
    missing = [path for path in data_files + video_files if not path.exists()]
    if not missing:
        return
    shown = ", ".join(str(path.relative_to(root)) for path in missing[:5])
    more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
    raise FileNotFoundError(
        f"{len(missing)} file(s) referenced by meta/episodes are missing under {root}: "
        f"{shown}{more}; the source dataset is incomplete"
    )


def _parquet_rows(files: list[Path]) -> tuple[int, int]:
    """Row count from the parquet footers, plus files whose footer is unreadable."""
    rows = 0
    unreadable = 0
    for path in files:
        try:
            rows += pq.read_metadata(path).num_rows
        except Exception:  # A file we are skipping must not fail the conversion.
            unreadable += 1
    return rows, unreadable


def _warn_orphans(root: Path, data_files: list[Path], video_files: list[Path]) -> None:
    """Report on-disk files that ``meta/episodes`` never references; they are skipped."""
    data_dir = root / "data"
    on_disk_data = set(data_dir.rglob("*.parquet")) if data_dir.is_dir() else set()
    orphan_data = sorted(on_disk_data - set(data_files))
    if orphan_data:
        rows, unreadable = _parquet_rows(orphan_data)
        detail = f"{rows:,} rows" if not unreadable else f"{rows:,} rows, {unreadable} unreadable"
        print(
            f"warning: skipping {len(orphan_data)} orphan parquet file(s) "
            f"not referenced by meta/episodes ({detail})"
        )
    video_dir = root / "videos"
    on_disk_videos = set(video_dir.rglob("*.mp4")) if video_dir.is_dir() else set()
    orphan_videos = sorted(on_disk_videos - set(video_files))
    if orphan_videos:
        print(
            f"warning: skipping {len(orphan_videos)} orphan video file(s) "
            "not referenced by meta/episodes"
        )


def _stamp_storage_format(meta_dir: Path) -> None:
    """Declare the Lance storage format in ``meta/info.json``.

    Readers select the storage backend from this field; layout detection is only
    a fallback for datasets converted before the field existed. Idempotent.
    """
    info_file = meta_dir / "info.json"
    info = json.loads(info_file.read_text())
    if info.get("storage_format") != "lance":
        info["storage_format"] = "lance"
        info_file.write_text(json.dumps(info, indent=4))


def _reject_damaged_source(src_root: Path, repo_id: str | None, allow_defects: bool) -> None:
    """Run the doctor's checks and refuse a source whose referenced files are damaged.

    Only defects in data an episode references block: those bytes end up in the
    output, so ``berkeley_autolab_ur5``'s short aggregated videos would convert
    silently and then crash the reader past the last real frame. Orphan parquet and
    video findings do not block: they are not referenced data, and the converter's
    file selection skips them (see ``_referenced_files``), so they never reach the
    output.

    ``allow_defects`` keeps the old copy-anything behavior for datasets that are
    already published or under investigation, but says clearly what it lets through.
    """
    try:
        result = audit(src_root, repo_id=repo_id)
    except Exception as err:
        raise DamagedSourceError(f"{src_root}: the source audit failed: {err}") from err

    problems = result.blocking_problems
    if not problems:
        return
    listed = [f"  - {problem.message}" for problem in problems[:10]]
    if len(problems) > 10:
        listed.append(f"  ... and {len(problems) - 10} more")
    if allow_defects:
        print(
            "WARNING: --allow-defects: converting a damaged source dataset. The output keeps "
            "these defects, and every read that needs the missing data fails:\n" + "\n".join(listed),
            file=sys.stderr,
        )
        return
    raise DamagedSourceError(
        f"{src_root} is damaged in files its metadata references; refusing to convert it:\n"
        + "\n".join(listed)
        + "\nThe output would copy the damage. Run `lerobot-lance-doctor --root <dataset>` "
        "for the full report, fix the source, or pass --allow-defects to convert anyway."
    )


def convert(
    out_dir: Path, repo_id: str | None = None, root: Path | None = None, allow_defects: bool = False
) -> None:
    """Convert one LeRobot v3.0 dataset to the three-table Lance layout.

    Pass ``root`` to convert a dataset already on disk (nothing is
    downloaded), or ``repo_id`` alone to download from the Hub first
    (into ``$HF_LEROBOT_HOME``, like any other lerobot tool).

    The source is audited with the doctor's checks before anything is written, so
    a dataset that is damaged where episodes reference it fails here instead of
    producing an unreadable output; ``allow_defects`` overrides that refusal.
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

    _reject_damaged_source(src_root, repo_id, allow_defects)

    # Select files from meta/episodes before creating the output, so an
    # incomplete source fails without leaving a half-built dataset behind.
    meta = LeRobotDatasetMetadata(repo_id or str(src_root), root=src_root)
    data_files, video_files = _referenced_files(meta, src_root)
    _check_referenced_files_exist(src_root, data_files, video_files)
    _warn_orphans(src_root, data_files, video_files)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "meta").exists():
        shutil.copytree(src_root / "meta", out_dir / "meta")
    _stamp_storage_format(out_dir / "meta")

    db = lancedb.connect(str(out_dir))

    print("building meta table ...")
    # Read from out_dir/meta (the stamped copy) so the meta table — what remote
    # roots materialize meta/ from — carries storage_format too.
    meta_schema = pa.schema([pa.field("path", pa.string()), pa.field("data", pa.large_binary())])
    meta_files = sorted(f for f in (out_dir / "meta").rglob("*") if f.is_file())
    db.create_table(
        META_TABLE,
        pa.Table.from_pylist(
            [{"path": str(f.relative_to(out_dir / "meta")), "data": f.read_bytes()} for f in meta_files],
            schema=meta_schema,
        ),
        mode="overwrite",
    )
    print(f"  {len(meta_files)} files")

    print("building frames table ...")
    db.create_table(FRAMES_TABLE, _frames_reader(data_files), mode="overwrite")
    print(f"  {db.open_table(FRAMES_TABLE).count_rows()} rows")

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
    # Build streamed batches with the blob column as its STORAGE type
    # (large_binary): pyarrow can't construct the blob-v2 extension array from raw
    # bytes, but lance encodes incoming large_binary into the managed blob column
    # on write (matched by name), same as the list-of-dicts add path did.
    storage_schema = pa.schema(
        [pa.field(VIDEO_BLOB_COLUMN, pa.large_binary()) if f.name == VIDEO_BLOB_COLUMN else f for f in schema]
    )

    def video_batches():
        # Yield ~VIDEO_BATCH_BYTES record batches lazily so peak producer memory
        # stays bounded, but stream them through ONE videos_table.add() below.
        # Lance keeps a single writer open across the batches, so the whole videos
        # table is one commit with Lance-sized fragments -- instead of one commit
        # (a new version + fragments) per 512 MB, which fragments the dataset and
        # multiplies manifest/scan/object-open work for every training worker.
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
                yield pa.RecordBatch.from_pylist(pending, schema=storage_schema)
                pending, pending_bytes = [], 0
        if pending:
            yield pa.RecordBatch.from_pylist(pending, schema=storage_schema)

    if video_files:
        videos_table.add(pa.RecordBatchReader.from_batches(storage_schema, video_batches()))
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
    parser.add_argument(
        "--allow-defects",
        action="store_true",
        help=(
            "convert even when the doctor finds defects in files episodes reference; "
            "the output keeps them"
        ),
    )
    args = parser.parse_args()
    if args.repo_id is None and args.root is None:
        parser.error("pass --repo-id and/or --root")
    try:
        convert(args.out, repo_id=args.repo_id, root=args.root, allow_defects=args.allow_defects)
    except DamagedSourceError as err:
        parser.exit(1, f"\n{err}\n")


if __name__ == "__main__":
    main()
