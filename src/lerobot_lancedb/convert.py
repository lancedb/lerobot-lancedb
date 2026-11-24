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
      videos.lance  # one row per source mp4: External Blob v2 URI or managed blob + byte index
      meta.lance    # one row per meta/ file (path, bytes) — metadata transport for remote roots

Usage:
    # Zero-copy from S3 or Hugging Face Hub (range-reads only MP4 headers; no MP4 payloads copied):
    lerobot-lance-convert --root s3://bucket/droid --out s3://bucket/droid-lance
    lerobot-lance-convert --repo-id lerobot/droid_1.0.1 --out ./droid-lance

    # Local dataset (copies MP4s by default; pass --video-mode external to reference in place):
    lerobot-lance-convert --root /path/to/local/dataset --out ./pusht-lance
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.language import LANGUAGE_COLUMNS
from numpy.typing import NDArray

from ._source import (
    DEFAULT_IO_WORKERS,
    LocalSource,
    _file_uri_to_path,
    is_remote_uri,
    normalize_base_uri,
    open_source,
    validate_rel_path,
)
from .doctor import audit
from .mp4_index import DEFAULT_HEADER_PROBE_BYTES, ProbedMp4, probe_mp4_byte_index

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
EXTERNAL_VIDEO_BATCH_BYTES = 64 * 1024 * 1024
VideoMode = Literal["auto", "external", "copy"]
_MANAGED_OUTPUT_ENTRIES = ("meta", f"{META_TABLE}.lance", f"{FRAMES_TABLE}.lance", f"{VIDEOS_TABLE}.lance")


class _StagedMetaMissing(RuntimeError):
    """Carries FileNotFoundError past LeRobotDatasetMetadata.__init__'s Hub fallback."""

    def __init__(self, original: FileNotFoundError | NotADirectoryError) -> None:
        super().__init__(str(original))
        self.original = original


class _LocalOnlyMetadata(LeRobotDatasetMetadata):
    """Load already-staged metadata strictly from ``root`` without calling out to the Hub."""

    def _load_metadata(self) -> None:
        try:
            super()._load_metadata()
        except (FileNotFoundError, NotADirectoryError) as err:
            raise _StagedMetaMissing(err) from err


def _load_local_metadata(repo_id: str, root: Path, revision: str | None = None) -> LeRobotDatasetMetadata:
    try:
        return _LocalOnlyMetadata(repo_id, root=root, revision=revision)
    except _StagedMetaMissing as wrapped:
        raise wrapped.original from None


@dataclass(frozen=True)
class _VideoRef:
    """One unique video file referenced by ``meta/episodes``."""

    video_key: str
    chunk_index: int
    file_index: int
    rel_path: str


class DamagedSourceError(RuntimeError):
    """The source dataset is damaged where its own metadata references it, or the
    audit that looks for that damage could not run.

    The converter copies or references parquet rows and mp4 files verbatim, so a
    defect the doctor can see becomes a defect of the output; the resulting
    dataset converts fine and then fails inside the video decoder at read time.
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


def _referenced_rel_paths(
    meta: LeRobotDatasetMetadata,
) -> tuple[list[str], list[_VideoRef], dict[str, float]]:
    """Derive referenced data/ and video/ relative paths plus max implied video end timestamps.

    Reads only the integer chunk/file columns from ``meta.episodes`` instead of
    indexing full episode rows (which materializes every per-episode stats column
    and takes >2 minutes on 95k-episode datasets like DROID).
    """
    episodes = meta.episodes
    if episodes is None or len(episodes) == 0:
        return [], [], {}

    data_path_tmpl = meta.data_path
    data_chunks = episodes["data/chunk_index"]
    data_files_col = episodes["data/file_index"]
    seen_data: dict[tuple[int, int], str] = {}
    for chunk_idx, file_idx in zip(data_chunks, data_files_col, strict=True):
        pair = (int(chunk_idx), int(file_idx))
        if pair not in seen_data:
            seen_data[pair] = data_path_tmpl.format(chunk_index=pair[0], file_index=pair[1])
    data_rels = list(seen_data.values())

    video_refs: list[_VideoRef] = []
    implied_end: dict[str, float] = {}
    if meta.video_keys:
        video_path_tmpl = meta.video_path
        if video_path_tmpl is None:
            raise ValueError(f"Dataset '{meta.repo_id}' has video_keys but no video_path template.")
        columns = set(episodes.column_names)
        starts = episodes["dataset_from_index"]
        ends = episodes["dataset_to_index"]
        fps = meta.fps
        for key in meta.video_keys:
            v_chunks = episodes[f"videos/{key}/chunk_index"]
            v_files = episodes[f"videos/{key}/file_index"]
            to_col = episodes[f"videos/{key}/to_timestamp"] if f"videos/{key}/to_timestamp" in columns else None
            from_col = episodes[f"videos/{key}/from_timestamp"]
            seen_vid: dict[tuple[int, int], str] = {}
            for ep_idx, (chunk_idx, file_idx) in enumerate(zip(v_chunks, v_files, strict=True)):
                pair = (int(chunk_idx), int(file_idx))
                rel = seen_vid.get(pair)
                if rel is None:
                    rel = video_path_tmpl.format(video_key=key, chunk_index=pair[0], file_index=pair[1])
                    seen_vid[pair] = rel
                    video_refs.append(
                        _VideoRef(
                            video_key=key,
                            chunk_index=pair[0],
                            file_index=pair[1],
                            rel_path=rel,
                        )
                    )
                end_ts = (
                    float(to_col[ep_idx])
                    if to_col is not None
                    else float(from_col[ep_idx]) + (int(ends[ep_idx]) - int(starts[ep_idx])) / fps
                )
                if end_ts > implied_end.get(rel, 0.0):
                    implied_end[rel] = end_ts

    return data_rels, video_refs, implied_end


def _referenced_files(meta: LeRobotDatasetMetadata, root: Path) -> tuple[list[Path], list[Path]]:
    """The data and video files ``meta/episodes`` references, in reading order."""
    data_rels, video_refs, _ = _referenced_rel_paths(meta)
    data_files = _unique(root / rel for rel in data_rels)
    video_files = _unique(root / ref.rel_path for ref in video_refs)
    return data_files, video_files


def _check_referenced_files_exist(root: Path, data_files: list[Path], video_files: list[Path]) -> None:
    """Fail before writing anything when a referenced file is missing."""
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


def _warn_remote_orphans(
    all_rel_paths: Iterable[str],
    referenced_data: set[str],
    referenced_videos: set[str],
) -> None:
    """Report remote source files that ``meta/episodes`` never references; they are skipped."""
    orphan_data = sorted(
        rel for rel in all_rel_paths if rel.startswith("data/") and rel.endswith(".parquet") and rel not in referenced_data
    )
    if orphan_data:
        print(
            f"warning: skipping {len(orphan_data)} orphan parquet file(s) "
            "not referenced by meta/episodes"
        )
    orphan_videos = sorted(
        rel for rel in all_rel_paths if rel.startswith("videos/") and rel.endswith(".mp4") and rel not in referenced_videos
    )
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


def _handle_blocking_defects(source_label: str, messages: list[str], allow_defects: bool) -> None:
    if not messages:
        return
    listed = [f"  - {msg}" for msg in messages[:10]]
    if len(messages) > 10:
        listed.append(f"  ... and {len(messages) - 10} more")
    if allow_defects:
        print(
            "WARNING: --allow-defects: converting a damaged source dataset. The output keeps "
            "these defects, and every read that needs the missing data fails:\n" + "\n".join(listed),
            file=sys.stderr,
        )
        return
    raise DamagedSourceError(
        f"{source_label} is damaged in files its metadata references; refusing to convert it:\n"
        + "\n".join(listed)
        + "\nRun `lerobot-lance-doctor --root <dataset>` for a local report, fix the source, "
        "or pass --allow-defects to convert anyway."
    )


def _reject_damaged_source(src_root: Path, repo_id: str | None, allow_defects: bool) -> None:
    """Run the doctor's checks and refuse a source whose referenced files are damaged."""
    try:
        result = audit(src_root, repo_id=repo_id)
    except Exception as err:
        raise DamagedSourceError(f"{src_root}: the source audit failed: {err}") from err

    problems = result.blocking_problems
    _handle_blocking_defects(str(src_root), [p.message for p in problems], allow_defects)


def _check_episode_boundaries(meta: LeRobotDatasetMetadata) -> list[str]:
    starts = [int(x) for x in meta.episodes["dataset_from_index"]]
    ends = [int(x) for x in meta.episodes["dataset_to_index"]]
    problems: list[str] = []
    if starts and starts[0] != 0:
        problems.append(f"first episode starts at {starts[0]}, not 0")
    if ends and ends[-1] != meta.total_frames:
        problems.append(f"last episode ends at {ends[-1]}, meta declares {meta.total_frames}")
    for i in range(len(starts) - 1):
        if starts[i + 1] != ends[i]:
            problems.append(f"gap/overlap between episode {i} (ends {ends[i]}) and {i + 1} (starts {starts[i + 1]})")
    return problems


def _video_blob_field() -> pa.Field:
    blob_field = lancedb.blob(VIDEO_BLOB_COLUMN)
    metadata = dict(blob_field.metadata or {})
    metadata[b"lancedb:logical_type"] = b"mp4"
    return blob_field.with_metadata(metadata)


def _videos_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("video_key", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("file_index", pa.int64()),
            pa.field("file_size", pa.int64()),
            pa.field("moov_offset", pa.int64()),
            pa.field("moov_size", pa.int64()),
            pa.field("kf_indices", pa.list_(pa.int64())),
            pa.field("kf_positions", pa.list_(pa.int64())),
            _video_blob_field(),
        ]
    )


def _resolve_video_mode(
    video_mode: VideoMode,
    repo_id: str | None,
    root: Path | str | None,
    video_base_uri: str | None = None,
) -> Literal["external", "copy"]:
    if video_mode not in ("auto", "external", "copy"):
        raise ValueError(f"Invalid video_mode={video_mode!r}: expected 'auto', 'external', or 'copy'")
    if video_mode == "copy" and video_base_uri is not None:
        raise ValueError("video_base_uri is only valid with video_mode='external' (or 'auto')")
    if video_mode == "auto":
        if video_base_uri is not None:
            return "external"
        if root is not None and not is_remote_uri(root) and not str(root).startswith("file://"):
            return "copy"
        return "external"
    if video_mode == "copy" and root is not None and is_remote_uri(root):
        raise ValueError(
            f"video_mode='copy' requires a local --root directory, got remote URI {str(root)!r}; "
            "use --video-mode external to convert remote datasets in place"
        )
    return video_mode


def _connect_output(out_target: Path | str, storage_options: dict[str, str] | None):
    uri = str(out_target)
    if storage_options:
        return lancedb.connect(uri, storage_options=dict(storage_options))
    return lancedb.connect(uri)


def _local_path(target: Path | str) -> Path:
    raw = str(target)
    return _file_uri_to_path(raw) if raw.startswith("file://") else Path(target)


@contextmanager
def _local_build_dir(out_path: Path):
    """Create a sibling build directory on the same filesystem as ``out_path``."""
    resolved = out_path.resolve()
    if resolved == resolved.parent:
        raise ValueError(f"Refusing to convert directly into filesystem root: {out_path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix=f".{resolved.name}.tmp-", dir=resolved.parent))
    try:
        yield build_dir
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)


def _commit_local_output(build_dir: Path, out_path: Path) -> None:
    """Publish a fully built local output directory without clobbering existing tables on failure."""
    resolved = out_path.resolve()
    if not resolved.exists():
        build_dir.rename(resolved)
        return
    if not resolved.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {resolved}")

    backup_dir = Path(tempfile.mkdtemp(prefix=f".{resolved.name}.old-", dir=resolved.parent))
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for name in _MANAGED_OUTPUT_ENTRIES:
            existing = resolved / name
            if existing.exists():
                existing.rename(backup_dir / name)
                moved_old.append(name)
        for name in _MANAGED_OUTPUT_ENTRIES:
            built = build_dir / name
            if built.exists():
                built.rename(resolved / name)
                moved_new.append(name)
    except BaseException:
        for name in reversed(moved_new):
            target = resolved / name
            if target.exists():
                target.rename(build_dir / name)
        for name in reversed(moved_old):
            saved = backup_dir / name
            if saved.exists():
                saved.rename(resolved / name)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


def _prepare_stamped_meta(src_meta_dir: Path, target_dir: Path) -> Path:
    """Copy ``src_meta_dir`` fresh into ``target_dir / 'meta'`` and stamp ``storage_format``."""
    meta_dir = target_dir / "meta"
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    shutil.copytree(src_meta_dir, meta_dir)
    _stamp_storage_format(meta_dir)
    return meta_dir


def _write_meta_table(db, meta_dir: Path, *, mode: str) -> None:
    print("building meta table ...")
    meta_schema = pa.schema([pa.field("path", pa.string()), pa.field("data", pa.large_binary())])
    meta_files = sorted(f for f in meta_dir.rglob("*") if f.is_file())
    db.create_table(
        META_TABLE,
        pa.Table.from_pylist(
            [{"path": f.relative_to(meta_dir).as_posix(), "data": f.read_bytes()} for f in meta_files],
            schema=meta_schema,
        ),
        mode=mode,
    )
    print(f"  {len(meta_files)} files")


def _write_frames_table(db, data_files: list[Path], *, mode: str):
    print("building frames table ...")
    frames_table = db.create_table(FRAMES_TABLE, _frames_reader(data_files), mode=mode)
    print(f"  {frames_table.count_rows()} rows")
    return frames_table


def _create_scalar_indexes(frames_table, videos_table=None) -> None:
    # Scalar indexes: not used by the training loader (it reads by _rowid),
    # but they make user-side SQL fast (episode/task filtering, exploration).
    frames_table.create_scalar_index("episode_index")
    frames_table.create_scalar_index("task_index", index_type="BITMAP")
    if videos_table is not None:
        videos_table.create_scalar_index("video_key", index_type="BITMAP")


def _print_done(out_target: Path | str, extra: str = "") -> None:
    suffix = f" ({extra})" if extra else ""
    if is_remote_uri(out_target):
        print(f"done: {out_target}{suffix}")
        return
    out_path = _local_path(out_target)
    total = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
    detail = f"{total / 1e6:.1f} MB" + (f", {extra}" if extra else "")
    print(f"done: {out_path} ({detail})")


def _merged_videos_write_options(
    storage_options: dict[str, str] | None,
    source_storage_options: dict[str, str] | None,
) -> dict[str, str] | None:
    merged = dict(storage_options or {})
    if source_storage_options:
        for key, value in source_storage_options.items():
            scoped = key if key.startswith("base_1.") else f"base_1.{key}"
            merged.setdefault(scoped, value)
    return merged or None


def _list_int64_array(chunks: list[NDArray[np.int64]]) -> pa.ListArray:
    offsets = np.empty(len(chunks) + 1, dtype=np.int32)
    offsets[0] = 0
    for idx, arr in enumerate(chunks):
        offsets[idx + 1] = offsets[idx] + len(arr)
    values = np.concatenate(chunks) if chunks and offsets[-1] > 0 else np.empty(0, dtype=np.int64)
    return pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(values, type=pa.int64()),
    )


def _convert_copy(
    out_target: Path | str,
    *,
    repo_id: str | None,
    root: Path | str | None,
    allow_defects: bool,
    revision: str | None,
    storage_options: dict[str, str] | None,
) -> None:
    if root is not None:
        src_root = _local_path(root)
        if not (src_root / "meta").is_dir():
            raise FileNotFoundError(f"{src_root} does not look like a LeRobot dataset (no meta/ dir)")
    elif repo_id is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds_kwargs = {"revision": revision} if revision is not None else {}
        ds = LeRobotDataset(repo_id, **ds_kwargs)  # downloads to $HF_LEROBOT_HOME if needed
        src_root = Path(ds.root)
    else:
        raise ValueError("pass --repo-id or --root")

    _reject_damaged_source(src_root, repo_id, allow_defects)

    # Select files from meta/episodes before creating the output, so an
    # incomplete source fails without leaving a half-built dataset behind.
    meta = _load_local_metadata(repo_id or str(src_root), root=src_root, revision=revision)
    data_files, video_files = _referenced_files(meta, src_root)
    _check_referenced_files_exist(src_root, data_files, video_files)
    _warn_orphans(src_root, data_files, video_files)

    schema = _videos_schema()
    # Build streamed batches with the blob column as its STORAGE type
    # (large_binary): pyarrow can't construct the blob-v2 extension array from
    # raw bytes, but lance encodes incoming large_binary into the managed blob
    # column on write (matched by name).
    storage_schema = pa.schema(
        [pa.field(VIDEO_BLOB_COLUMN, pa.large_binary()) if f.name == VIDEO_BLOB_COLUMN else f for f in schema]
    )

    def video_batches():
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

    def _build_into(db, meta_dir: Path, *, table_mode: str) -> None:
        frames_table = _write_frames_table(db, data_files, mode=table_mode)
        print(f"building videos table from {len(video_files)} files (managed copy) ...")
        videos_table = db.create_table(VIDEOS_TABLE, schema=schema, mode=table_mode)
        if video_files:
            videos_table.add(pa.RecordBatchReader.from_batches(storage_schema, video_batches()))
        print(f"  {videos_table.count_rows()} rows")
        _write_meta_table(db, meta_dir, mode=table_mode)
        _create_scalar_indexes(frames_table, videos_table if video_files else None)

    if is_remote_uri(out_target):
        with tempfile.TemporaryDirectory(prefix="lerobot-lance-convert-") as tmp:
            meta_dir = _prepare_stamped_meta(src_root / "meta", Path(tmp))
            db = _connect_output(out_target, storage_options)
            _build_into(db, meta_dir, table_mode="overwrite")
        _print_done(out_target)
    else:
        out_path = _local_path(out_target)
        with _local_build_dir(out_path) as build_dir:
            meta_dir = _prepare_stamped_meta(src_root / "meta", build_dir)
            db = _connect_output(build_dir, storage_options)
            _build_into(db, meta_dir, table_mode="create")
            _commit_local_output(build_dir, out_path)
        _print_done(out_target)


def _external_video_uri(base_uri: str, rel_path: str) -> str:
    clean_base = base_uri if base_uri == "file:///" else base_uri.rstrip("/")
    if clean_base.endswith("/videos") and rel_path.startswith("videos/"):
        sub_rel = rel_path[len("videos/") :]
    else:
        sub_rel = rel_path.lstrip("/")
    encoded_rel = quote(sub_rel, safe="/")
    sep = "" if clean_base.endswith("/") else "/"
    return f"{clean_base}{sep}{encoded_rel}"


def _dry_run_external_base(
    staging: Path,
    schema: pa.Schema,
    resolved_base_uri: str,
    sample_ref: _VideoRef,
    sample_probe: ProbedMp4,
    write_options: dict[str, str] | None,
) -> None:
    """Validate ``initial_bases`` + external URI acceptance locally before mutating a remote output."""
    check_dir = staging / "_external_base_check"
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    check_db = lancedb.connect(str(check_dir))
    sample_batch = pa.RecordBatch.from_arrays(
        [
            pa.array([sample_ref.video_key], type=pa.string()),
            pa.array([sample_ref.chunk_index], type=pa.int64()),
            pa.array([sample_ref.file_index], type=pa.int64()),
            pa.array([sample_probe.file_size], type=pa.int64()),
            pa.array([sample_probe.moov_offset], type=pa.int64()),
            pa.array([sample_probe.moov_size], type=pa.int64()),
            _list_int64_array([sample_probe.kf_indices[:1]]),
            _list_int64_array([sample_probe.kf_positions[:1]]),
            lancedb.blob_uri_array(
                [_external_video_uri(resolved_base_uri, sample_ref.rel_path)],
                positions=[0],
                sizes=[sample_probe.file_size],
            ),
        ],
        schema=schema,
    )
    create_kwargs = {"storage_options": write_options} if write_options else {}
    check_db.create_table(
        VIDEOS_TABLE,
        pa.RecordBatchReader.from_batches(schema, [sample_batch]),
        schema=schema,
        mode="create",
        initial_bases=[lancedb.BasePath(path=resolved_base_uri, name="source_videos", id=1)],
        **create_kwargs,
    )


def _convert_external(
    out_target: Path | str,
    *,
    repo_id: str | None,
    root: Path | str | None,
    allow_defects: bool,
    video_base_uri: str | None,
    revision: str | None,
    storage_options: dict[str, str] | None,
    source_storage_options: dict[str, str] | None,
    video_workers: int,
    header_probe_bytes: int,
) -> None:
    normalized_explicit_base = normalize_base_uri(video_base_uri) if video_base_uri is not None else None
    src_opts = source_storage_options if source_storage_options is not None else storage_options
    with (
        open_source(
            repo_id=repo_id,
            root=root,
            revision=revision,
            storage_options=src_opts,
            max_workers=video_workers,
        ) as source,
        tempfile.TemporaryDirectory(prefix="lerobot-lance-convert-") as tmp,
    ):
        staging = Path(tmp)
        source_files = source.list_files()
        meta_root = source.materialize_meta(staging)
        effective_revision = getattr(source, "commit_sha", None) or revision
        meta = _load_local_metadata(
            repo_id or source.root_label, root=meta_root, revision=effective_revision
        )
        data_rels, video_refs, implied_end = _referenced_rel_paths(meta)
        video_rels = [ref.rel_path for ref in video_refs]
        for rel in data_rels:
            validate_rel_path(rel)
        for rel in video_rels:
            validate_rel_path(rel)

        # Run BOUNDARIES, DATA, VIDEOS, and SUPPLY checks before writing any output,
        # without downloading MP4 payloads.
        blocking: list[str] = _check_episode_boundaries(meta)

        missing_data = [rel for rel in data_rels if rel not in source_files]
        for rel in missing_data:
            blocking.append(f"missing referenced parquet: {rel}")

        data_paths: list[Path] = []
        if not missing_data and data_rels:
            data_paths = source.materialize_data(data_rels, staging)
            parquet_rows = sum(pq.read_metadata(p).num_rows for p in data_paths)
            if parquet_rows != meta.total_frames:
                blocking.append(
                    f"referenced parquets hold {parquet_rows} rows, meta declares {meta.total_frames}"
                )

        missing_videos = [rel for rel in video_rels if rel not in source_files]
        zero_videos = [rel for rel in video_rels if rel in source_files and source_files[rel].size == 0]
        for rel in missing_videos:
            blocking.append(f"missing referenced video: {rel}")
        for rel in zero_videos:
            blocking.append(f"zero-byte video: {rel}")

        probe_targets = [rel for rel in video_rels if rel in source_files and source_files[rel].size > 0]
        probed: dict[str, ProbedMp4] = {}
        probe_errors: dict[str, Exception] = {}

        if probe_targets:
            print(f"probing {len(probe_targets)} video(s) via range reads ...")

            def _probe_one(rel: str) -> tuple[str, ProbedMp4]:
                res = probe_mp4_byte_index(
                    rel,
                    source.read_range,
                    file_size=source_files[rel].size,
                    header_probe_bytes=header_probe_bytes,
                )
                return rel, res

            with ThreadPoolExecutor(max_workers=max(1, video_workers)) as pool:
                futs = {pool.submit(_probe_one, rel): rel for rel in probe_targets}
                for fut in as_completed(futs):
                    rel = futs[fut]
                    try:
                        _, res = fut.result()
                        probed[rel] = res
                    except Exception as err:
                        probe_errors[rel] = err
                        blocking.append(f"unreadable video {rel}: {err}")

            for rel in probe_targets:
                res = probed.get(rel)
                if res is None:
                    continue
                implied_frames = round(implied_end[rel] * meta.fps)
                if res.sample_count < implied_frames:
                    blocking.append(
                        f"{rel}: container declares {res.sample_count} frames, episodes imply "
                        f"{implied_frames} (short by {implied_frames - res.sample_count})"
                    )

        _handle_blocking_defects(source.root_label, blocking, allow_defects)

        missing_all = missing_data + missing_videos
        if missing_all:
            shown = ", ".join(missing_all[:5])
            more = f" and {len(missing_all) - 5} more" if len(missing_all) > 5 else ""
            raise FileNotFoundError(
                f"{len(missing_all)} file(s) referenced by meta/episodes are missing under "
                f"{source.root_label}: {shown}{more}; the source dataset is incomplete"
            )
        if zero_videos or probe_errors:
            first_rel = zero_videos[0] if zero_videos else next(iter(probe_errors))
            if first_rel in probe_errors:
                raise probe_errors[first_rel]
            raise ValueError(f"Cannot index zero-byte video {first_rel!r}")

        if isinstance(source, LocalSource):
            _warn_orphans(source.root, data_paths, [source.root / r for r in video_rels])
        else:
            _warn_remote_orphans(source_files.keys(), set(data_rels), set(video_rels))

        schema = _videos_schema()
        resolved_base_uri = (
            normalized_explicit_base
            if normalized_explicit_base is not None
            else normalize_base_uri(source.default_video_base_uri(video_rels))
        ) if video_refs else ""
        videos_write_opts = _merged_videos_write_options(storage_options, source_storage_options)

        total_reads = sum(p.range_gets for p in probed.values())
        total_bytes_read = sum(p.bytes_read for p in probed.values())
        total_video_bytes = sum(p.file_size for p in probed.values())

        def ext_video_batches():
            video_keys: list[str] = []
            chunk_indices: list[int] = []
            file_indices: list[int] = []
            file_sizes: list[int] = []
            moov_offsets: list[int] = []
            moov_sizes: list[int] = []
            kf_idx_chunks: list[NDArray[np.int64]] = []
            kf_pos_chunks: list[NDArray[np.int64]] = []
            uris: list[str] = []
            positions: list[int] = []
            sizes: list[int] = []
            pending_bytes = 0

            def flush() -> pa.RecordBatch:
                arrays = [
                    pa.array(video_keys, type=pa.string()),
                    pa.array(chunk_indices, type=pa.int64()),
                    pa.array(file_indices, type=pa.int64()),
                    pa.array(file_sizes, type=pa.int64()),
                    pa.array(moov_offsets, type=pa.int64()),
                    pa.array(moov_sizes, type=pa.int64()),
                    _list_int64_array(kf_idx_chunks),
                    _list_int64_array(kf_pos_chunks),
                    lancedb.blob_uri_array(uris, positions=positions, sizes=sizes),
                ]
                return pa.RecordBatch.from_arrays(arrays, schema=schema)

            for ref in video_refs:
                p = probed[ref.rel_path]
                video_keys.append(ref.video_key)
                chunk_indices.append(ref.chunk_index)
                file_indices.append(ref.file_index)
                file_sizes.append(p.file_size)
                moov_offsets.append(p.moov_offset)
                moov_sizes.append(p.moov_size)
                kf_idx_chunks.append(p.kf_indices)
                kf_pos_chunks.append(p.kf_positions)
                uris.append(_external_video_uri(resolved_base_uri, ref.rel_path))
                positions.append(0)
                sizes.append(p.file_size)
                pending_bytes += len(p.kf_indices) * 16 + 256
                if pending_bytes >= EXTERNAL_VIDEO_BATCH_BYTES:
                    yield flush()
                    video_keys.clear()
                    chunk_indices.clear()
                    file_indices.clear()
                    file_sizes.clear()
                    moov_offsets.clear()
                    moov_sizes.clear()
                    kf_idx_chunks.clear()
                    kf_pos_chunks.clear()
                    uris.clear()
                    positions.clear()
                    sizes.clear()
                    pending_bytes = 0
            if video_keys:
                yield flush()

        def _build_into(db, meta_dir: Path, *, table_mode: str, drop_existing_videos: bool) -> None:
            videos_create_kwargs = {"storage_options": videos_write_opts} if videos_write_opts else {}
            if video_refs:
                print(
                    f"building videos table from {len(video_refs)} external video(s) "
                    f"(base: {resolved_base_uri}) ..."
                )
                if drop_existing_videos:
                    db.drop_table(VIDEOS_TABLE, ignore_missing=True)
                videos_table = db.create_table(
                    VIDEOS_TABLE,
                    pa.RecordBatchReader.from_batches(schema, ext_video_batches()),
                    schema=schema,
                    mode="create",
                    initial_bases=[lancedb.BasePath(path=resolved_base_uri, name="source_videos", id=1)],
                    **videos_create_kwargs,
                )
                print(
                    f"  {videos_table.count_rows()} rows "
                    f"({total_reads} range GETs, {total_bytes_read / 1e6:.1f} MB read vs "
                    f"{total_video_bytes / 1e9:.2f} GB source MP4s)"
                )
            else:
                if drop_existing_videos:
                    db.drop_table(VIDEOS_TABLE, ignore_missing=True)
                videos_table = db.create_table(VIDEOS_TABLE, schema=schema, mode="create")
                print("  0 rows")

            frames_table = _write_frames_table(db, data_paths, mode=table_mode)
            _write_meta_table(db, meta_dir, mode=table_mode)
            _create_scalar_indexes(frames_table, videos_table if video_refs else None)

        if is_remote_uri(out_target):
            if video_refs:
                _dry_run_external_base(
                    staging,
                    schema,
                    resolved_base_uri,
                    video_refs[0],
                    probed[video_refs[0].rel_path],
                    videos_write_opts,
                )
            meta_dir = _prepare_stamped_meta(meta_root / "meta", staging / "_stamped")
            db = _connect_output(out_target, storage_options)
            _build_into(db, meta_dir, table_mode="overwrite", drop_existing_videos=True)
            _print_done(out_target, extra=f"external base: {resolved_base_uri}" if video_refs else "")
        else:
            out_path = _local_path(out_target)
            with _local_build_dir(out_path) as build_dir:
                meta_dir = _prepare_stamped_meta(meta_root / "meta", build_dir)
                db = _connect_output(build_dir, storage_options)
                _build_into(db, meta_dir, table_mode="create", drop_existing_videos=False)
                _commit_local_output(build_dir, out_path)
            _print_done(out_target, extra=f"external base: {resolved_base_uri}" if video_refs else "")


def convert(
    out_dir: Path | str,
    repo_id: str | None = None,
    root: Path | str | None = None,
    allow_defects: bool = False,
    *,
    video_mode: VideoMode = "auto",
    video_base_uri: str | None = None,
    revision: str | None = None,
    storage_options: dict[str, str] | None = None,
    source_storage_options: dict[str, str] | None = None,
    video_workers: int = DEFAULT_IO_WORKERS,
    header_probe_bytes: int = DEFAULT_HEADER_PROBE_BYTES,
) -> None:
    """Convert one LeRobot v3.0 dataset to the three-table Lance layout.

    Parameters
    ----------
    out_dir:
        Local directory or object-store URI (``s3://``, ``gs://``, ``hf://``)
        where the Lance tables are written.
    repo_id:
        Hugging Face Hub dataset id (e.g. ``lerobot/droid_1.0.1``). In
        ``external`` mode (the default for ``repo_id``), no MP4s are downloaded
        and external descriptors are pinned to the resolved 40-char commit SHA.
    root:
        Local LeRobot dataset directory or object-store URI (``s3://``, ``gs://``,
        ``hf://``, ``file://``).
    allow_defects:
        Convert even when source checks find defects in referenced files.
    video_mode:
        ``"auto"`` (default): ``"external"`` for remote ``root`` URIs and
        ``repo_id``; ``"copy"`` for local directory ``root``. Pass ``"external"``
        to write zero-copy External Blob v2 descriptors without copying MP4s, or
        ``"copy"`` to embed MP4 bytes as managed blobs.
    video_base_uri:
        Optional explicit base URI registered in ``initial_bases`` for external
        video descriptors (e.g. an ``s3://`` prefix when indexing from a local
        mirror).
    revision:
        Optional Hugging Face Hub revision (branch, tag, or commit SHA).
    storage_options:
        Optional storage options forwarded to ``lancedb.connect`` (and to the
        remote source unless ``source_storage_options`` is given).
    source_storage_options:
        Optional separate storage options for reading the remote source.
    video_workers:
        Concurrent range-read workers for external MP4 probing.
    header_probe_bytes:
        Initial prefix read size per MP4 when locating ``moov`` (default 1 MiB).
    """
    if repo_id is None and root is None:
        raise ValueError("pass --repo-id or --root")

    mode = _resolve_video_mode(video_mode, repo_id, root, video_base_uri=video_base_uri)
    if mode == "external":
        _convert_external(
            out_dir,
            repo_id=repo_id,
            root=root,
            allow_defects=allow_defects,
            video_base_uri=video_base_uri,
            revision=revision,
            storage_options=storage_options,
            source_storage_options=source_storage_options,
            video_workers=video_workers,
            header_probe_bytes=header_probe_bytes,
        )
    else:
        _convert_copy(
            out_dir,
            repo_id=repo_id,
            root=root,
            allow_defects=allow_defects,
            revision=revision,
            storage_options=storage_options,
        )


def _parse_kv_options(items: list[str] | None, parser: argparse.ArgumentParser) -> dict[str, str] | None:
    if not items:
        return None
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            parser.error(f"invalid storage option {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            parser.error(f"invalid storage option {item!r}: empty key")
        parsed[key] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hub dataset id, e.g. lerobot/droid_1.0.1 (zero-copy external by default)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="local LeRobot dataset dir or remote URI (s3://, gs://, hf://, file://)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output local directory or remote URI (s3://, gs://, hf://) for the Lance layout",
    )
    parser.add_argument(
        "--video-mode",
        choices=["auto", "external", "copy"],
        default="auto",
        help=(
            "how to store videos in videos.lance: 'external' writes zero-copy External Blob v2 "
            "descriptors without downloading/copying MP4s; 'copy' embeds MP4 bytes as managed "
            "blobs; 'auto' (default) uses 'external' for remote/--repo-id sources and 'copy' for local dirs"
        ),
    )
    parser.add_argument(
        "--video-base-uri",
        default=None,
        help="explicit base URI registered in initial_bases for external video descriptors",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face Hub revision (branch, tag, or commit SHA; pinned to commit SHA in external mode)",
    )
    parser.add_argument(
        "--storage-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="storage option forwarded to LanceDB and remote source (repeatable, e.g. aws_region=us-east-2)",
    )
    parser.add_argument(
        "--source-storage-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="storage option for reading the remote source only (overrides --storage-option for source reads)",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=DEFAULT_IO_WORKERS,
        help=f"concurrent range-read workers in external mode (default: {DEFAULT_IO_WORKERS})",
    )
    parser.add_argument(
        "--allow-defects",
        action="store_true",
        help=(
            "convert even when source checks find defects in files episodes reference; "
            "the output keeps them"
        ),
    )
    args = parser.parse_args()
    if args.repo_id is None and args.root is None:
        parser.error("pass --repo-id and/or --root")

    storage_options = _parse_kv_options(args.storage_option, parser)
    source_storage_options = _parse_kv_options(args.source_storage_option, parser)

    try:
        convert(
            args.out,
            repo_id=args.repo_id,
            root=args.root,
            allow_defects=args.allow_defects,
            video_mode=args.video_mode,
            video_base_uri=args.video_base_uri,
            revision=args.revision,
            storage_options=storage_options,
            source_storage_options=source_storage_options,
            video_workers=args.video_workers,
        )
    except DamagedSourceError as err:
        parser.exit(1, f"\n{err}\n")


if __name__ == "__main__":
    main()
