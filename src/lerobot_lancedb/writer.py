#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert a parquet+mp4 :class:`LeRobotDataset` to a single Lance table.

Output layout::

    <output>/
      <table_name>.lance/         # the Lance table (one row per frame)
      meta/                       # verbatim copy of the source meta/
        info.json
        stats.json
        tasks.parquet
        subtasks.parquet (when present)
        episodes/chunk-NNN/file-MMM.parquet
"""

from __future__ import annotations

import io
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames


logger = logging.getLogger(__name__)


_DEFAULT_JPEG_QUALITY = 95


def _to_lance_name(name: str) -> str:
    return name.replace(".", "_")


def _encode_jpeg(frame: torch.Tensor | np.ndarray, jpeg_quality: int) -> bytes:
    """Encode an image (CHW or HWC, uint8 or float [0,1]) as JPEG bytes."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = (arr.clip(0, 1) * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def _build_schema(features: dict[str, dict], has_subtasks: bool) -> tuple[pa.Schema, dict[str, int]]:
    fields = [
        pa.field("episode_index", pa.int32()),
        pa.field("frame_index", pa.int32()),
        pa.field("index", pa.int64()),
        pa.field("timestamp", pa.float32()),
        pa.field("task_index", pa.int32()),
    ]
    if has_subtasks:
        fields.append(pa.field("subtask_index", pa.int32()))

    dims: dict[str, int] = {}
    reserved = {"episode_index", "frame_index", "index", "timestamp", "task_index", "subtask_index"}
    for key, ft in features.items():
        lance_key = _to_lance_name(key)
        if lance_key in reserved:
            continue
        if ft["dtype"] in ("image", "video"):
            fields.append(pa.field(lance_key, pa.binary()))
        else:
            shape = tuple(ft.get("shape", (1,))) or (1,)
            dim = int(np.prod(shape))
            dims[lance_key] = dim
            fields.append(pa.field(lance_key, pa.list_(pa.float32(), dim)))
    return pa.schema(fields), dims


def _episode_record_batch(
    src: LeRobotDataset,
    ep_idx: int,
    schema: pa.Schema,
    tabular_dims: dict[str, int],
    image_keys: set[str],
    video_keys: set[str],
    jpeg_quality: int,
    tolerance_s: float,
) -> pa.RecordBatch:
    """Materialize one episode → one ``pa.RecordBatch``."""
    meta = src.meta
    ep = meta.episodes[ep_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start
    if ep_len <= 0:
        raise ValueError(f"Episode {ep_idx} has non-positive length: {ep_len}")

    hf = src.hf_dataset  # triggers lazy load
    if src.reader._absolute_to_relative_idx is None:
        rel_indices = list(range(ep_start, ep_end))
    else:
        rel_indices = [src.reader._absolute_to_relative_idx[i] for i in range(ep_start, ep_end)]
    rows = hf[rel_indices]

    decoded_videos: dict[str, torch.Tensor] = {}
    if video_keys:
        ts_local = rows["timestamp"]
        if isinstance(ts_local, list):
            ts_local = (
                torch.stack(ts_local)
                if isinstance(ts_local[0], torch.Tensor)
                else torch.tensor(ts_local)
            )
        ts_local_list = [float(t) for t in ts_local.tolist()]

        for vid_key in video_keys:
            from_ts = float(ep[f"videos/{vid_key}/from_timestamp"])
            shifted = [from_ts + t for t in ts_local_list]
            video_path = src.root / meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted,
                tolerance_s,
                src._video_backend,
                return_uint8=True,
            )
            decoded_videos[vid_key] = frames

    arrays: dict[str, pa.Array] = {}

    def _to_numpy_scalar_column(values, dtype) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(dtype)
        if isinstance(values, list) and values and isinstance(values[0], torch.Tensor):
            return torch.stack(values).detach().cpu().numpy().astype(dtype)
        return np.asarray(values, dtype=dtype)

    arrays["episode_index"] = pa.array(np.full(ep_len, ep_idx, dtype=np.int32), type=pa.int32())
    arrays["frame_index"] = pa.array(np.arange(ep_len, dtype=np.int32), type=pa.int32())
    arrays["index"] = pa.array(np.arange(ep_start, ep_end, dtype=np.int64), type=pa.int64())
    arrays["timestamp"] = pa.array(
        _to_numpy_scalar_column(rows["timestamp"], np.float32), type=pa.float32()
    )
    arrays["task_index"] = pa.array(
        _to_numpy_scalar_column(rows["task_index"], np.int32), type=pa.int32()
    )
    if "subtask_index" in [f.name for f in schema]:
        sub_idx = rows.get("subtask_index", np.zeros(ep_len, dtype=np.int32))
        arrays["subtask_index"] = pa.array(
            _to_numpy_scalar_column(sub_idx, np.int32), type=pa.int32()
        )

    for key, _ft in meta.features.items():
        lance_key = _to_lance_name(key)
        if lance_key in arrays:
            continue
        if key in video_keys:
            frames = decoded_videos[key]
            blobs = [_encode_jpeg(frames[i], jpeg_quality) for i in range(ep_len)]
            arrays[lance_key] = pa.array(blobs, type=pa.binary())
        elif key in image_keys:
            blobs = []
            col = rows[key]
            for i in range(ep_len):
                v = col[i] if isinstance(col, list) else col[i]
                if isinstance(v, Image.Image):
                    buf = io.BytesIO()
                    v.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
                    blobs.append(buf.getvalue())
                else:
                    blobs.append(_encode_jpeg(v, jpeg_quality))
            arrays[lance_key] = pa.array(blobs, type=pa.binary())
        else:
            dim = tabular_dims[lance_key]
            col = rows[key]
            if isinstance(col, torch.Tensor):
                flat = col.detach().cpu().numpy().astype(np.float32).reshape(ep_len, dim).reshape(-1)
            elif isinstance(col, list) and col and isinstance(col[0], torch.Tensor):
                flat = (
                    torch.stack(col)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(ep_len, dim)
                    .reshape(-1)
                )
            else:
                flat = np.asarray(col, dtype=np.float32).reshape(ep_len, dim).reshape(-1)
            arrays[lance_key] = pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=pa.float32()), dim
            )

    ordered = [arrays[f.name] for f in schema]
    return pa.RecordBatch.from_arrays(ordered, schema=schema)


def _copy_metadata(src_root: Path, dst_root: Path) -> None:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    shutil.copytree(src_meta, dst_meta)
    logger.info("Copied metadata: %s → %s", src_meta, dst_meta)


def convert_to_lance(
    repo_id: str,
    output: str | Path,
    *,
    src_root: str | Path | None = None,
    revision: str | None = None,
    table_name: str | None = None,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
    tolerance_s: float = 1e-4,
    overwrite: bool = False,
    progress: bool = True,
    push_to_hub: str | None = None,
) -> Path:
    """Convert an existing LeRobot dataset to a single Lance table."""
    import lancedb

    output = Path(output)
    if table_name is None:
        table_name = repo_id.split("/")[-1]

    # ``return_uint8`` is a recent addition to LeRobotDataset; we only pass it
    # when the installed version accepts it. Older versions return float32
    # frames anyway, which the writer's ``_encode_jpeg`` handles transparently.
    import inspect

    src_kwargs: dict = {"root": src_root, "revision": revision, "tolerance_s": tolerance_s}
    if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
        src_kwargs["return_uint8"] = True
    src = LeRobotDataset(repo_id, **src_kwargs)
    meta = src.meta

    image_keys = set(meta.image_keys)
    video_keys = set(meta.video_keys)
    has_subtasks = "subtask_index" in meta.features and meta.subtasks is not None

    schema, tabular_dims = _build_schema(meta.features, has_subtasks)

    output.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(output))
    if table_name in db.list_tables().tables:
        if not overwrite:
            raise FileExistsError(
                f"Lance table '{table_name}' already exists at '{output}'. "
                "Pass overwrite=True (or --overwrite) to replace it."
            )
        db.drop_table(table_name)

    n_eps = meta.total_episodes

    def _episode_batches() -> Iterable[pa.RecordBatch]:
        for ep_idx in range(n_eps):
            if progress:
                logger.info("Converting episode %d / %d", ep_idx + 1, n_eps)
            yield _episode_record_batch(
                src,
                ep_idx,
                schema,
                tabular_dims,
                image_keys,
                video_keys,
                jpeg_quality,
                tolerance_s,
            )

    reader = pa.RecordBatchReader.from_batches(schema, _episode_batches())
    db.create_table(table_name, data=reader, schema=schema)

    _copy_metadata(meta.root, output)
    logger.info("Lance conversion complete: %s/%s.lance", output, table_name)

    if push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(push_to_hub, repo_type="dataset", exist_ok=True)
        logger.info("Uploading %s → hf://datasets/%s", output, push_to_hub)
        api.upload_large_folder(
            repo_id=push_to_hub,
            folder_path=str(output),
            repo_type="dataset",
        )

    return output


__all__ = ["convert_to_lance"]
