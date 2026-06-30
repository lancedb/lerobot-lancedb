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

import inspect
import io
import logging
import os
import queue
import shutil
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import torch
from huggingface_hub import HfApi
from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import DEFAULT_FEATURES, validate_episode_buffer, validate_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames
from PIL import Image

logger = logging.getLogger(__name__)


_DEFAULT_JPEG_QUALITY = 95


def _default_encode_workers() -> int:
    """Heuristic worker count for the image-encode thread pool.

    PIL's libjpeg-turbo / libpng paths release the GIL on encode, so threads
    do scale. Cap at the smaller of (CPU count, 8) — the writer iterates
    one episode at a time and additional threads beyond that gain little.
    """
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 8))


_ENCODE_WORKERS = _default_encode_workers()


def _decode_video_frames_compat(video_path, timestamps, tolerance_s, backend):
    """``decode_video_frames`` adds kwargs across lerobot versions; pass them
    conditionally so we work against 0.5.x as well as newer releases.
    """
    sig = inspect.signature(decode_video_frames)
    kwargs = {}
    if "return_uint8" in sig.parameters:
        kwargs["return_uint8"] = True
    return decode_video_frames(video_path, timestamps, tolerance_s, backend, **kwargs)


def _to_lance_name(name: str) -> str:
    return name.replace(".", "_")


def _encode_image(
    frame: torch.Tensor | np.ndarray,
    jpeg_quality: int,
    chroma_subsampling: int = 2,
) -> bytes:
    """Encode an image (CHW or HWC, uint8 or float [0,1]) as JPEG bytes.

    Tradeoff space (measured on lerobot/pusht, lerobot/aloha_static_cups_open,
    lerobot/koch_pick_place_5_lego; see the README for full numbers):

    * ``jpeg_quality=100, chroma_subsampling=0`` → "best JPEG": 4:4:4 chroma
      (no subsampling) + max quality. Nearly lossless on natural images,
      full NVJPEG speed. Not bit-exact — JPEG's DCT quantization still
      introduces small artifacts at hard edges.

    * ``jpeg_quality=95, chroma_subsampling=2`` (defaults): 4:2:0 chroma
      (half-resolution Cb/Cr) + q=95. Smallest, fastest, lossy enough to
      measurably hurt training accuracy on synthetic / hard-edged content
      (10pp env-success drop on pusht) and natural multi-camera data (~17%
      held-out RMSE penalty on ALOHA-class).

    For bit-exact storage of ``dtype=video`` sources, use
    :func:`convert_to_lance_video` (mp4 bytes verbatim via Lance blob v2).
    """
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
    Image.fromarray(arr).save(
        buf,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=chroma_subsampling,
    )
    return buf.getvalue()


def _encode_jpeg(frame: torch.Tensor | np.ndarray, jpeg_quality: int) -> bytes:
    """Back-compat alias — equivalent to ``_encode_image(jpeg_quality)``."""
    return _encode_image(frame, jpeg_quality)


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


def _to_numpy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _frame_to_chw_uint8(frame: Any) -> np.ndarray:
    """Normalize a LeRobot image/video frame value to CHW uint8."""
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    else:
        arr = np.asarray(_to_numpy_value(frame))

    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = (arr.clip(0, 1) * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        chw = arr
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        chw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expected CHW or HWC image frame, got shape {arr.shape}.")

    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)
    elif chw.shape[0] == 4:
        chw = chw[:3]
    return chw


def _compute_episode_stats_in_memory(
    episode_data: dict[str, Any],
    features: dict[str, dict],
    quantile_list: list[float] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute LeRobot-compatible stats without materializing images on disk."""
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    ep_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, data in episode_data.items():
        if key not in features or features[key]["dtype"] in {"string", "language"}:
            continue

        if features[key]["dtype"] in {"image", "video"}:
            frames = [_frame_to_chw_uint8(frame) for frame in data]
            ep_ft_array = np.stack(frames)
            stats = get_feature_stats(
                ep_ft_array,
                axis=(0, 2, 3),
                keepdims=True,
                quantile_list=quantile_list,
            )
            ep_stats[key] = {
                stat_key: stat_value if stat_key == "count" else np.squeeze(stat_value / 255.0, axis=0)
                for stat_key, stat_value in stats.items()
            }
        else:
            ep_ft_array = np.asarray(data)
            ep_stats[key] = get_feature_stats(
                ep_ft_array,
                axis=0,
                keepdims=ep_ft_array.ndim == 1,
                quantile_list=quantile_list,
            )

    return ep_stats


def _episode_buffer_record_batch(
    meta: LeRobotDatasetMetadata,
    episode_buffer: dict[str, Any],
    schema: pa.Schema,
    tabular_dims: dict[str, int],
    image_keys: set[str],
    video_keys: set[str],
    jpeg_quality: int,
    chroma_subsampling: int,
) -> pa.RecordBatch:
    ep_len = int(len(episode_buffer["index"]))
    arrays: dict[str, pa.Array] = {}

    def _to_numpy_scalar_column(values, dtype) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(dtype)
        if isinstance(values, list) and values and isinstance(values[0], torch.Tensor):
            return torch.stack(values).detach().cpu().numpy().astype(dtype)
        return np.asarray(values, dtype=dtype)

    arrays["episode_index"] = pa.array(
        _to_numpy_scalar_column(episode_buffer["episode_index"], np.int32), type=pa.int32()
    )
    arrays["frame_index"] = pa.array(
        _to_numpy_scalar_column(episode_buffer["frame_index"], np.int32), type=pa.int32()
    )
    arrays["index"] = pa.array(
        _to_numpy_scalar_column(episode_buffer["index"], np.int64), type=pa.int64()
    )
    arrays["timestamp"] = pa.array(
        _to_numpy_scalar_column(episode_buffer["timestamp"], np.float32), type=pa.float32()
    )
    arrays["task_index"] = pa.array(
        _to_numpy_scalar_column(episode_buffer["task_index"], np.int32), type=pa.int32()
    )
    if "subtask_index" in [f.name for f in schema]:
        sub_idx = episode_buffer.get("subtask_index", np.zeros(ep_len, dtype=np.int32))
        arrays["subtask_index"] = pa.array(
            _to_numpy_scalar_column(sub_idx, np.int32), type=pa.int32()
        )

    for key, _ft in meta.features.items():
        lance_key = _to_lance_name(key)
        if lance_key in arrays:
            continue
        if key in video_keys or key in image_keys:
            col = episode_buffer[key]

            def _enc_i(i: int, col=col):
                v = col[i] if isinstance(col, list) else col[i]
                if isinstance(v, Image.Image):
                    buf = io.BytesIO()
                    v.convert("RGB").save(
                        buf,
                        format="JPEG",
                        quality=jpeg_quality,
                        subsampling=chroma_subsampling,
                    )
                    return buf.getvalue()
                return _encode_image(v, jpeg_quality, chroma_subsampling)

            with ThreadPoolExecutor(max_workers=_ENCODE_WORKERS) as pool:
                blobs = list(pool.map(_enc_i, range(ep_len)))
            arrays[lance_key] = pa.array(blobs, type=pa.binary())
        else:
            dim = tabular_dims[lance_key]
            col = episode_buffer[key]
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


class _StreamingLanceTableWriter:
    """Append RecordBatches to an existing LanceDB table with one reader."""

    _STOP = object()

    def __init__(self, table: Any, schema: pa.Schema, max_pending_batches: int = 2) -> None:
        if max_pending_batches < 1:
            raise ValueError("max_pending_batches must be >= 1.")
        self._table = table
        self._schema = schema
        self._queue: queue.Queue[pa.RecordBatch | object] = queue.Queue(maxsize=max_pending_batches)
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="lerobot-lancedb-writer",
            daemon=True,
        )
        self._thread.start()

    def _iter_batches(self) -> Iterable[pa.RecordBatch]:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                break
            yield item

    def _run(self) -> None:
        try:
            reader = pa.RecordBatchReader.from_batches(self._schema, self._iter_batches())
            self._table.add(reader, mode="append")
        except BaseException as exc:
            with self._error_lock:
                self._error = exc

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Lance table writer failed.") from error

    def _put(self, item: pa.RecordBatch | object) -> None:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def write(self, batch: pa.RecordBatch) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed Lance table writer.")
        self._put(batch)
        self._raise_if_failed()

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._put(self._STOP)
        self._thread.join()
        self._raise_if_failed()


class LanceFramesWriter:
    """Write-mode companion for :class:`LeRobotLanceDataset` frames layout."""

    def __init__(
        self,
        *,
        meta: LeRobotDatasetMetadata,
        root: Path,
        table_name: str,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        chroma_subsampling: int = 2,
    ) -> None:
        self._meta = meta
        self._root = root
        self._table_name = table_name
        self._jpeg_quality = jpeg_quality
        self._chroma_subsampling = chroma_subsampling
        self._image_keys = set(meta.image_keys)
        self._video_keys = set(meta.video_keys)
        self._schema, self._tabular_dims = _build_schema(
            meta.features,
            has_subtasks="subtask_index" in meta.features and meta.subtasks is not None,
        )
        self.episode_buffer: dict = self._create_episode_buffer()
        self._db = None
        self._table = None
        self._table_writer: _StreamingLanceTableWriter | None = None
        self._finalized = False

    def _create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self._meta.total_episodes if episode_index is None else episode_index
        ep_buffer: dict[str, Any] = {"size": 0, "task": []}
        for key in self._meta.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def add_frame(self, frame: dict) -> None:
        frame = {key: _to_numpy_value(value) for key, value in frame.items()}
        validate_frame(frame, self._meta.features)

        frame_index = self.episode_buffer["size"]
        timestamp = frame_index / self._meta.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame["task"])

        for key, value in frame.items():
            if key == "task":
                continue
            if key not in self._meta.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in "
                    f"'{self._meta.features.keys()}'."
                )
            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def _looks_like_episode_buffer(self, episode_data: dict[str, Any]) -> bool:
        return "size" in episode_data and "episode_index" in episode_data

    def _episode_length_from_columns(self, episode_data: dict[str, Any]) -> int:
        lengths = []
        for key, value in episode_data.items():
            if key in DEFAULT_FEATURES:
                continue
            if key == "task":
                if isinstance(value, str):
                    continue
                lengths.append(len(value))
                continue
            if key not in self._meta.features:
                continue
            value = _to_numpy_value(value)
            if isinstance(value, Image.Image):
                continue
            if isinstance(value, np.ndarray):
                feature_shape = tuple(self._meta.features[key].get("shape", ()))
                lengths.append(1 if value.shape == feature_shape else len(value))
            else:
                lengths.append(len(value))

        if not lengths:
            raise ValueError("Could not infer episode length from episode_data.")
        if len(set(lengths)) != 1:
            raise ValueError(f"episode_data columns have inconsistent lengths: {lengths}")
        return int(lengths[0])

    def _split_column(self, key: str, value: Any, length: int) -> list[Any]:
        value = _to_numpy_value(value)
        if isinstance(value, Image.Image):
            if length != 1:
                raise ValueError(f"Feature '{key}' provides one image but episode length is {length}.")
            return [value]
        if isinstance(value, np.ndarray):
            feature_shape = tuple(self._meta.features[key].get("shape", ()))
            if value.shape == feature_shape:
                if length != 1:
                    raise ValueError(
                        f"Feature '{key}' provides one value with shape {value.shape} "
                        f"but episode length is {length}."
                    )
                return [value]
            if len(value) != length:
                raise ValueError(
                    f"Feature '{key}' length {len(value)} does not match episode length {length}."
                )
            return [value[i] for i in range(length)]
        if len(value) != length:
            raise ValueError(f"Feature '{key}' length {len(value)} does not match episode length {length}.")
        return [_to_numpy_value(v) for v in value]

    def _normalize_episode_data(self, episode_data: dict[str, Any]) -> dict:
        if self._looks_like_episode_buffer(episode_data):
            return {
                key: list(value) if isinstance(value, list) else value
                for key, value in episode_data.items()
            }

        length = self._episode_length_from_columns(episode_data)
        tasks_raw = episode_data.get("task")
        if tasks_raw is None:
            raise ValueError("episode_data must include a 'task' entry.")
        tasks = [tasks_raw] * length if isinstance(tasks_raw, str) else list(tasks_raw)
        if len(tasks) != length:
            raise ValueError(f"task length {len(tasks)} does not match episode length {length}.")

        columns = {
            key: self._split_column(key, value, length)
            for key, value in episode_data.items()
            if key != "task" and key not in DEFAULT_FEATURES
        }
        ep_buffer = self._create_episode_buffer()
        for i in range(length):
            frame = {key: values[i] for key, values in columns.items()}
            frame["task"] = tasks[i]
            frame = {key: _to_numpy_value(value) for key, value in frame.items()}
            validate_frame(frame, self._meta.features)
            ep_buffer["frame_index"].append(i)
            ep_buffer["timestamp"].append(i / self._meta.fps)
            ep_buffer["task"].append(frame["task"])
            for key, value in frame.items():
                if key != "task":
                    ep_buffer[key].append(value)
            ep_buffer["size"] += 1
        return ep_buffer

    def _ensure_table_writer(self) -> _StreamingLanceTableWriter:
        if self._db is None:
            self._db = lancedb.connect(str(self._root))
        if self._table is None:
            names = list(self._db.list_tables().tables)
            if self._table_name in names:
                raise FileExistsError(
                    f"Lance table '{self._table_name}' already exists at '{self._root}'."
                )
            self._table = self._db.create_table(self._table_name, data=None, schema=self._schema)
        if self._table_writer is None:
            self._table_writer = _StreamingLanceTableWriter(self._table, self._schema)
        return self._table_writer

    def _write_record_batch(self, batch: pa.RecordBatch) -> None:
        self._ensure_table_writer().write(batch)

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        del parallel_encoding
        episode_buffer = (
            self._normalize_episode_data(episode_data)
            if episode_data is not None
            else self.episode_buffer
        )
        validate_episode_buffer(episode_buffer, self._meta.total_episodes, self._meta.features)

        episode_length = int(episode_buffer.pop("size"))
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = int(episode_buffer["episode_index"])

        episode_buffer["index"] = np.arange(
            self._meta.total_frames, self._meta.total_frames + episode_length, dtype=np.int64
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index, dtype=np.int64)

        self._meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array(
            [self._meta.get_task_index(task) for task in tasks], dtype=np.int64
        )

        for key, ft in self._meta.features.items():
            if key in {"index", "episode_index", "task_index"} or ft["dtype"] in {"image", "video"}:
                continue
            stacked_values = np.stack(episode_buffer[key])
            if tuple(ft["shape"]) == (1,) and ft["dtype"] != "string":
                stacked_values = stacked_values.reshape(episode_length)
            episode_buffer[key] = stacked_values

        ep_stats = _compute_episode_stats_in_memory(episode_buffer, self._meta.features)
        batch = _episode_buffer_record_batch(
            self._meta,
            episode_buffer,
            self._schema,
            self._tabular_dims,
            self._image_keys,
            self._video_keys,
            self._jpeg_quality,
            self._chroma_subsampling,
        )
        self._write_record_batch(batch)

        # LeRobot's metadata writer builds parquet schemas from dict insertion
        # order. Seed the frame range keys so every metadata buffer flush keeps
        # the same schema order; the metadata layer still owns the final values.
        episode_metadata = {
            "dataset_from_index": int(self._meta.total_frames),
            "dataset_to_index": int(self._meta.total_frames) + episode_length,
        }
        self._meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, episode_metadata)

        if episode_data is None or self.episode_buffer["size"] == 0:
            self.clear_episode_buffer()

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        del delete_images
        self.episode_buffer = self._create_episode_buffer()

    def finalize(self) -> None:
        if self._finalized:
            return
        if self._table_writer is not None:
            self._table_writer.close()
        self._meta.finalize()
        self._finalized = True


def _episode_record_batch(
    src: LeRobotDataset,
    ep_idx: int,
    schema: pa.Schema,
    tabular_dims: dict[str, int],
    image_keys: set[str],
    video_keys: set[str],
    jpeg_quality: int,
    tolerance_s: float,
    chroma_subsampling: int,
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
            frames = _decode_video_frames_compat(
                video_path, shifted, tolerance_s, src._video_backend
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

            def _enc_v(i: int, frames=frames):
                return _encode_image(frames[i], jpeg_quality, chroma_subsampling)

            with ThreadPoolExecutor(max_workers=_ENCODE_WORKERS) as pool:
                blobs = list(pool.map(_enc_v, range(ep_len)))
            arrays[lance_key] = pa.array(blobs, type=pa.binary())
        elif key in image_keys:
            col = rows[key]

            def _enc_i(i: int, col=col):
                v = col[i] if isinstance(col, list) else col[i]
                if isinstance(v, Image.Image):
                    buf = io.BytesIO()
                    v.convert("RGB").save(
                        buf,
                        format="JPEG",
                        quality=jpeg_quality,
                        subsampling=chroma_subsampling,
                    )
                    return buf.getvalue()
                return _encode_image(v, jpeg_quality, chroma_subsampling)

            with ThreadPoolExecutor(max_workers=_ENCODE_WORKERS) as pool:
                blobs = list(pool.map(_enc_i, range(ep_len)))
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
    chroma_subsampling: int = 2,
    tolerance_s: float = 1e-4,
    overwrite: bool = False,
    progress: bool = True,
    push_to_hub: str | None = None,
) -> Path:
    """Convert an existing LeRobot dataset to a single Lance table.

    Frames are JPEG-encoded (per-row). Two knobs control the quality /
    size / fidelity tradeoff:

    * ``jpeg_quality`` (default 95) — bigger numbers, bigger files, fewer
      artifacts. 100 + ``chroma_subsampling=0`` gets you near-lossless
      JPEG while keeping the NVJPEG decode path.
    * ``chroma_subsampling`` (default 2 = 4:2:0) — 0 = 4:4:4 (no
      subsampling, max color fidelity); 1 = 4:2:2; 2 = 4:2:0 (half-res
      chroma, the default).

    For bit-exact storage of ``dtype=video`` sources, prefer
    :func:`convert_to_lance_video` — it copies the source mp4 bytes
    verbatim using Lance blob v2 and decodes on the fly with torchcodec.
    """
    output = Path(output)
    if table_name is None:
        table_name = repo_id.split("/")[-1]

    # ``return_uint8`` is a recent addition to LeRobotDataset; we only pass it
    # when the installed version accepts it. Older versions return float32
    # frames anyway, which the writer's ``_encode_jpeg`` handles transparently.

    src_kwargs: dict = {"root": src_root, "revision": revision, "tolerance_s": tolerance_s}
    if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
        src_kwargs["return_uint8"] = True
    src = LeRobotDataset(repo_id, **src_kwargs)
    meta = src.meta

    image_keys = set(meta.image_keys)
    video_keys = set(meta.video_keys)
    has_subtasks = "subtask_index" in meta.features and getattr(meta, "subtasks", None) is not None

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
                chroma_subsampling,
            )

    reader = pa.RecordBatchReader.from_batches(schema, _episode_batches())
    db.create_table(table_name, data=reader, schema=schema)

    _copy_metadata(meta.root, output)
    logger.info("Lance conversion complete: %s/%s.lance", output, table_name)

    if push_to_hub:
        api = HfApi()
        api.create_repo(push_to_hub, repo_type="dataset", exist_ok=True)
        logger.info("Uploading %s → hf://datasets/%s", output, push_to_hub)
        api.upload_large_folder(
            repo_id=push_to_hub,
            folder_path=str(output),
            repo_type="dataset",
        )

    return output


def _video_schema() -> pa.Schema:
    """Schema for the video-blob table.

    LeRobot's video layout stores many episodes per mp4 file, and different
    cameras may use *different* ``(chunk, file)`` indexing for the same
    episode — laptop and phone on Koch datasets diverge after a few episodes,
    for example. So we identify each row by ``(video_key, chunk_index,
    file_index)`` (no pivoting on the camera key), keeping a single
    ``video_bytes`` blob column. The reader joins on
    ``meta.episodes[ep_idx]["videos/{cam}/{chunk,file}_index"]`` to pick the
    right row, then shifts query timestamps by
    ``videos/{cam}/from_timestamp`` into the shared mp4's timeline.
    """
    blob_meta = {b"lance-encoding:blob": b"true"}
    return pa.schema([
        pa.field("video_key", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("file_index", pa.int32()),
        pa.field("video_bytes", pa.large_binary(), metadata=blob_meta),
    ])


def _frames_schema_no_images(features: dict[str, dict], has_subtasks: bool) -> tuple[pa.Schema, dict[str, int]]:
    """Like ``_build_schema`` but skips image/video columns (videos go in a
    sibling table)."""
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
        if lance_key in reserved or ft["dtype"] in ("image", "video"):
            continue
        shape = tuple(ft.get("shape", (1,))) or (1,)
        dim = int(np.prod(shape))
        dims[lance_key] = dim
        fields.append(pa.field(lance_key, pa.list_(pa.float32(), dim)))
    return pa.schema(fields), dims


def _frames_episode_batch_no_images(
    src: LeRobotDataset,
    ep_idx: int,
    schema: pa.Schema,
    tabular_dims: dict[str, int],
) -> pa.RecordBatch:
    """Per-frame batch for the frames table; image/video columns are omitted
    (those live in the sibling video-blob table). Adapted from
    ``_episode_record_batch`` but without the decode/encode of image data."""
    meta = src.meta
    ep = meta.episodes[ep_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start

    hf = src.hf_dataset
    if src.reader._absolute_to_relative_idx is None:
        rel_indices = list(range(ep_start, ep_end))
    else:
        rel_indices = [src.reader._absolute_to_relative_idx[i] for i in range(ep_start, ep_end)]
    rows = hf[rel_indices]

    arrays: dict[str, pa.Array] = {}
    arrays["episode_index"] = pa.array(np.full(ep_len, ep_idx, dtype=np.int32), type=pa.int32())
    arrays["frame_index"] = pa.array(np.arange(ep_len, dtype=np.int32), type=pa.int32())
    arrays["index"] = pa.array(np.arange(ep_start, ep_end, dtype=np.int64), type=pa.int64())

    def _np_col(values, dtype):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(dtype)
        if isinstance(values, list) and values and isinstance(values[0], torch.Tensor):
            return torch.stack(values).detach().cpu().numpy().astype(dtype)
        return np.asarray(values, dtype=dtype)

    arrays["timestamp"] = pa.array(_np_col(rows["timestamp"], np.float32), type=pa.float32())
    arrays["task_index"] = pa.array(_np_col(rows["task_index"], np.int32), type=pa.int32())
    if "subtask_index" in [f.name for f in schema]:
        sub = rows.get("subtask_index", np.zeros(ep_len, dtype=np.int32))
        arrays["subtask_index"] = pa.array(_np_col(sub, np.int32), type=pa.int32())

    image_or_video = set(meta.image_keys) | set(meta.video_keys)
    for key in meta.features:
        if key in image_or_video:
            continue
        lance_key = _to_lance_name(key)
        if lance_key in arrays:
            continue
        if lance_key not in tabular_dims:
            continue
        dim = tabular_dims[lance_key]
        col = rows[key]
        if isinstance(col, torch.Tensor):
            flat = col.detach().cpu().numpy().astype(np.float32).reshape(ep_len, dim).reshape(-1)
        elif isinstance(col, list) and col and isinstance(col[0], torch.Tensor):
            flat = (
                torch.stack(col).detach().cpu().numpy().astype(np.float32)
                .reshape(ep_len, dim).reshape(-1)
            )
        else:
            flat = np.asarray(col, dtype=np.float32).reshape(ep_len, dim).reshape(-1)
        arrays[lance_key] = pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), dim
        )

    ordered = [arrays[f.name] for f in schema]
    return pa.RecordBatch.from_arrays(ordered, schema=schema)


def _collect_unique_video_files(
    src: LeRobotDataset, video_keys: list[str]
) -> list[tuple[str, int, int, int]]:
    """Return sorted ``(video_key, chunk_index, file_index, representative_ep)``
    tuples — one per unique mp4 file across all cameras. Different cameras can
    use different ``(chunk, file)`` indexing for the same episode (Koch
    datasets do this), so the key includes the camera. The representative
    episode is any episode that maps to that ``(video_key, chunk, file)``
    triple — we use it to resolve the mp4 path via
    :py:meth:`LeRobotDatasetMetadata.get_video_file_path`."""
    meta = src.meta
    seen: dict[tuple[str, int, int], int] = {}
    for ep_idx in range(meta.total_episodes):
        ep = meta.episodes[ep_idx]
        for vid_key in video_keys:
            key = (
                vid_key,
                int(ep[f"videos/{vid_key}/chunk_index"]),
                int(ep[f"videos/{vid_key}/file_index"]),
            )
            seen.setdefault(key, ep_idx)
    return [(vk, c, f, ep) for (vk, c, f), ep in sorted(seen.items())]


def _video_file_batch(
    src: LeRobotDataset,
    video_key: str,
    chunk_index: int,
    file_index: int,
    representative_ep: int,
    schema: pa.Schema,
) -> pa.RecordBatch:
    """One row per (video_key, chunk_index, file_index)."""
    meta = src.meta
    video_path = src.root / meta.get_video_file_path(representative_ep, video_key)
    with open(video_path, "rb") as fh:
        blob = fh.read()
    arrays = {
        "video_key": pa.array([video_key], type=pa.string()),
        "chunk_index": pa.array([np.int32(chunk_index)], type=pa.int32()),
        "file_index": pa.array([np.int32(file_index)], type=pa.int32()),
        "video_bytes": pa.array([blob], type=pa.large_binary()),
    }
    ordered = [arrays[f.name] for f in schema]
    return pa.RecordBatch.from_arrays(ordered, schema=schema)


def convert_to_lance_video(
    repo_id: str,
    output: str | Path,
    *,
    src_root: str | Path | None = None,
    revision: str | None = None,
    table_name: str | None = None,
    tolerance_s: float = 1e-4,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Convert a parquet+mp4 LeRobotDataset to two Lance tables: per-frame
    tabular data + per-episode mp4 blobs (using Lance's blob v2 encoding).

    Why this exists
    ---------------
    The default :func:`convert_to_lance` re-encodes each video frame as
    JPEG. That gives random-access frame reads but introduces a second
    lossy step on top of upstream's AV1 video (measured ~10pp env-success
    drop on pusht and ~17% held-out RMSE penalty on ALOHA; see the README
    parity section). When you want bit-exact pixels AND the speed benefits
    of Lance's storage layer, the alternative is to keep the original mp4
    bytes verbatim and decode on the fly — exactly what upstream does,
    except Lance's blob v2 encoding handles the storage:

    * Bit-exact: blob bytes are byte-identical to the source mp4.
    * Streaming reads: blob columns aren't materialized into Arrow buffers;
      :py:meth:`lance.Dataset.take_blobs` returns file-like objects, and we
      hand the bytes straight to torchcodec.
    * Unified storage: one Lance dataset, no separate mp4 file tree.

    Output layout::

        <output>/
          <name>.lance/          # frames table (per-frame, no image columns)
          <name>_videos.lance/   # per-episode rows, one blob column per camera
          meta/                  # verbatim from upstream
    """
    output = Path(output)
    if table_name is None:
        table_name = repo_id.split("/")[-1]

    src_kwargs: dict = {"root": src_root, "revision": revision, "tolerance_s": tolerance_s}
    if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
        src_kwargs["return_uint8"] = True
    src = LeRobotDataset(repo_id, **src_kwargs)
    meta = src.meta

    video_keys = list(meta.video_keys)
    image_keys = list(meta.image_keys)
    if image_keys:
        raise NotImplementedError(
            f"convert_to_lance_video requires `dtype=video` features in the source "
            f"(got image_keys={image_keys}). Use convert_to_lance() for "
            "image-stored datasets."
        )
    if not video_keys:
        raise ValueError("No video features found in source dataset — nothing to convert.")

    has_subtasks = "subtask_index" in meta.features and getattr(meta, "subtasks", None) is not None
    frames_schema, tabular_dims = _frames_schema_no_images(meta.features, has_subtasks)
    videos_schema = _video_schema()

    output.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(output))
    frames_name = table_name
    videos_name = f"{table_name}_videos"
    existing = db.list_tables().tables
    for name in (frames_name, videos_name):
        if name in existing:
            if not overwrite:
                raise FileExistsError(
                    f"Lance table '{name}' already exists at '{output}'. Pass overwrite=True."
                )
            db.drop_table(name)

    n_eps = meta.total_episodes

    def _frames_batches() -> Iterable[pa.RecordBatch]:
        for ep_idx in range(n_eps):
            if progress:
                logger.info("frames: episode %d / %d", ep_idx + 1, n_eps)
            yield _frames_episode_batch_no_images(src, ep_idx, frames_schema, tabular_dims)

    unique_files = _collect_unique_video_files(src, video_keys)

    def _video_batches() -> Iterable[pa.RecordBatch]:
        for i, (vkey, chunk, fidx, rep_ep) in enumerate(unique_files):
            if progress:
                logger.info(
                    "videos: file %d / %d (%s chunk=%d file=%d)",
                    i + 1, len(unique_files), vkey, chunk, fidx,
                )
            yield _video_file_batch(src, vkey, chunk, fidx, rep_ep, videos_schema)

    db.create_table(
        frames_name,
        data=pa.RecordBatchReader.from_batches(frames_schema, _frames_batches()),
        schema=frames_schema,
    )
    db.create_table(
        videos_name,
        data=pa.RecordBatchReader.from_batches(videos_schema, _video_batches()),
        schema=videos_schema,
    )

    _copy_metadata(meta.root, output)
    logger.info("Lance video conversion complete: %s/{%s.lance, %s.lance}", output, frames_name, videos_name)
    return output


__all__ = ["convert_to_lance", "convert_to_lance_video"]
