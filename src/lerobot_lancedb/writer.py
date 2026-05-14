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


def _decode_video_frames_compat(video_path, timestamps, tolerance_s, backend):
    """``decode_video_frames`` adds kwargs across lerobot versions; pass them
    conditionally so we work against 0.5.x as well as newer releases.
    """
    import inspect

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
    lossless: bool,
    chroma_subsampling: int = 2,
) -> bytes:
    """Encode an image (CHW or HWC, uint8 or float [0,1]) as JPEG or PNG bytes.

    Tradeoff space (measured on lerobot/pusht and lerobot/aloha_static_cups_open;
    see README's "End-to-end training parity" section for full numbers):

    * ``lossless=True`` → PNG: bit-exact pixels, larger files, no NVJPEG GPU
      decode (falls back to PIL on CPU). The only safe choice if your data is
      synthetic / has hard edges (``lerobot/pusht``) or if upstream stored it
      bit-exact (``lerobot/pusht_image`` and other ``dtype=image`` datasets).

    * ``lossless=False, jpeg_quality=100, chroma_subsampling=0`` → "best JPEG":
      4:4:4 chroma (no subsampling) + max quality. Nearly lossless on natural
      images, ~5-10× smaller than PNG, full NVJPEG speed. Not bit-exact —
      JPEG's DCT quantization still introduces small artifacts at hard edges.

    * ``lossless=False, jpeg_quality=95, chroma_subsampling=2`` → default:
      4:2:0 chroma (half-resolution Cb/Cr) + q=95. Smallest, fastest, lossy
      enough to measurably hurt training accuracy on synthetic content
      (10pp env-success drop on pusht) and ~17% RMSE penalty on ALOHA-class
      multi-camera data.
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
    if lossless:
        Image.fromarray(arr).save(buf, format="PNG", optimize=False)
    else:
        Image.fromarray(arr).save(
            buf,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=chroma_subsampling,
        )
    return buf.getvalue()


def _encode_jpeg(frame: torch.Tensor | np.ndarray, jpeg_quality: int) -> bytes:
    """Back-compat alias — equivalent to ``_encode_image(.., lossless=False)``."""
    return _encode_image(frame, jpeg_quality, lossless=False)


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
    lossless: bool,
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
            blobs = [
                _encode_image(frames[i], jpeg_quality, lossless, chroma_subsampling)
                for i in range(ep_len)
            ]
            arrays[lance_key] = pa.array(blobs, type=pa.binary())
        elif key in image_keys:
            blobs = []
            col = rows[key]
            for i in range(ep_len):
                v = col[i] if isinstance(col, list) else col[i]
                if isinstance(v, Image.Image):
                    buf = io.BytesIO()
                    if lossless:
                        v.convert("RGB").save(buf, format="PNG", optimize=False)
                    else:
                        v.convert("RGB").save(
                            buf,
                            format="JPEG",
                            quality=jpeg_quality,
                            subsampling=chroma_subsampling,
                        )
                    blobs.append(buf.getvalue())
                else:
                    blobs.append(_encode_image(v, jpeg_quality, lossless, chroma_subsampling))
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
    lossless: bool = False,
    tolerance_s: float = 1e-4,
    overwrite: bool = False,
    progress: bool = True,
    push_to_hub: str | None = None,
) -> Path:
    """Convert an existing LeRobot dataset to a single Lance table.

    ``lossless=True`` stores frames as PNG (bit-exact) instead of JPEG. Use
    this for synthetic / hard-edged content (sim renders, sparse indicators,
    UI overlays, ``lerobot/pusht`` etc.) where JPEG ringing around edges
    measurably degrades downstream policy accuracy. The reader handles PNG
    transparently via PIL; only NVJPEG GPU decode is skipped for PNG bytes.
    For natural camera footage (ALOHA-class datasets), the JPEG default is
    fine — see the README's parity section for measurements.
    """
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
                lossless,
                chroma_subsampling,
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
    The default :func:`convert_to_lance` re-encodes each video frame as JPEG
    (or PNG with ``lossless=True``). That gives random-access frame reads
    but introduces a second lossy step on top of upstream's AV1 video
    (measured ~10pp env-success drop on pusht and ~17% held-out RMSE penalty
    on ALOHA; see the README parity section). When you want bit-exact pixels
    AND the speed benefits of Lance's storage layer, the alternative is to
    keep the original mp4 bytes verbatim and decode on the fly — exactly
    what upstream does, except Lance's blob v2 encoding handles the storage:

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
    import lancedb

    output = Path(output)
    if table_name is None:
        table_name = repo_id.split("/")[-1]

    import inspect

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
            f"(got image_keys={image_keys}). Use convert_to_lance(..., lossless=True) "
            "for image-stored datasets."
        )
    if not video_keys:
        raise ValueError("No video features found in source dataset — nothing to convert.")

    has_subtasks = "subtask_index" in meta.features and meta.subtasks is not None
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
