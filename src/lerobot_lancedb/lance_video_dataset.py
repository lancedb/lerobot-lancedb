#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Video-blob Lance dataset: bit-exact pixels, decoded on the fly.

:class:`LeRobotLanceVideoDataset` reads a two-table Lance dataset produced by
:func:`lerobot_lancedb.writer.convert_to_lance_video`:

* ``<name>.lance`` — one row per frame, tabular data (state, action,
  timestamps, indices). No image data here.
* ``<name>_videos.lance`` — one row per episode, with the original mp4
  bytes stored verbatim as Lance blob v2 columns (one column per camera).

At read time we use :py:meth:`lance.Dataset.take_blobs` to stream the mp4
bytes (no full materialization into Arrow buffers) and torchcodec decodes
just the frames at the requested timestamps. The bytes are bit-identical
to the upstream mp4, so this avoids the JPEG-roundtrip artifact that the
default :class:`LeRobotLanceDataset` introduces (~10pp env-success drop
on pusht, ~17 % held-out RMSE penalty on ALOHA — see the README parity
section).

When to prefer this over :func:`convert_to_lance`:

* Your source dataset is video-stored (``dtype=video`` features — most
  lerobot datasets). Image datasets (``dtype=image``, e.g.
  ``lerobot/pusht_image``) don't have source mp4 files; use
  :func:`convert_to_lance` with a high ``jpeg_quality`` for those.
* You want bit-exact pixels *and* per-camera storage size identical to
  the upstream mp4 footprint.

Performance characteristics:

* Bit-exact pixels (per-decoder semantics; torchcodec or pyav).
* Random access requires seeking inside a per-episode mp4 — the same
  seek cost the upstream LeRobotDataset pays. The cached VideoDecoder
  amortizes this across consecutive accesses to the same episode.
* No NVJPEG GPU decode path. CPU decode via torchcodec.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lance
import lancedb
import numpy as np
import pyarrow as pa
import torch
import torch.utils.data
from huggingface_hub import HfApi, snapshot_download
from lancedb.permutation import Permutation
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
from torchcodec.decoders import VideoDecoder

from ._spawn_compat import force_spawn_for_lance

try:
    from huggingface_hub import get_token as _hf_get_token
except ImportError:
    _hf_get_token = None


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="The given buffer is not writable", category=UserWarning)


_RESERVED_KEYS = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index", "subtask_index"}
)


def _to_lance_name(name: str) -> str:
    return name.replace(".", "_")


class _VideoDecoderCache:
    """Per-worker cache: (chunk_index, file_index, video_key) -> torchcodec.VideoDecoder.

    LeRobot mp4 layout packs many episodes into a single mp4 (one file per
    ``(chunk_index, file_index)``), so the natural cache key is the file
    identity, not the episode. Capacity-bounded LRU; sized for the typical
    4-worker × O(few cameras × few files) workload.
    """

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self._cache: dict[tuple[int, int, str], Any] = {}
        self._order: list[tuple[int, int, str]] = []

    def get(self, key: tuple[int, int, str], blob_bytes: bytes):
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        decoder = VideoDecoder(blob_bytes, seek_mode="approximate")
        self._cache[key] = decoder
        self._order.append(key)
        while len(self._order) > self.capacity:
            drop = self._order.pop(0)
            self._cache.pop(drop, None)
        return decoder


class LeRobotLanceVideoDataset(LeRobotDataset):
    """Lance-backed dataset that stores videos as blobs and decodes on the fly.

    Subclasses :class:`LeRobotDataset` so existing trainers / samplers /
    ``isinstance`` checks accept it transparently. See module docstring for
    why and when to use this over the default JPEG-per-frame format.

    Args mirror :class:`LeRobotLanceDataset`: ``root`` for local, ``uri``
    for cloud, ``repo_id`` for HF Hub. ``decode_device`` is accepted for
    API parity but currently CPU-only (NVDEC integration is a TODO).
    """

    # Lance serves every frame by absolute dataset index; the absolute->relative
    # row remap only exists for episode-filtered parquet readers upstream.
    # Shadow the upstream property so trainers (e.g. lerobot-train's
    # EpisodeAwareSampler wiring) can read it without touching a parquet reader.
    absolute_to_relative_idx = None


    def __init__(
        self,
        repo_id: str | None = None,
        root: str | Path | None = None,
        *,
        uri: str | None = None,
        meta_root: str | Path | None = None,
        table_name: str | None = None,
        revision: str | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 1e-4,
        return_uint8: bool = False,
        connect_kwargs: dict[str, Any] | None = None,
        # Accepted for drop-in compatibility with lerobot's dataset factory
        # (`make_dataset` passes these); they have no effect on Lance reads.
        video_backend: str | None = None,
        download_videos: bool | None = None,
        force_cache_sync: bool | None = None,
        depth_output_unit: str | None = None,
        decoder_cache_size: int = 16,
    ) -> None:
        torch.utils.data.Dataset.__init__(self)

        if repo_id is not None and root is None and uri is None and Path(repo_id).is_dir():
            # positional local path (pre-0.2 call style): treat as root
            repo_id, root = None, repo_id

        if root is None and uri is None and repo_id is None:
            raise TypeError("LeRobotLanceVideoDataset requires one of `root`, `uri`, or `repo_id`.")

        # HF Hub path: download the whole dataset locally and proceed via
        # ``root``. lance's pyarrow API has no ``hf://`` provider for blob v2
        # reads (the ``_videos.lance`` table), so the hybrid "stream frames
        # + download videos" was misleading — we'd be downloading anyway.
        # Full download is more elegant and matches the actual cost.
        if (repo_id is not None or (uri is not None and uri.startswith("hf://"))) and root is None:
            if repo_id is None:
                repo_id, revision = self._parse_hf_uri(uri, revision)
                uri = None
            logger.warning(
                "LeRobotLanceVideoDataset over HF Hub will download the full dataset "
                "to local disk (no true streaming for the video-blob layout). "
                "For frame-by-frame streaming from HF, use LeRobotLanceDataset instead."
            )
            root = self._materialize_from_hub(repo_id, revision, table_name)

        self._uri, self._frames_name = self._resolve_uri_and_table(root, uri, table_name)
        self._videos_name = f"{self._frames_name}_videos"
        meta_root_resolved = (
            Path(meta_root) if meta_root is not None else (Path(root) if root is not None else None)
        )
        if meta_root_resolved is None:
            raise ValueError(
                "When using `uri=` for a remote Lance table, `meta_root=` must point to a local "
                "directory holding `meta/info.json` etc."
            )

        self.repo_id = repo_id or self._frames_name
        self._requested_root = meta_root_resolved
        self.root = meta_root_resolved
        self.revision = revision
        self.episodes = list(episodes) if episodes is not None else None
        self.tolerance_s = tolerance_s
        self.delta_timestamps = delta_timestamps
        self.set_image_transforms(image_transforms)
        self._return_uint8 = return_uint8

        # video_backend kept for repr compatibility; we ignore it (always torchcodec on bytes).
        self._video_backend = "torchcodec-blob"
        self._batch_encoding_size = 1
        self._vcodec = None
        self._encoder_threads = None
        self.reader = None
        self.writer = None
        self._is_finalized = True

        self._connect_kwargs = self._auto_connect_kwargs(self._uri, connect_kwargs)
        force_spawn_for_lance()

        self.meta = self._load_metadata(self.repo_id, meta_root_resolved)

        # Probe both lance tables at init (see LeRobotLanceDataset for rationale).
        self._probe_tables_exist()

        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        self._original_keys = [k for k in self.meta.features.keys() if k not in _RESERVED_KEYS]
        self._lance_to_dot = {_to_lance_name(k): k for k in self._original_keys}
        self._video_keys = set(self.meta.video_keys)
        if set(self.meta.image_keys):
            raise NotImplementedError(
                "LeRobotLanceVideoDataset requires `dtype=video` source features. "
                "For `dtype=image` datasets (e.g. lerobot/pusht_image), use "
                "LeRobotLanceDataset with a high --jpeg-quality at conversion time."
            )
        if not self._video_keys:
            raise ValueError("No video features in dataset metadata — nothing to decode.")

        sorted_eps = sorted(self._extract_episode_bounds(self.meta).items(), key=lambda kv: kv[1][0])
        if sorted_eps:
            self._ep_starts = np.array([v[0] for _, v in sorted_eps], dtype=np.int64)
            self._ep_ends = np.array([v[1] for _, v in sorted_eps], dtype=np.int64)
            self._ep_ids = np.array([k for k, _ in sorted_eps], dtype=np.int64)
        else:
            raise ValueError("Dataset metadata has no episodes — cannot build episode bounds.")

        # Per-(episode, video_key) → (chunk_index, file_index, from_timestamp).
        # Multiple episodes can map to the same (chunk, file); the from_timestamp
        # tells us where each episode lives within that shared mp4's timeline.
        self._ep_video_file: dict[tuple[int, str], tuple[int, int, float]] = {}
        for ep_idx in range(self.meta.total_episodes):
            ep = self.meta.episodes[ep_idx]
            for vid_key in self._video_keys:
                self._ep_video_file[(ep_idx, vid_key)] = (
                    int(ep[f"videos/{vid_key}/chunk_index"]),
                    int(ep[f"videos/{vid_key}/file_index"]),
                    float(ep[f"videos/{vid_key}/from_timestamp"]),
                )

        # Lazy lance handles + decoder cache (rebuilt per worker).
        self._db = None
        self._frames_table = None
        self._videos_dataset = None
        self._frames_perm = None
        self._decoder_cache_size = decoder_cache_size
        self._decoder_cache: _VideoDecoderCache | None = None
        self._fetch_columns: list[str] | None = None

    # ── helpers reused from LeRobotLanceDataset (small enough to inline) ──

    @staticmethod
    def _parse_hf_uri(uri: str, revision: str | None) -> tuple[str, str | None]:
        """Split ``hf://datasets/<repo>[@rev]`` into ``(repo_id, revision)``."""
        rest = uri[len("hf://datasets/") :].rstrip("/")
        repo_id, _, embedded_rev = rest.partition("@")
        return repo_id, revision or (embedded_rev or None)

    @staticmethod
    def _materialize_from_hub(repo_id: str, revision: str | None, table_name: str | None) -> Path:
        """Download the entire dataset (``meta/`` + both ``*.lance`` tables) locally.

        Pre-flights by listing the repo's files: if no ``*.lance/`` directory
        exists on the Hub, fails immediately instead of paying a (potentially
        hundreds-of-MB) download just to error out later. lance can't read
        through HF's symlink cache layout, so we download via ``local_dir=``
        to a flat directory of real files.
        """
        api = HfApi()
        try:
            files = api.list_repo_files(repo_id, revision=revision, repo_type="dataset")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not list HF repo {repo_id!r}: {e}. "
                "Check the repo exists and credentials are set."
            ) from e
        has_lance = any("/" in f and f.split("/", 1)[0].endswith(".lance") for f in files)
        if not has_lance:
            raise FileNotFoundError(
                f"HF repo {repo_id!r} contains no '*.lance/' directory — doesn't "
                "look like a lerobot-lancedb dataset. Convert the source first with "
                "`lerobot-convert-to-lance` or `lerobot-convert-to-lance-video`."
            )
        local_dir = (
            Path(HF_LEROBOT_HUB_CACHE)
            / "lerobot-lancedb-datasets"
            / repo_id.replace("/", "--")
            / (revision or "main")
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=["meta/*", "*.lance/**"],
        )
        return local_dir

    @staticmethod
    def _resolve_uri_and_table(root, uri, table_name):
        if uri is not None:
            stripped = str(uri).rstrip("/")
            if stripped.lower().endswith(".lance"):
                sep = stripped.rfind("/")
                if sep < 0:
                    return ".", stripped[: -len(".lance")]
                return stripped[:sep], stripped[sep + 1 : -len(".lance")]
            if table_name is None:
                raise ValueError("`uri=` does not end in '.lance' and `table_name=` was not set.")
            return stripped, table_name

        root = Path(root)
        if root.name.lower().endswith(".lance"):
            return str(root.parent), root.stem
        if table_name is not None:
            return str(root), table_name
        # Prefer a *.lance whose name does NOT end in _videos (that's the frames table).
        candidates = [p for p in sorted(root.glob("*.lance")) if not p.stem.endswith("_videos")]
        if not candidates:
            raise FileNotFoundError(f"No frames '*.lance' table found under {root}.")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple '*.lance' tables under {root}: {[c.name for c in candidates]}. "
                "Pass `table_name=` explicitly."
            )
        return str(root), candidates[0].stem

    @staticmethod
    def _auto_connect_kwargs(uri, connect_kwargs):
        kw = dict(connect_kwargs or {})
        storage_options = dict(kw.get("storage_options") or {})
        if uri.startswith("s3://") and "region" not in storage_options:
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
            storage_options.setdefault("region", region)
            storage_options.setdefault("virtual_hosted_style_request", "true")
        if uri.startswith("hf://") and "token" not in storage_options:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token is None and _hf_get_token is not None:
                token = _hf_get_token()
            if token:
                storage_options["token"] = token
        if storage_options:
            kw["storage_options"] = storage_options
        return kw

    @staticmethod
    def _load_metadata(repo_id, meta_root):
        meta_root = Path(meta_root)
        if not (meta_root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Lance dataset at '{meta_root}' is missing 'meta/info.json'. "
                "Did you run convert_to_lance_video to produce it?"
            )
        return LeRobotDatasetMetadata(repo_id=repo_id, root=meta_root)

    @staticmethod
    def _extract_episode_bounds(meta):
        eps = meta.episodes
        if eps is None:
            return {}
        out: dict[int, tuple[int, int]] = {}
        for i in range(len(eps)):
            row = eps[i]
            ep_idx = int(row.get("episode_index", i))
            out[ep_idx] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        return out

    # ── lance handle management ───────────────────────────────────────

    def _probe_tables_exist(self) -> None:
        """Verify both lance tables exist at ``self._uri`` before lazy-open.

        Surfaces "this URI is not a lerobot-lancedb video dataset" at init
        time with a clear message, instead of leaking a lance HTTP 404
        from ``__getitem__`` much later.
        """
        try:
            db = lancedb.connect(self._uri, **self._connect_kwargs)
            names = list(db.list_tables().tables)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not list lance tables at {self._uri!r}: {e}. "
                "Check that the URI / repo exists and credentials are set."
            ) from e
        missing = [n for n in (self._frames_name, self._videos_name) if n not in names]
        if missing:
            raise FileNotFoundError(
                f"Missing lance table(s) {missing} at {self._uri!r} "
                f"(existing tables: {names}). This doesn't look like a "
                "lerobot-lancedb video dataset — convert the source with "
                "`lerobot-convert-to-lance-video` first."
            )

    def _ensure_open(self) -> None:
        if self._frames_perm is not None:
            return

        self._db = lancedb.connect(self._uri, **self._connect_kwargs)
        self._frames_table = self._db.open_table(self._frames_name)
        # ``take_blobs`` lives on ``lance.LanceDataset``. ``lance.dataset()``
        # honors storage_options for s3/gs/az. HF Hub URIs never reach here
        # (HF is fully materialized in __init__ via _materialize_from_hub).
        videos_uri = f"{self._uri}/{self._videos_name}.lance"
        storage_options = self._connect_kwargs.get("storage_options")
        self._videos_dataset = lance.dataset(videos_uri, storage_options=storage_options)

        # Build a (video_key, chunk_index, file_index) → row offset map for take_blobs lookup.
        # Different cameras can use different (chunk, file) indexing for the same episode
        # (e.g. Koch's laptop and phone cameras diverge after a few episodes), so the row
        # identity has to include the camera key, not just (chunk, file).
        chunks = self._videos_dataset.to_table(
            columns=["video_key", "chunk_index", "file_index"]
        ).to_pylist()
        self._file_row_index: dict[tuple[str, int, int], int] = {
            (str(r["video_key"]), int(r["chunk_index"]), int(r["file_index"])): i
            for i, r in enumerate(chunks)
        }

        all_cols = list(self._frames_table.schema.names)
        wanted_dot_keys = list(self._original_keys) + [
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            "task_index",
        ]
        if "subtask_index" in self.meta.features:
            wanted_dot_keys.append("subtask_index")
        wanted_lance_keys = [_to_lance_name(k) for k in wanted_dot_keys if k not in self._video_keys]
        self._fetch_columns = [c for c in wanted_lance_keys if c in all_cols]
        self._frames_perm = (
            Permutation.identity(self._frames_table)
            .select_columns(self._fetch_columns)
            .with_format("arrow")
        )

        # Per-worker decoder cache (one per dataset instance, reset on pickle).
        self._decoder_cache = _VideoDecoderCache(capacity=self._decoder_cache_size)

    # ── pickling for spawn-mode workers ───────────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_db"] = None
        state["_frames_table"] = None
        state["_videos_dataset"] = None
        state["_frames_perm"] = None
        state["_decoder_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ── public surface mirrored from LeRobotDataset ───────────────────

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def num_frames(self) -> int:
        return self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    def __len__(self) -> int:
        return self.num_frames

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({{\n"
            f"    Frames Lance: '{self._uri}/{self._frames_name}.lance',\n"
            f"    Videos Lance: '{self._uri}/{self._videos_name}.lance',\n"
            f"    Number of episodes: '{self.num_episodes}',\n"
            f"    Number of frames: '{self.num_frames}',\n"
            f"    Video keys: {sorted(self._video_keys)},\n"
            f"}})"
        )

    def set_image_transforms(self, image_transforms):
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms

    def clear_image_transforms(self):
        self.set_image_transforms(None)

    # ── delta-window helpers (same as LeRobotLanceDataset) ────────────

    def _episode_for_index(self, abs_idx: int) -> int:
        pos = int(np.searchsorted(self._ep_starts, abs_idx, side="right")) - 1
        if pos < 0 or abs_idx >= int(self._ep_ends[pos]):
            raise IndexError(f"Frame index {abs_idx} outside any episode bound.")
        return int(self._ep_ids[pos])

    def _build_query_indices(self, abs_idx: int, ep_idx: int):
        pos = int(np.searchsorted(self._ep_ids, ep_idx))
        ep_start = int(self._ep_starts[pos])
        ep_end = int(self._ep_ends[pos])
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in (self.delta_indices or {}).items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) or (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in (self.delta_indices or {}).items()
        }
        return query_indices, padding

    # ── core read path ────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        return self.__getitems__([int(idx)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        if not indices:
            return []
        self._ensure_open()

        # 1. Lay out per-sample row positions.
        all_rows: list[int] = []
        base_pos: list[int] = []
        delta_pos: list[dict[str, tuple[int, int]]] = []
        padding_per_sample: list[dict[str, torch.Tensor]] = []
        for abs_idx in indices:
            base_pos.append(len(all_rows))
            all_rows.append(int(abs_idx))
            sample_delta: dict[str, tuple[int, int]] = {}
            sample_pad: dict[str, torch.Tensor] = {}
            if self.delta_indices is not None:
                ep_idx = self._episode_for_index(int(abs_idx))
                qi, sample_pad = self._build_query_indices(int(abs_idx), ep_idx)
                for key, rows in qi.items():
                    start = len(all_rows)
                    all_rows.extend(rows)
                    sample_delta[key] = (start, len(rows))
            delta_pos.append(sample_delta)
            padding_per_sample.append(sample_pad)

        # 2. Dedupe + fetch frames-table rows.
        unique_rows = sorted(set(all_rows))
        unique_batch = self._frames_perm.__getitems__(unique_rows)
        if all_rows == unique_rows:
            big_batch = unique_batch
        else:
            row_lookup = {r: i for i, r in enumerate(unique_rows)}
            gather = pa.array([row_lookup[r] for r in all_rows], type=pa.int64())
            big_batch = unique_batch.take(gather)

        # 3. Extract tabular columns.
        np_columns: dict[str, np.ndarray] = {}
        for col_name in big_batch.schema.names:
            col = big_batch.column(col_name)
            ctype = col.type
            if pa.types.is_fixed_size_list(ctype):
                dim = ctype.list_size
                flat = np.array(col.flatten().to_numpy(zero_copy_only=False), copy=True)
                np_columns[col_name] = flat.reshape(len(col), dim)
            else:
                np_columns[col_name] = np.array(col.to_numpy(zero_copy_only=False), copy=True)

        ep_arr = np_columns["episode_index"]
        fi_arr = np_columns["frame_index"]
        idx_arr = np_columns["index"]
        ts_arr = np_columns["timestamp"]
        task_arr = np_columns["task_index"]
        subtask_arr = np_columns.get("subtask_index")

        # 4. Plan video decodes per (file_id, video_key). Episodes that share an
        # mp4 file get one take_blobs + one VideoDecoder; from_timestamp shifts
        # each sample's query timestamps into the shared file's timeline.
        # video_plan[(chunk, file, vid_key)] -> list of (sample_idx, shifted_timestamps)
        video_plan: dict[tuple[int, int, str], list[tuple[int, list[float]]]] = {}
        for s in range(len(indices)):
            bp = base_pos[s]
            ep_idx = int(ep_arr[bp])
            for vid_key in self._video_keys:
                chunk, fidx, from_ts = self._ep_video_file[(ep_idx, vid_key)]
                use_delta = self.delta_indices is not None and vid_key in self.delta_indices
                if use_delta:
                    start, length = delta_pos[s][vid_key]
                    ts = [float(ts_arr[start + i]) + from_ts for i in range(length)]
                else:
                    ts = [float(ts_arr[bp]) + from_ts]
                video_plan.setdefault((chunk, fidx, vid_key), []).append((s, ts))

        # 5. Build tabular results.
        results: list[dict[str, Any]] = []
        for s in range(len(indices)):
            item: dict[str, Any] = {}
            bp = base_pos[s]

            item["episode_index"] = torch.tensor(int(ep_arr[bp]), dtype=torch.int64)
            item["frame_index"] = torch.tensor(int(fi_arr[bp]), dtype=torch.int64)
            item["index"] = torch.tensor(int(idx_arr[bp]), dtype=torch.int64)
            item["timestamp"] = torch.tensor(float(ts_arr[bp]), dtype=torch.float32)
            task_idx = int(task_arr[bp])
            item["task_index"] = torch.tensor(task_idx, dtype=torch.int64)
            if subtask_arr is not None:
                item["subtask_index"] = torch.tensor(int(subtask_arr[bp]), dtype=torch.int64)
            try:
                item["task"] = self.meta.tasks.iloc[task_idx].name
            except (IndexError, AttributeError):
                item["task"] = ""
            if subtask_arr is not None and self.meta.subtasks is not None:
                try:
                    item["subtask"] = self.meta.subtasks.iloc[int(item["subtask_index"])].name
                except (IndexError, AttributeError):
                    pass

            for pad_key, mask in padding_per_sample[s].items():
                item[pad_key] = mask

            for dot_key in self._original_keys:
                if dot_key in self._video_keys:
                    continue  # filled in step 6
                lance_key = _to_lance_name(dot_key)
                if lance_key not in np_columns:
                    continue
                use_delta = self.delta_indices is not None and dot_key in self.delta_indices
                if use_delta:
                    start, length = delta_pos[s][dot_key]
                    item[dot_key] = torch.from_numpy(np_columns[lance_key][start : start + length].copy())
                else:
                    item[dot_key] = torch.from_numpy(np_columns[lance_key][bp].copy())

            results.append(item)

        # 6. Per-file-key decode. Unique (chunk, file, vid_key) triples cost one
        # take_blobs + one VideoDecoder; samples using that file share the decoder.
        # Misses come from the per-worker cache; hits skip the take_blobs entirely.
        files_needing_blobs = [
            (chunk, fidx, vid_key)
            for (chunk, fidx, vid_key) in video_plan.keys()
            if (chunk, fidx, vid_key) not in self._decoder_cache._cache
        ]
        # Fetch all needed blobs from the single video_bytes column in one call.
        if files_needing_blobs:
            row_idxs = [
                self._file_row_index[(vid_key, chunk, fidx)]
                for (chunk, fidx, vid_key) in files_needing_blobs
            ]
            blob_files = self._videos_dataset.take_blobs(
                blob_column="video_bytes", indices=row_idxs
            )
            for (chunk, fidx, vid_key), blob_file in zip(
                files_needing_blobs, blob_files, strict=True
            ):
                self._decoder_cache.get((chunk, fidx, vid_key), blob_file.readall())
                blob_file.close()

        for (chunk, fidx, vid_key), plan in video_plan.items():
            decoder = self._decoder_cache._cache[(chunk, fidx, vid_key)]
            avg_fps = decoder.metadata.average_fps
            use_delta = self.delta_indices is not None and vid_key in self.delta_indices

            # Flatten across samples for one decode call per file/key.
            all_frame_indices: list[int] = []
            offsets: list[tuple[int, int, int]] = []  # (sample_idx, start, length)
            for sample_idx, timestamps in plan:
                start = len(all_frame_indices)
                all_frame_indices.extend(round(ts * avg_fps) for ts in timestamps)
                offsets.append((sample_idx, start, len(timestamps)))
            frames_batch = decoder.get_frames_at(indices=all_frame_indices)
            frames_all = frames_batch.data  # (N, C, H, W) uint8
            if not self._return_uint8:
                frames_all = frames_all.to(torch.float32) / 255.0

            for sample_idx, lo, length in offsets:
                frames = frames_all[lo : lo + length]
                if not use_delta and frames.shape[0] == 1:
                    frames = frames.squeeze(0)
                results[sample_idx][vid_key] = frames

        # 7. Optional image_transforms.
        if self.image_transforms is not None:
            for s in range(len(results)):
                for cam in self.meta.camera_keys:
                    if cam in results[s]:
                        results[s][cam] = self.image_transforms(results[s][cam])

        return results


__all__ = ["LeRobotLanceVideoDataset"]
