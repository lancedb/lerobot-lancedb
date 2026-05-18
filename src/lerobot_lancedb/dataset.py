#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Lance-backed :class:`LeRobotDataset` subclass.

This module exposes :class:`LeRobotLanceDataset`, a subclass of
:class:`lerobot.datasets.LeRobotDataset` that reads frames from a single Lance
table (one row per frame, JPEG-encoded images, columnar tabular features)
instead of the upstream parquet + per-episode-mp4 layout.

Because it's a real subclass, ``isinstance(x, LeRobotDataset)`` returns
``True``, so any existing LeRobot code that accepts a ``LeRobotDataset`` — the
training factory, :class:`EpisodeAwareSampler`, third-party trainers,
``hasattr`` checks — accepts a Lance-backed one too. We override the read
path (``__getitem__`` / ``__getitems__``) and skip the parent's parquet-only
``__init__`` heavy lifting, then set the attributes the parent class would
have set so downstream code remains backend-agnostic.
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ._spawn_compat import force_spawn_for_lance


logger = logging.getLogger(__name__)

# Suppress the "buffer is not writable" UserWarning emitted by torch.frombuffer
# when wrapping JPEG bytes returned from Lance. We never write through the view.
warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable",
    category=UserWarning,
)

# Optional fast-path: torchvision's libjpeg-turbo-backed batch JPEG decoder.
try:
    from torchvision.io import (
        ImageReadMode as _TVImageReadMode,
        decode_jpeg as _tv_decode_jpeg,
    )

    _TV_RGB = _TVImageReadMode.RGB
except (ImportError, AttributeError):
    _tv_decode_jpeg = None
    _TV_RGB = None


_RESERVED_KEYS = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index", "subtask_index"}
)


def _to_lance_name(name: str) -> str:
    """Lance rejects ``.`` in field names; rename ``foo.bar`` → ``foo_bar``."""
    return name.replace(".", "_")


class LeRobotLanceDataset(LeRobotDataset):
    """Map-style dataset that reads frames from a Lance table.

    Subclasses :class:`LeRobotDataset` so existing trainers / samplers /
    ``isinstance`` checks accept it transparently. Recording / writing is
    intentionally not supported — convert from an existing
    :class:`LeRobotDataset` via
    :func:`lerobot_lancedb.writer.convert_to_lance`.

    Args:
        root: Local directory containing the Lance table and ``meta/`` sidecar.
            One of ``root``, ``uri``, or ``repo_id`` must be provided.
        uri: Cloud URI for the Lance table (``s3://``, ``gs://``,
            ``hf://buckets/...``, ``hf://datasets/...``, ...). When set,
            ``meta_root`` must also be provided since metadata lives outside
            the Lance table.
        meta_root: Local directory holding the ``meta/`` sidecar. Defaults to
            ``root`` when ``root`` is given. For ``uri`` and ``repo_id``,
            metadata is auto-downloaded into a revision-safe cache when
            ``meta_root`` is not provided.
        table_name: Lance table name. Auto-detected from a single ``*.lance``
            subdirectory under ``root`` when omitted; defaults to the second
            segment of ``repo_id`` when streaming from the Hub.
        repo_id: HF Hub dataset repo (e.g. ``'me/pusht_lance'``). Lance reads
            natively from ``hf://datasets/{repo_id}``; the ``meta/`` sidecar
            is snapshot-downloaded into the standard cache. Useful when
            you've pushed a converted dataset back to the Hub.
        revision: HF Hub revision (branch, tag, or commit). Used only when
            ``repo_id`` is set.
        episodes: Optional episode-index filter. **Stored only** for the
            sampler to read — the Lance reader serves every frame; the
            sampler (e.g. :class:`EpisodeAwareSampler`) is the actual filter.
        delta_timestamps: Same shape as in :class:`LeRobotDataset`.
        image_transforms: Callable applied to each camera tensor.
        tolerance_s: Forwarded to :func:`check_delta_timestamps`.
        return_uint8: When ``True``, video features are returned as
            ``(C, H, W)`` ``uint8`` instead of ``float32 / 255``. Image
            features (``dtype='image'`` upstream) are always returned as
            ``float32 / 255`` to match the upstream reader.
        connect_kwargs: Forwarded to :func:`lancedb.connect` — e.g.
            ``storage_options={...}`` to override env-derived defaults.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        uri: str | None = None,
        meta_root: str | Path | None = None,
        table_name: str | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 1e-4,
        return_uint8: bool = False,
        connect_kwargs: dict[str, Any] | None = None,
        decode_device: str | torch.device | None = "auto",
    ) -> None:
        """``decode_device`` decodes JPEGs on the named torch device.

        * ``"auto"`` (default) — use ``cuda`` if available, otherwise ``cpu``.
          Recommended for GPU training: NVJPEG is typically ~10× faster than
          libjpeg-turbo and decoded tensors land on the GPU directly, saving
          the H2D transfer in the training loop. The decoded-batch memory
          (~230 MB per batch on a 4-cam 480×640 dataset) is memory the
          forward pass needs on-GPU anyway, so this isn't extra pressure.
        * ``"cpu"`` — force CPU decode. Useful if you're memory-constrained
          on a small GPU (each spawn-mode worker spawns its own CUDA context,
          ~500 MB-1 GB each), or if you want to control where tensors land.
        * ``"cuda"`` / ``torch.device(...)`` — explicit device. Use when you
          have multiple GPUs and want decode to land on a specific one.
        """
        # Skip LeRobotDataset.__init__ — it does Hub downloads + builds a
        # DatasetReader for the parquet+mp4 path, neither of which apply
        # here. Go straight to torch.utils.data.Dataset for the bookkeeping
        # that PyTorch's loader needs.
        torch.utils.data.Dataset.__init__(self)

        if root is None and uri is None and repo_id is None:
            raise TypeError(
                "LeRobotLanceDataset requires one of `root` (local), "
                "`uri` (cloud), or `repo_id` (HF Hub)."
            )

        # Hub locator: snapshot-download just `meta/*` and let Lance stream
        # the table from `hf://datasets/<repo_id>` natively.
        if repo_id is not None and uri is None and root is None:
            uri, hub_meta_root, table_name = self._materialize_from_hub(
                repo_id, revision, table_name
            )
            if meta_root is None:
                meta_root = hub_meta_root

        # Resolve URI and table name. ``root`` is the directory holding the
        # ``<table>.lance/`` and ``meta/`` siblings.
        self._uri, self._table_name = self._resolve_uri_and_table(root, uri, table_name)
        meta_root_resolved = (
            Path(meta_root) if meta_root is not None else (Path(root) if root is not None else None)
        )
        if meta_root_resolved is None:
            raise ValueError(
                "When using `uri=` for a remote Lance table, `meta_root=` must point "
                "to a local directory holding `meta/info.json` etc."
            )

        # --- Attributes that LeRobotDataset would have set ----------------
        # These are read by EpisodeAwareSampler, the training factory,
        # `__repr__`, and other LeRobot code. Set them up exactly as the
        # parent does so subclass instances are indistinguishable from the
        # outside.
        self.repo_id = repo_id or self._table_name
        self._requested_root = meta_root_resolved
        self.root = meta_root_resolved
        self.revision = revision
        # Episodes are a *sampler hint*, not a reader filter (see the
        # docstring). EpisodeAwareSampler(...episode_indices_to_use=ds.episodes)
        # is the actual filter.
        self.episodes = list(episodes) if episodes is not None else None
        self.tolerance_s = tolerance_s
        self.delta_timestamps = delta_timestamps
        self.set_image_transforms(image_transforms)
        self._return_uint8 = return_uint8
        # Resolve ``decode_device``:
        #   "auto" (default) → cuda if available, else cpu
        #   None             → cpu (back-compat with the original API)
        #   "cuda"/"cpu"/torch.device(...) → use as-is
        if decode_device == "auto":
            decode_device = "cuda" if torch.cuda.is_available() else None
        if decode_device in (None, "cpu"):
            self._decode_device = None
        else:
            self._decode_device = torch.device(decode_device)

        # The parent has a few video-encoder-specific attributes used only on
        # the write path. We won't ever write, but pyright/static checks and
        # `LeRobotDataset.__repr__` touch some of these — initialize them.
        self._video_backend = None
        self._batch_encoding_size = 1
        self._vcodec = None
        self._encoder_threads = None
        self.reader = None  # No upstream DatasetReader — Lance has its own
        self.writer = None  # Read-only dataset
        self._is_finalized = True

        # --- Lance-specific state ----------------------------------------
        self._connect_kwargs = self._auto_connect_kwargs(self._uri, connect_kwargs)

        # Worker-spawn safety. Must happen before any DataLoader spawns.
        force_spawn_for_lance()

        # Metadata: reuse the same loader as the parquet+mp4 path. The
        # sidecar layout is identical, so meta.fps, meta.features,
        # meta.stats, meta.tasks, meta.episodes all work without
        # modification.
        self.meta = self._load_metadata(self.repo_id, meta_root_resolved)

        self._probe_table_exists()

        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        # meta.features includes auto-populated index columns; the user's
        # actual feature columns are everything else.
        self._original_keys = [k for k in self.meta.features.keys() if k not in _RESERVED_KEYS]
        self._lance_to_dot = {_to_lance_name(k): k for k in self._original_keys}
        self._image_only_keys = set(self.meta.image_keys)
        self._video_keys = set(self.meta.video_keys)
        self._image_keys = self._image_only_keys | self._video_keys
        self._image_keys_lance = frozenset(_to_lance_name(k) for k in self._image_keys)

        # Episode bounds for delta-window clamping and binary-search lookup.
        sorted_eps = sorted(self._extract_episode_bounds(self.meta).items(), key=lambda kv: kv[1][0])
        if sorted_eps:
            self._ep_starts = np.array([v[0] for _, v in sorted_eps], dtype=np.int64)
            self._ep_ends = np.array([v[1] for _, v in sorted_eps], dtype=np.int64)
            self._ep_ids = np.array([k for k, _ in sorted_eps], dtype=np.int64)
        else:
            self._ep_starts = np.empty(0, dtype=np.int64)
            self._ep_ends = np.empty(0, dtype=np.int64)
            self._ep_ids = np.empty(0, dtype=np.int64)

        # Lazy lance handles; rebuilt per worker after spawn pickling.
        self._db = None
        self._table = None
        self._perm = None
        self._fetch_columns: list[str] | None = None
        self._all_lance_columns: list[str] | None = None

    # ── construction helpers ──────────────────────────────────────────

    @staticmethod
    def _materialize_from_hub(
        repo_id: str, revision: str | None, table_name: str | None
    ) -> tuple[str, Path, str]:
        """Resolve a HF Hub ``repo_id`` to a lance URI + local meta sidecar."""
        from huggingface_hub import snapshot_download

        from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

        local_root = Path(
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=revision,
                cache_dir=HF_LEROBOT_HUB_CACHE,
                allow_patterns="meta/*",
            )
        )
        if table_name is None:
            table_name = repo_id.split("/")[-1]
        suffix = f"@{revision}" if revision else ""
        lance_uri = f"hf://datasets/{repo_id}{suffix}"
        return lance_uri, local_root, table_name

    @staticmethod
    def _resolve_uri_and_table(
        root: str | Path | None, uri: str | None, table_name: str | None
    ) -> tuple[str, str]:
        """Decide the lancedb connect URI and the table name to open."""
        if uri is not None:
            stripped = str(uri).rstrip("/")
            if stripped.lower().endswith(".lance"):
                sep = stripped.rfind("/")
                if sep < 0:
                    return ".", stripped[: -len(".lance")]
                return stripped[:sep], stripped[sep + 1 : -len(".lance")]
            if table_name is None:
                raise ValueError(
                    "`uri=` does not end in '.lance' and `table_name=` was not set; "
                    "cannot infer which table to open."
                )
            return stripped, table_name

        root = Path(root)
        if root.name.lower().endswith(".lance"):
            return str(root.parent), root.stem
        if table_name is not None:
            return str(root), table_name
        candidates = sorted(root.glob("*.lance"))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No '*.lance' table found under {root}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple '*.lance' tables under {root}: {[c.name for c in candidates]}. "
                "Pass `table_name=` explicitly."
            )
        return str(root), candidates[0].stem

    @staticmethod
    def _auto_connect_kwargs(
        uri: str, connect_kwargs: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Inject default ``storage_options`` for recognized cloud schemes."""
        kw = dict(connect_kwargs or {})
        storage_options = dict(kw.get("storage_options") or {})

        if uri.startswith("s3://") and "region" not in storage_options:
            region = (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-east-1"
            )
            storage_options.setdefault("region", region)
            storage_options.setdefault("virtual_hosted_style_request", "true")

        if uri.startswith("hf://") and "token" not in storage_options:
            try:
                from huggingface_hub import get_token
            except ImportError:
                get_token = None  # type: ignore[assignment]
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token is None and get_token is not None:
                token = get_token()
            if token:
                storage_options["token"] = token

        if storage_options:
            kw["storage_options"] = storage_options
        return kw

    @staticmethod
    def _load_metadata(repo_id: str, meta_root: Path) -> LeRobotDatasetMetadata:
        meta_root = Path(meta_root)
        if not (meta_root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Lance dataset at '{meta_root}' is missing 'meta/info.json'. "
                "Did you run lerobot-convert-to-lance to produce it?"
            )
        return LeRobotDatasetMetadata(repo_id=repo_id, root=meta_root)

    @staticmethod
    def _extract_episode_bounds(meta: LeRobotDatasetMetadata) -> dict[int, tuple[int, int]]:
        eps = meta.episodes
        if eps is None:
            return {}
        out: dict[int, tuple[int, int]] = {}
        for i in range(len(eps)):
            row = eps[i]
            ep_idx = int(row.get("episode_index", i))
            out[ep_idx] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        return out

    # ── pickling for spawn-mode workers ───────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_db"] = None
        state["_table"] = None
        state["_perm"] = None
        return state

    def __setstate__(self, state: dict) -> None:
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
            f"    Lance URI: '{self._uri}/{self._table_name}.lance',\n"
            f"    Number of episodes: '{self.num_episodes}',\n"
            f"    Number of frames: '{self.num_frames}',\n"
            f"    Features: '{list(self.features)}',\n"
            f"}})"
        )

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        self.set_image_transforms(None)

    # ── lance handle management ───────────────────────────────────────

    def _probe_table_exists(self) -> None:
        """Verify the lance table exists at ``self._uri`` before lazy-open.

        Surfaces "this URI is not a lerobot-lancedb dataset" at init time
        with a clear message, instead of leaking a lance HTTP 404 from
        ``__getitem__`` much later.
        """
        import lancedb

        try:
            db = lancedb.connect(self._uri, **self._connect_kwargs)
            names = list(db.list_tables().tables)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not list lance tables at {self._uri!r}: {e}. "
                "Check that the URI / repo exists and credentials are set."
            ) from e
        if self._table_name not in names:
            raise FileNotFoundError(
                f"No lance table named '{self._table_name}' at {self._uri!r} "
                f"(existing tables: {names}). This doesn't look like a "
                "lerobot-lancedb dataset — convert the source with "
                "`lerobot-convert-to-lance` (frames) or "
                "`lerobot-convert-to-lance-video` (video) first."
            )

    def _ensure_open(self) -> None:
        """Connect to the table and build a Permutation read handle."""
        if self._perm is not None:
            return
        import lancedb
        from lancedb.permutation import Permutation

        self._db = lancedb.connect(self._uri, **self._connect_kwargs)
        self._table = self._db.open_table(self._table_name)
        if self._all_lance_columns is None:
            self._all_lance_columns = list(self._table.schema.names)
        wanted_dot_keys = list(self._original_keys) + [
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            "task_index",
        ]
        if "subtask_index" in self.meta.features:
            wanted_dot_keys.append("subtask_index")
        wanted_lance_keys = [_to_lance_name(k) for k in wanted_dot_keys]
        self._fetch_columns = [c for c in wanted_lance_keys if c in self._all_lance_columns]
        self._perm = (
            Permutation.identity(self._table)
            .select_columns(self._fetch_columns)
            .with_format("arrow")
        )

    # ── decoding ──────────────────────────────────────────────────────

    def _decode_jpeg_blobs(self, blobs: list[bytes]) -> torch.Tensor:
        """Decode JPEG ``bytes`` → ``(N, C, H, W)`` ``uint8`` tensor.

        When ``self._decode_device`` is set (e.g. ``cuda``), torchvision uses
        NVJPEG and returns tensors on that device — ~10× faster than CPU
        libjpeg-turbo on a typical NVIDIA GPU and saves the H2D copy.
        """
        if not blobs:
            return torch.empty(
                0, dtype=torch.uint8, device=self._decode_device or "cpu"
            )

        if _tv_decode_jpeg is not None:
            try:
                byte_tensors = [
                    torch.frombuffer(b if isinstance(b, (bytes, bytearray)) else bytes(b), dtype=torch.uint8)
                    for b in blobs
                ]
                if self._decode_device is not None:
                    decoded = _tv_decode_jpeg(
                        byte_tensors, mode=_TV_RGB, device=self._decode_device
                    )
                else:
                    decoded = _tv_decode_jpeg(byte_tensors, mode=_TV_RGB)
                return torch.stack(decoded)
            except (RuntimeError, TypeError):
                pass  # malformed blob — fall through to PIL

        out = []
        for b in blobs:
            with Image.open(io.BytesIO(b)) as img:
                arr = np.array(img.convert("RGB"))
            out.append(torch.from_numpy(arr).permute(2, 0, 1))
        stacked = torch.stack(out)
        if self._decode_device is not None:
            stacked = stacked.to(self._decode_device)
        return stacked

    def _decode_image_column(self, blobs: list[bytes], dot_key: str) -> torch.Tensor:
        """Decode JPEGs and apply per-feature normalization.

        Mirrors upstream :class:`LeRobotDataset`: video features honor
        ``return_uint8`` (the trainer's IPC-saver flag), but
        ``dtype='image'`` features always come back as ``float32`` in
        ``[0, 1]``.
        """
        frames = self._decode_jpeg_blobs(blobs)
        if dot_key in self._video_keys and self._return_uint8:
            return frames
        return frames.to(torch.float32) / 255.0

    # ── delta-window helpers ──────────────────────────────────────────

    def _episode_for_index(self, abs_idx: int) -> int:
        """Absolute frame index → episode index via binary search."""
        pos = int(np.searchsorted(self._ep_starts, abs_idx, side="right")) - 1
        if pos < 0 or abs_idx >= int(self._ep_ends[pos]):
            raise IndexError(f"Frame index {abs_idx} is outside any episode bound.")
        return int(self._ep_ids[pos])

    def _build_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Clamp delta offsets to the episode + emit padding flags.

        Same shape as upstream :meth:`DatasetReader._get_query_indices`.
        """
        ep_start = int(self._ep_starts[np.searchsorted(self._ep_ids, ep_idx)])
        ep_end = int(self._ep_ends[np.searchsorted(self._ep_ids, ep_idx)])
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
        """Return one frame. Overrides :meth:`LeRobotDataset.__getitem__`."""
        return self.__getitems__([int(idx)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched ``__getitem__`` (PyTorch DataLoader picks this up via duck typing).

        1. Lay out every row we need per sample (base + delta expansions).
        2. Dedupe the union and call ``Permutation.__getitems__`` exactly once.
        3. Bulk-extract each column into numpy / Python lists.
        4. Slice per sample with cheap numpy / list indexing.
        5. Batched JPEG decode per camera key.
        """
        if not indices:
            return []
        self._ensure_open()
        import pyarrow as pa

        # Step 1.
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

        # Step 2.
        unique_rows = sorted(set(all_rows))
        unique_batch = self._perm.__getitems__(unique_rows)
        if all_rows == unique_rows:
            big_batch = unique_batch
        else:
            row_lookup = {r: i for i, r in enumerate(unique_rows)}
            gather = pa.array([row_lookup[r] for r in all_rows], type=pa.int64())
            big_batch = unique_batch.take(gather)

        # Step 3.
        np_columns: dict[str, np.ndarray] = {}
        blob_columns: dict[str, list[bytes]] = {}
        for col_name in big_batch.schema.names:
            col = big_batch.column(col_name)
            ctype = col.type
            if pa.types.is_binary(ctype) or pa.types.is_large_binary(ctype):
                blob_columns[col_name] = col.to_pylist()
            elif pa.types.is_fixed_size_list(ctype):
                dim = ctype.list_size
                flat = np.array(col.flatten().to_numpy(zero_copy_only=False), copy=True)
                np_columns[col_name] = flat.reshape(len(col), dim)
            else:
                np_columns[col_name] = np.array(
                    col.to_numpy(zero_copy_only=False), copy=True
                )

        # Step 4.
        results: list[dict[str, Any]] = []
        blob_layout: dict[str, list[tuple[int, list[bytes]]]] = {}

        ep_arr = np_columns["episode_index"]
        fi_arr = np_columns["frame_index"]
        idx_arr = np_columns["index"]
        ts_arr = np_columns["timestamp"]
        task_arr = np_columns["task_index"]
        subtask_arr = np_columns.get("subtask_index")

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
                lance_key = _to_lance_name(dot_key)
                is_image = lance_key in self._image_keys_lance
                use_delta = self.delta_indices is not None and dot_key in self.delta_indices

                if use_delta:
                    start, length = delta_pos[s][dot_key]
                    if is_image:
                        blobs = blob_columns[lance_key][start : start + length]
                        blob_layout.setdefault(dot_key, []).append((s, blobs))
                    else:
                        item[dot_key] = torch.from_numpy(
                            np_columns[lance_key][start : start + length].copy()
                        )
                else:
                    if is_image:
                        blob_layout.setdefault(dot_key, []).append(
                            (s, [blob_columns[lance_key][bp]])
                        )
                    else:
                        item[dot_key] = torch.from_numpy(np_columns[lance_key][bp].copy())

            results.append(item)

        # Step 5: batched JPEG decode per camera.
        for dot_key, per_sample in blob_layout.items():
            flat: list[bytes] = []
            offsets: list[tuple[int, int]] = []
            for _s, blobs in per_sample:
                start = len(flat)
                flat.extend(blobs)
                offsets.append((start, start + len(blobs)))
            decoded = self._decode_image_column(flat, dot_key)
            for (s, _blobs), (lo, hi) in zip(per_sample, offsets, strict=True):
                tensor = decoded[lo:hi]
                if tensor.shape[0] == 1 and (
                    self.delta_indices is None or dot_key not in self.delta_indices
                ):
                    tensor = tensor.squeeze(0)
                results[s][dot_key] = tensor

        # Step 6: user image_transforms.
        if self.image_transforms is not None:
            for s in range(len(results)):
                for cam in self.meta.camera_keys:
                    if cam in results[s]:
                        results[s][cam] = self.image_transforms(results[s][cam])

        return results


__all__ = ["LeRobotLanceDataset"]
