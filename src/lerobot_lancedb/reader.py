#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import bisect
import hashlib
import io
import json
import multiprocessing
import operator
import os
import re
import shutil
import stat
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import av
import huggingface_hub
import numpy as np
import pyarrow as pa
import torch
from lerobot.configs.video import DEFAULT_DEPTH_UNIT, DepthEncoderConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.depth_utils import dequantize_depth
from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices
from lerobot.utils.constants import HF_LEROBOT_HOME

from ._source import _file_uri_to_path, _is_commit_sha, parse_hf_dataset_uri
from ._video_compat import FrameTimestampError, decode_video_frames_pyav
from .mp4_index import ParsedSampleIndex

try:
    from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
except ImportError:  # older lerobot without the hub snapshot cache constant
    HF_LEROBOT_HUB_CACHE = HF_LEROBOT_HOME / "hub"
from lerobot.utils.import_utils import is_package_available, require_package

# This standalone loader shares LeRobot metadata and decoding semantics.
# External Blob v2 table routing and sparse range planning are maintained here.
_lancedb_available = is_package_available("lancedb")

if TYPE_CHECKING or _lancedb_available:
    import lancedb
    from lancedb.permutation import Permutation

FRAMES_TABLE = "frames"
VIDEOS_TABLE = "videos"
META_TABLE = "meta"
VIDEO_BLOB_COLUMN = "video_bytes"
# Byte-index columns on the videos table Map a frame window to its byte ranges so a batch's
# bytes fetch can be batched . Keyframe columns assume constant frame rate. mp4-only
VIDEO_INDEX_COLUMNS = ("file_size", "moov_offset", "moov_size", "kf_indices", "kf_positions")
# ffmpeg reads more bytes than the frames requested. Padding each prefetched range
# to cover those known reads keeps them off the slow fallback path.
_OPEN_PROBE_BYTES = 256 * 1024
# Per-call limits for multi-sample RGB decoding within each DataLoader worker.
# Concurrent file decodes have independent budgets; final sample outputs,
# scatter copies, and decoder-internal memory are excluded from the byte estimate.
_RGB_DECODE_MAX_FRAMES = 64
_RGB_DECODE_MAX_BYTES = 64 << 20

# Limit the temporal span when merging multiple depth windows, further bounded
# by the byte budget. Individual windows keep the direct decode path; these
# limits do not split an individual window.
_DEPTH_DECODE_MAX_SPAN_FRAMES = 64
_DEPTH_DECODE_MAX_BYTES = 64 << 20
# PyAV retains decoded uint16 frames while dequantization creates float32 output.
# Leave extra headroom for the selected-frame stack and allocator overhead.
_DEPTH_DECODE_BYTES_PER_VALUE = 8


def _decode_frame_limit(
    shape: Sequence[int] | None,
    *,
    max_frames: int,
    max_bytes: int,
    bytes_per_value: int,
) -> int:
    """Bound frame capacity by estimated memory, allowing at least one frame.

    Missing or zero-sized shapes fall back to the frame cap. Callers apply the
    capacity to unique RGB frames or the temporal span of merged depth windows.
    """
    frame_values = int(np.prod(shape)) if shape else 0
    if not frame_values:
        return max_frames
    return min(max_frames, max(1, max_bytes // (frame_values * bytes_per_value)))


def _range_slack(meta: dict, is_depth: bool) -> int:
    """Pad by one average GOP, bounded for FFmpeg read-ahead and depth decode."""
    positions = meta["kf_positions"]
    if len(positions) > 1:
        gop_bytes = (int(positions[-1]) - int(positions[0])) // (len(positions) - 1)
    else:
        gop_bytes = meta["file_size"]
    # PyAV's sequential depth decode reads farther ahead than RGB seeking.
    return min(max(32 * 1024, gop_bytes), 64 * 1024) * (4 if is_depth else 1)


def _merge_spans(spans: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Coalesce overlapping or nearby byte ranges into fewer, larger requests."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _find_moov(read_at, file_size: int) -> tuple[int, int]:
    """Locate the mp4 ``moov`` box by walking top-level box headers."""
    offset = 0
    while offset < file_size:
        header = read_at(offset, 16)
        box_size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]
        if box_size == 1:
            box_size = int.from_bytes(header[8:16], "big")
        elif box_size == 0:
            box_size = file_size - offset
        if box_type == b"moov":
            return offset, box_size
        offset += box_size
    raise ValueError("no moov box found")


def build_video_byte_index(path: str | Path) -> dict:
    """Compute the byte-index columns for one video file.
    Works for any container/codec pyav can demux; frame indices assume constant frame rate. mp4-only.
    """
    path = Path(path)
    file_size = path.stat().st_size
    kf_entries = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for packet in container.demux(stream):
            if packet.pts is None or not packet.is_keyframe or packet.pos is None:
                continue
            kf_entries.append((round(float(packet.pts * packet.time_base) * fps), packet.pos))
    kf_entries.sort()
    with open(path, "rb") as f:

        def read_at(offset: int, length: int) -> bytes:
            f.seek(offset)
            return f.read(length)

        moov_offset, moov_size = _find_moov(read_at, file_size)
    return {
        "file_size": file_size,
        "moov_offset": moov_offset,
        "moov_size": moov_size,
        "kf_indices": [index for index, _ in kf_entries],
        "kf_positions": [position for _, position in kf_entries],
    }


class _SparseBlobSource(io.RawIOBase):
    """Adapter between range fetches and the decoders' file API."""

    def __init__(self, size: int, fallback):
        super().__init__()
        self._size = size
        self._fallback = fallback
        self._starts: list[int] = []
        self._chunks: list[bytes] = []
        self._pos = 0
        self.buffered = 0
        self.fallback_bytes = 0

    def add(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        lo = bisect.bisect_left(self._starts, offset)
        if lo > 0 and self._starts[lo - 1] + len(self._chunks[lo - 1]) >= offset:
            lo -= 1
        hi = bisect.bisect_right(self._starts, end)
        if lo == hi:
            self._starts.insert(lo, offset)
            self._chunks.insert(lo, data)
        else:
            merged_start = min(offset, self._starts[lo])
            merged_end = max(end, max(self._starts[i] + len(self._chunks[i]) for i in range(lo, hi)))
            merged = bytearray(merged_end - merged_start)
            for i in range(lo, hi):
                at = self._starts[i] - merged_start
                merged[at : at + len(self._chunks[i])] = self._chunks[i]
            merged[offset - merged_start : offset - merged_start + len(data)] = data
            del self._starts[lo:hi]
            del self._chunks[lo:hi]
            self._starts.insert(lo, merged_start)
            self._chunks.insert(lo, bytes(merged))
        self.buffered = sum(len(chunk) for chunk in self._chunks)

    def covers(self, start: int, end: int) -> bool:
        index = bisect.bisect_right(self._starts, start) - 1
        return index >= 0 and self._starts[index] + len(self._chunks[index]) >= end

    def missing(self, start: int, end: int):
        i = max(0, bisect.bisect_right(self._starts, start) - 1)
        while i < len(self._starts) and self._starts[i] < end:
            at = self._starts[i]
            if start < at:
                yield start, at
            start = max(start, at + len(self._chunks[i]))
            i += 1
        if start < end:
            yield start, end

    def cached_span(self, start: int, end: int) -> bytes:
        if not self.covers(start, end):
            raise ValueError("Incomplete compressed video span")
        i = bisect.bisect_right(self._starts, start) - 1
        return self._chunks[i][start - self._starts[i]:end - self._starts[i]]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer) -> int:
        if self._pos >= self._size:
            return 0
        want = min(len(buffer), self._size - self._pos)
        index = bisect.bisect_right(self._starts, self._pos) - 1
        if index >= 0:
            start, chunk = self._starts[index], self._chunks[index]
            inside = self._pos - start
            if inside < len(chunk):
                data = chunk[inside : inside + want]
                buffer[: len(data)] = data
                self._pos += len(data)
                return len(data)
        # Cap the miss at the next buffered range so we never re-fetch bytes
        # we already hold.
        next_index = bisect.bisect_right(self._starts, self._pos)
        if next_index < len(self._starts):
            want = min(want, self._starts[next_index] - self._pos)
        data = self._fallback.read_range(self._pos, want)
        self.fallback_bytes += len(data)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


class _EpisodeSubsetIndexMapping(Mapping[int, int]):
    """Index selected episode intervals without allocating an entry per frame."""

    def __init__(self, starts: np.ndarray, ends: np.ndarray):
        self._starts = starts
        self._relative_ends = np.cumsum(ends - starts)
        self.num_frames = int(self._relative_ends[-1]) if len(starts) else 0
        # A repeated episode contributes rows twice. As in the original dict,
        # absolute lookups return the last occurrence, in first-insertion order.
        self._intervals: dict[int, tuple[int, int]] = {}
        relative = 0
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            if start < end:
                self._intervals[start] = (end, relative)
            relative += end - start
        self._absolute_starts = sorted(self._intervals)
        self._num_keys = sum(end - start for start, (end, _) in self._intervals.items())

    def rel_to_abs(self, index: int) -> int:
        index = operator.index(index)
        if index < 0:
            index += self.num_frames
        if not 0 <= index < self.num_frames:
            raise IndexError(index)
        block = int(self._relative_ends.searchsorted(index, side="right"))
        base = int(self._relative_ends[block - 1]) if block else 0
        return int(self._starts[block]) + index - base

    def __getitem__(self, index: int) -> int:
        if not isinstance(index, (int, float, np.integer, np.floating)):
            raise KeyError(index)
        block = bisect.bisect_right(self._absolute_starts, index) - 1
        if block >= 0:
            start = self._absolute_starts[block]
            end, base = self._intervals[start]
            if index < end and int(index) == index:
                return base + int(index) - start
        raise KeyError(index)

    def __len__(self) -> int:
        return self._num_keys

    def __iter__(self) -> Iterator[int]:
        for start, (end, _) in self._intervals.items():
            yield from range(start, end)

    def __eq__(self, other):
        # Mapping.__eq__ would reconstruct a per-frame dict for each operand.
        if not isinstance(other, Mapping):
            return NotImplemented
        return len(self) == len(other) and all(other.get(key) == value for key, value in self.items())


@dataclass
class _MiniFile:
    index: ParsedSampleIndex
    source: _SparseBlobSource


class _VideoDecoderLRU:
    """Per-worker LRU of torchcodec decoders keyed by (video_key, chunk, file).
    eviction is bounded by ``byte_budget`` too, not just entry count.
    """

    def __init__(self, capacity: int, byte_budget: int | None = None):
        self.capacity = capacity
        self.byte_budget = byte_budget
        self._items: OrderedDict[tuple, tuple[object, int]] = OrderedDict()
        self._total_bytes = 0

    def __contains__(self, key: tuple) -> bool:
        return key in self._items

    def get(self, key: tuple):
        self._items.move_to_end(key)
        return self._items[key][0]

    def put(self, key: tuple, decoder, nbytes: int = 0) -> None:
        if key in self._items:
            self._total_bytes -= self._items.pop(key)[1]
        if self.byte_budget is not None and nbytes > self.byte_budget:
            return
        self._items[key] = (decoder, nbytes)
        self._items.move_to_end(key)
        self._total_bytes += nbytes
        while len(self._items) > 1 and (
            len(self._items) > self.capacity
            or (self.byte_budget is not None and self._total_bytes > self.byte_budget)
        ):
            _, (_, evicted_bytes) = self._items.popitem(last=False)
            self._total_bytes -= evicted_bytes


def to_lance_column(key: str) -> str:
    return key.replace(".", "_")


def _is_remote_uri(path) -> bool:
    return "://" in str(path)


_METADATA_ROUTING_OPTIONS = {
    "endpoint",
    "endpoint_url",
    "region",
    "default_region",
    "virtual_hosted_style_request",
    "aws_endpoint",
    "aws_endpoint_url",
    "aws_region",
    "aws_default_region",
    "aws_virtual_hosted_style_request",
    "account_name",
    "azure_storage_account_name",
    "azure_storage_endpoint",
    "azure_endpoint",
    "oss_endpoint",
    "oss_region",
    "tos_endpoint",
    "tos_region",
    "cos_endpoint",
    "cos_region",
}


def _storage_options(
    db_uri: str, storage_options: dict | None, revision: str | None, token: str | bool | None = None
) -> dict:
    """Resolve HF options once for the connection and each table it opens."""
    options = dict(storage_options or {})
    if db_uri.startswith("hf://"):
        if "token" not in options and "hf_token" not in options and token is not False:
            resolved_token = token if isinstance(token, str) else huggingface_hub.get_token()
            if resolved_token:
                options["token"] = resolved_token
        uri_revision = parse_hf_dataset_uri(db_uri)[1] if db_uri.startswith("hf://datasets/") else None
        if uri_revision or revision:
            options.setdefault("revision", uri_revision or revision)
        effective_revision = options.get("hf_revision", options.get("revision"))
        if (
            _is_commit_sha(effective_revision)
            and "enable_resolve_cache" not in options
        ):
            options.setdefault("hf_enable_resolve_cache", "true")
    return options


def _apply_external_hf_options(options: dict, bases, token: str | bool | None) -> None:
    if not isinstance(bases, dict):
        return
    for base_id, base in bases.items():
        if not re.fullmatch(r"[1-9][0-9]*", str(base_id)) or not isinstance(base, dict):
            continue
        uri = base.get("uri")
        if not isinstance(uri, str) or not uri.startswith("hf://datasets/"):
            continue
        try:
            _, uri_ref, _ = parse_hf_dataset_uri(uri)
        except ValueError:
            continue
        pin = uri_ref or base.get("hf_revision")
        if not _is_commit_sha(pin):
            continue
        prefix = f"base_{base_id}."
        if prefix + "hf_revision" not in options:
            options.setdefault(prefix + "revision", pin)
        ref = options.get(prefix + "hf_revision", options.get(prefix + "revision"))
        if _is_commit_sha(ref) and not any(prefix + name in options for name in ("enable_resolve_cache", "hf_enable_resolve_cache")):
            cache = options.get("hf_enable_resolve_cache", options.get("enable_resolve_cache", "true"))
            options.setdefault(prefix + "hf_enable_resolve_cache", cache)
        hub_token = options.get("hf_token", options.get("token", token))
        if isinstance(hub_token, str) and prefix + "hf_token" not in options:
            options.setdefault(prefix + "token", hub_token)
        if "hf_endpoint" in options and prefix + "endpoint" not in options:
            options.setdefault(prefix + "hf_endpoint", options["hf_endpoint"])


def _local_video_path(descriptor, file_size: int) -> Path | None:
    if not isinstance(descriptor, dict) or descriptor.get("kind") != 3 or descriptor.get("blob_id") != 0:
        return None
    uri = descriptor.get("blob_uri")
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return None
    if descriptor.get("position") != 0 or descriptor.get("size") not in (0, file_size):
        return None
    try:
        path = _file_uri_to_path(uri)
        info = path.stat()
        if path.is_absolute() and stat.S_ISREG(info.st_mode) and info.st_size == file_size:
            return path
    except (OSError, ValueError):
        pass
    return None


def _connect(db_uri: str, options: dict):
    if _is_remote_uri(db_uri):
        os.environ.setdefault("LANCE_IO_THREADS", "256")
    return lancedb.connect(db_uri, **({"storage_options": options} if options else {}))


def _open_table(db, name: str, storage_options: dict | None = None):
    if storage_options:
        return db.open_table(name, storage_options=storage_options)
    return db.open_table(name)


def _materialize_meta(db, local_root: Path, storage_options: dict | None = None) -> None:
    """Write ``meta/`` from the meta table to a local cache, once."""
    meta_dir = local_root / "meta"
    if meta_dir.exists():
        return
    try:
        table = _open_table(db, META_TABLE, storage_options)
    except Exception as error:
        raise FileNotFoundError(
            f"Dataset has no '{META_TABLE}' table. Re-convert it with a converter that "
            "ingests meta/, or create the table in place from an existing meta/ copy."
        ) from error

    tmp_dir = local_root / f"meta.tmp-{os.getpid()}"
    try:
        tmp_resolved = tmp_dir.resolve()
        for batch in table.search().select(["path", "data"]).to_batches():
            paths = batch.column("path").to_pylist()
            for i, rel_path in enumerate(paths):
                dst = (tmp_dir / rel_path).resolve()
                if not dst.is_relative_to(tmp_resolved):
                    raise ValueError(f"meta table entry escapes the cache directory: {rel_path!r}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(batch.column("data")[i].as_py())
        try:
            tmp_dir.rename(meta_dir)
        except OSError:
            if not meta_dir.exists():
                raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def lance_mp_context() -> str:
    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


def _storage_format_from_info(info_file: Path) -> str | None:
    """``storage_format`` declared in a ``meta/info.json``, or None (absent/unreadable)."""
    try:
        return json.loads(info_file.read_text()).get("storage_format")
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=32)
def is_lance_dataset(
    repo_id: str | None = None,
    root: str | Path | None = None,
    revision: str | None = None,
    *,
    token: str | bool | None = None,
    endpoint: str | None = None,
) -> bool:
    """True if the dataset's storage format is Lance.

    Resolution order: the ``storage_format`` field in LeRobot metadata decides when
    present; layout detection is only a fallback for datasets converted before the
    field existed. Datasets without either continue to the Parquet/MP4 loader.
    """
    if root is None and repo_id is None:
        return False
    if root is not None and _is_remote_uri(root):
        # Object-store URIs are read in place; only the Lance reader serves them.
        return True
    local_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    fmt = _storage_format_from_info(local_root / "meta" / "info.json")
    if fmt is not None:
        return fmt == "lance"
    if (local_root / f"{FRAMES_TABLE}.lance").exists():
        return True
    if (local_root / "meta" / "info.json").exists() and (local_root / "data").exists():
        # A complete local parquet/MP4 layout: definitively not Lance, skip the Hub
        # lookups below. (A Lance dataset's local cache holds `meta/` but never `data/`.)
        return False
    if repo_id is None:
        return False
    # No decisive local copy: read the Hub metadata. Every loader needs info.json
    # anyway and this lands in the same cache lerobot's metadata loader uses, so
    # the Parquet path pays no extra request.
    try:
        info_file = huggingface_hub.hf_hub_download(
            repo_id,
            "meta/info.json",
            repo_type="dataset",
            revision=revision,
            cache_dir=HF_LEROBOT_HUB_CACHE,
            token=token,
            endpoint=endpoint,
        )
        fmt = _storage_format_from_info(Path(info_file))
    except Exception:
        fmt = None
    if fmt is not None:
        return fmt == "lance"
    try:  # legacy Lance datasets (converted before `storage_format`): layout probe
        paths = huggingface_hub.HfApi(token=token, endpoint=endpoint).get_paths_info(
            repo_id, [f"{FRAMES_TABLE}.lance"], repo_type="dataset", revision=revision
        )
    except Exception:  # fall back to the parquet loader
        return False
    return len(paths) > 0


def resolve_lance_root(
    repo_id: str | None,
    root: str | Path | None,
    storage_options: dict | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> tuple[str, Path]:
    """Resolve a Lance dataset to its connect URI and the local root holding ``meta/``.

    For an object-store URI (``s3://``, ``gs://``, ``hf://`` ...) the ``meta/``
    directory is materialized once into a local cache. Shared by LanceDBDataset and
    the training factory so both agree on remote roots (a bare ``Path(root)`` would
    collapse ``s3://b/k`` to ``s3:/b/k`` and silently fall back to a Hub download).
    """
    if root is not None and _is_remote_uri(root):
        db_uri = str(root).rstrip("/")
        options = _storage_options(db_uri, storage_options, revision, token)
        cache_key = db_uri
        routing = {k.lower(): v for k, v in options.items() if k.lower() in _METADATA_ROUTING_OPTIONS}
        if db_uri.startswith("hf://"):
            ref = options.get("hf_revision", options.get("revision"))
            if ref:
                routing["revision"] = ref
        if routing:
            identity = json.dumps(routing, sort_keys=True, separators=(",", ":"))
            cache_key += "-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        local_root = HF_LEROBOT_HOME / "remote" / re.sub(r"[^A-Za-z0-9._-]+", "_", cache_key)
        if not (local_root / "meta").exists():
            _materialize_meta(_connect(db_uri, options), local_root, options or None)
        return db_uri, local_root
    root_path = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if (root_path / f"{FRAMES_TABLE}.lance").exists():
        return str(root_path), root_path
    if repo_id is not None:
        options = storage_options or {}
        ref = options.get("hf_revision", options.get("revision", revision))
        hub_token = options.get("hf_token", options.get("token", token))
        if (ref or hub_token is not None) and (root is None or not (root_path / "meta").exists()):
            # Released LeRobot versions differ in token support. Download just
            # metadata into a revision-specific Hub snapshot before loading it.
            root_path = Path(
                huggingface_hub.snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=ref,
                    token=hub_token,
                    allow_patterns="meta/**",
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    local_dir=root,
                )
            )
        return f"hf://datasets/{repo_id}", root_path
    raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {root_path}.")


def lance_metadata(
    repo_id: str | None,
    root: str | Path | None,
    revision: str | None = None,
    storage_options: dict | None = None,
    token: str | bool | None = None,
) -> LeRobotDatasetMetadata:
    """LeRobotDatasetMetadata for a Lance dataset, materializing remote-URI ``meta/``.

    Lets the training factory resolve metadata (for delta-timestamp resolution)
    without tripping over a remote-URI root.
    """
    uri, local_root = resolve_lance_root(repo_id, root, storage_options, revision, token)
    if uri.startswith("hf://"):
        options = _storage_options(uri, storage_options, revision, token)
        revision = options.get("hf_revision", options.get("revision"))
    return LeRobotDatasetMetadata(
        repo_id if repo_id is not None else str(local_root), root=local_root, revision=revision
    )


class LanceDBDataset(torch.utils.data.Dataset):
    """Map-style dataset over a Lance-backed LeRobot dataset.

    Returns the same item dict as :class:`LeRobotDataset` and satisfies the same
    duck-typed contract, so the training pipeline consumes it unchanged. Implements
    batched ``__getitems__``: a batch's rows (including delta-timestamp windows)
    are fetched from the frames table in one deduplicated read.

    Args:
        repo_id: Hub dataset repo; tables stream over ``hf://``, only ``meta/`` downloads.
        root: Local dir with ``meta/`` and ``.lance`` tables, or an object-store
            URI (``s3://...``) with the same layout. Local tables win when both given.
        episodes: Episode indices to select. ``None`` means all.
        image_transforms: Optional torchvision v2 transform for camera frames.
        delta_timestamps: Feature key -> relative timestamp offsets (seconds), as
            in :class:`LeRobotDataset`.
        tolerance_s: Timestamp synchronization tolerance in seconds.
        revision: Hub revision for the ``meta/`` download.
        return_uint8: Return RGB frames as raw uint8 instead of normalized float32.
        storage_options: Options passed to the connection and each table, including External blob bases.
        token: Hub token, or False to disable implicit Hub authentication.
        video_decoder_cache_size: Max decoders per worker (default 16, also
            bounded by a 2 GiB per-worker byte budget).
        columns: Requested features and optional task label. Defaults to all features and task.
        use_mini_mp4: Decode indexed RGB blob ranges using small MP4 containers in memory.
        depth_output_unit: Unit depth features dequantize to (``'mm'`` or ``'m'``).
            Depth decodes through pyav (16-bit planes torchcodec cannot emit).
    """

    def __init__(
        self,
        repo_id: str | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        return_uint8: bool = False,
        storage_options: dict | None = None,
        video_decoder_cache_size: int | None = None,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        token: str | bool | None = None,
        *,
        columns: Sequence[str] | None = None,
        use_mini_mp4: bool = True,
    ):
        super().__init__()
        require_package("lancedb", extra="lance")
        if repo_id is None and root is None:
            raise ValueError("Provide `repo_id`, `root`, or both.")

        self.repo_id = repo_id
        self.tolerance_s = tolerance_s
        self._db_uri, self.root = resolve_lance_root(repo_id, root, storage_options, revision, token)
        self._storage_options = _storage_options(self._db_uri, storage_options, revision, token)
        if self._db_uri.startswith("hf://"):
            revision = self._storage_options.get("hf_revision", self._storage_options.get("revision"))
        self.meta = LeRobotDatasetMetadata(
            repo_id if repo_id is not None else str(self.root), root=self.root, revision=revision
        )
        if self._db_uri == f"hf://datasets/{repo_id}" and not (root and _is_remote_uri(root)):
            # The metadata loader may have resolved a LeRobot version tag.
            # A metadata-only Hub download has a commit-specific snapshot path.
            resolved = self.meta.revision
            if self.root.parent.name == "snapshots" and _is_commit_sha(self.root.name):
                resolved = self.root.name
                self._storage_options.pop("hf_revision", None)
                self._storage_options["revision"] = resolved
            self._storage_options = _storage_options(self._db_uri, self._storage_options, resolved, token)

        # LeRobot 0.7 discards extension fields when it constructs DatasetInfo.
        info = json.loads((self.meta.root / "meta/info.json").read_text())
        _apply_external_hf_options(self._storage_options, info.get("lance_external_bases"), token)

        selected = set(columns) if columns is not None else None
        if selected is not None:
            if not selected:
                raise ValueError("columns must be non-empty")
            unknown = selected.difference(self.meta.features, {"task"})
            if unknown:
                raise ValueError(f"Unknown columns: {sorted(unknown)}")
            missing = set(delta_timestamps or {}) - selected
            if missing:
                raise ValueError(f"delta_timestamps keys missing from columns: {sorted(missing)}")

        def active(keys):
            return keys if selected is None else [key for key in keys if key in selected]

        self._selected = selected
        self._include_task = selected is None or "task" in selected
        self._video_keys = active(self.meta.video_keys)
        self._camera_keys = active(self.meta.camera_keys)
        self._depth_keys = active(self.meta.depth_keys)
        if active(self.meta.image_keys):
            raise NotImplementedError(
                f"Image-backed features are not supported by LanceDBDataset: {self.meta.image_keys}. "
                "Re-encode them as video."
            )
        # Depth videos carry 16-bit planes torchcodec cannot emit; they decode
        # through pyav over the same prefetched sources.
        self._depth_output_unit = depth_output_unit
        if self._depth_keys:
            self.meta.rescale_depth_stats(self._depth_output_unit)
        self._depth_encoder_configs = {
            key: DepthEncoderConfig.from_video_info(self.meta.features[key].get("info"))
            for key in self._depth_keys
        }
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        self.return_uint8 = return_uint8

        self.episodes = sorted(episodes) if episodes is not None else None
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        # Episode boundaries (absolute frame index space) for delta clamping + padding.
        self._ep_from = self._episode_numpy("dataset_from_index", np.int64)
        self._ep_to = self._episode_numpy("dataset_to_index", np.int64)
        # Integrity: episode ranges must tile [0, total_frames) exactly. Public
        # datasets ship broken meta often (droid orphans, berkeley tails, agibot empties).
        if len(self._ep_from) and (
            int(self._ep_from[0]) != 0
            or int(self._ep_to[-1]) != self.meta.total_frames
            or (self._ep_from[1:] != self._ep_to[:-1]).any()
        ):
            raise ValueError(
                "Episode boundaries in meta do not tile [0, total_frames) contiguously; "
                "the dataset metadata is inconsistent."
            )

        self._subset_index = None
        if self.episodes is not None:
            self._subset_index = _EpisodeSubsetIndexMapping(
                self._ep_from[self.episodes], self._ep_to[self.episodes]
            )

        self._task_names = list(self.meta.tasks.index) if self._include_task else []

        self._output_tabular_keys = [
            key
            for key in active(self.meta.features)
            if key not in self.meta.video_keys and key not in self.meta.image_keys
        ]
        self._fetch_tabular_keys = self._output_tabular_keys.copy()
        if self._video_keys and "timestamp" not in self._fetch_tabular_keys:
            self._fetch_tabular_keys.append("timestamp")
        if self._include_task and "task_index" not in self._fetch_tabular_keys:
            self._fetch_tabular_keys.append("task_index")
        self._fetch_columns = [to_lance_column(key) for key in self._fetch_tabular_keys]
        self._feature_shapes = {
            key: tuple(self.meta.features[key].get("shape") or ()) for key in self._output_tabular_keys
        }
        # String features pass through as python strings, language columns
        # (list<struct>, lerobot#3467) as python lists of dicts, like upstream.
        self._string_keys = {
            key for key in self._output_tabular_keys if self.meta.features[key].get("dtype") == "string"
        }
        self._language_keys = {
            key for key in self._output_tabular_keys if self.meta.features[key].get("dtype") == "language"
        }

        # Which (chunk, file) mp4 holds each episode and where it starts inside
        # it (episodes share files in v3.0; timestamps shift by from_timestamp).
        self._video_locator = {
            key: (
                self._episode_numpy(f"videos/{key}/chunk_index", np.int64),
                self._episode_numpy(f"videos/{key}/file_index", np.int64),
                self._episode_numpy(f"videos/{key}/from_timestamp", np.float64),
            )
            for key in self._video_keys
        }

        self._frames_perm = None
        self._videos_table = None
        self._video_row_ids: dict[tuple, int] | None = None
        self._use_mini_mp4 = use_mini_mp4
        self._index_cache = _VideoDecoderLRU(128, byte_budget=64 << 20)
        self._mini_ineligible = set()
        self._file_meta: OrderedDict[tuple, dict] = OrderedDict()
        self._prefetch_pool: ThreadPoolExecutor | None = None
        self._decode_pool: ThreadPoolExecutor | None = None
        if video_decoder_cache_size is None:
            video_decoder_cache_size = 16
        self._decoder_cache = _VideoDecoderLRU(video_decoder_cache_size, byte_budget=2 << 30)  # 2GB cap

    def _episode_numpy(self, name: str, dtype: type[np.generic]) -> np.ndarray:
        # Read straight from the underlying Arrow column, not HF Dataset __getitem__
        # which row-formats every element (~1M calls at droid scale, ~10s in __init__).
        column = self.meta.episodes.data.column(name).to_numpy(zero_copy_only=False)
        return column.astype(dtype, copy=False)

    def _ensure_open(self) -> None:
        if self._frames_perm is not None:
            return
        if self._video_keys:
            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
            self._decode_pool = ThreadPoolExecutor(max_workers=16)
        options = self._storage_options
        db = _connect(self._db_uri, options)
        table = _open_table(db, FRAMES_TABLE, options or None)
        n_rows = table.count_rows()
        if n_rows != self.meta.total_frames:
            raise ValueError(
                f"frames table has {n_rows} rows but meta declares "
                f"{self.meta.total_frames} frames; the dataset is truncated or corrupt."
            )
        self._frames_perm = (
            Permutation.identity(table).select_columns(self._fetch_columns).with_format("arrow")
        )
        if self._video_keys:
            self._videos_table = _open_table(db, VIDEOS_TABLE, options or None)
            self._has_sample_index = self._use_mini_mp4 and "sample_index" in self._videos_table.schema.names
            # TODO: for million-file datasets, resolve row ids lazily per batch.
            index = (
                self._videos_table.search()
                .select(["video_key", "chunk_index", "file_index"])
                .with_row_id(True)
                .to_arrow()
            )
            self._video_row_ids = {
                (row["video_key"], row["chunk_index"], row["file_index"]): row["_rowid"]
                for row in index.to_pylist()
            }
            referenced = {
                (key, int(chunk), int(file))
                for key, (chunks, files, _) in self._video_locator.items()
                for chunk, file in zip(chunks, files, strict=True)
            }
            missing = referenced - self._video_row_ids.keys()
            if missing:
                raise ValueError(
                    f"videos table is missing {len(missing)} file(s) referenced by episode "
                    f"metadata, e.g. {sorted(missing)[:3]}. The dataset is incomplete or "
                    "was converted against different metadata."
                )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_frames_perm"] = None
        state["_videos_table"] = None
        state["_video_row_ids"] = None
        state["_mini_ineligible"] = set()
        state["_index_cache"] = _VideoDecoderLRU(self._index_cache.capacity, self._index_cache.byte_budget)
        state["_file_meta"] = OrderedDict()
        state["_prefetch_pool"] = None
        state["_decode_pool"] = None
        state["_decoder_cache"] = _VideoDecoderLRU(
            self._decoder_cache.capacity, byte_budget=self._decoder_cache.byte_budget
        )
        return state

    @property
    def num_frames(self) -> int:
        return self._subset_index.num_frames if self._subset_index is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> Mapping[int, int] | None:
        return self._subset_index

    @property
    def features(self) -> dict[str, dict]:
        if self._selected is None:
            return self.meta.features
        return {key: value for key, value in self.meta.features.items() if key in self._selected}

    @property
    def fps(self) -> int:
        return self.meta.fps

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched fetch: one deduplicated frames-table read and one blob fetch per batch."""
        self._ensure_open()
        plans = self._plan_batch(indices)
        rows, row_pos = self._batch_rows(plans)

        # Video prep (byte-index, header ranges, decoder creation, window ranges)
        # needs only the batch's files and frame indices, so it overlaps the
        # frames-table fetch. The timestamp coverage check in _ensure_decoders is
        # the safety net: any drift costs a re-fetch, never a wrong frame.
        prepared_future = None
        if self._video_keys:
            windows = self._plan_file_windows(plans)
            prepared_future = self._prefetch_pool.submit(self._prepare_files, sorted(windows), windows)

        columns = self._fetch_rows(rows)
        items = [self._build_item(plan, columns, row_pos) for plan in plans]

        if self._video_keys:
            decoded = self._decode_videos(plans, columns, row_pos, prepared_future.result())
            for item, frames in zip(items, decoded, strict=True):
                item.update(frames)
        if self.image_transforms is not None:
            for item in items:
                for cam_key in self._camera_keys:
                    if cam_key in self._depth_keys:
                        continue
                    item[cam_key] = self.image_transforms(item[cam_key])
        return items

    def _plan_file_windows(self, plans: list[dict]) -> dict[tuple, list[tuple[int, int]]]:
        """Map each batch sample's video windows to (file key -> frame spans).

        Positions come from episode metadata alone; stage 2 re-derives ranges
        from real timestamps and fetches anything missed.
        """
        fps = float(self.meta.fps)
        windows: dict[tuple, list[tuple[int, int]]] = {}
        for plan in plans:
            ep_idx = plan["ep_idx"]
            for key in self._video_keys:
                file_key, span = self._planned_file_window(key, ep_idx, plan, fps)
                spans = windows.setdefault(file_key, [])
                if self._use_mini_mp4 and key not in self._depth_keys and key in plan["windows"]:
                    rows = plan["windows"][key]
                    base = span[0] - min(rows)
                    spans.extend((base + row, base + row) for row in set(rows))
                else:
                    spans.append(span)
        return windows

    def _plan_batch(self, indices: list[int]) -> list[dict]:
        """Resolve each sample to the absolute rows it needs and its padding masks."""
        plans = []
        for idx in indices:
            abs_idx = self._resolve_abs_idx(idx)
            ep_idx = self._episode_index_for_abs_idx(abs_idx)
            start, end = self._episode_bounds(ep_idx)
            plan = {"abs_idx": abs_idx, "ep_idx": ep_idx, "rows": {abs_idx}, "windows": {}, "padding": {}}
            if self.delta_indices is not None:
                for key, deltas in self.delta_indices.items():
                    window = [min(max(abs_idx + delta, start), end - 1) for delta in deltas]
                    plan["windows"][key] = window
                    plan["rows"].update(window)
                    plan["padding"][f"{key}_is_pad"] = torch.BoolTensor(
                        [not (start <= abs_idx + delta < end) for delta in deltas]
                    )
            plans.append(plan)
        return plans

    def _resolve_abs_idx(self, idx: int) -> int:
        return self._subset_index.rel_to_abs(idx) if self._subset_index is not None else int(idx)

    def _episode_index_for_abs_idx(self, abs_idx: int) -> int:
        return int(np.searchsorted(self._ep_from, abs_idx, side="right") - 1)

    def _episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        return int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])

    def _batch_rows(self, plans: list[dict]) -> tuple[list[int], dict[int, int]]:
        rows = sorted({row for plan in plans for row in plan["rows"]})
        return rows, {row: pos for pos, row in enumerate(rows)}

    def _planned_file_window(
        self, key: str, ep_idx: int, plan: dict, fps: float
    ) -> tuple[tuple[str, int, int], tuple[int, int]]:
        ep_start, _ = self._episode_bounds(ep_idx)
        _, _, from_ts_arr = self._video_locator[key]
        window = plan["windows"].get(key, [plan["abs_idx"]])
        base = round(float(from_ts_arr[ep_idx]) * fps) - ep_start
        first = base + min(window) - (1 if key in self._depth_keys else 0)
        return self._video_file_key(key, ep_idx), (first, base + max(window))

    def _fetch_rows(self, rows: list[int]) -> dict[str, np.ndarray]:
        batch = self._frames_perm.__getitems__(rows)
        columns = {}
        for key, lance_name in zip(self._fetch_tabular_keys, self._fetch_columns, strict=True):
            array = batch.column(lance_name)
            if hasattr(array, "combine_chunks"):
                array = array.combine_chunks()
            if key in self._language_keys:
                rows = array.to_pylist()
                for row in rows:
                    for msg in row or ():
                        if msg.get("tool_calls"):
                            msg["tool_calls"] = [
                                json.loads(call) if isinstance(call, str) else call
                                for call in msg["tool_calls"]
                            ]
                columns[key] = rows
            elif hasattr(array, "flatten") and hasattr(array.type, "value_type"):
                values = array.flatten().to_numpy(zero_copy_only=False)
                columns[key] = values.reshape(len(array), -1)
            else:
                columns[key] = array.to_numpy(zero_copy_only=False)
        return columns

    def _decode_videos(
        self,
        plans: list[dict],
        columns: dict[str, np.ndarray],
        row_pos: dict[int, int],
        prepared: dict[tuple, tuple | _MiniFile],
    ) -> list[dict[str, torch.Tensor]]:
        """Decode all camera frames a batch needs, one blob fetch + one decode pass per file."""
        requests = self._build_video_requests(plans, columns["timestamp"], row_pos)
        mini = {k: v for k, v in prepared.items() if isinstance(v, _MiniFile)}
        legacy = {k: v for k, v in prepared.items() if k not in mini}
        mini_indices = {}
        for key, entry in mini.items():
            idx = entry.index
            origin = int(idx.pts[idx.order[0]]) / idx.timescale
            mini_indices[key] = [
                [round((ts - origin) * idx.fps) for ts in timestamps] for _, timestamps in requests[key]
            ]
        spans = self._prefetch_mini(
            mini, {k: {i for row in rows for i in row} for k, rows in mini_indices.items()}
        )
        entries = self._ensure_decoders({k: requests[k] for k in legacy}, legacy)
        results: list[dict[str, torch.Tensor]] = [{} for _ in plans]

        def decode_file(file_key, file_requests):
            if file_key in mini:
                return self._decode_mini_file(
                    file_key, file_requests, mini[file_key], spans[file_key], mini_indices[file_key]
                )
            decoder, source = entries[file_key]
            if file_key[0] in self._depth_keys:
                return self._decode_depth_file_requests(source, file_requests, file_key)
            return self._decode_rgb_file_requests(decoder, file_requests, file_key)

        futures = {
            k: self._decode_pool.submit(decode_file, k, r)
            for k, r in requests.items()
            if k not in mini or k in spans
        }
        retry = {k: requests[k] for k in mini if k not in spans}
        for key, future in futures.items():
            try:
                decoded = future.result()
            except Exception:
                if key not in mini:
                    raise
                retry[key] = requests[key]
                continue
            for sample_idx, frames in decoded:
                results[sample_idx][key[0]] = frames
        if retry:
            self._mini_ineligible.update(retry)
            for key in retry:
                mini.pop(key)
            entries.update(self._ensure_decoders(retry, self._prepare_files(list(retry))))
            futures = {key: self._decode_pool.submit(decode_file, key, req) for key, req in retry.items()}
            for key, future in futures.items():
                for sample_idx, frames in future.result():
                    results[sample_idx][key[0]] = frames
        return results

    def _decode_mini_file(self, key, requests, entry, spans, indices):
        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

        # Approximate ordinal seeks use the average frame rate. Across a hole in
        # decode time they can land on an entirely different picture. Keep adjacent
        # GOPs in one container and split only where samples were omitted.
        groups = []
        for span in spans:
            if not groups or groups[-1][-1][1] + 1 != span[0]:
                groups.append([])
            groups[-1].append(span)
        index = entry.index
        needed = {int(index.order[i]): i for row in indices for i in row}
        locations = {}
        movies = []
        for group_id, group in enumerate(groups):
            movie, order = index.mini_mp4(group, entry.source.cached_span)
            movies.append(movie)
            for pos, sample in enumerate(order):
                if sample in needed:
                    locations[needed[sample]] = (group_id, pos)

        @lru_cache(maxsize=1)
        def open_group(group_id):
            return VideoDecoder(movies[group_id], seek_mode="approximate")

        def decode(indices):
            by_group = defaultdict(list)
            for output_pos, i in enumerate(indices):
                group, local = locations[i]
                by_group[group].append((output_pos, local))
            for part, (group, positions) in enumerate(by_group.items()):
                output, local = zip(*positions, strict=True)
                batch = open_group(group).get_frames_at(indices=list(local))
                expected = index.pts[index.order[[indices[pos] for pos in output]]]
                if not np.array_equal(np.rint(batch.pts_seconds.cpu().numpy() * index.timescale), expected):
                    raise ValueError("Mini-MP4 decoded a different sample PTS")
                if len(by_group) == 1:
                    return batch.data, batch.pts_seconds
                if part == 0:
                    frames = batch.data.new_empty((len(indices), *batch.data.shape[1:]))
                    pts = batch.pts_seconds.new_empty(len(indices))
                frames[list(output)] = batch.data
                pts[list(output)] = batch.pts_seconds
            return frames, pts

        return self._decode_rgb_file_requests(None, requests, key, indices, decode)

    def _decode_rgb_file_requests(
        self,
        decoder,
        file_requests: list[tuple[int, list[float]]],
        file_key: tuple,
        request_indices=None,
        decode_mini=None,
    ) -> list[tuple[int, torch.Tensor]]:
        """Decode shared frame indices once, checking PTS against every original query."""
        if request_indices is None:
            fps = decoder.metadata.average_fps
            request_indices = [[round(ts * fps) for ts in timestamps] for _, timestamps in file_requests]

        def decode(indices):
            if decode_mini is None:
                batch = decoder.get_frames_at(indices=indices)
                frames, pts = batch.data, batch.pts_seconds
            else:
                frames, pts = decode_mini(indices)
            return frames if self.return_uint8 else (frames / 255.0).float(), pts

        def finish(request, frames, pts):
            sample_idx, timestamps = request
            query = torch.tensor(timestamps, dtype=torch.float64, device=pts.device)
            if not ((query - pts).abs() < self.tolerance_s).all():
                key, chunk, file = file_key
                raise FrameTimestampError(
                    f"Query timestamps violate tolerance_s={self.tolerance_s} for video "
                    f"'{key}' (chunk {chunk}, file {file}): queried {timestamps}, loaded {pts.tolist()}."
                )
            return sample_idx, frames.squeeze(0)

        if len(file_requests) == 1 and decode_mini is None:
            return [finish(file_requests[0], *decode(request_indices[0]))]

        unique = sorted({idx for indices in request_indices for idx in indices})
        limit = _decode_frame_limit(
            self.meta.features[file_key[0]].get("shape"),
            max_frames=_RGB_DECODE_MAX_FRAMES,
            max_bytes=_RGB_DECODE_MAX_BYTES,
            bytes_per_value=1 if self.return_uint8 else 5,  # uint8 input plus float32 output
        )
        if len(unique) <= limit:
            frames, pts = decode(unique)
            positions = {index: pos for pos, index in enumerate(unique)}
            decoded = []
            for request, indices in zip(file_requests, request_indices, strict=True):
                take = [positions[index] for index in indices]
                decoded.append(finish(request, frames[take], pts[take]))
            return decoded

        # Scatter bounded decode chunks directly into the final sample tensors.
        # Retaining all decoded chunks until gather would defeat the byte limit.
        for start in range(0, len(unique), limit):
            chunk = unique[start : start + limit]
            frames, pts = decode(chunk)
            if start == 0:
                sample_frames = [frames.new_empty((len(req), *frames.shape[1:])) for req in request_indices]
                sample_pts = [pts.new_empty(len(req)) for req in request_indices]
            positions = {index: pos for pos, index in enumerate(chunk)}
            for sample, indices in enumerate(request_indices):
                target = [pos for pos, index in enumerate(indices) if index in positions]
                if target:
                    take = [positions[indices[pos]] for pos in target]
                    sample_frames[sample][target] = frames[take]
                    sample_pts[sample][target] = pts[take]
            del frames, pts
        return [finish(*args) for args in zip(file_requests, sample_frames, sample_pts, strict=True)]

    def _build_video_requests(
        self,
        plans: list[dict],
        timestamps: np.ndarray,
        row_pos: dict[int, int],
    ) -> dict[tuple, list[tuple[int, list[float]]]]:
        requests: dict[tuple, list[tuple[int, list[float]]]] = defaultdict(list)
        for sample_idx, plan in enumerate(plans):
            ep_idx = plan["ep_idx"]
            for key in self._video_keys:
                window = plan["windows"].get(key, [plan["abs_idx"]])
                requests[self._video_file_key(key, ep_idx)].append(
                    (sample_idx, self._shifted_video_timestamps(key, ep_idx, window, timestamps, row_pos))
                )
        return requests

    def _prepare_files(
        self, file_keys: list[tuple], windows: dict[tuple, list[tuple[int, int]]] | None = None
    ) -> dict[tuple, tuple | _MiniFile]:
        """Stage 1 of video decoding: everything that doesn't need timestamps.

        Loads byte-index metadata, fetches container head/moov/first-packet ranges
        for uncached files, creates their decoders, and prefetches the batch's
        frame-window byte ranges. Timestamp-independent, so it runs in a background thread.
        """
        # Lazy: torchcodec is a platform-conditional extra (no wheel on some
        # platforms); upstream video_utils imports it lazily for the same reason.
        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

        self._load_file_meta([key for key in file_keys if key not in self._file_meta])

        prepared: dict[tuple, tuple] = {}
        new_files = []
        direct = {}
        for key in file_keys:
            if key in self._decoder_cache:
                prepared[key] = self._decoder_cache.get(key)
            elif (path := self._file_meta[key]["local_path"]) is not None:
                direct[key] = self._decode_pool.submit(VideoDecoder, str(path), seek_mode="approximate")
            else:
                new_files.append(key)
        mini = {}
        if windows and new_files and self._has_sample_index:
            candidates = [
                key
                for key in new_files
                if key[0] not in self._depth_keys and key not in self._mini_ineligible
            ]
            mini = self._load_mini_files(candidates)
            planned = {
                key: (i for first, last in windows[key] for i in range(first, last + 1)) for key in mini
            }
            ready = self._prefetch_mini(mini, planned)
            mini = {key: mini[key] for key in ready}
            new_files = [key for key in new_files if key not in mini]

        for key, future in direct.items():
            try:
                prepared[key] = (future.result(), None)
            except Exception:
                # The path can disappear or the native file opener can reject it.
                new_files.append(key)

        sources = {key: source for key, (_, source) in prepared.items()}
        spans_by_key: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        if new_files:
            handles = self._videos_table.fetch_blob_files(
                VIDEO_BLOB_COLUMN, [self._video_row_ids[key] for key in new_files]
            )
            for key, handle in zip(new_files, handles, strict=True):
                meta = self._file_meta[key]
                size = meta["file_size"]
                sources[key] = _SparseBlobSource(size, handle)
                slack = _range_slack(meta, key[0] in self._depth_keys)
                spans_by_key[key].extend(
                    [
                        (0, min(_OPEN_PROBE_BYTES, size)),
                        (meta["moov_offset"], min(meta["moov_offset"] + meta["moov_size"] + slack, size)),
                    ]
                )
                if len(meta["kf_positions"]):
                    first_packet = int(meta["kf_positions"][0])
                    spans_by_key[key].append((first_packet, min(first_packet + _OPEN_PROBE_BYTES, size)))

        for key, frame_windows in (windows or {}).items():
            if key in mini or sources[key] is None:
                continue
            for first, last in frame_windows:
                span = self._window_byte_range(key[0], self._file_meta[key], first, last)
                if not sources[key].covers(*span):
                    spans_by_key[key].append(span)
        if spans_by_key:
            self._fetch_spans(spans_by_key, sources)

        def open_decoder(key):
            source = sources[key]
            decoder = None
            if key[0] not in self._depth_keys:
                decoder = VideoDecoder(source, seek_mode="approximate")
            return key, (decoder, source)

        prepared.update(self._decode_pool.map(open_decoder, new_files))
        return prepared | mini

    def _load_mini_files(self, keys):
        missing = [key for key in keys if key not in self._index_cache]
        indices = {key: self._index_cache.get(key) for key in keys if key in self._index_cache}
        if missing:
            batch = (
                self._videos_table.take_row_ids([self._video_row_ids[k] for k in missing])
                .select(["sample_index"])
                .to_arrow()
            )
            for key, scalar in zip(missing, batch.column("sample_index"), strict=True):
                try:
                    idx = ParsedSampleIndex.parse(scalar.as_py() or b"")
                    if int(idx.offsets[-1]) + int(idx.sizes[-1]) > self._file_meta[key]["file_size"]:
                        raise ValueError("Sample outside video blob")
                except ValueError:
                    self._mini_ineligible.add(key)
                    continue
                indices[key] = idx
                self._index_cache.put(key, idx, idx.nbytes)
        files = {}
        for key, idx in indices.items():
            cache_key = ("mini", *key)
            source = (
                self._decoder_cache.get(cache_key)
                if cache_key in self._decoder_cache
                else _SparseBlobSource(self._file_meta[key]["file_size"], None)
            )
            files[key] = _MiniFile(idx, source)
        return files

    def _prefetch_mini(self, files, indices):
        plans = {}
        requests = []
        targets = []
        for key, entry in files.items():
            try:
                spans = entry.index.spans(indices[key])
            except ValueError:
                self._mini_ineligible.add(key)
                continue
            plans[key] = spans
            ranges = [entry.index.byte_span(*span) for span in spans]
            missing = [gap for span in ranges for gap in entry.source.missing(*span)]
            for first, last in _merge_spans(missing, 64 << 10):
                for start, end in entry.source.missing(first, last):
                    requests.append((self._video_row_ids[key], start, end - start))
                    targets.append((key, start, end))
        if requests:
            try:
                blobs = self._videos_table.fetch_blob_ranges(VIDEO_BLOB_COLUMN, requests)
                for (key, start, end), blob in zip(targets, blobs, strict=True):
                    data = blob.as_py()
                    if len(data) != end - start:
                        raise ValueError("Truncated sample range")
                    files[key].source.add(start, data)
            except Exception:
                for key, _, _ in targets:
                    plans.pop(key, None)
                    self._mini_ineligible.add(key)
        for key in plans:
            source = files[key].source
            self._decoder_cache.put(("mini", *key), source, source.buffered)
        return plans

    def _video_file_key(self, key: str, ep_idx: int) -> tuple[str, int, int]:
        chunk_arr, file_arr, _ = self._video_locator[key]
        return key, int(chunk_arr[ep_idx]), int(file_arr[ep_idx])

    def _shifted_video_timestamps(
        self,
        key: str,
        ep_idx: int,
        window: list[int],
        timestamps: np.ndarray,
        row_pos: dict[int, int],
    ) -> list[float]:
        _, _, from_ts_arr = self._video_locator[key]
        base = float(from_ts_arr[ep_idx])
        return [base + float(timestamps[row_pos[row]]) for row in window]

    def _ensure_decoders(self, requests: dict, prepared: dict[tuple, tuple]) -> dict:
        """Stage 2 of video decoding: fetch this batch's frame windows."""
        decoders = {key: decoder for key, (decoder, _) in prepared.items()}
        sources = {key: source for key, (_, source) in prepared.items()}

        spans_by_key: dict[tuple, list[tuple[int, int]]] = {}
        fps = float(self.meta.fps)
        for key, file_requests in requests.items():
            meta = self._file_meta[key]
            source = sources[key]
            if source is None:
                continue
            spans = []
            reach_back = 1 if key[0] in self._depth_keys else 0
            for _, shifted_ts in file_requests:
                first = round(min(shifted_ts) * fps)
                last = round(max(shifted_ts) * fps)
                start, end = self._window_byte_range(key[0], meta, first - reach_back, last)
                if not source.covers(start, end):
                    spans.append((start, end))
            spans_by_key[key] = spans
        if spans_by_key:
            self._fetch_spans(spans_by_key, sources)

        for key in requests:
            source = sources[key]
            # Re-account each batch as sources grow, so the byte bounds memory
            decoder = decoders[key]
            if decoder is not None:
                height = decoder.metadata.height or 0
                width = decoder.metadata.width or 0
            else:  # depth: no torchcodec decoder. size from the feature shape
                height, width = (tuple(self.meta.features[key[0]].get("shape") or (0, 0)) + (0, 0))[:2]
            context_cost = (8 << 20) + 8 * height * width
            buffered = source.buffered if source is not None else 0
            self._decoder_cache.put(key, (decoder, source), nbytes=buffered + context_cost)
        return prepared

    def _decode_depth_frames(self, source, shifted_ts: list[float], file_key: tuple) -> torch.Tensor:
        """Decode and dequantize to (N, C, H, W), preserving timestamp order and the time axis."""
        source.seek(0)
        frames = decode_video_frames_pyav(
            source, shifted_ts, self.tolerance_s, return_uint8=False, is_depth=True
        )
        config = self._depth_encoder_configs[file_key[0]]
        return dequantize_depth(
            frames,
            depth_min=config.depth_min,
            depth_max=config.depth_max,
            shift=config.shift,
            use_log=config.use_log,
            output_unit=self._depth_output_unit,
        )

    def _decode_depth_file_requests(
        self, source, file_requests: list[tuple[int, list[float]]], file_key: tuple
    ) -> list[tuple[int, torch.Tensor]]:
        """Merge nearby sample windows so each depth span opens and seeks PyAV once."""
        if len(file_requests) == 1:
            sample_idx, shifted_ts = file_requests[0]
            return [(sample_idx, self._decode_depth_frames(source, shifted_ts, file_key).squeeze(0))]

        fps = float(self.meta.fps)
        max_span_frames = _decode_frame_limit(
            self.meta.features[file_key[0]].get("shape"),
            max_frames=_DEPTH_DECODE_MAX_SPAN_FRAMES,
            max_bytes=_DEPTH_DECODE_MAX_BYTES,
            bytes_per_value=_DEPTH_DECODE_BYTES_PER_VALUE,
        )
        merge_gap_s = 1.0 / fps + self.tolerance_s

        ordered = sorted(file_requests, key=lambda request: min(request[1]))
        groups: list[list[tuple[int, list[float]]]] = []
        group: list[tuple[int, list[float]]] = []
        group_start = group_end = 0.0
        for request in ordered:
            start, end = min(request[1]), max(request[1])
            merged_end = max(group_end, end)
            merged_span_frames = int(np.ceil((merged_end - group_start) * fps)) + 1
            if group and start <= group_end + merge_gap_s and merged_span_frames <= max_span_frames:
                group.append(request)
                group_end = merged_end
            else:
                if group:
                    groups.append(group)
                group = [request]
                group_start, group_end = start, end
        groups.append(group)

        decoded: list[tuple[int, torch.Tensor]] = []
        for group in groups:
            if len(group) == 1:
                sample_idx, timestamps = group[0]
                decoded.append(
                    (sample_idx, self._decode_depth_frames(source, timestamps, file_key).squeeze(0))
                )
                continue
            unique_ts = sorted({timestamp for _, timestamps in group for timestamp in timestamps})
            frames = self._decode_depth_frames(source, unique_ts, file_key)
            positions = {timestamp: position for position, timestamp in enumerate(unique_ts)}
            for sample_idx, timestamps in group:
                decoded.append(
                    (sample_idx, frames[[positions[timestamp] for timestamp in timestamps]].squeeze(0))
                )
        return decoded

    def _fetch_spans(
        self, spans_by_key: dict[tuple, list[tuple[int, int]]], sources: dict[tuple, _SparseBlobSource]
    ) -> None:
        range_requests: list[tuple[int, int, int]] = []
        range_targets: list[tuple[tuple, int]] = []
        for key, spans in spans_by_key.items():
            for start, end in _merge_spans(
                spans, _range_slack(self._file_meta[key], key[0] in self._depth_keys)
            ):
                range_requests.append((self._video_row_ids[key], start, end - start))
                range_targets.append((key, start))
        if not range_requests:
            return
        payloads = self._videos_table.fetch_blob_ranges(VIDEO_BLOB_COLUMN, range_requests)
        for (key, offset), payload in zip(range_targets, payloads, strict=True):
            sources[key].add(offset, payload.as_py())

    def _load_file_meta(self, missing: list[tuple]) -> None:
        """Fetch byte-index columns for files not yet in the per-worker cache."""
        if not missing:
            return
        row_ids = [self._video_row_ids[file_key] for file_key in missing]
        batch = (
            self._videos_table.take_row_ids(row_ids)
            .select(["video_key", "chunk_index", "file_index", *VIDEO_INDEX_COLUMNS, VIDEO_BLOB_COLUMN])
            .to_arrow()
        )
        if batch.num_rows != len(missing):
            raise ValueError(
                "videos table is missing byte-index columns or rows "
                f"({batch.num_rows} matches for {len(missing)} files). Re-convert the dataset "
                f"with a converter that writes {VIDEO_INDEX_COLUMNS}."
            )
        scalars = {
            name: batch.column(name).to_pylist()
            for name in ("video_key", "chunk_index", "file_index", "file_size", "moov_offset", "moov_size")
        }
        kf_index_column = batch.column("kf_indices").combine_chunks()
        kf_position_column = batch.column("kf_positions").combine_chunks()
        index_offsets = kf_index_column.offsets.to_numpy(zero_copy_only=False)
        index_values = kf_index_column.values.to_numpy(zero_copy_only=False)
        position_offsets = kf_position_column.offsets.to_numpy(zero_copy_only=False)
        position_values = kf_position_column.values.to_numpy(zero_copy_only=False)
        descriptors = batch.column(VIDEO_BLOB_COLUMN)
        has_descriptors = pa.types.is_struct(descriptors.type)
        for i in range(batch.num_rows):
            file_key = (scalars["video_key"][i], scalars["chunk_index"][i], scalars["file_index"][i])
            local_path = None
            if has_descriptors and file_key[0] not in self._depth_keys:
                local_path = _local_video_path(descriptors[i].as_py(), scalars["file_size"][i])
            self._file_meta[file_key] = {
                "local_path": local_path,
                "file_size": scalars["file_size"][i],
                "moov_offset": scalars["moov_offset"][i],
                "moov_size": scalars["moov_size"][i],
                "kf_indices": index_values[index_offsets[i] : index_offsets[i + 1]],
                "kf_positions": position_values[position_offsets[i] : position_offsets[i + 1]],
            }
        while len(self._file_meta) > 2048:
            self._file_meta.popitem(last=False)

    def _window_byte_range(self, key: str, meta: dict, first_frame: int, last_frame: int) -> tuple[int, int]:
        """Byte range covering frames [first, last]: preceding keyframe to next keyframe."""
        kf_indices, kf_positions = meta["kf_indices"], meta["kf_positions"]
        start_idx = max(int(np.searchsorted(kf_indices, first_frame, side="right")) - 1, 0)
        end_idx = int(np.searchsorted(kf_indices, last_frame, side="right"))
        end = int(kf_positions[end_idx]) if end_idx < len(kf_positions) else meta["file_size"]
        slack = _range_slack(meta, key in self._depth_keys)
        # TorchCodec fills a 64 KiB AVIO buffer after a seek. A small GOP needs
        # one whole read from its keyframe, including when it ends a wider window.
        last_gop_start = int(kf_positions[max(0, end_idx - 1)])
        padded_end = max(end + slack, last_gop_start + (64 << 10))
        return int(kf_positions[start_idx]), min(padded_end, meta["file_size"])

    def _build_item(self, plan: dict, columns: dict[str, np.ndarray], row_pos: dict[int, int]) -> dict:
        base = row_pos[plan["abs_idx"]]
        item = {}
        for key in self._output_tabular_keys:
            data = columns[key]
            shape = self._feature_shapes[key]
            if key in self._language_keys:
                item[key] = data[base]
            elif key in self._string_keys:
                value = data[base]
                item[key] = value if isinstance(value, str) or value is None else str(value)
            elif key in plan["windows"]:
                window = data[[row_pos[row] for row in plan["windows"][key]]]
                if len(shape) > 1:
                    window = window.reshape(len(window), *shape)
                item[key] = torch.from_numpy(np.ascontiguousarray(window))
            elif data.ndim > 1:
                value = data[base]
                if len(shape) > 1:
                    value = value.reshape(shape)
                item[key] = torch.from_numpy(value.copy())
            else:
                item[key] = torch.tensor(data[base])
        item.update(plan["padding"])
        if self._include_task:
            item["task"] = self._task_names[int(columns["task_index"][base])]
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_id={self.repo_id},\n"
            f"  uri={self._db_uri},\n"
            f"  episodes={self.num_episodes} selected / {self.meta.total_episodes} total,\n"
            f"  frames={self.num_frames},\n"
            f")"
        )
