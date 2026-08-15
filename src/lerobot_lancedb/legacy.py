#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LeRobot v2.0 / v2.1 source layout helpers.

The Lance reader (and current ``LeRobotDatasetMetadata``) speak v3.0 metadata:
parquet episode records, ``tasks.parquet``, ``stats.json``, and
``(video_key, chunk_index, file_index)`` locators. v2.x datasets instead keep
one parquet/mp4 per episode and jsonl metadata. Conversion does **not**
re-chunk or re-encode those files — each episode stays its own videos-table
row, with ``from_timestamp=0`` — but the output ``meta/`` is rewritten so the
loader can open the result.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

V3 = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
DEFAULT_V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_V2_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_V2_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

_EPISODE_STEM = re.compile(r"episode[_-](\d+)$", re.IGNORECASE)
_FILE_STEM = re.compile(r"file[_-](\d+)$", re.IGNORECASE)
_CHUNK_DIR = re.compile(r"chunk-(\d+)$", re.IGNORECASE)
_TRAILING_INT = re.compile(r"(\d+)$")


def read_info(root: Path) -> dict[str, Any]:
    info_file = Path(root) / "meta" / "info.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"{root} does not look like a LeRobot dataset (no meta/info.json)")
    return json.loads(info_file.read_text())


def normalize_codebase_version(raw: str | None) -> str:
    """``v2.1`` / ``2.1.0`` / ``v3.0`` → ``vMAJOR.MINOR``."""
    if not raw:
        return V3
    text = str(raw).strip()
    if text.startswith(("v", "V")):
        text = text[1:]
    parts = text.split(".")
    major = parts[0] if parts and parts[0].isdigit() else "3"
    minor = parts[1] if len(parts) > 1 and parts[1].isdigit() else "0"
    return f"v{major}.{minor}"


def codebase_version(root: Path) -> str:
    return normalize_codebase_version(read_info(root).get("codebase_version"))


def is_v2(version: str) -> bool:
    return version.startswith("v2.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trailing_int(name: str) -> int:
    match = _TRAILING_INT.search(name)
    if match is None:
        raise ValueError(f"no trailing integer in {name!r}")
    return int(match.group(1))


def parse_video_locator(mp4: Path, videos_root: Path) -> tuple[str, int, int]:
    """``(video_key, chunk_index, file_index)`` for a source mp4.

    v3.0: ``videos/{video_key}/chunk-XXX/file-YYY.mp4``
    v2.x: ``videos/chunk-XXX/{video_key}/episode_YYYYYY.mp4``
          (or ``videos/{video_key}/chunk-XXX/episode_YYYYYY.mp4``)
    """
    rel = Path(mp4).resolve().relative_to(Path(videos_root).resolve())
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"video path is not a LeRobot layout: {mp4}")
    stem = Path(mp4).stem
    file_index = trailing_int(stem)
    if _FILE_STEM.search(stem):
        # v3: videos/{key}/chunk-000/file-000.mp4
        video_key = "/".join(parts[:-2])
        chunk_index = trailing_int(parts[-2])
        return video_key, chunk_index, file_index
    if _EPISODE_STEM.search(stem):
        if _CHUNK_DIR.search(parts[0]):
            chunk_index = trailing_int(parts[0])
            video_key = "/".join(parts[1:-1])
        elif _CHUNK_DIR.search(parts[-2]):
            video_key = "/".join(parts[:-2])
            chunk_index = trailing_int(parts[-2])
        else:
            raise ValueError(f"cannot find chunk directory in v2 video path: {mp4}")
        return video_key, chunk_index, file_index
    raise ValueError(f"unrecognized video filename (expected file-* or episode_*): {mp4}")


def data_parquet_files(src_root: Path) -> list[Path]:
    """Parquet files under ``data/``, in the order the frames table must be written.

    v2.x ``episode_NNNNNN.parquet`` files sort by episode index; v3.0
    ``file-XXX.parquet`` files sort by (chunk, file). Both layouts write frames
    in ascending global ``index``, so this order is what ``_frames_reader``
    verifies.
    """
    files = [p for p in (Path(src_root) / "data").rglob("*.parquet") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"no parquet files under {Path(src_root) / 'data'}")

    def sort_key(path: Path) -> tuple:
        episode = _EPISODE_STEM.search(path.stem)
        if episode:
            return (0, int(episode.group(1)))
        chunk = _CHUNK_DIR.search(path.parent.name)
        file_m = _FILE_STEM.search(path.stem)
        return (
            1,
            int(chunk.group(1)) if chunk else 0,
            int(file_m.group(1)) if file_m else 0,
            str(path),
        )

    return sorted(files, key=sort_key)


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    return [key for key, spec in features.items() if spec.get("dtype") == "video"]


def _format_v2_path(template: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    episode_chunk = episode_index // chunks_size
    kwargs: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_chunk": episode_chunk,
        "chunk_index": episode_chunk,
        "file_index": episode_index,
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        fallback = DEFAULT_V2_VIDEO_PATH if video_key is not None else DEFAULT_V2_DATA_PATH
        return fallback.format(**kwargs)


def v2_data_path(src_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    template = info.get("data_path") or DEFAULT_V2_DATA_PATH
    return Path(src_root) / _format_v2_path(template, episode_index, chunks_size)


def v2_video_path(src_root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunks_size = int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    template = info.get("video_path") or DEFAULT_V2_VIDEO_PATH
    return Path(src_root) / _format_v2_path(template, episode_index, chunks_size, video_key=video_key)


def load_v2_episodes(src_root: Path) -> list[dict[str, Any]]:
    path = Path(src_root) / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"v2 dataset is missing {path}")
    episodes = sorted(load_jsonl(path), key=lambda row: int(row["episode_index"]))
    if not episodes:
        raise ValueError(f"{path} is empty")
    return episodes


def load_v2_tasks(src_root: Path, episodes: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return ``(task_string, task_index)`` rows, synthesizing from episodes if needed."""
    path = Path(src_root) / "meta" / "tasks.jsonl"
    if path.is_file():
        rows = sorted(load_jsonl(path), key=lambda row: int(row["task_index"]))
        return [(str(row["task"]), int(row["task_index"])) for row in rows]
    seen: dict[str, int] = {}
    for episode in episodes:
        for task in _episode_tasks(episode):
            if task not in seen:
                seen[task] = len(seen)
    if not seen:
        seen[""] = 0
    return [(task, index) for task, index in seen.items()]


def _episode_tasks(episode: dict[str, Any]) -> list[str]:
    if "tasks" in episode:
        tasks = episode["tasks"]
        if isinstance(tasks, str):
            return [tasks]
        return [str(t) for t in tasks]
    if "task" in episode and episode["task"] is not None:
        return [str(episode["task"])]
    return []


def load_v2_episodes_stats(src_root: Path) -> dict[int, dict[str, Any]]:
    path = Path(src_root) / "meta" / "episodes_stats.jsonl"
    if not path.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        out[int(row["episode_index"])] = row.get("stats") or {}
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _numpy_feature_stats(stats: dict[str, Any], count: int | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in stats.items():
        array = np.atleast_1d(np.array(value))
        out[key] = array
    if "count" not in out and count is not None:
        out["count"] = np.atleast_1d(np.array([count]))
    return out


def _aggregate_stats(stats_list: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    try:
        from lerobot.datasets.compute_stats import aggregate_stats

        return aggregate_stats(stats_list)
    except Exception:
        pass
    keys = {key for stats in stats_list for key in stats}
    aggregated: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        parts = [stats[key] for stats in stats_list if key in stats]
        means = np.stack([p["mean"] for p in parts])
        variances = np.stack([p["std"] ** 2 for p in parts])
        counts = np.stack([p.get("count", np.array([1])) for p in parts]).astype(np.float64)
        total = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        weighted = (means * counts).sum(axis=0) / total
        delta = means - weighted
        variance = ((variances + delta**2) * counts).sum(axis=0) / total
        aggregated[key] = {
            "min": np.min(np.stack([p["min"] for p in parts]), axis=0),
            "max": np.max(np.stack([p["max"] for p in parts]), axis=0),
            "mean": weighted,
            "std": np.sqrt(np.maximum(variance, 0)),
            "count": np.atleast_1d(total),
        }
    return aggregated


def _write_stats(src_root: Path, out_meta: Path, episode_lengths: dict[int, int]) -> None:
    src_stats = Path(src_root) / "meta" / "stats.json"
    per_episode = load_v2_episodes_stats(src_root)
    if per_episode:
        prepared = []
        for episode_index, stats in sorted(per_episode.items()):
            length = episode_lengths.get(episode_index)
            prepared.append(
                {feat: _numpy_feature_stats(feat_stats, count=length) for feat, feat_stats in stats.items()}
            )
        try:
            aggregated = _aggregate_stats(prepared)
            (out_meta / "stats.json").write_text(json.dumps(_jsonable(aggregated), indent=4) + "\n")
            return
        except Exception as err:
            print(f"  warning: could not aggregate v2.1 episode stats ({err})")
    if src_stats.is_file():
        shutil.copy2(src_stats, out_meta / "stats.json")


def _write_tasks_parquet(out_meta: Path, tasks: list[tuple[str, int]]) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {"task_index": [index for _, index in tasks]},
        index=pd.Index([task for task, _ in tasks], name="task"),
    )
    frame.to_parquet(out_meta / "tasks.parquet")


def _write_info_v3(info: dict[str, Any], out_meta: Path) -> None:
    new_info = dict(info)
    new_info["codebase_version"] = V3
    new_info.pop("total_chunks", None)
    new_info.pop("total_videos", None)
    new_info["data_files_size_in_mb"] = int(
        new_info.get("data_files_size_in_mb") or DEFAULT_DATA_FILE_SIZE_IN_MB
    )
    new_info["video_files_size_in_mb"] = int(
        new_info.get("video_files_size_in_mb") or DEFAULT_VIDEO_FILE_SIZE_IN_MB
    )
    new_info["data_path"] = DEFAULT_V3_DATA_PATH
    if info.get("video_path"):
        new_info["video_path"] = DEFAULT_V3_VIDEO_PATH
    else:
        new_info["video_path"] = None
    fps = int(info["fps"])
    new_info["fps"] = fps
    features = copy.deepcopy(new_info.get("features") or {})
    for spec in features.values():
        if spec.get("dtype") != "video":
            spec.setdefault("fps", fps)
    new_info["features"] = features
    new_info.setdefault("chunks_size", int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE))
    (out_meta / "info.json").write_text(json.dumps(new_info, indent=4) + "\n")


def _locator_for_video(
    src_root: Path, info: dict[str, Any], episode_index: int, video_key: str
) -> tuple[int, int]:
    path = v2_video_path(src_root, info, episode_index, video_key)
    videos_root = Path(src_root) / "videos"
    if path.is_file() and videos_root.is_dir():
        _, chunk_index, file_index = parse_video_locator(path, videos_root)
        return chunk_index, file_index
    chunks_size = int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    return episode_index // chunks_size, episode_index


def _locator_for_data(src_root: Path, info: dict[str, Any], episode_index: int) -> tuple[int, int]:
    path = v2_data_path(src_root, info, episode_index)
    chunks_size = int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    if path.is_file():
        chunk = _CHUNK_DIR.search(path.parent.name)
        episode = _EPISODE_STEM.search(path.stem)
        if chunk and episode:
            return int(chunk.group(1)), int(episode.group(1))
    return episode_index // chunks_size, episode_index


def write_v3_meta_from_v2(src_root: Path, out_meta: Path) -> str:
    """Rewrite v2.x jsonl metadata into the v3.0 files ``LanceDBDataset`` loads.

    Returns the source ``codebase_version`` (already normalized). Does not
    concatenate parquet/mp4 files: each episode keeps its own ``file_index``.
    """
    src_root = Path(src_root)
    out_meta = Path(out_meta)
    info = read_info(src_root)
    version = normalize_codebase_version(info.get("codebase_version"))
    if not is_v2(version):
        raise ValueError(f"write_v3_meta_from_v2 expected a v2.x dataset, got {version}")

    episodes = load_v2_episodes(src_root)
    tasks = load_v2_tasks(src_root, episodes)
    keys = video_keys_from_info(info)
    fps = float(info["fps"])

    out_meta.mkdir(parents=True, exist_ok=True)
    _write_info_v3(info, out_meta)
    _write_tasks_parquet(out_meta, tasks)

    rows: list[dict[str, Any]] = []
    cursor = 0
    lengths: dict[int, int] = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        data_path = v2_data_path(src_root, info, episode_index)
        if data_path.is_file():
            length = int(pq.read_metadata(data_path).num_rows)
        else:
            length = int(episode.get("length") or 0)
        lengths[episode_index] = length
        data_chunk, data_file = _locator_for_data(src_root, info, episode_index)
        duration_s = length / fps if fps else 0.0
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": _episode_tasks(episode),
            "length": length,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + length,
            "data/chunk_index": data_chunk,
            "data/file_index": data_file,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for key in keys:
            chunk_index, file_index = _locator_for_video(src_root, info, episode_index, key)
            row[f"videos/{key}/chunk_index"] = chunk_index
            row[f"videos/{key}/file_index"] = file_index
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = duration_s
        rows.append(row)
        cursor += length

    episodes_dir = out_meta / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), episodes_dir / "file-000.parquet")

    _write_stats(src_root, out_meta, lengths)

    declared = int(info.get("total_frames") or cursor)
    if cursor != declared:
        print(f"  warning: episode lengths sum to {cursor} frames, info.json declares {declared}")

    return version


@dataclass
class SourceAudit:
    """Read-only view of an upstream (parquet/mp4) dataset for ``lerobot-lance-doctor``."""

    version: str
    total_episodes: int
    total_frames: int
    fps: float
    video_keys: list[str]
    starts: list[int]
    ends: list[int]
    data_files: list[Path]
    video_end_ts: dict[Path, float]


def load_source_audit(root: Path, repo_id: str | None = None) -> SourceAudit:
    """Describe a v2.x or v3.x upstream dataset without converting it."""
    root = Path(root)
    version = codebase_version(root)
    if is_v2(version):
        return _v2_source_audit(root, version)
    return _v3_source_audit(root, repo_id, version)


def _v2_source_audit(root: Path, version: str) -> SourceAudit:
    info = read_info(root)
    episodes = load_v2_episodes(root)
    fps = float(info["fps"])
    keys = video_keys_from_info(info)
    starts: list[int] = []
    ends: list[int] = []
    data_files: list[Path] = []
    video_end_ts: dict[Path, float] = {}
    cursor = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        path = v2_data_path(root, info, episode_index)
        data_files.append(path)
        if path.is_file():
            length = int(pq.read_metadata(path).num_rows)
        else:
            length = int(episode.get("length") or 0)
        starts.append(cursor)
        ends.append(cursor + length)
        duration_s = length / fps if fps else 0.0
        for key in keys:
            video_path = v2_video_path(root, info, episode_index, key)
            video_end_ts[video_path] = max(video_end_ts.get(video_path, 0.0), duration_s)
        cursor += length
    return SourceAudit(
        version=version,
        total_episodes=int(info.get("total_episodes") or len(episodes)),
        total_frames=int(info.get("total_frames") or cursor),
        fps=fps,
        video_keys=keys,
        starts=starts,
        ends=ends,
        data_files=data_files,
        video_end_ts=video_end_ts,
    )


def _v3_source_audit(root: Path, repo_id: str | None, version: str) -> SourceAudit:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id or str(root), root=root)
    starts = [int(x) for x in meta.episodes["dataset_from_index"]]
    ends = [int(x) for x in meta.episodes["dataset_to_index"]]
    data_files = [root / meta.get_data_file_path(ep) for ep in range(meta.total_episodes)]
    video_end_ts: dict[Path, float] = {}
    columns = meta.episodes.column_names
    for key in meta.video_keys:
        to_ts = (
            meta.episodes[f"videos/{key}/to_timestamp"] if f"videos/{key}/to_timestamp" in columns else None
        )
        from_ts = meta.episodes[f"videos/{key}/from_timestamp"]
        for ep in range(meta.total_episodes):
            path = root / meta.get_video_file_path(ep, key)
            end = (
                float(to_ts[ep])
                if to_ts is not None
                else float(from_ts[ep]) + (ends[ep] - starts[ep]) / meta.fps
            )
            video_end_ts[path] = max(video_end_ts.get(path, 0.0), end)
    return SourceAudit(
        version=version,
        total_episodes=meta.total_episodes,
        total_frames=meta.total_frames,
        fps=float(meta.fps),
        video_keys=list(meta.video_keys),
        starts=starts,
        ends=ends,
        data_files=data_files,
        video_end_ts=video_end_ts,
    )
