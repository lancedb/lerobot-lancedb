"""Prepare native LeRobot v3 fixtures with deterministic inter-frame video."""

from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# width, height, cameras, GOP, frames per episode. Each clip spans at least two GOPs.
PROFILES = {"droid-short-gop": (320, 180, 3, 2, 16), "long-gop-720p": (1280, 720, 2, 30, 62)}
FPS = 15


def write_clip(path, width, height, gop, length, camera, episode):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Moving texture makes adjacent timestamps and different cameras distinguishable.
    rng = np.random.default_rng(100 * episode + camera)
    texture = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    with av.open(str(path), "w", options={"movflags": "faststart"}) as output:
        stream = output.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.thread_count = 1
        stream.options = {"g": str(gop), "sc_threshold": "0", "bf": "0", "preset": "veryfast"}
        for frame in range(length):
            rgb = np.roll(texture, frame * 2, axis=1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def synthetic(root, profile, *, vector_width=0):
    width, height, cameras, gop, length = PROFILES[profile]
    videos = [f"observation.images.cam{i}" for i in range(cameras)]
    meta = LeRobotDatasetMetadata.create(
        repo_id=f"local/{profile}",
        root=root,
        fps=FPS,
        features={
            "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            **({"wide": {"dtype": "float32", "shape": (vector_width,), "names": None}} if vector_width else {}),
            **{
                k: {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
                for k in videos
            },
        },
    )
    meta.save_episode_tasks(["reach"])
    for ep in range(2):
        start = ep * length
        locations = {
            "data/chunk_index": 0,
            "data/file_index": ep,
            "dataset_from_index": start,
            "dataset_to_index": start + length,
        }
        for key in videos:
            locations.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": ep,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": length / FPS,
                }
            )
        meta.save_episode(
            episode_index=ep,
            episode_length=length,
            episode_tasks=["reach"],
            episode_stats={},
            episode_metadata=locations,
        )
    meta.finalize()
    for ep in range(2):
        states = np.arange(length * 7, dtype=np.float32).reshape(length, 7) + ep
        rows = pa.table(
            {
                "index": range(ep * length, (ep + 1) * length),
                "frame_index": range(length),
                "episode_index": [ep] * length,
                "task_index": [0] * length,
                "timestamp": pa.array(np.arange(length) / FPS, pa.float32()),
                "observation.state": pa.array(states.tolist(), pa.list_(pa.float32(), 7)),
                "action": pa.array((-states).tolist(), pa.list_(pa.float32(), 7)),
            }
        )
        if vector_width:
            values = pa.array(np.arange(length * vector_width, dtype=np.float32))
            rows = rows.append_column("wide", pa.FixedSizeListArray.from_arrays(values, vector_width))
        path = root / meta.get_data_file_path(ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rows, path)
        for camera, key in enumerate(videos):
            write_clip(root / meta.get_video_file_path(ep, key), width, height, gop, length, camera, ep)
    return root


def source_path(workload, scratch, path, repo_id):
    if path:
        return Path(path).expanduser().resolve()
    if workload in PROFILES and not repo_id:
        return synthetic(scratch / "source", workload)
    repo_id = repo_id or "lerobot/pusht"
    try:
        return Path(snapshot_download(repo_id, repo_type="dataset", local_files_only=True))
    except LocalEntryNotFoundError:
        snapshots = Path(constants.HF_HUB_CACHE) / ("datasets--" + repo_id.replace("/", "--")) / "snapshots"
        candidates = sorted(p for p in snapshots.glob("*") if (p / "meta/info.json").is_file())
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"Select a cached snapshot for {repo_id} with --dataset-path: {candidates}"
        ) from None
