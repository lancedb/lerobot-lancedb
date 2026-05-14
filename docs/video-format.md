# Video format (`LeRobotLanceVideoDataset`)

The video layout keeps the original mp4 bytes verbatim, stored in a Lance column with the [blob v2 encoding](https://lancedb.github.io/lance/format/index.html). At read time, [`Dataset.take_blobs`](https://lancedb.github.io/lance/api/python/lance.html) returns the bytes as a file-like object; we hand them straight to torchcodec and ask for the specific frames we need.

This is the closest the Lance loader gets to the upstream parquet+mp4 path — same pixels (bit-exact), same disk size (within 0.5 %), and a single self-contained Lance dataset.

## When to use it

- Your source dataset is `dtype=video` (most lerobot datasets — ALOHA, Koch, pusht, etc.).
- You care about byte-exact pixel fidelity for training-accuracy parity.
- You don't want to ship a separate mp4 directory tree alongside the parquet/Lance metadata.

If your source is `dtype=image` (e.g. `lerobot/pusht_image`), there's no mp4 to copy — use [`convert_to_lance --lossless`](frames-format.md) instead.

## Schema

```
<table>.lance/                   # one row per frame
  episode_index:     int32
  frame_index:       int32
  index:             int64
  timestamp:         float32
  task_index:        int32
  observation.state: list<float32>[D]
  action:            list<float32>[A]
  ...other tabular features
  (no image columns)

<table>_videos.lance/            # one row per source mp4 file
  video_key:    string           # e.g. "observation.images.cam_high"
  chunk_index:  int32
  file_index:   int32
  video_bytes:  large_binary     # metadata: lance-encoding:blob = true
```

Each row in the videos table is one unique mp4 file across all cameras. LeRobot lays out mp4s as `videos/{video_key}/chunk-NNN/file-MMM.mp4` and stores `(chunk_index, file_index, from_timestamp)` per episode in `meta/episodes/*.parquet`; the reader joins on those.

!!! note "Why `(video_key, chunk, file)` instead of just `(chunk, file)`?"
    On bimanual ALOHA datasets all cameras share the same mp4 file for a given episode, but on single-arm setups like Koch the laptop and phone cameras can use *different* file indices for the same episode. The row identity has to include the camera key.

## Reader

```python
from lerobot_lancedb import LeRobotLanceVideoDataset

ds = LeRobotLanceVideoDataset(root="./aloha_cups_open_lance_video")
sample = ds[0]
print(sample["observation.images.cam_high"].shape)   # (C, H, W) uint8 (or float)
print(sample["action"].shape)                        # (A,) float32
```

Like the frames-format reader, this subclasses `LeRobotDataset`, supports cloud URIs and `repo_id`, and exposes `delta_timestamps`.

### How a frame read works

1. Look up the frame in the per-frame table to get `(episode_index, timestamp, tabular fields)`.
2. For each requested camera, look up `(chunk_index, file_index, from_timestamp)` from `meta.episodes`.
3. For unique `(video_key, chunk, file)` triples not already cached, fetch the blob via `take_blobs(blob_column="video_bytes", indices=[...])` and hand the bytes to torchcodec.
4. Compute the in-file frame index as `round((timestamp + from_timestamp) * average_fps)` and decode.
5. Cache the `VideoDecoder` per `(chunk, file, video_key)` in a per-worker LRU.

For batch reads (`__getitems__`) the decoder cache amortizes across samples that share a file; for shuffled training this means at most one decode set-up per file per worker.

### `delta_timestamps`

```python
ds = LeRobotLanceVideoDataset(
    root="./aloha_cups_open_lance_video",
    delta_timestamps={
        "observation.images.cam_high": [-0.1, -0.05, 0.0],
        "action": [0.0, 0.05, 0.1, 0.15, 0.2],
    },
)
```

Multiple timestamps for the same camera in one access are batched into a single `decoder.get_frames_at(indices=[...])` call.

### Decoder device

```python
LeRobotLanceVideoDataset(root="...", decoder_cache_size=16)
```

`decoder_cache_size` bounds the per-worker decoder cache (default 16, LRU). Bigger caches help when many episodes share a few files (typical for ALOHA-style consolidated mp4s).

There's no `decode_device="cuda"` shortcut yet — torchcodec's CUDA decode path needs a CUDA-enabled torchcodec build (pip wheels ship the `ffmpeg` variant only). See [Known issues in the README](https://github.com/lancedb/lerobot-lancedb#known-issues--todo).

## Cloud reads

Same env-var auth as the frames format:

```python
ds = LeRobotLanceVideoDataset(
    uri="s3://bucket/path/aloha_cups_open.lance",
    meta_root="./aloha_cups_open_meta",
)
```

Because blob v2 columns aren't materialized into Arrow buffers, the reader only pays a byte-range fetch for the specific mp4 file each access touches — well-suited to remote object stores.

## Trade-offs vs the frames format

| | Frames format | Video format |
|---|---|---|
| Bit-exact pixels | only with `--lossless` | **always** |
| Disk size | ~7-25× larger than upstream mp4 | **~same as upstream mp4** |
| Single-frame throughput | fastest (esp. with NVJPEG) | slower (torchcodec per-frame seek + decode) |
| `delta_timestamps` throughput | fast, but loader bottleneck on multi-cam | **~tied with JPEG-95 on ALOHA, faster than upstream parquet+mp4** |
| GPU decode | NVJPEG (`decode_device="auto"`) | needs CUDA-built torchcodec |
| Cloud-friendly | yes, byte-range fetches | yes, byte-range fetches |

See [Benchmarks](benchmarks.md) for full numbers.
