# Video format (`LeRobotLanceVideoDataset`)

Keeps the original mp4 bytes verbatim in a Lance column with the [blob v2 encoding](https://lancedb.github.io/lance/format/index.html). At read time, [`Dataset.take_blobs`](https://lancedb.github.io/lance/api/python/lance.html) returns the bytes as a file-like object; we hand them to torchcodec and ask for the specific frames we need.

This is the closest the Lance loader gets to the upstream parquet+mp4 path:

- Same pixels (bit-exact).
- Same disk size (within 0.5 % on every dataset we measured).
- One self-contained Lance dataset — no separate mp4 file tree.

## When to use

- Your source dataset is `dtype=video` (most lerobot datasets).
- You care about byte-exact training-accuracy parity with upstream.
- You don't want to ship a separate mp4 directory tree alongside the Lance data.

If your source is `dtype=image` (`lerobot/pusht_image`, etc.), there's no mp4 to copy — use [`convert_to_lance`](frames-format.md) instead.

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

Each row in the videos table is **one unique mp4 file**. LeRobot lays out source mp4s as `videos/{video_key}/chunk-NNN/file-MMM.mp4` and stores `(chunk_index, file_index, from_timestamp)` per episode in `meta/episodes/*.parquet`; the reader joins on those.

!!! note "Why `(video_key, chunk, file)` instead of just `(chunk, file)`?"
    On bimanual ALOHA datasets all cameras share the same mp4 file for a given episode. On single-arm setups like Koch the laptop and phone cameras can use *different* file indices for the same episode. The row identity has to include the camera key.

## Reader

```python
from lerobot_lancedb import LeRobotLanceVideoDataset

ds = LeRobotLanceVideoDataset(root="./aloha_cups_open_lance_video")

sample = ds[0]
sample["observation.images.cam_high"].shape   # (C, H, W) uint8 (or float)
sample["action"].shape                        # (A,) float32
```

Same constructor surface as the frames format — `root=`, `repo_id=`, `uri=` + `meta_root=`, plus `delta_timestamps`.

## How a frame read works

For each `__getitems__([i0, i1, ...])`:

1. Look up tabular fields for the requested frames from the per-frame table.
2. For each unique `(video_key, chunk, file)` triple needed by this batch, fetch the blob via `take_blobs(blob_column="video_bytes", indices=[...])`.
3. Cache the resulting `torchcodec.VideoDecoder` per `(chunk, file, video_key)` in a per-worker LRU.
4. Compute the in-file frame index as `round((timestamp + from_timestamp) * average_fps)` and decode.

For shuffled training this means at most one decoder setup per file per worker, regardless of how many frames the batch pulls.

## `delta_timestamps`

```python
ds = LeRobotLanceVideoDataset(
    root="./aloha_cups_open_lance_video",
    delta_timestamps={
        "observation.images.cam_high": [-0.1, -0.05, 0.0],
        "action":                       [0.0, 0.05, 0.1, 0.15, 0.2],
    },
)
```

Multiple timestamps for the same camera collapse into a single `decoder.get_frames_at(indices=[...])` call.

## Decoder cache size

```python
LeRobotLanceVideoDataset(root="...", decoder_cache_size=16)
```

`decoder_cache_size` bounds the per-worker decoder LRU (default 16). Bigger caches help when many episodes share a few files (typical for ALOHA-style consolidated mp4s).

## GPU decode

There's no `decode_device="cuda"` shortcut yet. torchcodec's CUDA decode path needs:

- A torchcodec build linked against the NVIDIA Video Codec SDK (the default pip wheel ships the `ffmpeg` variant, which raises `Unsupported device: cuda`).
- A GPU that NVDECs the source codec (AV1 NVDEC requires RTX 40-series / Hopper or newer).

Once both are in place, wiring it through is a one-line change (we already pass `device=` to `VideoDecoder`).

## Cloud auth

Same env vars as the frames format:

- **S3** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- **GCS** — `GOOGLE_APPLICATION_CREDENTIALS`
- **HF Hub** — `HF_TOKEN` (or `huggingface-cli login`)

```python
ds = LeRobotLanceVideoDataset(
    uri="s3://bucket/path/aloha_cups_open.lance",
    meta_root="./aloha_cups_open_meta",
)
```

Because blob v2 columns aren't materialized into Arrow buffers, the reader only pays a byte-range fetch for the specific mp4 file each access touches — well-suited to remote object stores.

## Trade-offs vs the frames format

| | Frames format | **Video format** |
|---|---|---|
| Bit-exact pixels | only with `--jpeg-quality=100 --jpeg-subsampling=0` (near-bit-exact) | **always** |
| Disk size | 5–25× larger than upstream | **~same as upstream** |
| Single-frame throughput | fastest (NVJPEG-eligible) | slower (torchcodec seek + decode) |
| `delta_timestamps` throughput | fast | **~tied with JPEG-95, faster than upstream** |
| GPU decode | NVJPEG (`decode_device="auto"`) | needs CUDA-built torchcodec |
| Cloud-friendly | yes, byte-range fetches | yes, byte-range fetches |

See [Benchmarks](benchmarks.md) for measured numbers.
