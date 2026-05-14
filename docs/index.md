# lerobot-lancedb

Lance-backed datasets for [LeRobot](https://github.com/huggingface/lerobot) — frame-level random access on local disk and cloud (S3 / GCS / HF Hub / HF Buckets), with **two interchangeable storage layouts**.

`LeRobotLanceDataset` (per-frame JPEG / PNG bytes) and `LeRobotLanceVideoDataset` (per-file mp4 blobs via [Lance blob v2 encoding](https://lancedb.github.io/lance/format/index.html), decoded on the fly with torchcodec) both subclass `LeRobotDataset`, so existing trainers / samplers / `isinstance` checks accept them transparently.

## Install

```bash
pip install lerobot-lancedb
```

Brings in `lerobot[dataset]`, `lancedb` / `pylance`, and `torchcodec`.

## 30-second tour

```bash
# Convert a video-stored dataset (recommended path — bit-exact, fast)
lerobot-convert-to-lance-video \
    --repo-id=lerobot/aloha_static_cups_open \
    --output=./aloha_cups_open_lance_video \
    --overwrite
```

```python
from lerobot_lancedb import LeRobotLanceVideoDataset

ds = LeRobotLanceVideoDataset(root="./aloha_cups_open_lance_video")
# Plug into any code that expects a LeRobotDataset — including the upstream
# training factory, EpisodeAwareSampler, third-party trainers.
```

See [Conversion](conversion.md) for the full CLI reference and the trade-offs between the four storage modes.

## Why two layouts

Most lerobot datasets ship as **AV1-encoded mp4 chunks** with one file per episode (or per N episodes for large datasets). When you read a frame, you decode mp4 → array. We measured what happens when you re-encode those frames as something Lance can store:

| layout | what we store in Lance | bit-exact vs upstream? | recommended for |
|---|---|:---:|---|
| **Frames format** (`convert_to_lance`) | per-row JPEG / PNG bytes | only with `--lossless` (PNG) | `dtype=image` sources, anything with sharp edges where lossy JPEG ringing matters |
| **Video format** (`convert_to_lance_video`) | per-file mp4 bytes verbatim, blob v2 | always | `dtype=video` sources (most lerobot datasets) — same disk size as upstream |

Both readers expose the same API; the difference is what happens at convert and decode time.

## Headline benchmark

Real-world dataset (`lerobot/aloha_static_cups_open`, 4 cameras × 480 × 640, batch 32, num_workers 4, CPU decode, H100):

| format | size | delta_ts throughput | pixel fidelity vs upstream |
|---|---:|---:|---|
| upstream parquet+mp4 | 486 MB | 18.7 fps | reference |
| **`convert_to_lance_video`** | **487 MB** | **45.6 fps** | bit-exact |
| `convert_to_lance` (JPEG-95) | 3626 MB | 46.0 fps | 1.4% pixels visibly differ |
| `convert_to_lance --lossless` (PNG) | 12 581 MB | 13.7 fps | bit-exact |

The video format hits the same disk footprint as upstream while being 2.5× faster at the realistic training read pattern. JPEG-95 default trades pixel fidelity for size, which [costs training accuracy](benchmarks.md#training-accuracy-parity).

Full numbers + methodology: [Benchmarks](benchmarks.md).

## Where to next

| If you want to… | Go to |
|---|---|
| Convert a dataset | [Conversion](conversion.md) |
| Understand the JPEG / PNG per-frame layout | [Frames format](frames-format.md) |
| Understand the mp4-blob layout | [Video format](video-format.md) |
| See training + benchmark scripts | [Examples](examples.md) |
| See the full size × throughput × accuracy tables | [Benchmarks](benchmarks.md) |
