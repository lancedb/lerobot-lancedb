# Conversion

Two converters are exposed as console scripts. Pick by source dtype:

| Source dataset `dtype` | Use |
|---|---|
| `video` (most lerobot datasets) | [`lerobot-convert-to-lance-video`](#video-format) |
| `image` (`lerobot/pusht_image`, etc.) | [`lerobot-convert-to-lance --lossless`](#frames-format) |

Either command lays out a self-contained directory:

```
<output>/
  <table>.lance/         # one row per frame, tabular data
  <table>_videos.lance/  # only for video-format: per-file mp4 blobs (lance-encoding:blob)
  meta/                  # verbatim copy of upstream meta/
```

Both can optionally `--push-to-hub` to publish back to the HF Hub.

## Video format

```bash
lerobot-convert-to-lance-video \
    --repo-id=lerobot/aloha_static_cups_open \
    --output=./aloha_cups_open_lance_video \
    --overwrite
```

What it does:

1. Walks every episode in the source dataset.
2. For each `(video_key, chunk_index, file_index)` triple — that is, every unique source mp4 file — copies the mp4 bytes **verbatim** into a Lance blob-v2-encoded column.
3. Writes per-frame tabular columns (state, action, timestamps) into a sibling table.

Decoding happens on the fly at read time via torchcodec. Because the bytes are bit-identical to upstream, pixel fidelity is preserved.

Conversion is fast — usually under a minute per dataset, regardless of the dataset's frame count, because no frame is ever decoded at write time.

### CLI reference

```
--repo-id REPO_ID           Source HF dataset (required)
--output OUTPUT             Local output dir (required)
--src-root SRC_ROOT         Override the source root (else fetched from HF cache)
--revision REVISION         HF revision / branch / tag
--table-name TABLE_NAME     Override the frames-table name (default: last segment of repo_id)
--tolerance-s TOLERANCE_S   Frame-timestamp tolerance
--overwrite                 Drop and rewrite if the target already exists
```

## Frames format

```bash
# Default: JPEG-95 (smallest, fastest single-frame, lossy)
lerobot-convert-to-lance \
    --repo-id=lerobot/pusht \
    --output=./pusht_lance --overwrite

# Bit-exact PNG (recommended for image-stored sources)
lerobot-convert-to-lance \
    --repo-id=lerobot/pusht_image \
    --output=./pusht_image_lance \
    --lossless --overwrite

# Near-lossless JPEG: q=100 + 4:4:4 chroma. Keeps the NVJPEG decode path.
lerobot-convert-to-lance \
    --repo-id=lerobot/pusht \
    --output=./pusht_lance_high_quality \
    --jpeg-quality=100 --jpeg-subsampling=0 --overwrite
```

What it does:

1. For each episode, decode the source frames (mp4 or per-frame parquet bytes).
2. Re-encode every frame as JPEG (or PNG with `--lossless`) using a 4–8-thread pool.
3. Store the encoded bytes inline on each per-frame row.

The encoding happens at convert time, so this path takes meaningfully longer than the video format — and on PNG-encoded multi-camera datasets it's the dominant cost (~13 min for `aloha_static_cups_open` at full PNG quality after the threading optimization).

The JPEG default has measurable pixel artifacts (1.4 – 13.5 % of pixels visibly differ depending on dataset). The artifacts [translate to a real training-accuracy hit](benchmarks.md#training-accuracy-parity); use `--lossless` or the video format if accuracy parity matters.

### CLI reference

```
--repo-id REPO_ID           Source HF dataset (required)
--output OUTPUT             Local output dir (required)
--src-root SRC_ROOT         Override the source root
--revision REVISION         HF revision / branch / tag
--table-name TABLE_NAME     Override the Lance table name
--jpeg-quality {1..100}     JPEG quality (default 95)
--jpeg-subsampling {0,1,2}  JPEG chroma subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0 (default)
--lossless                  Use PNG instead of JPEG — bit-exact, larger, no NVJPEG GPU decode
--overwrite                 Drop and rewrite if the target already exists
--push-to-hub REPO          Optional HF Hub repo to upload to
```

## Python API

```python
from lerobot_lancedb import convert_to_lance, convert_to_lance_video

# Video format
convert_to_lance_video(
    repo_id="lerobot/aloha_static_cups_open",
    output="./aloha_lance_video",
    overwrite=True,
)

# Frames format with explicit knobs
convert_to_lance(
    repo_id="lerobot/pusht",
    output="./pusht_lance",
    jpeg_quality=100,
    chroma_subsampling=0,  # 4:4:4 — near-lossless, NVJPEG-compatible
    overwrite=True,
)

convert_to_lance(
    repo_id="lerobot/pusht_image",
    output="./pusht_image_lance",
    lossless=True,         # PNG, bit-exact
    overwrite=True,
)
```

## Push to the Hub

The frames CLI supports `--push-to-hub=<user>/<repo>`. The video CLI doesn't yet — call the Python API and upload separately:

```bash
lerobot-convert-to-lance-video --repo-id=lerobot/pusht \
    --output=./pusht_lance_video --overwrite
huggingface-cli upload <user>/pusht_lance_video ./pusht_lance_video --repo-type=dataset
```

## Cloud sources

Both converters read upstream sources via the HF cache. For cloud Lance reads at training time, see the [Frames](frames-format.md#cloud-reads) and [Video](video-format.md#cloud-reads) pages.
