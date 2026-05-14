# Conversion

Two converters exposed as console scripts. Pick by source `dtype`:

- **`dtype=video`** (most lerobot datasets) → [`lerobot-convert-to-lance-video`](#video-format) (recommended — bit-exact, ~same disk size as upstream)
- **`dtype=image`** (`lerobot/pusht_image`, etc.) → [`lerobot-convert-to-lance`](#frames-format) (JPEG; turn quality up for near-bit-exact storage)

Either command lays out a self-contained directory next to a verbatim
`meta/` copy from upstream.

---

## Video format

```bash
lerobot-convert-to-lance-video \
    --repo-id=lerobot/aloha_static_cups_open \
    --output=./aloha_cups_open_lance_video \
    --overwrite
```

What it produces:

```
aloha_cups_open_lance_video/
  aloha_static_cups_open.lance/         # one row per frame, tabular data
  aloha_static_cups_open_videos.lance/  # one row per source mp4 file (blob v2)
  meta/
```

What it does:

1. Walks every episode in the source dataset.
2. For each unique `(video_key, chunk_index, file_index)` triple — i.e. every distinct source mp4 file — copies the bytes verbatim into a Lance blob-v2-encoded `video_bytes` column.
3. Writes per-frame tabular columns (state, action, timestamps, ...) into the sibling frames table.

Decoding happens on the fly at read time via torchcodec. Conversion is fast (usually under a minute) because no frame is ever decoded at write time.

### CLI flags

```
--repo-id REPO_ID           Source HF dataset (required)
--output OUTPUT             Local output dir (required)
--src-root SRC_ROOT         Override the source root (else fetched from HF cache)
--revision REVISION         HF revision / branch / tag
--table-name TABLE_NAME     Override the frames-table name (default: last segment of repo_id)
--tolerance-s TOLERANCE_S   Frame-timestamp tolerance
--overwrite                 Drop and rewrite if the target already exists
```

---

## Frames format

The frames CLI re-encodes every frame as JPEG and stores the bytes per-row.

Three useful presets:

- **Smallest / fastest** (default):
  ```bash
  lerobot-convert-to-lance \
      --repo-id=lerobot/pusht \
      --output=./pusht_lance --overwrite
  ```
  JPEG-95 + 4:2:0 chroma. Lossy — measurable pixel artifacts (1–14 % of pixels visibly differ depending on dataset). Use only when size / single-frame throughput trumps fidelity.

- **Near-lossless JPEG** (best fidelity that still NVJPEG-decodes):
  ```bash
  lerobot-convert-to-lance \
      --repo-id=lerobot/pusht_image \
      --output=./pusht_image_lance \
      --jpeg-quality=100 --jpeg-subsampling=0 --overwrite
  ```
  q=100 + 4:4:4 chroma. ~1.8× larger than the default. Visible-pixel artifacts drop 25–90× — typically below the threshold where it matters for training, but **not bit-exact**.

- **Push to the Hub** after converting locally:
  ```bash
  lerobot-convert-to-lance \
      --repo-id=lerobot/pusht --output=./pusht_lance --overwrite \
      --push-to-hub=me/pusht_lance
  ```

### CLI flags

```
--repo-id REPO_ID           Source HF dataset (required)
--output OUTPUT             Local output dir (required)
--src-root SRC_ROOT         Override the source root
--revision REVISION         HF revision / branch / tag
--table-name TABLE_NAME     Override the Lance table name
--jpeg-quality {1..100}     JPEG quality (default 95)
--jpeg-subsampling {0,1,2}  JPEG chroma subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0 (default)
--overwrite                 Drop and rewrite if the target already exists
--push-to-hub REPO          Optional HF Hub repo to upload to
```

---

## Python API

```python
from lerobot_lancedb import convert_to_lance, convert_to_lance_video

# Video format
convert_to_lance_video(
    repo_id="lerobot/aloha_static_cups_open",
    output="./aloha_lance_video",
    overwrite=True,
)

# Frames format — defaults
convert_to_lance(
    repo_id="lerobot/pusht",
    output="./pusht_lance",
    overwrite=True,
)

# Frames format — near-lossless JPEG
convert_to_lance(
    repo_id="lerobot/pusht_image",
    output="./pusht_image_lance",
    jpeg_quality=100,
    chroma_subsampling=0,
    overwrite=True,
)
```

If you want bit-exact pixel parity on a `dtype=video` source, use `convert_to_lance_video` — that's the whole reason it exists.
