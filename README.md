# lerobot-lancedb

Lance-backed datasets for [LeRobot](https://github.com/huggingface/lerobot) — frame-level random access on local disk and cloud (S3 / GCS / HF Hub / HF Buckets), with two storage layouts you can choose between depending on what you care about.

`LeRobotLanceDataset` and `LeRobotLanceVideoDataset` both subclass `LeRobotDataset`, so any code that accepts a `LeRobotDataset` (the training factory, `EpisodeAwareSampler`, third-party trainers, `isinstance` checks) accepts a Lance-backed one too.

## At a glance

- **Two storage layouts** for the image / video data, picked per dataset:
  - `convert_to_lance` — per-frame rows with JPEG or PNG bytes (use `--lossless` for bit-exact PNG).
  - `convert_to_lance_video` — per-file rows with the original mp4 bytes stored verbatim using **Lance blob v2** encoding; decoded on the fly with torchcodec. Bit-exact pixels, same disk size as upstream.
- **Backend-agnostic API**: same `__getitem__` / `__getitems__` surface as upstream `LeRobotDataset`, including `delta_timestamps`, multi-camera, multi-task, multi-subtask.
- **Cloud-native reads**: `s3://`, `gs://`, `hf://datasets/...`, `hf://buckets/...` — Lance does byte-range fetches; no separate mp4 directory tree to ship around.
- **GPU NVJPEG decode** for the JPEG / PNG layout, auto-enabled on CUDA boxes via `decode_device="auto"`.
- **Spawn-mode-safe DataLoader workers**, persistent-workers compatible.
- **Drop-in replacement validated end-to-end**: trained `DiffusionPolicy` on pusht and `ACT` on ALOHA from Lance-backed data; matches the upstream parquet+mp4 path within seed noise when you pick the right storage format. (See [parity table](#end-to-end-training-parity).)

## Install

```bash
pip install lerobot-lancedb
# or, for development
pip install -e .[dev]
```

The package brings in `lerobot[dataset]`, `lancedb` / `pylance`, and the `torchcodec` decoder used by the video-blob layout.

## Quickstart

### Pick a storage layout

Most lerobot datasets are `dtype=video` — the upstream stores frames inside per-episode mp4 chunks. A small minority (e.g. `lerobot/pusht_image`) are `dtype=image` — frames stored bit-exact as encoded bytes in the parquet directly.

| Source dtype | Recommended converter | Why |
|---|---|---|
| `video` (most datasets) | `convert_to_lance_video` | Bit-exact pixels, ~same disk size as upstream, fast random access via Lance blob v2. |
| `image` (`pusht_image`, similar) | `convert_to_lance(..., lossless=True)` | Bit-exact PNG storage. The upstream is bit-exact too; JPEG roundtrip would silently degrade it. |
| Either, when you care about size more than fidelity | `convert_to_lance` (default JPEG-95) | Smallest single-frame decode latency. **Measurably lossy** — see the [trade-offs](#storage-format-trade-offs). |

### Convert: video-stored datasets (recommended path)

```bash
python -c "from lerobot_lancedb import convert_to_lance_video; \
    convert_to_lance_video('lerobot/pusht', './pusht_lance_video')"
```

This produces:

```
pusht_lance_video/
  pusht.lance/             # one row per frame; tabular (state, action, timestamps)
  pusht_videos.lance/      # one row per source mp4 file; raw bytes as blob v2 columns
  meta/                    # verbatim copy of upstream meta/
```

### Convert: image-stored or quality-flexible datasets

```bash
# Bit-exact PNG (recommended for accuracy-sensitive work on image-stored sources)
lerobot-convert-to-lance \
    --repo-id=lerobot/pusht_image \
    --output=./pusht_image_lance \
    --lossless --overwrite

# Default JPEG-95 (smaller, faster single-frame, lossy)
lerobot-convert-to-lance \
    --repo-id=lerobot/pusht \
    --output=./pusht_lance \
    --overwrite

# Near-lossless JPEG (q=100 + 4:4:4 chroma) — kept NVJPEG-compatible
lerobot-convert-to-lance \
    --repo-id=lerobot/pusht \
    --output=./pusht_lance_nearly_lossless \
    --jpeg-quality=100 --jpeg-subsampling=0 --overwrite
```

Optional `--push-to-hub=<user>/<repo>` to upload the converted dataset.

### Train

Same API regardless of which converter you used:

```python
from lerobot_lancedb import LeRobotLanceDataset, LeRobotLanceVideoDataset

# Per-frame JPEG/PNG layout
ds = LeRobotLanceDataset(root="./pusht_lance")

# Per-file mp4-blob layout
ds = LeRobotLanceVideoDataset(root="./pusht_lance_video")

# Both work with cloud URIs and HF Hub:
ds = LeRobotLanceDataset(uri="s3://bucket/pusht.lance", meta_root="./pusht_lance")
ds = LeRobotLanceVideoDataset(repo_id="me/pusht_lance_video")
```

For a complete training script, see [`examples/train_and_eval_lance.py`](examples/train_and_eval_lance.py) — it trains `DiffusionPolicy` with the official `lerobot/diffusion_pusht` recipe and writes a checkpoint that `lerobot-eval` can load directly.

## Storage format trade-offs

We measured size, throughput, and pixel fidelity for all four formats on three datasets covering very different shapes:

- `lerobot/pusht` — synthetic 96×96 single-camera, video-stored (a tiny toy reference).
- `lerobot/aloha_static_cups_open` — real-world 480×640 four-camera bimanual, video-stored.
- `lerobot/koch_pick_place_5_lego` — real-world 480×640 two-camera single-arm, video-stored.

**Methodology**: `batch_size=32`, `num_workers=4`, 30 batches (5 warmup), H100 box, CPU decode for apples-to-apples across formats. `delta_ts fps` uses an 8-frame-per-sample delta window (the realistic training read pattern). Pixel diffs sampled across 16 random frames per camera and averaged. Reproducible via [`examples/benchmark_formats.py`](examples/benchmark_formats.py).

### `lerobot/pusht` (synthetic, 96×96, 1-cam)

| format | size MB | single-frame fps | delta_ts fps | mean abs pixel diff | visibly-different pixels |
|---|---:|---:|---:|---:|---:|
| upstream parquet+mp4 | **7.3** | 4296 | 750 | 0 | 0.0% |
| `convert_to_lance` (JPEG-95, default) | 60.0 | **9148** | **3510** | 0.0020 | 6.2% |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 105.6 | 8321 | 2909 | 0.0003 | 0.07% |
| `convert_to_lance --lossless` (PNG) | 58.5 | 7746 | 1835 | **0** | **0.0%** |
| **`convert_to_lance_video`** | **8.0** | 8446 | 2853 | **0** | **0.0%** |

### `lerobot/aloha_static_cups_open` (real-world, 480×640, 4-cam bimanual)

| format | size MB | single-frame fps | delta_ts fps | mean abs pixel diff | visibly-different pixels |
|---|---:|---:|---:|---:|---:|
| upstream parquet+mp4 | **485.6** | 121.3 | 18.7 | 0 | 0.0% |
| `convert_to_lance` (JPEG-95, default) | 3626.0 | **444.2** | **46.0** | 0.0021 | 1.4% |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 8735.4 | 317.0 | 32.5 | 0.0012 | 0.06% |
| `convert_to_lance --lossless` (PNG) | 12 581 | 130.3 | 13.7 | **0** | **0.0%** |
| **`convert_to_lance_video`** | **487.4** | 271.6 | 45.6 | **0** | **0.0%** |

### `lerobot/koch_pick_place_5_lego` (real-world, 480×640, 2-cam single-arm)

| format | size MB | single-frame fps | delta_ts fps | mean abs pixel diff | visibly-different pixels |
|---|---:|---:|---:|---:|---:|
| upstream parquet+mp4 | **2014.1** | 185.3 | 26.6 | 0 | 0.0% |
| `convert_to_lance` (JPEG-95, default) | 8541.0 | **578.6** | **70.8** | 0.0047 | **13.5%** |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 17 335.3 | 426.0 | 49.0 | 0.0016 | 0.14% |
| **`convert_to_lance_video`** | **2015.9** | 291.1 | 53.8 | **0** | **0.0%** |

(PNG row omitted — same story as ALOHA, ~26× larger than upstream and slower than mp4 decode at delta_ts on multi-camera 480×640. Use `convert_to_lance_video` for video-stored datasets at this scale.)

Note how the JPEG-95 visibly-different-pixel fraction varies a lot with content: ALOHA's natural backgrounds compress cleanly (1.4%), Koch's high-contrast Lego on a desk gives JPEG ringing 10× worse (13.5%). The dataset shape matters, not just the resolution.

### Take-aways

- **`convert_to_lance_video` is the right default for video-stored datasets.** Almost identical disk size to the upstream mp4 layout (within ~0.5% on both datasets), bit-exact pixels, and 2.5× faster than upstream at the realistic delta_timestamps pattern on ALOHA. The trade-off is no NVJPEG GPU decode — torchcodec on CPU does the work.
- **JPEG-100 + 4:4:4 subsampling is the practical "good enough" JPEG.** Reduces visibly-different pixels by ~25-90× vs the JPEG-95 default (e.g. 6.2% → 0.07% on pusht, 1.4% → 0.06% on ALOHA) while keeping the full NVJPEG decode path. ~1.8× larger than JPEG-95 but still NVJPEG-compatible. Recommended when you want most of the speed of JPEG and most of the fidelity of lossless, and can't afford the size of PNG.
- **`--lossless` (PNG) is the right default for image-stored datasets** like `lerobot/pusht_image`: bit-exact AND smaller than upstream parquet, because the upstream's stored-bytes encoding is less aggressive than PNG. For video-stored multi-camera datasets (ALOHA-class), PNG is a poor fit — 26× larger than upstream/video, and large-frame deflate decode in PIL is actually slower than mp4 decode under delta_timestamps. Use `convert_to_lance_video` there instead.
- **JPEG-95 default is the fastest single-frame format** but it has measurable pixel artifacts — 6% of pixels visibly differ on pusht, 1.4% on ALOHA. Those artifacts translate to training-accuracy hits (see next section). Keep this default only when you're size- or speed-bound and don't care about training parity.

## End-to-end training parity

To validate that the Lance loader is a drop-in replacement, we trained models end-to-end on Lance-backed data and compared against the upstream parquet+mp4 path at the same seed and recipe.

### pusht (DiffusionPolicy, 200k steps, env eval)

Recipe matches the published [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht) config: batch 64, `crop_shape=(84, 84)` with random crop, grad-clip 10, cosine LR with 500 warmup, ImageNet image normalization stats, seed=42. Eval is 500 `gym-pusht` rollouts at seed=100000.

| storage format | env success rate | avg max overlap |
|---|---:|---:|
| Lance JPEG-95 default | 58.0% | 0.919 |
| **Lance `--lossless` (PNG)** | **65.8%** | **0.938** |
| upstream parquet+mp4 (head-to-head, seed=42) | 68.0% | 0.9586 |
| HF model card (seed=100000) | 65.4% | 0.955 |

The **10-percentage-point gap** between JPEG-95 and the upstream loader is not seed variance — two of our pusht runs with different train seeds landed at 57.4% and 58.0% (max-overlap 0.916 and 0.919). The gap is entirely the JPEG roundtrip the writer introduces on top of the upstream AV1 video. Re-running with `--lossless` (or `convert_to_lance_video`, which is also bit-exact) closes the gap to within seed noise of the upstream checkpoint.

### ALOHA (ACT, 30k steps, held-out action MSE)

Training: `lerobot/aloha_static_cups_open` with ACT defaults, seed=42, 30k steps. Eval: held-out action MSE on the last 10% of episodes.

| storage format | train loss @ 30k | held-out action RMSE |
|---|---:|---:|
| Lance JPEG-95 default | 0.0962 | 0.0927 |
| upstream parquet+mp4 (head-to-head, seed=42) | 0.0635 | 0.0790 |

Training with JPEG-95 storage gives **17% higher held-out RMSE** and **52% higher train loss** at 30k steps. Again, this is the JPEG roundtrip cost — `convert_to_lance_video` is the recommended fix on ALOHA-class data because PNG storage on 4-camera 480×640 video would cost ~12 GB per dataset.

## Throughput vs upstream parquet+mp4 (with GPU NVJPEG)

The trade-off tables above use CPU decode for both backends to isolate the format-vs-format comparison. The historical motivation for this project was the GPU win: on H100 with NVJPEG enabled (the `decode_device="auto"` default of `LeRobotLanceDataset`), Lance JPEG-95 beats upstream parquet+mp4 by **5-7×** under realistic delta_timestamps on multi-camera ALOHA data — measured at 10.87 vs 2.14 bps on `aloha_static_cups_open`, 11.11 vs 1.63 bps on `aloha_static_ziploc_slide`, both at batch 32 × num_workers 4. See [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md) for the full reproduction recipe.

Caveat: that 5-7× was measured against upstream's `pyav` default. With upstream switched to `torchcodec` (faster than pyav on the same mp4s), the gap narrows; on pusht specifically the dataloader is no longer the bottleneck and Lance wins by ~1.5× end-to-end. For ALOHA-class natural-image multi-camera data the win is real and large; for tiny synthetic datasets the win is modest.

## Cloud / Hub reads

The reader accepts cloud URIs directly. Authentication picks up from the standard env vars:

```python
# S3 — uses AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
ds = LeRobotLanceDataset(uri="s3://bucket/path/pusht.lance",
                         meta_root="./pusht_lance")

# Google Cloud Storage — uses GOOGLE_APPLICATION_CREDENTIALS
ds = LeRobotLanceDataset(uri="gs://bucket/path/pusht.lance",
                         meta_root="./pusht_lance")

# Hugging Face Hub — uses HF_TOKEN or `huggingface-cli login`
ds = LeRobotLanceDataset(repo_id="me/pusht_lance")
```

Lance reads byte ranges from the object store on demand — no full-dataset download, no separate mp4 directory to ship.

## Feature status

| Feature | Supported |
|---|---|
| Local lance dir (`root=`) | ✓ |
| HF Hub via `repo_id=` (`hf://datasets/...` natively) | ✓ |
| Cloud URI (S3 / GCS / HF Buckets) | ✓ |
| `delta_timestamps` (temporal windows) | ✓ |
| Multi-camera, multi-task, multi-subtask features | ✓ |
| Spawn-mode DataLoader workers (`persistent_workers=True`) | ✓ |
| GPU NVJPEG decode (`decode_device="auto"`) for JPEG / PNG layout | ✓ |
| Per-frame JPEG / PNG storage (`convert_to_lance`, `LeRobotLanceDataset`) | ✓ |
| Per-file mp4-blob storage (`convert_to_lance_video`, `LeRobotLanceVideoDataset`) | ✓ |
| Writer flags `--lossless`, `--jpeg-quality`, `--jpeg-subsampling` | ✓ |
| Bit-exact pixel verification + training-accuracy parity tests | ✓ ([`examples/benchmark_formats.py`](examples/benchmark_formats.py)) |
| Per-epoch reshuffle via `PermutationBuilder.shuffle` | not yet (planned) |
| Writing / recording brand-new datasets | no — use upstream `LeRobotDataset.create` and then convert |

## CLI reference

```bash
lerobot-convert-to-lance --help
```

Highlights:

```
--repo-id REPO_ID           Source HF dataset (required)
--output OUTPUT             Local output dir (required)
--src-root SRC_ROOT         Override the source root (else fetched from HF cache)
--revision REVISION         HF revision / branch / tag
--table-name TABLE_NAME     Override the Lance table name (default: last segment of repo_id)
--jpeg-quality {1..100}     JPEG quality (default 95)
--jpeg-subsampling {0,1,2}  JPEG chroma subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0 (default)
--lossless                  Use PNG instead of JPEG — bit-exact, larger, no NVJPEG path
--overwrite                 Drop and rewrite if the target already exists
--push-to-hub REPO          Optional HF Hub repo to upload to
```

For `convert_to_lance_video`, use the Python API (no separate CLI yet):

```python
from lerobot_lancedb import convert_to_lance_video
convert_to_lance_video(
    repo_id="lerobot/aloha_static_cups_open",
    output="./aloha_cups_open_lance_video",
    overwrite=True,
)
```

## Examples

| Script | What it does |
|---|---|
| [`examples/conversion.py`](examples/conversion.py) | Batch-convert a list of HF datasets to the default JPEG-95 Lance layout. Also runs the legacy NVJPEG throughput benchmark. |
| [`examples/benchmark_formats.py`](examples/benchmark_formats.py) | Cross-format benchmark: size × throughput × pixel fidelity per dataset. Produces the tables in this README. |
| [`examples/train_and_eval_lance.py`](examples/train_and_eval_lance.py) | Train `DiffusionPolicy` on a Lance-backed dataset using the published `lerobot/diffusion_pusht` recipe; saves a checkpoint `lerobot-eval` can load. |
| [`examples/aloha_loader_parity.py`](examples/aloha_loader_parity.py) | ACT head-to-head between Lance and upstream parquet+mp4 on ALOHA; produces held-out action MSE. |
| [`examples/train_with_lance.py`](examples/train_with_lance.py) | Minimal 10-step demo that the Lance loader fits into a stock LeRobot training loop unchanged. |

## Status

Experimental. APIs and on-disk layout may change between 0.x releases as we gather feedback. The JPEG-95 writer default predates the training-accuracy validation in this README — it's kept for backwards compatibility, but the recommended converter for new datasets is `convert_to_lance_video` (for `dtype=video` sources) or `convert_to_lance(..., lossless=True)` (for `dtype=image` sources).

## Contributing

Issues and PRs welcome. The package is intentionally small:

- `src/lerobot_lancedb/dataset.py` — JPEG/PNG-layout reader (`LeRobotLanceDataset`).
- `src/lerobot_lancedb/lance_video_dataset.py` — mp4-blob reader (`LeRobotLanceVideoDataset`).
- `src/lerobot_lancedb/writer.py` — both converters.
- `src/lerobot_lancedb/benchmark.py` — shared throughput-benchmark utilities.
- `src/lerobot_lancedb/auto.py` — `make_lerobot_dataset` auto-detection.

## Known issues / TODO

- **PNG writer is slow on large multi-camera datasets.** Conversion is single-threaded; deflate-compressing every pixel of every frame across 4 cameras at 480×640 takes ~75 min for `lerobot/aloha_static_cups_open` (vs ~11 min for the JPEG-95 path and <1 min for `convert_to_lance_video`). For `dtype=video` sources, prefer `convert_to_lance_video` — it's both faster to write and bit-exact. Parallelizing the PNG writer across episodes would be a worthwhile optimization but isn't done yet.
- **`convert_to_lance_video` has no standalone CLI yet.** Use the Python API.
- **Per-epoch reshuffle** via `PermutationBuilder.shuffle` isn't wired up yet — the current shuffling relies on the DataLoader sampler.
- **NVDEC for the video-blob path** isn't enabled. torchcodec on CPU is fast enough that this hasn't been a bottleneck in practice, but on huge multi-camera datasets a GPU decode path would close more of the gap to JPEG/NVJPEG.

## License

Apache 2.0.
