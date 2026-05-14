# Benchmarks

All numbers reproducible via [`examples/benchmark_formats.py`](https://github.com/lancedb/lerobot-lancedb/blob/main/examples/benchmark_formats.py) and [`examples/aloha_loader_parity.py`](https://github.com/lancedb/lerobot-lancedb/blob/main/examples/aloha_loader_parity.py). Hardware for the runs below: H100 80 GB box, CPU decode (no NVJPEG) for apples-to-apples comparison across formats.

## Storage + throughput + pixel fidelity

Methodology: `batch_size=32`, `num_workers=4`, 30 batches (5 warmup). `delta_ts fps` uses an 8-frame-per-sample delta window — the realistic training read pattern. Pixel diffs sampled across 16 random frames per camera, averaged.

### `lerobot/pusht` (synthetic 96×96, 1-cam)

| format | size MB | single fps | delta_ts fps | mean abs pixel diff | visibly different pixels |
|---|---:|---:|---:|---:|---:|
| upstream parquet+mp4 | **7.3** | 4296 | 750 | 0 | 0.0% |
| `convert_to_lance` (JPEG-95) | 60.0 | **9148** | **3510** | 0.0020 | 6.2% |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 105.6 | 8321 | 2909 | 0.0003 | 0.07% |
| `convert_to_lance --lossless` (PNG) | 58.5 | 7746 | 1835 | **0** | **0.0%** |
| **`convert_to_lance_video`** | **8.0** | 8446 | 2853 | **0** | **0.0%** |

### `lerobot/aloha_static_cups_open` (real-world 480×640, 4-cam bimanual)

| format | size MB | single fps | delta_ts fps | mean abs pixel diff | visibly different pixels |
|---|---:|---:|---:|---:|---:|
| upstream parquet+mp4 | **485.6** | 121.3 | 18.7 | 0 | 0.0% |
| `convert_to_lance` (JPEG-95) | 3626.0 | **444.2** | **46.0** | 0.0021 | 1.4% |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 8735.4 | 317.0 | 32.5 | 0.0012 | 0.06% |
| `convert_to_lance --lossless` (PNG) | 12 581 | 130.3 | 13.7 | **0** | **0.0%** |
| **`convert_to_lance_video`** | **487.4** | 271.6 | 45.6 | **0** | **0.0%** |

### `lerobot/koch_pick_place_5_lego` (real-world 480×640, 2-cam single-arm)

| format | size MB | single fps | delta_ts fps | mean abs pixel diff | visibly different pixels |
|---|---:|---:|---:|---:|---:|
| upstream parquet+mp4 | **2014.1** | 185.3 | 26.6 | 0 | 0.0% |
| `convert_to_lance` (JPEG-95) | 8541.0 | **578.6** | **70.8** | 0.0047 | **13.5%** |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 17 335.3 | 426.0 | 49.0 | 0.0016 | 0.14% |
| **`convert_to_lance_video`** | **2015.9** | 291.1 | 53.8 | **0** | **0.0%** |

(Koch PNG row omitted to save disk — same shape as ALOHA.)

## Training-accuracy parity

Pixel-level fidelity matters because it propagates into training accuracy. Two end-to-end checks:

### pusht — `DiffusionPolicy`, 200k steps, env eval

Same recipe as [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht), seed=42, 500-episode `gym-pusht` rollout at seed=100000.

| storage format | env success rate | avg max overlap |
|---|---:|---:|
| Lance JPEG-95 | 58.0% | 0.919 |
| **Lance `--lossless` (PNG)** | **65.8%** | **0.938** |
| upstream parquet+mp4 (head-to-head, seed=42) | 68.0% | 0.9586 |
| HF model card (seed=100000) | 65.4% | 0.955 |

The **10-percentage-point gap** between JPEG-95 and the upstream loader is not seed variance (two different seeds of the JPEG-95 run landed at 57.4% and 58.0%) — it's the JPEG roundtrip the writer introduces on top of the upstream AV1 video. PNG (or the video format) closes the gap to within seed noise.

### ALOHA cups_open — ACT, 30k steps, held-out action MSE

Same recipe (ACT defaults + ImageNet image norm + grad-clip 10), seed=42, held-out action MSE on the last 10% of episodes.

| storage format | train loss @ 30k | held-out action RMSE |
|---|---:|---:|
| Lance JPEG-95 | 0.0962 | 0.0927 |
| upstream parquet+mp4 (head-to-head) | 0.0635 | 0.0790 |

JPEG-95 storage gives **52% higher train loss** and **17% higher held-out RMSE** vs the upstream loader at the same seed and step count. On video-stored multi-camera data, `convert_to_lance_video` is the right fix — it matches upstream's disk footprint while staying bit-exact.

## Reading the numbers

A few patterns worth knowing when picking a format:

- **JPEG-95 visibly-different-pixel fraction varies a lot with content** — 6.2% on synthetic pusht, 1.4% on ALOHA's natural backgrounds, 13.5% on Koch's high-contrast Lego scenes. The dataset shape matters, not just the resolution.
- **PNG isn't always smaller than upstream parquet+mp4** — for `dtype=image` sources (e.g. `pusht_image`) PNG storage is actually smaller than upstream because upstream uses a less aggressive PNG. For `dtype=video` sources (everything ALOHA-class) PNG is several times larger than the source mp4 and, on multi-camera 480×640 data, slower to decode than the mp4 path.
- **`convert_to_lance_video` matches upstream's disk size** within 0.5% on all three datasets we measured — Lance blob v2 stores the mp4 bytes verbatim with negligible per-file metadata.
- **JPEG-100 + 4:4:4** reduces visible-pixel artifacts by 25–90× vs the default while keeping the NVJPEG decode path. ~1.8× larger than the JPEG-95 default; still not bit-exact at sharp edges.
