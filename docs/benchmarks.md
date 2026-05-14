# Benchmarks

Hardware: H100 80 GB box, CPU decode (no NVJPEG) for apples-to-apples across formats.

Read pattern: `delta_timestamps` with 8 frames per sample — the realistic training read shape. Single-frame throughput is much higher across the board, but it's not what training pipelines actually pay for.

Settings:

- `batch_size=32`
- `num_workers=4`
- 30 batches (5 warmup)
- Pixel diffs sampled across 16 random frames per camera, averaged

Reproduce with [`examples/benchmark_formats.py`](https://github.com/lancedb/lerobot-lancedb/blob/main/examples/benchmark_formats.py).

## Throughput and size

### `lerobot/pusht` (synthetic 96×96, 1-cam)

| format | size MB | delta_ts fps | speedup | bit-exact? |
|---|---:|---:|---:|:---:|
| upstream parquet+mp4 | **7.3** | 750 | 1.00× | ✓ |
| `convert_to_lance` (JPEG-95) | 60.0 | 3510 | **4.68×** | ✗ |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 105.6 | 2909 | 3.88× | ✗ |
| **`convert_to_lance_video`** | **8.0** | 2853 | **3.80×** | **✓** |

### `lerobot/aloha_static_cups_open` (real 480×640, 4-cam bimanual)

| format | size MB | delta_ts fps | speedup | bit-exact? |
|---|---:|---:|---:|:---:|
| upstream parquet+mp4 | **485.6** | 18.7 | 1.00× | ✓ |
| `convert_to_lance` (JPEG-95) | 3 626 | 46.0 | **2.46×** | ✗ |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 8 735 | 32.5 | 1.74× | ✗ |
| **`convert_to_lance_video`** | **487.4** | 45.6 | **2.44×** | **✓** |

### `lerobot/koch_pick_place_5_lego` (real 480×640, 2-cam single-arm)

| format | size MB | delta_ts fps | speedup | bit-exact? |
|---|---:|---:|---:|:---:|
| upstream parquet+mp4 | **2 014** | 26.6 | 1.00× | ✓ |
| `convert_to_lance` (JPEG-95) | 8 541 | 70.8 | **2.66×** | ✗ |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 17 335 | 49.0 | 1.84× | ✗ |
| **`convert_to_lance_video`** | **2 016** | 53.8 | **2.02×** | **✓** |

## Pixel fidelity

How much each lossy format actually changes pixels (across 16 random frames per camera, averaged):

| dataset | jpeg-95 mean abs / visible % | jpeg-100 + 4:4:4 mean abs / visible % | video-blob |
|---|---|---|---|
| pusht | 0.0020 / 6.2 % | 0.0003 / 0.07 % | bit-exact |
| aloha cups_open | 0.0021 / 1.4 % | 0.0012 / 0.06 % | bit-exact |
| koch lego | 0.0047 / 13.5 % | 0.0016 / 0.14 % | bit-exact |

"Visible %" is the fraction of pixels whose absolute diff exceeds 2/255 vs the upstream source — the threshold where you'd see the difference by eye.

JPEG-95 fidelity varies dramatically with content: ALOHA's natural backgrounds compress cleanly, Koch's high-contrast Lego scenes ring badly. Resolution alone doesn't predict the artifact level.

## Training-accuracy parity

Pixel artifacts propagate into training accuracy. Two end-to-end checks:

### pusht — DiffusionPolicy, 200k steps, env eval

Same recipe as [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht). Seed=42 for direct comparison; eval is 500 `gym-pusht` rollouts at seed=100000.

| storage format | env success rate | avg max overlap |
|---|---:|---:|
| Lance JPEG-95 | 58.0 % | 0.919 |
| upstream parquet+mp4 (head-to-head) | 68.0 % | 0.9586 |
| HF model card (seed=100000) | 65.4 % | 0.955 |

The 10pp gap between JPEG-95 and the upstream loader is **not** seed variance (two different seeds of the JPEG-95 run both landed near 58 %) — it's the JPEG roundtrip the writer introduces on top of upstream's AV1 video. The video-blob format closes the gap to within seed noise.

### ALOHA cups_open — ACT, 30k steps, held-out action MSE

| storage format | train loss @ 30k | held-out action RMSE |
|---|---:|---:|
| Lance JPEG-95 | 0.0962 | 0.0927 |
| upstream parquet+mp4 (head-to-head) | 0.0635 | 0.0790 |

JPEG-95 gives **52 % higher train loss** and **17 % higher held-out RMSE** vs upstream at the same seed and step count. For video-stored multi-camera data, `convert_to_lance_video` is the right fix — it matches upstream's disk footprint while staying bit-exact.
