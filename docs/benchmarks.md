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

End-to-end checks that the loader produces models that behave the same as the upstream parquet+mp4 path.

### ALOHA cups_open — ACT, 30k steps, held-out action MSE

Same recipe (ACT defaults + ImageNet image norm + grad-clip 10), seed=42, held-out MSE on the last 10 % of episodes. `num_workers=4`.

| storage format | train loss @ 30k | held-out RMSE |
|---|---:|---:|
| Lance JPEG-95 (default) | 0.0962 | 0.0927 |
| Lance JPEG-100 + 4:4:4 | 0.0961 | 0.0872 |
| Lance video-blob | 0.0972 | 0.0901 |
| upstream parquet+mp4 | 0.0635 | 0.0790 |

What this says about the storage formats:

- **All three Lance modes give equivalent training accuracy** within ~6 % of each other. Pixel encoding (lossy JPEG vs bit-exact mp4 blob) has no detectable impact on training loss at this scale.
- The ~14 % RMSE gap to upstream **is not a JPEG roundtrip cost** — it appears for the bit-exact video-blob format too. Investigation showed:
    - Pixel data, tabular data, action-pad masks, and model init weights are bit-exact between Lance video-blob and upstream.
    - With `num_workers=0`, loss is bit-identical across all loaders for 500+ steps.
    - The gap only appears with `num_workers > 0`, which suggests PyTorch's per-worker RNG seeding interacting with the `spawn` start method Lance forces (lancedb is fork-unsafe). Worth more investigation; not blocking.

### pusht — DiffusionPolicy, 200k steps, env eval

Same recipe as [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht), seed=42; eval is 500 `gym-pusht` rollouts at seed=100000.

| storage format | env success rate | avg max overlap |
|---|---:|---:|
| Lance JPEG-95 | 58.0 % | 0.919 |
| upstream parquet+mp4 (head-to-head) | 68.0 % | 0.9586 |
| HF model card (seed=100000) | 65.4 % | 0.955 |

The 10pp gap was originally attributed to the JPEG roundtrip. Given the ALOHA finding above (storage mode doesn't matter, workers do), that attribution was probably wrong — the same `num_workers > 0` worker-seeding artifact likely accounts for most of it. Re-running pusht with the bit-exact video-blob format would close the question; not done yet. Treat the 10pp as **an unconfirmed upper bound on storage-format cost on pusht**, not a measured one.
