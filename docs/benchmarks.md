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

End-to-end check that the loader trains models that actually learn and match upstream's behaviour. Two complementary tests: pusht with `DiffusionPolicy` + env rollouts (the rigorous one — measures actual task success), and ALOHA cups_open with ACT + held-out action MSE (no sim env exists for this dataset).

### pusht — `DiffusionPolicy`, 200k steps, env eval

Same recipe as [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht). Seed=42 for direct comparison; eval is 500 `gym-pusht` rollouts at seed=100000.

| storage format | env success rate | avg max overlap |
|---|---:|---:|
| Lance JPEG-95 | 58.0 % | 0.919 |
| Lance video-blob | **68.4 %** | **0.936** |
| upstream parquet+mp4 (head-to-head) | 68.0 % | 0.9586 |
| HF model card (seed=100000) | 65.4 % | 0.955 |

What this says:

- The **video-blob format trains a working policy that matches upstream's env-eval result** within seed-to-seed noise. The HF reference at seed=100000 lands at 65.4 % and our seed=42 upstream run lands at 68.0 %, putting video-blob's 68.4 % squarely in that range.
- **JPEG-95 storage costs ~10 pp success rate on pusht.** pusht has sharp synthetic edges (a flat T-shape on flat background) — exactly the content type JPEG handles worst (6.2 % of pixels visibly differ vs source). 200 k diffusion training amplifies that into a measurable env-eval gap.
- The fix on pusht is `convert_to_lance_video`. JPEG-100 + 4:4:4 would also help (visible-pixel artifacts drop ~90× vs JPEG-95) but we didn't run it end-to-end at 200 k.

### ALOHA cups_open — ACT, 30k steps, held-out action MSE

Same recipe (ACT defaults + ImageNet image norm + grad-clip 10), seed=42, `num_workers=4`. Held-out MSE on the last 10 % of episodes.

| storage format | train loss @ 30k | held-out RMSE |
|---|---:|---:|
| Lance JPEG-95 (default) | 0.0962 | 0.0927 |
| Lance JPEG-100 + 4:4:4 | 0.0961 | 0.0872 |
| Lance video-blob | 0.0972 | 0.0901 |

All three modes land within ~6 % of each other. On natural multi-camera footage at this training scale the format choice doesn't surface a measurable accuracy effect — JPEG-95's visible-pixel artifact rate on ALOHA is only 1.4 %, an order of magnitude lower than on pusht.

So the two findings are consistent rather than contradictory: storage format matters when the content has sharp edges that JPEG rings on, and when the training pipeline is sensitive enough to surface it.

### Bit-exact substantiation

For `convert_to_lance_video` specifically, we verified by direct comparison against upstream:

- Pixel bytes are bit-identical (multiple sample frames, all cameras).
- Tabular fields (state, action, timestamps, indices) are bit-identical.
- Action-pad masks under `delta_timestamps` match including at episode boundaries.
- Model init weights at seed=42 are bit-identical regardless of which loader's metadata is loaded first.

Reproduce: [`examples/train_and_eval_lance.py`](https://github.com/lancedb/lerobot-lancedb/blob/main/examples/train_and_eval_lance.py) (pusht) and [`examples/aloha_loader_parity.py`](https://github.com/lancedb/lerobot-lancedb/blob/main/examples/aloha_loader_parity.py) (ALOHA).
