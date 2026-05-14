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

End-to-end check that all three Lance storage modes train models to the same place. ALOHA cups_open, ACT, 30k steps, seed=42, `num_workers=4`. Held-out action MSE on the last 10 % of episodes.

| storage format | train loss @ 30k | held-out RMSE |
|---|---:|---:|
| Lance JPEG-95 (default) | 0.0962 | 0.0927 |
| Lance JPEG-100 + 4:4:4 | 0.0961 | 0.0872 |
| Lance video-blob | 0.0972 | 0.0901 |

All three modes land within ~6 % of each other. **Pixel encoding choice has no detectable impact on training loss at this scale.** Whether to pick JPEG-95, JPEG-100/4:4:4, or video-blob is a size/throughput/fidelity decision (see the tables above), not a training-accuracy decision.

The bit-exact checks the parity claim relies on:

- Pixel bytes — verified bit-identical between Lance video-blob and upstream parquet+mp4.
- Tabular fields (state, action, timestamps, indices) — bit-identical.
- Action-pad masks under `delta_timestamps` — bit-identical including at episode boundaries.
- Model init weights at seed=42 — bit-identical regardless of which loader's metadata is loaded first.

Reproduce with [`examples/aloha_loader_parity.py`](https://github.com/lancedb/lerobot-lancedb/blob/main/examples/aloha_loader_parity.py).
