# lerobot-lancedb

📖 **Docs: <https://lancedb.github.io/lerobot-lancedb/>**

Lance-backed datasets for [LeRobot](https://github.com/huggingface/lerobot). Drop-in replacement for `LeRobotDataset` with two storage layouts:

- **`LeRobotLanceDataset`** — per-frame JPEG bytes (lossy, fastest at single-frame access, optional GPU NVJPEG decode).
- **`LeRobotLanceVideoDataset`** — per-file mp4 bytes stored via Lance blob v2, decoded on the fly with torchcodec. Bit-exact pixels, ~same disk size as upstream.

Both subclass `LeRobotDataset` so existing trainers / samplers / `isinstance` checks accept them transparently.

## Install

Until the first PyPI release, install from GitHub:

```bash
pip install git+https://github.com/lancedb/lerobot-lancedb.git
```

Or, for local development:

```bash
git clone https://github.com/lancedb/lerobot-lancedb.git
cd lerobot-lancedb
pip install -e '.[dev]'
```

## Quickstart

```bash
# Convert (recommended path for dtype=video sources)
lerobot-convert-to-lance-video \
    --repo-id=lerobot/aloha_static_cups_open \
    --output=./aloha_cups_open_lance_video --overwrite
```

```python
from lerobot_lancedb import LeRobotLanceVideoDataset
ds = LeRobotLanceVideoDataset(root="./aloha_cups_open_lance_video")
```

For the JPEG layout, use `lerobot-convert-to-lance` and `LeRobotLanceDataset` instead. See the [docs](https://lancedb.github.io/lerobot-lancedb/) for the full CLI / API reference.

## Benchmark

Realistic training read pattern (`delta_timestamps`, 8 frames / sample, batch 32, num_workers 4, CPU decode, H100):

| dataset | format | size MB | delta_ts fps | **speedup** |
|---|---|---:|---:|---:|
| **pusht** (96×96, 1-cam) | upstream parquet+mp4 | 7.3 | 750 | 1.00× |
| | `convert_to_lance` (JPEG-95) | 60.0 | 3510 | **4.68×** |
| | `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 105.6 | 2909 | 3.88× |
| | **`convert_to_lance_video`** | **8.0** | 2853 | **3.80×** |
| **ALOHA cups_open** (480×640, 4-cam) | upstream parquet+mp4 | 485.6 | 18.7 | 1.00× |
| | `convert_to_lance` (JPEG-95) | 3626.0 | 46.0 | **2.46×** |
| | `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 8735.4 | 32.5 | 1.74× |
| | **`convert_to_lance_video`** | **487.4** | 45.6 | **2.44×** |
| **Koch lego** (480×640, 2-cam) | upstream parquet+mp4 | 2014.1 | 26.6 | 1.00× |
| | `convert_to_lance` (JPEG-95) | 8541.0 | 70.8 | **2.66×** |
| | `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 17 335.3 | 49.0 | 1.84× |
| | **`convert_to_lance_video`** | **2015.9** | 53.8 | **2.02×** |

Reproducible via [`examples/benchmark_formats.py`](examples/benchmark_formats.py).

## Training parity

End-to-end check that the loader trains models that **actually learn and match the upstream parquet+mp4 path in env-eval**.

**pusht — `DiffusionPolicy` 200k, seed=42, gym-pusht 500-rollout eval:**

| storage format | env success rate | avg max overlap |
|---|---:|---:|
| `convert_to_lance` (JPEG-95) | 58.0 % | 0.919 |
| **`convert_to_lance_video`** | **68.4 %** | **0.936** |
| upstream parquet+mp4 (head-to-head) | 68.0 % | 0.9586 |
| HF model card ([`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht), seed=100000) | 65.4 % | 0.955 |

`convert_to_lance_video` matches upstream's env-eval result within seed noise. JPEG-95 storage costs ~10 pp on this dataset — pusht has sharp synthetic edges where JPEG ringing artifacts concentrate (6.2 % of pixels visibly differ) and 200 k diffusion training is sensitive to that. Pick the video format when your source is `dtype=video` and you care about accuracy.

**ALOHA cups_open — ACT 30k, seed=42, held-out action RMSE:**

| storage format | train loss @ 30k | held-out RMSE |
|---|---:|---:|
| `convert_to_lance` (JPEG-95) | 0.0962 | 0.0927 |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 0.0961 | 0.0872 |
| `convert_to_lance_video` | 0.0972 | 0.0901 |

Same recipe, same seed. All three modes land within ~6 % of each other — on natural multi-camera footage at this training scale the format choice doesn't surface a measurable accuracy effect.

Reproduce: [`examples/train_and_eval_lance.py`](examples/train_and_eval_lance.py) (pusht) and [`examples/aloha_loader_parity.py`](examples/aloha_loader_parity.py) (ALOHA). Full discussion in [`docs/benchmarks.md`](https://lancedb.github.io/lerobot-lancedb/benchmarks/).

## Cloud / Hub

Both readers accept `s3://`, `gs://`, `hf://datasets/...`, `hf://buckets/...` URIs and pick up credentials from the usual env vars (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, `HF_TOKEN`). Lance does byte-range fetches — no full-dataset download.

## License

Apache 2.0.
