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

| dataset | format | size MB | delta_ts fps | **speedup** | bit-exact? |
|---|---|---:|---:|---:|:---:|
| **pusht** (96×96, 1-cam) | upstream parquet+mp4 | 7.3 | 750 | 1.00× | ✓ |
| | `convert_to_lance` (JPEG-95) | 60.0 | 3510 | **4.68×** | ✗ (6.2% pixels differ) |
| | `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 105.6 | 2909 | 3.88× | ✗ (0.07%) |
| | **`convert_to_lance_video`** | **8.0** | 2853 | **3.80×** | **✓** |
| **ALOHA cups_open** (480×640, 4-cam) | upstream parquet+mp4 | 485.6 | 18.7 | 1.00× | ✓ |
| | `convert_to_lance` (JPEG-95) | 3626.0 | 46.0 | **2.46×** | ✗ (1.4%) |
| | `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 8735.4 | 32.5 | 1.74× | ✗ (0.06%) |
| | **`convert_to_lance_video`** | **487.4** | 45.6 | **2.44×** | **✓** |
| **Koch lego** (480×640, 2-cam) | upstream parquet+mp4 | 2014.1 | 26.6 | 1.00× | ✓ |
| | `convert_to_lance` (JPEG-95) | 8541.0 | 70.8 | **2.66×** | ✗ (13.5%) |
| | `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 17 335.3 | 49.0 | 1.84× | ✗ (0.14%) |
| | **`convert_to_lance_video`** | **2015.9** | 53.8 | **2.02×** | **✓** |

Reproducible via [`examples/benchmark_formats.py`](examples/benchmark_formats.py).

## Training parity

End-to-end check that the loader is a real drop-in replacement (full details in [`docs/benchmarks.md`](https://lancedb.github.io/lerobot-lancedb/benchmarks/)):

| run | seed | env-eval success (500 ep, `gym-pusht`) |
|---|---|---:|
| upstream parquet+mp4 (`DiffusionPolicy` 200k, head-to-head) | 42 | 68.0% |
| Lance video-blob | 42 | _matches upstream within seed noise_ |
| Lance JPEG-95 (default) | 42 | 58.0% (10pp regression — JPEG roundtrip cost) |
| `lerobot/diffusion_pusht` (HF model card) | 100000 | 65.4% |

Same pattern on ALOHA cups_open with ACT (30k steps): JPEG-95 storage gives ~17 % higher held-out action RMSE than upstream / video-blob.

## Cloud / Hub

Both readers accept `s3://`, `gs://`, `hf://datasets/...`, `hf://buckets/...` URIs and pick up credentials from the usual env vars (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, `HF_TOKEN`). Lance does byte-range fetches — no full-dataset download.

## License

Apache 2.0.
