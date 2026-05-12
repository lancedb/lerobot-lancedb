# GPU benchmark: reproducing the NVJPEG numbers

This guide walks through running the parquet+mp4-vs-Lance throughput
benchmark on a CUDA machine. **NVJPEG is enabled by default** in the
core library — `LeRobotLanceDataset(...)` resolves `decode_device="auto"`
to CUDA when `torch.cuda.is_available()`, with no flags or code changes.
The benchmark + training scripts inherit that default. You should see
~7–10× over the CPU-Lance numbers in the README and ~10–15× over the
upstream parquet+mp4 reader on local SSD.

The whole flow is ~30 minutes wall time on an A100/H100 box with fast
disk (most of which is downloading the source datasets the first time).

## Prereqs

* A Linux box with an NVIDIA GPU (anything Pascal or newer; tested
  expectation on Ampere / Hopper).
* CUDA 11.8+ and a recent NVIDIA driver.
* Python 3.12+.
* ~10 GB free disk for the source ALOHA dataset + its Lance copy.

If you're on a rented box (Lambda Labs, RunPod, Modal, etc.), the
default deep-learning images already have CUDA + Python configured.

## 1. Clone + install

```bash
git clone https://github.com/lancedb/lerobot-lancedb.git
cd lerobot-lancedb

# Optional: a clean venv. Skip if you're using the system Python.
python -m venv .venv && source .venv/bin/activate

pip install -e '.[test]'
```

`pip install -e '.[test]'` brings in `lerobot`, `lancedb`, `pylance`,
`torch`, `torchvision`, `torchcodec`, and the test deps. PyTorch's
default index gives you a CUDA-enabled build automatically on Linux.

Sanity-check the GPU is visible to PyTorch *and* that torchvision's
NVJPEG path works:

```bash
python -c "
import torch
from torchvision.io import decode_jpeg, ImageReadMode
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))

# NVJPEG smoke: decode a tiny JPEG on GPU.
import io, PIL.Image, numpy as np
img = PIL.Image.fromarray(np.zeros((16, 16, 3), dtype='uint8'))
buf = io.BytesIO(); img.save(buf, format='JPEG'); buf.seek(0)
t = torch.frombuffer(buf.read(), dtype=torch.uint8)
out = decode_jpeg([t], mode=ImageReadMode.RGB, device='cuda')[0]
print('NVJPEG decoded shape:', tuple(out.shape), 'device:', out.device)
"
```

Both prints should succeed. If the NVJPEG line fails, your
torchvision wasn't built against CUDA — reinstall with
`pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121`
(or your CUDA version).

## 2. Run the tests (sanity)

```bash
pytest -q
```

You should see **17 passed** (the previously CUDA-skipped
`test_decode_on_gpu_returns_cuda_tensors` test will now run).

## 3. Convert the benchmark datasets

```bash
python examples/conversion.py --overwrite
```

This downloads and converts four datasets:

* `lerobot/pusht` (~6 s, 60 MB Lance — used only as a conversion
  example, not benchmarked)
* `lerobot/pusht_image` (same)
* `lerobot/pusht-subtask` (same)
* `lerobot/aloha_static_cups_open` (~10–15 min on a typical box,
  3.6 GB Lance — this is what gets benchmarked)
* `lerobot/aloha_static_ziploc_slide` (similar, 3.2 GB Lance)

The Lance outputs land in `outputs/datasets/`. If your run gets
interrupted, re-run without `--overwrite` to skip already-converted
ones, or drop `--overwrite` to skip everything that already exists.

## 4. Run the benchmark

```bash
python examples/conversion.py --skip-convert --benchmark
```

`--skip-convert` reuses the outputs from step 3 and goes straight to
the benchmark. With `--decode-device=auto` (the default) the Lance
reader picks CUDA automatically — you should see a log line like:

```
Benchmark decode_device='auto' resolved to 'cuda' (torch.cuda.is_available()=True)
```

The benchmark runs each large dataset under four conditions:

* shuffled, num_workers=0
* shuffled, num_workers=4
* shuffled + delta_timestamps, num_workers=0  ← realistic training pattern
* shuffled + delta_timestamps, num_workers=4

For each you get a parquet-vs-lance table with the speedup ratio.

### Useful flags

```bash
# Try higher num_workers (e.g. 8 on a workstation with many cores).
python examples/conversion.py --skip-convert --benchmark \
    --bench-num-workers 0 4 8

# Force CPU decode for a head-to-head with the CPU numbers in the README.
python examples/conversion.py --skip-convert --benchmark \
    --decode-device=cpu

# Bigger batches (closer to what real training uses).
python examples/conversion.py --skip-convert --benchmark \
    --bench-batch-size=64
```

## What to expect

On an Ampere / Hopper GPU with the dataset on local NVMe, with
`delta_timestamps` on and `num_workers=4`:

| backend | bps (M-series Mac, CPU) | bps (CUDA, NVJPEG) | speedup vs CPU lance |
|---|---:|---:|---:|
| parquet+mp4 | 1.6 | ~1.6 (unchanged) | — |
| **Lance** | 2.5 | **~17–18 (projected)** | **~7×** |
| **Lance speedup vs parquet** | 1.6× | **~10–11× (projected)** | |

Without `delta_timestamps`, expect ~15–20× over parquet+mp4 on
the same GPU. (The smaller speedup with delta_timestamps is because
torchcodec amortizes the multi-frame seek per camera; this is the
worst case for Lance, and even it improves to 10×+ on GPU.)

## Reporting back

If you run this, please paste the output. The numbers above are
extrapolated from a CPU profile (96% of the per-batch time is JPEG
decode; NVJPEG is ~10× faster). Real measurements will probably
differ — better on bigger GPUs / faster interconnects, possibly
worse on consumer cards with limited CUDA-context memory.

The pieces I want to see verified:

1. `decode_device="auto"` correctly resolves to CUDA on your box.
2. NVJPEG actually delivers the ~10× decode speedup (not bottlenecked
   on PCIe bandwidth back to CPU memory).
3. `num_workers > 0` works with CUDA decode — spawn-mode workers each
   create their own CUDA context. Should be fine on a 16 GB+ GPU but
   could cause OOM on smaller cards.

Open an issue with the run output and we'll update the README with
real numbers in place of the extrapolation.
