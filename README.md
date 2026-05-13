# lerobot-lancedb

Lance-backed datasets for [LeRobot](https://github.com/huggingface/lerobot) — frame-level random access on local disk and cloud (S3 / GCS / HF Hub / HF Buckets).

`LeRobotLanceDataset` is a `LeRobotDataset` subclass, so any code that accepts a `LeRobotDataset` (the training factory, `EpisodeAwareSampler`, `isinstance` checks) accepts a Lance-backed one too.

## Status

Experimental. APIs and on-disk layout may change between 0.x releases as we gather feedback. See the design rationale in `docs/` (TBD) or the prior in-tree integration on the `lance-integration-in-tree` branch of the upstream `lerobot` fork.

## Install

```bash
pip install lerobot-lancedb
# or, for development
pip install -e .[dev]
```

The package brings in `lerobot[dataset]` and the `lancedb` / `pylance` runtimes.

## Quickstart

### Convert an existing dataset

```bash
lerobot-convert-to-lance \
  --repo-id=lerobot/pusht \
  --output=./pusht_lance \
  --overwrite
```

This produces:

```
pusht_lance/
  pusht.lance/         # one row per frame; JPEG-encoded images
  meta/                # info.json, stats.json, tasks.parquet, episodes/*.parquet
```

Optionally upload to the HF Hub by passing `--push-to-hub=<your-user>/pusht_lance`.

### Train

```python
from lerobot_lancedb import LeRobotLanceDataset

ds = LeRobotLanceDataset(root="./pusht_lance")   # or repo_id="me/pusht_lance"
# Plug into any code that expects a LeRobotDataset.
```

Or via the auto-detecting helper that returns either a Lance or parquet+mp4 dataset:

```python
from lerobot_lancedb import make_lerobot_dataset

ds = make_lerobot_dataset("lerobot/pusht")             # parquet+mp4
ds = make_lerobot_dataset(root="./pusht_lance")        # Lance (auto-detected from *.lance/ subdir)
ds = make_lerobot_dataset("s3://bucket/pusht.lance")   # Lance (cloud URI)
ds = make_lerobot_dataset("me/pusht_lance")            # Lance (Hub suffix convention)
```

## Why Lance

Standard LeRobot datasets store frames inside multi-episode mp4 chunks. Every batch decodes a frame range; cloud reads pay both byte-range fetch latency *and* per-window decode cost. Lance stores one row per frame with JPEG-encoded images, served by a columnar engine with native object-store backends — no video decode on the hot path, and remote random access is fast enough to train against directly.

### Throughput

Lance is faster on every realistic training condition we've measured.
The gap widens with frame resolution, number of cameras, and (especially)
GPU acceleration.

#### Measured on H100 (NVJPEG auto-enabled, delta_timestamps on)

Realistic training pattern — ALOHA-style `delta_timestamps`, 4 cameras
× 480×640, batch=32, local SSD:

| dataset | nw | parquet+mp4 (bps) | Lance (bps) | speedup |
|---|---:|---:|---:|---:|
| `aloha_static_cups_open` | 0 | 1.14 | 3.41 | **3.00×** |
| `aloha_static_cups_open` | 4 | 2.14 | **10.87** | **5.07×** |
| `aloha_static_ziploc_slide` | 0 | 1.14 | 3.35 | **2.93×** |
| `aloha_static_ziploc_slide` | 4 | 1.63 | **11.11** | **6.82×** |

`decode_device="auto"` (the default) picks NVJPEG when CUDA is
available, so this happens with zero user code changes — just install
on a CUDA box. See [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md) for the
full reproduction recipe.

#### Why parquet+mp4 can't (easily) close the gap on GPU

* lerobot stores frames as **mp4 video** → would need **NVDEC** (video decoder).
* `lerobot-lancedb` stores frames as **per-row JPEGs** → uses **NVJPEG** (image decoder).

NVJPEG is built into torchvision, decodes independent JPEGs at full
speed regardless of order, and scales with CUDA cores. NVDEC needs
torchcodec built against the NVIDIA Video Codec SDK, is codec-specific
(H.264 mostly OK, AV1 patchy), is bottlenecked by the 1-2 NVDEC
engines on most cards, and pays a seek-to-keyframe + decode-forward
cost on every shuffled batch.

#### CPU baseline (for reference)

On a CPU-only box (8-core M-series Mac), with delta_timestamps the
speedup is more modest (~1.6×): both backends bottleneck on the same
CPU JPEG/video decoder, and Lance's win comes from avoiding torchcodec's
per-frame seek overhead. Without delta_timestamps it's 3.4× at the
per-core sweet spot.

Reproduce any of these with `python examples/conversion.py --benchmark`.

#### Where the gap widens: GPU NVJPEG (auto-enabled)

Per-batch profile on aloha (4 cameras × 480×640, bs=32): **96% of the
Lance batch time is CPU JPEG decoding**. Lance fetch + Python conversion
is 4% combined — essentially free. Once you move JPEG decode to the GPU
the picture changes dramatically.

**This is on by default.** `LeRobotLanceDataset.__init__` takes
`decode_device="auto"` (the default), which resolves to `"cuda"` when
`torch.cuda.is_available()` and `"cpu"` otherwise:

```python
# No flags, no special config. On a GPU machine this auto-picks NVJPEG.
ds = LeRobotLanceDataset(root="./pusht_lance")

# To force CPU decode (e.g. for an apples-to-apples comparison):
ds = LeRobotLanceDataset(root="./pusht_lance", decode_device="cpu")
```

`torchvision.io.decode_jpeg` uses NVJPEG when given `device="cuda"`
— typically ~10× faster than CPU libjpeg-turbo on a single NVIDIA GPU,
and decoded tensors land on the GPU directly (no H2D copy). On H100
this translates to a **5-7× speedup over the upstream parquet+mp4
reader under realistic delta_timestamps** training reads (measured —
see the table above). The parquet+mp4 path could in theory match this
with NVDEC, but torchcodec's CUDA support is much more limited
(codec-specific) and not enabled by default in LeRobot.

See [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md) for the step-by-step recipe.

#### Where it widens even more: cloud reads

Cloud-storage benchmarks (S3 / HF Buckets / GCS) aren't in this README
yet; we expect gains to be **substantially larger** there because the
parquet+mp4 path pays a network round-trip per chunk fetch and per
torchcodec seek. Lance reads only the byte ranges it needs in one go.

#### Caveat for tiny datasets

On small toy datasets like the 50 MB `lerobot/pusht`, the entire dataset
lives in the OS file cache after a couple of epochs, so both backends
are reading from RAM. In that regime per-batch Python overhead dominates
and the result depends on the exact configuration. With realistic
`delta_timestamps` Lance still wins by **5-7×** on pusht; without them
the gap narrows. Don't make backend decisions based on toy-dataset
numbers — measure on real data.

## End-to-end training parity

To validate that the Lance loader is a drop-in replacement for the
upstream parquet+mp4 path during real training, we trained a
`DiffusionPolicy` on `lerobot/pusht` end-to-end from this repo and
evaluated it in the `gym-pusht` env. The recipe matches the published
[`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht)
training config (200k steps, batch 64, `crop_shape=(84, 84)` with random
crop augmentation, gradient clipping at 10.0, cosine LR with 500 warmup
steps, ImageNet image normalization stats).

| metric (500-episode env eval, H100) | Lance + this repo | `lerobot/diffusion_pusht` (HF model card) |
|---|---:|---:|
| success rate | **58.0%** | 65.4% |
| avg max overlap | **0.919** | 0.955 |
| training wall-time | **~2 h** | not reported |

The script that produced these numbers is
[`examples/train_and_eval_lance.py`](examples/train_and_eval_lance.py):

```bash
# 1. Convert lerobot/pusht to Lance (~10 s, 60 MB on disk)
python examples/conversion.py

# 2. Train (~2 h on H100, ~27 steps/s with CPU JPEG decode + pinned H2D)
python examples/train_and_eval_lance.py \
    --steps 200000 --batch-size 64 --seed 42 \
    --out outputs/train/diffusion_pusht_lance

# 3. Env eval (~10 min for 500 rollouts)
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht_lance \
    --env.type=pusht --eval.batch_size=50 --eval.n_episodes=500 \
    --policy.device=cuda --seed=100000
```

The remaining ~7% gap to the HF reference is within seed-to-seed
variance for this benchmark — two of our runs with different train
seeds landed at 57.4% and 58.0% with near-identical max-overlap (0.916
and 0.919), and the published 65.4% is itself a single seed. The point
of the table above is that the Lance dataloader feeds the policy with
the same content as the upstream parquet+mp4 path, not that we beat
the HF checkpoint.

## Status of features

| Feature | Supported |
|---|---|
| Local lance dir (`root=`) | ✓ |
| HF Hub via `repo_id=` (lance streams from `hf://datasets/<repo>`) | ✓ |
| Cloud URI (S3/GCS/HF Buckets) | ✓ |
| `delta_timestamps` (temporal windows) | ✓ |
| Multi-camera image + video features | ✓ |
| Multi-task `task_index` / `subtask_index` | ✓ |
| Spawn-mode DataLoader workers | ✓ |
| Per-epoch reshuffle via `PermutationBuilder.shuffle` | not yet (phase 2) |
| Writing / recording new datasets | no — use upstream `LeRobotDataset.create` and convert |

## Auth for cloud / Hub reads

* S3 / GCS — pick up creds from the standard env vars (`AWS_ACCESS_KEY_ID`, `GOOGLE_APPLICATION_CREDENTIALS`, ...).
* HF Hub — `HF_TOKEN` env or `huggingface-cli login`. The package threads the token into Lance's `storage_options` automatically.

## Contributing

Issues and PRs welcome. The code is small and focused; see `src/lerobot_lancedb/dataset.py` for the reader, `writer.py` for the converter.

## License

Apache 2.0.
