# Examples

All scripts live under [`examples/`](https://github.com/lancedb/lerobot-lancedb/tree/main/examples) in the repo.

## End-to-end training: `train_and_eval_lance.py`

Trains a `DiffusionPolicy` on a Lance-backed dataset using the same recipe as the published [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht) (200k steps, batch 64, `crop_shape=(84,84)` with random crop, grad-clip 10, cosine LR with 500 warmup, ImageNet image norm). Saves a checkpoint that `lerobot-eval` can load directly.

```bash
# 1. Convert pusht (recommended: video format)
lerobot-convert-to-lance-video \
    --repo-id=lerobot/pusht --output=./pusht_lance_video --overwrite

# 2. Train (~2 h on H100)
python examples/train_and_eval_lance.py \
    --steps 200000 --batch-size 64 --seed 42 \
    --video-loader \
    --lance-root ./pusht_lance_video \
    --out outputs/train/diffusion_pusht_lance

# 3. Run env rollouts
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht_lance \
    --env.type=pusht --eval.batch_size=50 --eval.n_episodes=500 \
    --policy.device=cuda --seed=100000
```

Useful flags:

| Flag | Effect |
|---|---|
| `--video-loader` | Use `LeRobotLanceVideoDataset` (mp4 blob layout). |
| `--upstream-loader` | Use upstream parquet+mp4 — for head-to-head comparisons. |
| `--no-imagenet-stats` | Disable the ImageNet image-norm override. |
| `--no-crop` | Disable the (84, 84) random crop augmentation. |
| `--decode-device {auto,cpu,cuda}` | JPEG decode device (frames format only). |

For the offline (held-out frames, no env) eval path use `--eval-fraction 0.1` and `--eval-max-frames N`.

## ALOHA loader parity: `aloha_loader_parity.py`

Head-to-head training of an ACT policy across all storage layouts on `lerobot/aloha_static_cups_open`. Reports held-out action MSE / RMSE so you can compare storage modes apples-to-apples.

```bash
# Lance JPEG-95 (current default)
python examples/aloha_loader_parity.py --loader lance \
    --lance-root ./aloha_cups_open_lance \
    --steps 30000 --seed 42 \
    --out outputs/train/aloha_parity_lance

# Lance video-blob (recommended for video-stored sources)
python examples/aloha_loader_parity.py --loader lance-video \
    --lance-root ./aloha_cups_open_lance_video \
    --steps 30000 --seed 42 \
    --out outputs/train/aloha_parity_lance_video

# Upstream parquet+mp4 head-to-head
python examples/aloha_loader_parity.py --loader upstream \
    --steps 30000 --seed 42 \
    --out outputs/train/aloha_parity_upstream
```

Each run writes a `parity_metrics.json` next to the checkpoint with the loss curve + final held-out MSE / RMSE.

## Cross-format benchmark: `benchmark_formats.py`

Reproduces the tables in the [Benchmarks](benchmarks.md) page — size, throughput, and pixel fidelity across all four storage formats on a list of datasets.

```bash
# Single-frame read pattern
python examples/benchmark_formats.py \
    --repos lerobot/pusht lerobot/aloha_static_cups_open lerobot/koch_pick_place_5_lego \
    --num-batches 30 --warmup 5 --n-pixel-samples 16

# Delta-timestamps read pattern (realistic training)
python examples/benchmark_formats.py \
    --repos lerobot/pusht lerobot/aloha_static_cups_open \
    --num-batches 30 --warmup 5 --skip-pixel-diff \
    --delta-timestamps
```

Conversions are cached under `--out-root` (default `outputs/datasets/`); re-runs skip already-converted formats.

| Flag | Effect |
|---|---|
| `--formats jpeg-95 jpeg-100-444 png video` | Subset the format list. |
| `--delta-timestamps` | Use a realistic 8-frame-per-sample delta window. |
| `--skip-pixel-diff` | Don't measure pixel fidelity (faster). |
| `--skip-throughput` | Just measure size + pixel fidelity. |

## Conversion driver: `conversion.py`

Batch-converts a list of datasets to the JPEG layout (the historical default). Also has a `--benchmark` mode that runs the GPU NVJPEG throughput benchmark used in the original README. Mostly useful for reproducing the initial release results; prefer `lerobot-convert-to-lance-video` / `lerobot-convert-to-lance` for new conversions.

## Minimal smoke: `train_with_lance.py`

Ten-step DiffusionPolicy training loop on pusht, just to show the loader plugs into a stock LeRobot training loop with no other changes:

```bash
python examples/train_with_lance.py   # expects ./pusht_lance/ to exist
```
