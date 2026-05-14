# Examples

All scripts live under [`examples/`](https://github.com/lancedb/lerobot-lancedb/tree/main/examples).

## `train_and_eval_lance.py` — end-to-end training

Trains `DiffusionPolicy` on a Lance-backed dataset using the same recipe as the published [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht) (200k steps, batch 64, `crop_shape=(84,84)` with random crop, grad-clip 10, cosine LR with 500 warmup, ImageNet image norm). Saves a checkpoint that `lerobot-eval` can load directly.

Typical flow:

```bash
# 1. Convert
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

- `--video-loader` — use `LeRobotLanceVideoDataset` (mp4 blob layout).
- `--upstream-loader` — use upstream parquet+mp4 (head-to-head comparisons).
- `--no-imagenet-stats` — disable ImageNet image-norm override.
- `--no-crop` — disable the (84, 84) random crop augmentation.
- `--decode-device {auto,cpu,cuda}` — JPEG decode device (frames format only).
- `--eval-fraction 0.1` — offline (held-out frames) eval instead of env rollouts.

## `aloha_loader_parity.py` — head-to-head storage comparison

Trains an ACT policy on `lerobot/aloha_static_cups_open` from three different loaders and reports held-out action MSE / RMSE.

```bash
# Lance JPEG-95
python examples/aloha_loader_parity.py --loader lance \
    --lance-root ./aloha_cups_open_lance \
    --steps 30000 --seed 42 \
    --out outputs/train/aloha_parity_lance

# Lance video-blob
python examples/aloha_loader_parity.py --loader lance-video \
    --lance-root ./aloha_cups_open_lance_video \
    --steps 30000 --seed 42 \
    --out outputs/train/aloha_parity_lance_video

# Upstream parquet+mp4
python examples/aloha_loader_parity.py --loader upstream \
    --steps 30000 --seed 42 \
    --out outputs/train/aloha_parity_upstream
```

Each run writes `parity_metrics.json` (loss curve + final held-out MSE / RMSE) next to the checkpoint.

## `benchmark_formats.py` — cross-format benchmark

Reproduces the [Benchmarks](benchmarks.md) tables. Cached: re-runs skip already-converted formats.

```bash
# Single-frame read pattern
python examples/benchmark_formats.py \
    --repos lerobot/pusht lerobot/aloha_static_cups_open lerobot/koch_pick_place_5_lego \
    --num-batches 30 --warmup 5 --n-pixel-samples 16

# Delta-timestamps read pattern (realistic training)
python examples/benchmark_formats.py \
    --repos lerobot/aloha_static_cups_open \
    --num-batches 30 --warmup 5 --skip-pixel-diff \
    --delta-timestamps
```

Useful flags:

- `--formats jpeg-95 jpeg-100-444 video` — subset the format list.
- `--delta-timestamps` — realistic 8-frame-per-sample read pattern.
- `--skip-pixel-diff` — don't measure pixel fidelity (faster).
- `--skip-throughput` — just measure size + pixel fidelity.

## `conversion.py` — batch driver

Converts a list of datasets to the default JPEG layout. Has a `--benchmark` mode that runs the legacy NVJPEG throughput benchmark used in pre-v0.x releases. Prefer `lerobot-convert-to-lance-video` / `lerobot-convert-to-lance` for new conversions.

## `train_with_lance.py` — minimal smoke test

Ten-step DiffusionPolicy training on pusht — proves the loader plugs into a stock LeRobot training loop unchanged.

```bash
python examples/train_with_lance.py
```
