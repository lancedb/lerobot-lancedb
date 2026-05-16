#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Train a DiffusionPolicy on pusht (Lance-backed) and run offline eval.

End-to-end demo of using a Lance-backed dataset for both training and
prediction. The defaults reproduce the published ``lerobot/diffusion_pusht``
recipe — 200k steps, batch 64, cosine schedule with 500 warmup steps,
``crop_shape=(84, 84)``, gradient clipping at 10.0, and ImageNet image
normalization stats. With these defaults the resulting model matches
the upstream env-eval numbers (see the README table).

Prerequisites:
    # Convert pusht once (~10 s, 60 MB Lance)
    python examples/conversion.py            # produces outputs/datasets/pusht_lance/

Train + env eval:
    python examples/train_and_eval_lance.py \\
        --steps 200000 --out outputs/train/diffusion_pusht_lance
    lerobot-eval --policy.path=outputs/train/diffusion_pusht_lance \\
        --env.type=pusht --eval.n_episodes=500 --policy.device=cuda

Smaller offline-only smoke run (eval against held-out frames, MSE only):
    python examples/train_and_eval_lance.py --steps 2000 --eval-fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets import EpisodeAwareSampler, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES

from lerobot.datasets import LeRobotDataset

from lerobot_lancedb import LeRobotLanceDataset, LeRobotLanceVideoDataset

log = logging.getLogger("train_and_eval_lance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--lance-root",
        type=Path,
        default=Path("outputs/datasets/pusht_lance"),
        help="Directory holding the converted pusht Lance table + meta/ sidecar.",
    )
    p.add_argument("--repo-id", default="lerobot/pusht")
    p.add_argument("--out", type=Path, default=Path("outputs/train/diffusion_pusht_lance"))
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--decode-device", default="cpu",
                   help="Lance JPEG decode device. Default 'cpu' avoids per-worker CUDA contexts "
                        "growing unboundedly during long runs; use 'cuda' or 'auto' if you trust "
                        "the per-worker GPU footprint for your dataset size.")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=0,
                   help="If >0, also save checkpoints at every N steps under --out/checkpoints/<step>/.")
    p.add_argument("--crop-shape", type=int, nargs=2, default=(84, 84), metavar=("H", "W"),
                   help="DiffusionPolicy crop_shape (default (84, 84) matches lerobot/diffusion_pusht).")
    p.add_argument("--no-crop", action="store_true", help="Disable image cropping (override --crop-shape).")
    p.add_argument("--grad-clip-norm", type=float, default=10.0,
                   help="Gradient clipping max-norm (default 10.0 matches lerobot recipe).")
    p.add_argument("--no-imagenet-stats", action="store_true",
                   help="By default we override image mean/std with ImageNet stats, which is what "
                        "lerobot's `use_imagenet_stats=True` (the default) does. Pass this flag to "
                        "use per-dataset stats from meta.stats instead.")
    p.add_argument("--eval-fraction", type=float, default=0.0,
                   help="Episodes held out for offline eval. 0 disables offline eval (matches lerobot recipe).")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-max-frames", type=int, default=2048,
                   help="Cap on offline-eval frames (set 0 to use all held-out frames).")
    p.add_argument("--skip-train", action="store_true", help="Load checkpoint from --out and only run eval.")
    p.add_argument("--skip-eval", action="store_true", help="Train only, no offline eval.")
    p.add_argument("--seed", type=int, default=100000,
                   help="Default matches lerobot/diffusion_pusht (seed=100000).")
    p.add_argument("--upstream-loader", action="store_true",
                   help="Use the upstream parquet+mp4 LeRobotDataset instead of LeRobotLanceDataset. "
                        "Useful for head-to-head comparisons. Ignores --lance-root and --decode-device.")
    p.add_argument("--video-loader", action="store_true",
                   help="Use LeRobotLanceVideoDataset (Lance blob v2 + torchcodec on-the-fly decode) "
                        "instead of the default JPEG/PNG-encoded LeRobotLanceDataset. Requires the "
                        "dataset to have been converted with convert_to_lance_video.")
    return p.parse_args()


def split_episodes(num_episodes: int, eval_fraction: float) -> tuple[list[int], list[int]]:
    n_eval = max(1, int(round(num_episodes * eval_fraction)))
    eval_eps = list(range(num_episodes - n_eval, num_episodes))
    train_eps = list(range(0, num_episodes - n_eval))
    return train_eps, eval_eps


def build_eval_indices(meta, eval_episodes: list[int]) -> np.ndarray:
    starts = meta.episodes["dataset_from_index"]
    ends = meta.episodes["dataset_to_index"]
    chunks = [np.arange(starts[e], ends[e], dtype=np.int64) for e in eval_episodes]
    return np.concatenate(chunks)


def main() -> None:
    args = parse_args()
    # lerobot's own imports call logging.basicConfig; force=True wins.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s lance_root=%s", device, args.lance_root)

    if not args.upstream_loader and not args.lance_root.exists():
        raise FileNotFoundError(
            f"{args.lance_root} not found. Run `python examples/conversion.py` first "
            "or pass --upstream-loader to use the upstream parquet+mp4 path."
        )

    if args.upstream_loader:
        meta = LeRobotDatasetMetadata(repo_id=args.repo_id)
    else:
        meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.lance_root)
    if args.eval_fraction > 0:
        train_eps, eval_eps = split_episodes(meta.total_episodes, args.eval_fraction)
    else:
        # Match the official lerobot recipe: train on every episode; eval happens in the env.
        train_eps = list(range(meta.total_episodes))
        eval_eps = []
    log.info("episodes: total=%d train=%d eval=%d", meta.total_episodes, len(train_eps), len(eval_eps))

    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    crop_shape = None if args.no_crop else tuple(args.crop_shape)
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        crop_shape=crop_shape,
    )
    policy = DiffusionPolicy(cfg).to(device)

    dataset_stats = meta.stats
    if not args.no_imagenet_stats:
        # Match `lerobot.datasets.factory.make_dataset` with use_imagenet_stats=True:
        # the camera-key mean/std get replaced with ImageNet stats before being passed to
        # make_pre_post_processors. Defaults-on in lerobot — required for parity with the
        # published lerobot/diffusion_pusht checkpoint.
        imagenet_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32)
        imagenet_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32)
        for key in cfg.image_features.keys():
            dataset_stats[key]["mean"] = imagenet_mean
            dataset_stats[key]["std"] = imagenet_std
        log.info("normalizing images with ImageNet stats (lerobot default)")

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)

    delta_timestamps = {
        "observation.image": [t / meta.fps for t in cfg.observation_delta_indices],
        "observation.state": [t / meta.fps for t in cfg.observation_delta_indices],
        ACTION: [t / meta.fps for t in cfg.action_delta_indices],
    }

    def make_dataset() -> torch.utils.data.Dataset:
        if args.upstream_loader:
            return LeRobotDataset(
                repo_id=args.repo_id,
                delta_timestamps=delta_timestamps,
            )
        if args.video_loader:
            return LeRobotLanceVideoDataset(
                root=args.lance_root,
                repo_id=args.repo_id,
                delta_timestamps=delta_timestamps,
                return_uint8=True,
            )
        return LeRobotLanceDataset(
            root=args.lance_root,
            repo_id=args.repo_id,
            delta_timestamps=delta_timestamps,
            return_uint8=True,
            decode_device=args.decode_device,
        )

    if not args.skip_train:
        train_dataset = make_dataset()
        sampler = EpisodeAwareSampler(
            meta.episodes["dataset_from_index"],
            meta.episodes["dataset_to_index"],
            episode_indices_to_use=train_eps,
            drop_n_last_frames=cfg.drop_n_last_frames,
            shuffle=True,
        )
        # pin_memory only works for CPU tensors. With decode_device='cpu' (the safer default)
        # or the upstream parquet+mp4 loader, tensors are on CPU and can be pinned.
        # With decode_device='cuda'/'auto', the Lance dataloader yields CUDA tensors and we can't.
        pin = device.type == "cuda" and (args.upstream_loader or args.video_loader or args.decode_device == "cpu")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            sampler=sampler,
            pin_memory=pin,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        # Use the policy's official optimizer + scheduler preset so this run
        # matches lerobot's train.py recipe exactly (Adam betas=(0.95, 0.999),
        # wd=1e-6, cosine schedule with 500 warmup steps for DiffusionPolicy).
        optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
        scheduler = cfg.get_scheduler_preset().build(optimizer, args.steps)
        policy.train()
        log.info("training: steps=%d batch_size=%d optimizer=%s scheduler=%s",
                 args.steps, args.batch_size,
                 type(optimizer).__name__, type(scheduler).__name__ if scheduler else "none")

        step = 0
        running = 0.0
        t0 = time.perf_counter()
        done = False
        while not done:
            for batch in train_loader:
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                running += float(loss.item())
                step += 1
                if step % args.log_every == 0:
                    avg = running / args.log_every
                    elapsed = time.perf_counter() - t0
                    sps = step / elapsed
                    eta = (args.steps - step) / sps if sps > 0 else float("inf")
                    lr = optimizer.param_groups[0]["lr"]
                    log.info("step %6d/%d  loss=%.4f  lr=%.2e  steps/s=%.2f  eta=%.0fs",
                             step, args.steps, avg, lr, sps, eta)
                    running = 0.0
                if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
                    ckpt = args.out / "checkpoints" / f"{step:07d}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(ckpt)
                    preprocessor.save_pretrained(ckpt)
                    postprocessor.save_pretrained(ckpt)
                    log.info("checkpoint → %s", ckpt)
                if step >= args.steps:
                    done = True
                    break

        log.info("saving policy + processors to %s", args.out)
        policy.save_pretrained(args.out)
        preprocessor.save_pretrained(args.out)
        postprocessor.save_pretrained(args.out)
    else:
        log.info("loading policy from %s", args.out)
        policy = DiffusionPolicy.from_pretrained(args.out).to(device)
        # preprocessor/postprocessor already constructed from meta.stats above

    # ---- offline eval ----
    if args.skip_eval or not eval_eps:
        log.info("skipping offline eval (eval_eps=%d, --skip-eval=%s)", len(eval_eps), args.skip_eval)
        return
    eval_indices = build_eval_indices(meta, eval_eps)
    if args.eval_max_frames > 0 and len(eval_indices) > args.eval_max_frames:
        rng = np.random.default_rng(args.seed)
        eval_indices = np.sort(rng.choice(eval_indices, size=args.eval_max_frames, replace=False))
    log.info("eval: frames=%d batches~=%d", len(eval_indices), int(np.ceil(len(eval_indices) / args.eval_batch_size)))

    # New dataset instance; we drive sample selection ourselves via a SubsetSampler.
    eval_dataset = make_dataset()
    eval_pin = device.type == "cuda" and (args.upstream_loader or args.video_loader or args.decode_device == "cpu")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=args.num_workers,
        batch_size=args.eval_batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(eval_indices.tolist(), generator=torch.Generator().manual_seed(args.seed)),
        pin_memory=eval_pin,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    image_keys = list(cfg.image_features.keys())
    start = cfg.n_obs_steps - 1
    end = start + cfg.n_action_steps

    policy.eval()
    preds, gts, ep_idxs, frame_idxs = [], [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch_raw in enumerate(eval_loader):
            # Move to device. The dataloader may already produce CUDA tensors
            # when decode_device=auto resolves to cuda — `.to(device)` is a no-op there.
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_raw.items()}
            batch_norm = preprocessor(batch_dev)
            stacked = torch.stack([batch_norm[k] for k in image_keys], dim=-4)
            gen_batch = {**batch_norm, OBS_IMAGES: stacked}
            predicted_norm = policy.diffusion.generate_actions(gen_batch)  # (B, n_action_steps, A)
            predicted = postprocessor(predicted_norm)  # un-normalized
            gt = batch_dev[ACTION][:, start:end]  # (B, n_action_steps, A)

            preds.append(predicted.cpu())
            gts.append(gt.cpu())
            ep_idxs.append(batch_dev["episode_index"].cpu())
            frame_idxs.append(batch_dev["frame_index"].cpu())
            if (i + 1) % 10 == 0:
                log.info("  eval batch %d", i + 1)

    preds_t = torch.cat(preds).numpy()      # (N, n_action_steps, A)
    gts_t = torch.cat(gts).numpy()
    ep_t = torch.cat(ep_idxs).numpy()
    fr_t = torch.cat(frame_idxs).numpy()

    per_sample_mse = ((preds_t - gts_t) ** 2).mean(axis=(1, 2))
    mse = float(per_sample_mse.mean())
    rmse = float(np.sqrt(mse))
    per_step_mse = ((preds_t - gts_t) ** 2).mean(axis=(0, 2)).tolist()  # over time
    metrics = {
        "n_samples": int(preds_t.shape[0]),
        "n_action_steps": int(preds_t.shape[1]),
        "action_dim": int(preds_t.shape[2]),
        "mse": mse,
        "rmse": rmse,
        "per_step_mse": per_step_mse,
        "eval_seconds": time.perf_counter() - t0,
        "device": str(device),
        "eval_episodes": eval_eps,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics: mse=%.4f rmse=%.4f over %d samples", mse, rmse, preds_t.shape[0])

    # Save predictions parquet: one row per (sample, step), columns
    # episode_index, frame_index (the conditioning frame), step_offset, predicted_action_*, gt_action_*.
    n_samples, n_steps, action_dim = preds_t.shape
    rows_episode = np.repeat(ep_t, n_steps)
    rows_frame = np.repeat(fr_t, n_steps)
    rows_step = np.tile(np.arange(n_steps), n_samples)
    pred_flat = preds_t.reshape(-1, action_dim)
    gt_flat = gts_t.reshape(-1, action_dim)

    table = pa.table({
        "episode_index": rows_episode,
        "frame_index": rows_frame,
        "step_offset": rows_step,
        **{f"pred_action_{i}": pred_flat[:, i] for i in range(action_dim)},
        **{f"gt_action_{i}": gt_flat[:, i] for i in range(action_dim)},
    })
    out_parquet = args.out / "predictions.parquet"
    pq.write_table(table, out_parquet)
    log.info("predictions → %s (rows=%d)", out_parquet, table.num_rows)


if __name__ == "__main__":
    main()
