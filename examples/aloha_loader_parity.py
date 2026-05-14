#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Lance-JPEG vs upstream-mp4 head-to-head on an ALOHA dataset.

Trains an ACT policy on `lerobot/aloha_static_cups_open` twice — once with
the Lance-backed loader (default JPEG-95 encoding), once with the upstream
parquet+mp4 loader — using the same recipe and seed, then reports the
held-out action MSE side by side. The point is to test whether the JPEG
roundtrip in the Lance writer measurably affects training accuracy on
natural-image (multi-camera, 480x640) ALOHA data, the way it does on
synthetic pusht.

Prerequisites:
    # 1. ALOHA already converted (~12 min, 3.6 GB)
    python examples/conversion.py            # produces outputs/datasets/aloha_static_cups_open_lance/

Run:
    # Train both, save predictions/MSE
    python examples/aloha_loader_parity.py --loader=lance     --steps 30000
    python examples/aloha_loader_parity.py --loader=upstream  --steps 30000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION

from lerobot_lancedb import LeRobotLanceDataset, LeRobotLanceVideoDataset

log = logging.getLogger("aloha_loader_parity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--loader",
        choices=("lance", "lance-video", "upstream"),
        required=True,
        help="lance: JPEG/PNG per-frame layout. lance-video: mp4-blob layout. upstream: parquet+mp4.",
    )
    p.add_argument("--repo-id", default="lerobot/aloha_static_cups_open")
    p.add_argument("--lance-root", type=Path, default=Path("outputs/datasets/aloha_static_cups_open_lance"))
    p.add_argument("--out", type=Path, default=None,
                   help="Default: outputs/train/aloha_parity_<loader>/")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--decode-device", default="cpu",
                   help="Lance JPEG decode device. Default 'cpu' (stable for long runs).")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="ACT default lr (1e-5 backbone, 1e-5 main) per lerobot recipe.")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-fraction", type=float, default=0.1,
                   help="Last N% of episodes held out for offline action-MSE eval.")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-max-frames", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_episodes(num_episodes: int, eval_fraction: float) -> tuple[list[int], list[int]]:
    n_eval = max(1, int(round(num_episodes * eval_fraction)))
    return list(range(num_episodes - n_eval)), list(range(num_episodes - n_eval, num_episodes))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.out is None:
        args.out = Path(f"outputs/train/aloha_parity_{args.loader}")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("loader=%s device=%s out=%s", args.loader, device, args.out)

    if args.loader in ("lance", "lance-video"):
        meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.lance_root)
    else:
        meta = LeRobotDatasetMetadata(repo_id=args.repo_id)
    log.info("dataset: %s  episodes=%d  frames=%d  fps=%d",
             args.repo_id, meta.total_episodes, meta.total_frames, meta.fps)

    train_eps, eval_eps = split_episodes(meta.total_episodes, args.eval_fraction)
    log.info("train_eps=%d eval_eps=%d", len(train_eps), len(eval_eps))

    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    # Use ACT defaults except seed; pin chunk size = ACT default = 100.
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg).to(device)

    # ImageNet image-stats override (matches lerobot use_imagenet_stats=True default).
    dataset_stats = meta.stats
    imagenet_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32)
    imagenet_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32)
    for key in cfg.image_features.keys():
        dataset_stats[key]["mean"] = imagenet_mean
        dataset_stats[key]["std"] = imagenet_std

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)

    # ACT loads single-frame observations (no observation delta indices) and an
    # action chunk of length cfg.chunk_size.
    delta_timestamps = {ACTION: [t / meta.fps for t in cfg.action_delta_indices]}

    def make_dataset() -> torch.utils.data.Dataset:
        if args.loader == "lance":
            return LeRobotLanceDataset(
                root=args.lance_root,
                repo_id=args.repo_id,
                delta_timestamps=delta_timestamps,
                return_uint8=True,
                decode_device=args.decode_device,
            )
        if args.loader == "lance-video":
            return LeRobotLanceVideoDataset(
                root=args.lance_root,
                repo_id=args.repo_id,
                delta_timestamps=delta_timestamps,
                return_uint8=True,
            )
        return LeRobotDataset(repo_id=args.repo_id, delta_timestamps=delta_timestamps)

    train_dataset = make_dataset()
    sampler = EpisodeAwareSampler(
        meta.episodes["dataset_from_index"],
        meta.episodes["dataset_to_index"],
        episode_indices_to_use=train_eps,
        shuffle=True,
    )
    # lance-video and upstream always return CPU tensors; lance only does if decode_device='cpu'.
    pin = device.type == "cuda" and (args.loader in ("upstream", "lance-video") or args.decode_device == "cpu")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    scheduler_preset = cfg.get_scheduler_preset()
    scheduler = scheduler_preset.build(optimizer, args.steps) if scheduler_preset is not None else None
    policy.train()
    log.info("training: steps=%d batch_size=%d optimizer=%s", args.steps, args.batch_size, type(optimizer).__name__)

    step = 0
    running = 0.0
    t0 = time.perf_counter()
    done = False
    losses_log: list[tuple[int, float]] = []
    while not done:
        for batch in train_loader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
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
                log.info("step %6d/%d loss=%.4f sps=%.2f eta=%.0fs", step, args.steps, avg, sps, eta)
                losses_log.append((step, avg))
                running = 0.0
            if step >= args.steps:
                done = True
                break

    train_wall = time.perf_counter() - t0
    log.info("training finished: %.0fs (%.2f sps)", train_wall, args.steps / train_wall)
    policy.save_pretrained(args.out)
    preprocessor.save_pretrained(args.out)
    postprocessor.save_pretrained(args.out)

    # Held-out MSE
    starts = meta.episodes["dataset_from_index"]
    ends = meta.episodes["dataset_to_index"]
    eval_indices = np.concatenate([np.arange(starts[e], ends[e], dtype=np.int64) for e in eval_eps])
    rng = np.random.default_rng(args.seed)
    if 0 < args.eval_max_frames < len(eval_indices):
        eval_indices = np.sort(rng.choice(eval_indices, size=args.eval_max_frames, replace=False))
    log.info("eval frames=%d", len(eval_indices))

    eval_dataset = make_dataset()
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=args.num_workers,
        batch_size=args.eval_batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(
            eval_indices.tolist(), generator=torch.Generator().manual_seed(args.seed)
        ),
        pin_memory=pin,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    policy.eval()
    sq_err_acc = 0.0
    count = 0
    t_eval = time.perf_counter()
    with torch.no_grad():
        for batch_raw in eval_loader:
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_raw.items()}
            batch_norm = preprocessor(batch_dev)
            pred_norm = policy.predict_action_chunk(batch_norm)  # (B, chunk, A)
            pred = postprocessor(pred_norm).to(device)
            gt = batch_dev[ACTION][:, : pred.shape[1]]
            mask = (~batch_dev["action_is_pad"][:, : pred.shape[1]]).unsqueeze(-1).float()
            sq = ((pred - gt) ** 2) * mask
            sq_err_acc += float(sq.sum().item())
            count += int(mask.sum().item() * pred.shape[-1])

    mse = sq_err_acc / max(count, 1)
    rmse = float(np.sqrt(mse))
    metrics = {
        "loader": args.loader,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_wall_seconds": train_wall,
        "train_steps_per_second": args.steps / train_wall,
        "eval_seconds": time.perf_counter() - t_eval,
        "eval_n_samples": int(count / max(pred.shape[-1], 1)),
        "eval_mse": mse,
        "eval_rmse": rmse,
        "loss_log": losses_log,
    }
    (args.out / "parity_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics: train=%.0fs (%.2f sps)  eval_mse=%.4f rmse=%.4f", train_wall, args.steps / train_wall, mse, rmse)


if __name__ == "__main__":
    main()
