#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Train a diffusion policy from a Lance-backed LeRobot dataset.

Three ways to load a Lance dataset (all subclass LeRobotDataset; any code
that takes a LeRobotDataset works):

  # Local directory produced by the converter
  ds = LeRobotLanceDataset(root="./pusht_lance")

  # HF Hub repo (lance streams natively from hf://datasets/...)
  ds = LeRobotLanceDataset(repo_id="me/pusht_lance")

  # Cloud URI (S3 / GCS / HF Buckets)
  ds = LeRobotLanceDataset(uri="s3://bucket/pusht.lance",
                           meta_root="./pusht_lance")

Or with auto-detection::

  from lerobot_lancedb import make_lerobot_dataset
  ds = make_lerobot_dataset("./pusht_lance")            # → Lance
  ds = make_lerobot_dataset("lerobot/pusht")            # → parquet+mp4

Prerequisites::

  pip install lerobot-lancedb
  lerobot-convert-to-lance --repo-id=lerobot/pusht \\
      --output=./pusht_lance --overwrite
"""

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features

from lerobot_lancedb import LeRobotLanceDataset


def main() -> None:
    output_directory = Path("outputs/train/example_lance_dataset")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    training_steps = 10
    log_freq = 1

    lance_root = Path("pusht_lance")
    if not lance_root.exists():
        raise FileNotFoundError(
            f"{lance_root} does not exist. Run "
            "`lerobot-convert-to-lance --repo-id=lerobot/pusht --output=pusht_lance` first."
        )

    # Metadata reuse: LeRobotDatasetMetadata reads the same meta/ sidecar that
    # the converter copied alongside the Lance table.
    meta = LeRobotDatasetMetadata(repo_id="lerobot/pusht", root=lance_root)
    features = dataset_to_policy_features(meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    delta_timestamps = {
        "observation.image": [t / meta.fps for t in cfg.observation_delta_indices],
        "observation.state": [t / meta.fps for t in cfg.observation_delta_indices],
        ACTION: [t / meta.fps for t in cfg.action_delta_indices],
    }

    dataset = LeRobotLanceDataset(
        root=lance_root,
        repo_id="lerobot/pusht",
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=16,
        pin_memory=device.type != "cpu",
        drop_last=True,
        shuffle=True,
        persistent_workers=True,
    )

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
