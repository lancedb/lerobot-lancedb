#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Lance-backed datasets for LeRobot."""

from __future__ import annotations

from .auto import make_lerobot_dataset
from .benchmark import benchmark_throughput, print_throughput_table
from .dataset import LeRobotLanceDataset
from .lance_video_dataset import LeRobotLanceVideoDataset
from .writer import convert_to_lance, convert_to_lance_video

__version__ = "0.2.0"


__all__ = [
    "LeRobotLanceDataset",
    "LeRobotLanceVideoDataset",
    "benchmark_throughput",
    "convert_to_lance",
    "convert_to_lance_video",
    "make_lerobot_dataset",
    "print_throughput_table",
    "__version__",
]
