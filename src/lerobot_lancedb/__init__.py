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
from .dataset import LeRobotLanceDataset
from .writer import convert_to_lance


__version__ = "0.1.0.dev0"


__all__ = [
    "LeRobotLanceDataset",
    "convert_to_lance",
    "make_lerobot_dataset",
    "__version__",
]
