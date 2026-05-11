#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert an existing LeRobot parquet+mp4 dataset to a single Lance table.

Usage::

    lerobot-convert-to-lance --repo-id=lerobot/pusht --output=./pusht_lance
"""

from __future__ import annotations

import argparse
import logging

from lerobot.utils.utils import init_logging

from lerobot_lancedb.writer import convert_to_lance


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repo id.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--src-root", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--table-name", type=str, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional HF Hub repo id to upload the converted dataset to.",
    )
    args = parser.parse_args()

    convert_to_lance(
        repo_id=args.repo_id,
        output=args.output,
        src_root=args.src_root,
        revision=args.revision,
        table_name=args.table_name,
        jpeg_quality=args.jpeg_quality,
        tolerance_s=args.tolerance_s,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
