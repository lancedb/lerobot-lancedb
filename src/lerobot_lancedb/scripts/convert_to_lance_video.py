#!/usr/bin/env python

# Copyright 2025 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Convert a parquet+mp4 LeRobotDataset to the Lance blob-v2 video layout.

Produces two Lance tables side-by-side:

* ``<output>/<table_name>.lance``       — one row per frame, tabular data only.
* ``<output>/<table_name>_videos.lance`` — one row per source mp4 file
  ``(video_key, chunk_index, file_index)``; the raw mp4 bytes are stored
  verbatim as a Lance blob-v2-encoded ``video_bytes`` column. Bit-exact
  pixels, ~same disk size as the upstream layout.

Use this for ``dtype=video`` source datasets where you want to avoid the
JPEG roundtrip that ``lerobot-convert-to-lance`` introduces. For
``dtype=image`` datasets (e.g. ``lerobot/pusht_image``), use
``lerobot-convert-to-lance --lossless`` instead.

Usage::

    lerobot-convert-to-lance-video \\
        --repo-id=lerobot/aloha_static_cups_open \\
        --output=./aloha_cups_open_lance_video
"""

from __future__ import annotations

import argparse
import logging

from lerobot.utils.utils import init_logging

from lerobot_lancedb.writer import convert_to_lance_video


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repo id.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--src-root", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="Override the frames-table name (default: last segment of repo_id).",
    )
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_to_lance_video(
        repo_id=args.repo_id,
        output=args.output,
        src_root=args.src_root,
        revision=args.revision,
        table_name=args.table_name,
        tolerance_s=args.tolerance_s,
        overwrite=args.overwrite,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
