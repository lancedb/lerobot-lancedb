"""Measure warm local batch reads with identical anchors and selected tensor values."""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from bench_loader import check_source
from bench_workloads import source_path, synthetic
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb.convert import convert


def measure(root, columns, batches):
    ds = LanceDBDataset(root=root, columns=columns, return_uint8=True)
    ds.__getitems__(batches[0])
    start = time.perf_counter()
    for indices in batches:
        ds.__getitems__(indices)
    return sum(map(len, batches)) / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--batches", type=int, default=64)
    args = parser.parse_args()
    if min(args.repeats, args.batches) < 1:
        parser.error("repeats and batches must be positive")
    check_source()
    torch.set_num_threads(1)
    for workload in ("pusht", "wide"):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            if workload == "wide":
                src = synthetic(work / "source", "droid-short-gop", vector_width=8192)
                columns = ["observation.state", "action", "observation.images.cam0"]
            else:
                src = source_path("pusht", work, None, None)
                columns = list(LeRobotDatasetMetadata("lerobot/pusht", root=src).features) + ["task"]
            root = work / "local"
            convert(root, root=src, video_mode="external")
            ds = LanceDBDataset(root=root, return_uint8=True)
            subset = LanceDBDataset(root=root, return_uint8=True, columns=columns)
            batches = np.random.default_rng(42).integers(len(ds), size=(args.batches, 8)).tolist()
            for indices in batches:
                full = ds.__getitems__(indices)
                for item, ref in zip(subset.__getitems__(indices), full, strict=True):
                    assert set(item) == set(columns)
                    for key in columns:
                        if isinstance(item[key], torch.Tensor):
                            assert torch.equal(item[key], ref[key])
                        else:
                            assert item[key] == ref[key]
            rates = {"full": [], "projected": []}
            for trial in range(args.repeats):
                for mode in reversed(rates) if trial % 2 else rates:
                    rates[mode].append(measure(root, None if mode == "full" else columns, batches))
            full, projected = (statistics.median(values) for values in rates.values())
            print(
                f"{workload}: full {full:.1f}, projected {projected:.1f} samples/s ({projected / full:.2f}x)"
            )


if __name__ == "__main__":
    main()
