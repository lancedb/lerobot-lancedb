# lerobot-lancedb

Lance-backed datasets for [LeRobot](https://github.com/huggingface/lerobot) — frame-level random access on local disk and cloud (S3 / GCS / HF Hub / HF Buckets).

`LeRobotLanceDataset` is a `LeRobotDataset` subclass, so any code that accepts a `LeRobotDataset` (the training factory, `EpisodeAwareSampler`, `isinstance` checks) accepts a Lance-backed one too.

## Status

Experimental. APIs and on-disk layout may change between 0.x releases as we gather feedback. See the design rationale in `docs/` (TBD) or the prior in-tree integration on the `lance-integration-in-tree` branch of the upstream `lerobot` fork.

## Install

```bash
pip install lerobot-lancedb
# or, for development
pip install -e .[dev]
```

The package brings in `lerobot[dataset]` and the `lancedb` / `pylance` runtimes.

## Quickstart

### Convert an existing dataset

```bash
lerobot-convert-to-lance \
  --repo-id=lerobot/pusht \
  --output=./pusht_lance \
  --overwrite
```

This produces:

```
pusht_lance/
  pusht.lance/         # one row per frame; JPEG-encoded images
  meta/                # info.json, stats.json, tasks.parquet, episodes/*.parquet
```

Optionally upload to the HF Hub by passing `--push-to-hub=<your-user>/pusht_lance`.

### Train

```python
from lerobot_lancedb import LeRobotLanceDataset

ds = LeRobotLanceDataset(root="./pusht_lance")   # or repo_id="me/pusht_lance"
# Plug into any code that expects a LeRobotDataset.
```

Or via the auto-detecting helper that returns either a Lance or parquet+mp4 dataset:

```python
from lerobot_lancedb import make_lerobot_dataset

ds = make_lerobot_dataset("lerobot/pusht")             # parquet+mp4
ds = make_lerobot_dataset(root="./pusht_lance")        # Lance (auto-detected from *.lance/ subdir)
ds = make_lerobot_dataset("s3://bucket/pusht.lance")   # Lance (cloud URI)
ds = make_lerobot_dataset("me/pusht_lance")            # Lance (Hub suffix convention)
```

## Why Lance

Standard LeRobot datasets store frames inside multi-episode mp4 chunks. Every batch decodes a frame range; cloud reads pay both byte-range fetch latency *and* per-window decode cost. Lance stores one row per frame with JPEG-encoded images, served by a columnar engine with native object-store backends — no video decode on the hot path, and remote random access is fast enough to train against directly.

### Throughput

Lance is faster on every realistic training condition we've measured. The
gap widens with frame resolution, number of cameras, and worker count.

Measured on `lerobot/aloha_static_cups_open` (4 cameras × 480×640 × 20 000
frames; local SSD, M-series Mac, batch=32):

| condition | parquet+mp4 (bps) | Lance (bps) | speedup |
|---|---:|---:|---:|
| shuffled, nw=0 | 2.4 | 5.8 | **2.39×** |
| shuffled, nw=4 | 3.6 | 13.9 | **3.88×** |
| shuffled + delta_timestamps, nw=0 | 1.0 | 1.8 | **1.78×** |
| shuffled + delta_timestamps, nw=4 | 1.6 | 2.5 | **1.62×** |

Reproduce with `python examples/conversion.py --benchmark` (full numbers
across more datasets are embedded in the file).

**Caveat for tiny datasets:** on small toy datasets like the 50 MB
`lerobot/pusht`, the entire dataset lives in the OS file cache after a
couple of epochs, so both backends are reading from RAM. In that regime
per-batch Python overhead dominates and the result depends on the exact
configuration. With realistic `delta_timestamps` Lance still wins by
**5-7×** on pusht; without delta_timestamps the gap narrows or inverts.
Don't make decisions based on toy-dataset numbers — measure on real data.

Cloud-storage benchmarks (S3 / HF Buckets / GCS) aren't in this README
yet; we expect gains to be **substantially larger** there because the
parquet+mp4 path pays per-decode network round-trips.

## Status of features

| Feature | Supported |
|---|---|
| Local lance dir (`root=`) | ✓ |
| HF Hub via `repo_id=` (lance streams from `hf://datasets/<repo>`) | ✓ |
| Cloud URI (S3/GCS/HF Buckets) | ✓ |
| `delta_timestamps` (temporal windows) | ✓ |
| Multi-camera image + video features | ✓ |
| Multi-task `task_index` / `subtask_index` | ✓ |
| Spawn-mode DataLoader workers | ✓ |
| Per-epoch reshuffle via `PermutationBuilder.shuffle` | not yet (phase 2) |
| Writing / recording new datasets | no — use upstream `LeRobotDataset.create` and convert |

## Auth for cloud / Hub reads

* S3 / GCS — pick up creds from the standard env vars (`AWS_ACCESS_KEY_ID`, `GOOGLE_APPLICATION_CREDENTIALS`, ...).
* HF Hub — `HF_TOKEN` env or `huggingface-cli login`. The package threads the token into Lance's `storage_options` automatically.

## Contributing

Issues and PRs welcome. The code is small and focused; see `src/lerobot_lancedb/dataset.py` for the reader, `writer.py` for the converter.

## License

Apache 2.0.
