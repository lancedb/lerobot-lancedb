# lerobot-lancedb

Companion tooling for [LeRobot](https://github.com/huggingface/lerobot)'s native Lance dataset support. Two tools:

- **`lerobot-lance-convert`** — converts a LeRobot v3.0 dataset (local dir or Hub repo id) into the three-table Lance layout that lerobot's `LanceDBDataset` reads.
- **`lerobot-lance-doctor`** — audits an upstream-format dataset for silent defects before you convert or train on it. Every large public dataset we converted failed at least one check.

The **loader is not in this repo**. It lives in lerobot core as `lerobot.datasets.lancedb_dataset.LanceDBDataset` (upstream PR pending; development happened on [AyushExel/lerobot#1](https://github.com/AyushExel/lerobot/pull/1)). This repo only produces and audits the data that loader consumes.

## The layout

The converter writes three Lance tables next to a verbatim copy of the standard `meta/` directory:

```
<out>/
  meta/             # byte-identical LeRobot v3.0 metadata
  frames.lance      # one row per frame: all tabular features (dots -> underscores in names)
  videos.lance      # one row per source mp4: whole file in a blob v2 column + byte-index columns
  meta.lance        # one row per meta/ file (path, bytes): the metadata transport for remote roots
```

**`frames.lance`** holds every tabular feature, one row per frame, sorted by frame index. Row position equals absolute frame index, so the loader needs no index structure at all: a batch of frame indices is one batched point-read. Fixed-size numeric vectors are stored as fixed-size lists; language columns (lerobot#3467) keep their nested `list<struct>` layout with Arrow extension types stripped to their storage types.

**`videos.lance`** holds each source mp4 verbatim in a Lance blob v2 column, plus byte-index columns (`file_size`, `moov_offset`, `moov_size`, `kf_indices`, `kf_positions`) computed at conversion time. These let the loader translate a frame window into keyframe-aligned byte ranges and fetch a whole batch's video bytes in one parallel `fetch_blob_ranges` call: an 8-frame window costs ~100 KB of transfer instead of the whole file.

**`meta.lance`** exists because `meta/` used to be the one part of a dataset the tables didn't carry, so remote roots (S3, GCS, HF) needed bespoke side-channels to fetch it. Now the loader materializes `meta/` byte-identical from this table through the same Lance connection it already has. The table is the transport; the `meta/` directory on disk stays the source of truth for every consumer, and metadata is not always small enough to smuggle in elsewhere (droid's per-episode stats are 566 MB).

The converter also builds scalar indexes (`episode_index`, `task_index`, `video_key`). The training loader doesn't use them, but they make ad-hoc SQL over your dataset fast.

## Install

```bash
pip install --extra-index-url https://pypi.fury.io/lancedb/ lerobot-lancedb
```

The extra index is needed because the required `lancedb>=0.37.1b0` is a beta: lancedb publishes beta wheels on fury.io, not PyPI. Version floors that matter:

| dependency | floor | why |
|---|---|---|
| `lerobot` | `>=0.6.0` | `lerobot.datasets.language` and `lerobot.datasets.dataset_metadata` imports |
| `lancedb` | `>=0.37.1b0` | blob v2 columns + `fetch_blob_ranges` (what makes remote training reads fast) |
| `av` | `>=12` | byte-index construction and container inspection |

## Quickstart

Convert pusht (downloads from the Hub if not cached; pass `--root` instead to convert a dataset already on disk without touching the network):

```bash
lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance
```

Load it with lerobot's native loader — one line, and the items are bit-exact with `LeRobotDataset`'s:

```python
from lerobot.datasets.lancedb_dataset import LanceDBDataset

ds = LanceDBDataset("lerobot/pusht", root="./pusht-lance")
item = ds[0]  # same keys, same tensors, same pixels as LeRobotDataset
```

`root` can also be `s3://...`, `gs://...`, or an `hf://` URI; the loader does ranged reads, so nothing downloads up front. Training needs zero changes: `lerobot-train` auto-detects the layout.

## Dataset doctor

Silent defects in public datasets are common, and they surface as confusing failures at convert or train time (or worse, don't surface at all). `lerobot-lance-doctor` runs five read-only checks against an upstream-format dataset: metadata loads, episode ranges tile `[0, total_frames)`, every referenced parquet exists with the right total row count and no orphans, every referenced video exists non-empty and readable with no orphans, and every video actually contains the frames the metadata implies it should (container metadata only, no decoding).

Real example: `lerobot/berkeley_autolab_ur5` on the Hub ships aggregated videos that are short a tail of frames relative to what its own metadata implies. The doctor catches it as a SUPPLY failure:

```
$ lerobot-lance-doctor --root ~/.cache/.../lerobot/berkeley_autolab_ur5 --repo-id lerobot/berkeley_autolab_ur5
[ ok ] META: 1000 episodes, 97939 frames, fps 5
[ ok ] BOUNDARIES: episode ranges tile [0, total_frames)
[ ok ] DATA: parquet files referenced, accounted, no orphans
[ ok ] VIDEOS: referenced, non-empty, readable, no orphans
[FAIL] SUPPLY: every video holds the frames meta implies
         videos/observation.images.image/chunk-000/file-000.mp4: container declares 30091 frames, episodes imply 30175 (short by 84)
         videos/observation.images.image/chunk-000/file-001.mp4: container declares 30093 frames, episodes imply 30191 (short by 98)
         videos/observation.images.image_with_depth/chunk-000/file-000.mp4: container declares 3839 frames, episodes imply 3953 (short by 114)
         ... and 25 more

1 of 5 checks failed
```

The exit code is the number of failed checks, so it drops straight into CI. Other defects we've hit in the wild and now check for: droid 1.0.1 has 44% orphan parquet rows plus orphan videos (it loads correctly only by filename-sort luck), and agibot shipped zero-byte videos inside otherwise-complete episodes.

## Note on the old plugin

Versions of this package before 0.3.0 were a different thing: a standalone loader plugin (`LeRobotLanceDataset`, `LeRobotLanceVideoDataset`) with its own storage layouts. That implementation is superseded by the native loader in lerobot core, and 0.3.0 removes it. `pip install lerobot-lancedb` keeps working; if you depend on the old loader classes, pin `lerobot-lancedb<0.3`, and plan to move to `LanceDBDataset` in lerobot once the upstream PR merges — datasets converted with the old plugin's converters are not compatible with the native loader, so re-convert with `lerobot-lance-convert`.

## License

Apache 2.0.
