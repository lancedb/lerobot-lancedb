# lerobot-lancedb

Convert a [LeRobot](https://github.com/huggingface/lerobot) dataset to a Lance
layout once, then train from it - local disk or object storage - faster than
upstream and without downloading the whole dataset first.

Three pieces:

- **`lerobot-lance-convert`** - turns a LeRobot v2.0, v2.1, or v3.0 dataset
  (local dir or Hub id) into a three-table Lance layout. v2.x sources keep
  their per-episode parquet/mp4 files as-is; `meta/` is rewritten to v3.0 so
  the loader can read the result. You do not need to run lerobot's
  `convert_dataset_v21_to_v30` first.
- **`lerobot-lance-doctor`** - audits an upstream-format dataset for silent
  defects before you convert or train. Every large public dataset we converted
  failed at least one check.
- **`LanceDBDataset`** - the map-style training loader for the Lance layout, so
  the round trip works from this one package.

## Why convert

The Lance layout is map-style random access over object storage: a training
worker fetches the exact byte ranges one batch needs, so you get a true global
shuffle straight from S3 without downloading the dataset and without holding a
big reservoir buffer in RAM. Upstream's only remote mode is an iterable streamer
(reservoir shuffle, one worker per shard, OOMs on large frames).

Batch 32, 8 workers, steady-state samples/s (global shuffle for lance; upstream's
streamer only does its windowed reservoir shuffle):

| dataset | lance local | lance S3 | upstream local | upstream Hub stream |
|---|---:|---:|---:|---:|
| pusht | 4,160 | 2,508 | 2,142 | 430 |
| aloha | 113 | 107 | 109 | 9.7 (1-worker cap) |
| koch | 189 | 182 | 120 | 11.7 |
| soarm | 79 | 74 | 69 | crashes |
| berkeley | 108 | 92 | 61 | 6.1 |
| droid (386 GB) | 227 | 136 | 132 | **OOM** |

The column that matters is **lance S3 vs upstream Hub stream**: same remote data,
lance is 6-15x faster and simply runs where the streamer exhausts RAM or a decode
worker crashes. Lance from local disk matches or beats upstream local too, so
converting doesn't cost you anything on the machine you already have. A
same-bucket, same-access apples-to-apples comparison (only the loader/format
differs) and the full methodology are in the [converter
walkthrough](https://github.com/lancedb/lerobot-lancedb/blob/main/docs/walkthrough.md).

## Install

```bash
pip install --extra-index-url https://pypi.fury.io/lancedb/ lerobot-lancedb
```

The extra index is needed because `lancedb>=0.37.1b0` (blob v2 + `fetch_blob_ranges`,
what makes the remote reads fast) is a beta, published on fury.io, not PyPI yet.

| dependency | floor | why |
|---|---|---|
| `lerobot` | `>=0.6.0` | metadata, feature, depth and video utilities the converter and loader reuse |
| `lancedb` | `>=0.37.1b0` | blob v2 columns + `fetch_blob_ranges` |
| `av` | `>=12` | byte-index construction and container inspection |

`lerobot` also supplies the loader's heavy deps (torch, torchcodec, numpy), so
they're pinned there.

## Convert

```bash
# from the Hub (downloads if not cached):
lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance
# or a dataset already on disk (nothing downloaded):
lerobot-lance-convert --root /path/to/dataset --out ./my-lance
```

Then push `./pusht-lance/` to object storage (`aws s3 cp --recursive`, etc.) and
point training at the URI.

## Load it back

```python
from lerobot_lancedb import LanceDBDataset, lance_mp_context
from torch.utils.data import DataLoader

ds = LanceDBDataset(root="./pusht-lance")              # or "s3://bucket/pusht-lance"
item = ds[0]                                           # same keys/tensors as LeRobotDataset

loader = DataLoader(ds, batch_size=32, num_workers=8,  # pair with EpisodeAwareSampler
                    multiprocessing_context=lance_mp_context())
```

It is a map-style `torch.utils.data.Dataset` returning items bit-exact with
`LeRobotDataset`. `root` may be a local dir or an `s3://` / `gs://` / `hf://` URI
(ranged reads, nothing downloads up front).

> **Two things to know.** (1) `LanceDBDataset` here is a **vendored copy** of
> lerobot's loader (the open reader PR), included so this package works end to end
> against released lerobot. Once the loader lands in lerobot core this re-exports
> `lerobot.datasets.lancedb_dataset.LanceDBDataset` and the copy is deleted; your
> import keeps working either way. (2) The `lerobot-train` CLI auto-detecting a
> Lance root is part of the lerobot reader PR (in `make_dataset`), so training
> straight off `--dataset.root ...-lance` needs that PR merged; until then, build
> the `DataLoader` yourself as above.

## The layout

Three Lance tables next to a copy of the standard `meta/` directory. The converter
stamps `"storage_format": "lance"` into `meta/info.json` (and into the meta table),
so readers can pick the storage backend from LeRobot metadata alone. v3.0 `meta/`
is copied verbatim; v2.0 / v2.1 jsonl metadata is rewritten to the v3.0 parquet
layout the loader expects (episode locators, `tasks.parquet`, `stats.json`):

```
<out>/
  meta/             # LeRobot v3.0 metadata (copied from a v3 source, or rewritten from v2.x)
  frames.lance      # one row per frame: tabular features (dots -> underscores)
  videos.lance      # one row per source mp4: bytes in a blob v2 column + byte index
  meta.lance        # one row per meta/ file (path, bytes): metadata transport for remote roots
```

- **`frames.lance`** - every tabular feature, one row per frame, sorted by index.
  Row N is frame N, so a batch of indices is one point-read, no index structure
  needed. Numeric vectors become fixed-size lists; language columns
  (lerobot#3467) keep their nested `list<struct>` with extension types stripped.
- **`videos.lance`** - each mp4 verbatim in a blob v2 column, plus byte-index
  columns (`file_size`, `moov_offset`, `moov_size`, `kf_indices`, `kf_positions`).
  The loader turns a frame window into a keyframe-aligned byte range and fetches a
  whole batch's video bytes in one `fetch_blob_ranges`: an 8-frame window costs
  ~100 KB of transfer, not the whole file.
- **`meta.lance`** - the `meta/` files as `(path, bytes)`, so a remote root can
  materialize `meta/` through the same Lance connection instead of a side channel
  (droid's per-episode stats alone are 566 MB).

Both large tables are written as a single streaming commit (bounded memory, no
per-512 MB fragmentation), and scalar indexes (`episode_index`, `task_index`,
`video_key`) are built for ad-hoc SQL - the training loader reads by row id and
doesn't need them.

## Dataset doctor

Silent defects in public datasets are common, and they surface as confusing
failures at convert or train time (or worse, don't surface at all).
`lerobot-lance-doctor` runs five read-only checks against an upstream-format
v2.0, v2.1, or v3.0 dataset: metadata loads, episode ranges tile `[0, total_frames)`, every
referenced parquet exists with the right total row count and no orphans, every
referenced video exists non-empty and readable with no orphans, and every video
actually contains the frames the metadata implies it should (container metadata
only, no decoding).

Real example: `lerobot/berkeley_autolab_ur5` ships aggregated videos short a tail
of frames relative to what its own metadata implies. The doctor catches it:

```
$ lerobot-lance-doctor --root ~/.cache/.../lerobot/berkeley_autolab_ur5 --repo-id lerobot/berkeley_autolab_ur5
[ ok ] META: 1000 episodes, 97939 frames, fps 5
[ ok ] BOUNDARIES: episode ranges tile [0, total_frames)
[ ok ] DATA: parquet files referenced, accounted, no orphans
[ ok ] VIDEOS: referenced, non-empty, readable, no orphans
[FAIL] SUPPLY: every video holds the frames meta implies
         videos/observation.images.image/chunk-000/file-000.mp4: container declares 30091 frames, episodes imply 30175 (short by 84)
         ... and 27 more

1 of 5 checks failed
```

The exit code is the number of failed checks, so it drops straight into CI. Other
defects we've hit and now check for: droid 1.0.1 has 44% orphan parquet rows plus
orphan videos (it loads correctly only by filename-sort luck), and agibot shipped
zero-byte videos inside otherwise-complete episodes.

## Note on the old plugin

Versions before 0.3.0 were a different thing: a standalone loader plugin
(`LeRobotLanceDataset`, `LeRobotLanceVideoDataset`) with its own storage layouts,
superseded by the native loader. If you depend on those classes, pin
`lerobot-lancedb<0.3` and plan to move to `LanceDBDataset` - datasets converted
with the old plugin are not compatible with the native loader, so re-convert with
`lerobot-lance-convert`.

## License

Apache-2.0.
