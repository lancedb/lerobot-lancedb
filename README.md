# lerobot-lancedb

Convert a LeRobot v3 dataset to Lance for map-style training with a global
shuffle. Keep the frame table on local disk and read the original videos from
object storage, or keep both on the training machine.

Three pieces:

- **`lerobot-lance-convert`** - turns a LeRobot v3.0 dataset (local dir or Hub id)
  into a three-table Lance layout.
- **`lerobot-lance-doctor`** - audits an upstream-format dataset for silent
  defects before you convert or train. Every large public dataset we converted
  failed at least one check.
- **`LanceDBDataset`** - the map-style training loader for the Lance layout, so
  the round trip works from this one package.

## Hybrid training

For a dataset already on the Hub or S3, put the three Lance tables on the
training machine. Tabular reads stay local. Only selected cameras need remote
MP4 ranges, and workers can seek directly to a shuffled batch's samples.
Conversion reads MP4 headers to build the index.

```bash
lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance
# For an S3 source, install the s3 extra and use its dataset prefix:
lerobot-lance-convert --root s3://my-bucket/pusht --out ./pusht-s3
```

```python
from lerobot_lancedb import LanceDBDataset

training = LanceDBDataset(root="./pusht-lance", columns=["observation.image", "observation.state", "action"])
item = training[0]
```

Hub conversion pins the video base to a commit. For a private dataset pass
`token` to the reader. For an S3-compatible service, supply an `s3://`
base and pass its HTTPS endpoint in `storage_options`.

On an AMD EPYC 4585PX, a global shuffle with one worker and batch size two gave
the following steady-state samples/s. Hybrid read the original MP4s over loopback
S3 with 20 ms request latency. Both local loaders read warm local files.

| Workload | Local Parquet | Local external Lance | Hybrid |
|---|---:|---:|---:|
| PushT | 521.8 | 944.0 | 81.8 |
| Three-camera DROID-shaped fixture | 150.2 | 198.2 | 134.2 |
| Two-camera 720p fixture | 20.6 | 26.4 | 25.6 |

The walkthrough includes the exact commands, streaming replay on the same samples,
and a wide-table projection measurement. [Loader measurements](docs/walkthrough.md#loader-measurements)

## Install

```bash
# Base install (local disk and Hugging Face Hub):
pip install --extra-index-url https://pypi.fury.io/lancedb/ lerobot-lancedb

# Add cloud source adapters for zero-copy conversion from S3 / GCS buckets:
pip install --extra-index-url https://pypi.fury.io/lancedb/ "lerobot-lancedb[s3]"
pip install --extra-index-url https://pypi.fury.io/lancedb/ "lerobot-lancedb[gcs]"
```

The extra index is needed because `lancedb>=0.42.0b26` (External Blob v2
`initial_bases`, commit-pinned `hf://`, and `fetch_blob_ranges`) is a beta,
published on fury.io, not PyPI yet.

| dependency | floor | why |
|---|---|---|
| `lerobot` | `>=0.6.0` | metadata, feature, depth and video utilities the converter and loader reuse |
| `lancedb` | `>=0.42.0b26` | External Blob v2 `initial_bases`, commit-pinned `hf://`, and `fetch_blob_ranges` |
| `av` | `>=12` | byte-index construction (`copy` mode) and container inspection (`doctor`) |
| `boto3` (`[s3]` / `[cloud]`) | `>=1.28` | optional S3 source adapter for zero-copy `s3://` conversion |
| `gcsfs` (`[gcs]` / `[cloud]`) | `>=2024.2` | optional GCS source adapter for zero-copy `gs://` conversion |

`lerobot` also supplies the loader's heavy deps (torch, torchcodec, numpy,
fsspec, requests), so they're pinned there.

## Convert

```bash
# Zero-copy from S3 to S3 (requires `lerobot-lancedb[s3]`; range-reads only MP4 headers,
# never downloads or copies MP4 payloads into videos.lance):
lerobot-lance-convert --root s3://bucket/droid --out s3://bucket/droid-lance

# Zero-copy from Hugging Face Hub (pins external base to the resolved 40-char commit SHA):
lerobot-lance-convert --repo-id lerobot/droid_1.0.1 --out ./droid-lance

# Local dataset (copies MP4s by default; pass --video-mode external to reference in place):
lerobot-lance-convert --root /path/to/dataset --out ./my-lance
```

`--video-mode` defaults to `auto`: remote roots (`s3://`, `gs://`, `hf://`, and
`--repo-id`) write zero-copy External Blob v2 descriptors (`external`) without
copying MP4 payloads into `videos.lance`, while local directory roots copy MP4s
into managed blobs (`copy`) so removing the source directory does not break the
output. External conversion transfers only the bounded Range-GET bytes needed to
read each MP4's `moov` box (1 MiB initial probe by default) and stages `meta/`
plus referenced `data/*.parquet` files under `TMPDIR` during preflight. External MP4 indexing supports standard non-fragmented single-`mdat` MP4s
(`stsz`/`stz2` and rate-one edit lists that do not truncate trailing samples)
and rejects truncated, tail-clipped, fragmented (`moof`/`mvex`), or multi-`mdat`
containers before writing output. Remote `--out` prefixes refuse overwriting
existing tables in place (convert to a new prefix or remove the prior tables
first), while local `--out` directories build in a sibling staging directory and
swap cleanly on success. Pass `--video-mode external` or `--video-mode copy` to
override, and `--video-base-uri` when indexing from a local mirror of an
object-store bucket. For local outputs converted in `copy` mode, push
`./my-lance/` to object storage (`aws s3 cp --recursive`, etc.) and point
training at the URI.

The conversion runs the doctor's checks on the source first and refuses a dataset
that is damaged where its own metadata references it - short aggregated videos,
missing or zero-byte files - because the converter copies those bytes verbatim and
the output would fail at read time instead. `--allow-defects` skips that gate and
keeps the referenced damage, printing every defect it lets through; a referenced
file that is missing still fails before output, with or without the flag, because
there are no bytes to copy.

Only the data and video files `meta/episodes` references are converted. Stray
files under `data/` and `videos/` are skipped with a warning that reports how
many files and parquet rows were skipped; unreferenced files are never copied,
with or without `--allow-defects`.

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

With `columns`, samples include the selected features, the requested `task`
label and padding masks for selected delta windows. A tabular-only selection
skips opening the video table. For a new local `--video-mode external`
conversion, eligible RGB clips open directly in TorchCodec. Keep their original
absolute paths accessible to every worker. Managed, depth and remote clips
continue to use indexed blob reads.

Unmodified LeRobot 0.7.0 can also open this layout with `LeRobotDataset`.
Use this package's `LanceDBDataset` for feature projection and local direct decode.

## The layout

Three Lance tables next to a copy of the standard `meta/` directory. The converter
stamps `"storage_format": "lance"` into `meta/info.json` (and into the meta table),
so readers can pick the storage backend from LeRobot metadata. Commit-pinned Hub
video bases also carry a `lance_external_bases` hint in that file.

```
<out>/
  meta/             # LeRobot v3.0 metadata with the Lance storage stamp
  frames.lance      # one row per frame: tabular features (dots -> underscores)
  videos.lance      # one row per source mp4: bytes in a blob v2 column + byte index
  meta.lance        # one row per meta/ file (path, bytes): metadata transport for remote roots
```

- **`frames.lance`** - every tabular feature, one row per frame, sorted by index.
  Row N is frame N, so a batch of indices is one point-read, no index structure
  needed. Numeric vectors become fixed-size lists; language columns
  (lerobot#3467) keep their nested `list<struct>` with extension types stripped.
- **`videos.lance`** - one row per source mp4: either an External Blob v2
  descriptor referencing the original mp4 in place (`--video-mode external`) or
  the mp4 verbatim in a managed blob v2 column (`--video-mode copy`), plus
  byte-index columns (`file_size`, `moov_offset`, `moov_size`, `kf_indices`,
  `kf_positions`). The loader turns a frame window into a keyframe-aligned byte
  range and fetches a whole batch's video bytes in one `fetch_blob_ranges`: an
  8-frame window costs ~100 KB of transfer, not the whole file.
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
dataset: metadata loads, episode ranges tile `[0, total_frames)`, every
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
