# Converting and training with a Hybrid dataset

Hybrid keeps `frames.lance`, `videos.lance`, and metadata on the training machine.
The video table contains descriptors and a byte index. The MP4s stay at the source.
This gives the sampler random access to episodes while most table operations use local disk.

```bash
lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance
```

```python
from lerobot_lancedb import LanceDBDataset

# Use the same feature names as the original LeRobot dataset.
ds = LanceDBDataset(root="./pusht-lance", columns=["observation.image", "observation.state", "action"])
print(ds[0].keys())
state_only = LanceDBDataset(root="./pusht-lance", columns=["observation.state", "task"])
```

Each worker fetches the selected tabular columns and opens only the decoders it needs.
A video selection also reads timestamps for alignment. A task label needs the task index.
Request either internal column by name to include it in a sample. List every feature with
`delta_timestamps` in `columns` to get its window and padding mask.

## Choosing a video base

- **Hugging Face Hub.** `--repo-id` resolves the source revision to a commit. The converter
  records the pinned video base in `meta/info.json`. The companion reader enables resolve
  caching for that base. Pass `token=...` for private videos. Set
  `storage_options={"base_1.hf_enable_resolve_cache": "false"}` to disable this cache.
  With LanceDB 0.42.0b26, set `HF_ENDPOINT` in the worker environment for a custom Hub endpoint.
- **S3 or S3-compatible storage.** Use an `s3://` video base. For a custom HTTPS service
  provide the endpoint separately, at conversion and when reading. For example, with
  credentials available through the AWS SDK and `S3_ENDPOINT` set to your service URL:

  ```bash
  lerobot-lance-convert --root s3://my-bucket/pusht --out ./pusht-s3 \
      --source-storage-option "aws_endpoint=$S3_ENDPOINT"
  ```

  ```python
  import os
  from lerobot_lancedb import LanceDBDataset

  ds = LanceDBDataset(root="./pusht-s3", storage_options={"base_1.aws_endpoint": os.environ["S3_ENDPOINT"]})
  print(ds[0]["action"])
  ```

  To index a local mirror, use `--root ./pusht --video-base-uri s3://my-bucket/pusht/videos`.
  Use `hf://datasets/owner/repo@COMMIT/videos` for a Hub mirror. A raw HTTPS URL is not a Lance video base.
- **Local disk.** `--root ./pusht --video-mode external` records the absolute, percent-encoded file
  URI. For whole RGB files whose size matches the index, TorchCodec opens that path directly.
  Depth files, older relative descriptors and managed blobs use the sparse blob reader. If native
  open fails, the reader retries through Lance. Use `--video-mode copy` for a self-contained dataset.

## Loader measurements

On an AMD EPYC 4585PX (16 cores, Linux x86-64), Suite A gave Parquet, local
external Lance and Hybrid the same global shuffle. Suite B captured upstream's
streaming batches and replayed their episode/frame indices on Hybrid. Both remote
modes used the same loopback service with 20 ms latency per request. Streaming
read Parquet and video remotely. Hybrid read its Lance tables locally. Backing
files were warm. Throughput excludes initialization and two warmup batches.

PushT used the cached `lerobot/pusht` snapshot `7628202a2180972f291ba1bc6723834921e72c19`.
The DROID-shaped fixture had three 320×180 cameras and a two-frame GOP. The 720p
fixture had two cameras and a 30-frame GOP. Each measured row used one worker,
ten batches of two samples, and three-frame camera/action windows. The runtime
was Python 3.12.16, LeRobot 0.7.0, LanceDB 0.42.0b26 and TorchCodec 0.17.0.

Steady-state samples per second on commit `8441827`:

| Workload | A · local Parquet | A · local Lance | A · Hybrid | B · streaming | B · Hybrid replay |
|---|---:|---:|---:|---:|---:|
| PushT | 521.8 | 944.0 | 81.8 | 1,368.6 | 91.0 |
| DROID-shaped, short GOP | 150.2 | 198.2 | 134.2 | 164.1 | 148.6 |
| 720p, long GOP | 20.6 | 26.4 | 25.6 | 22.6 | 21.4 |

On the matched P1 run (`0a172dd`), local Lance delivered 621.7, 177.5 and 24.8
samples/s respectively. The P2/P1 ratios were 1.52×, 1.12× and 1.06×. These are
short single-run comparisons. Startup, batch latency and bytes transferred are
reported separately by the harness.

From the repository root, with the PushT snapshot in the Hugging Face cache:

```bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
python scripts/bench_loader.py --workloads pusht droid 720p --suites a b \
    --steps 10 --num-workers 1 --rtt-ms 20 --verify-correctness \
    --output /tmp/lerobot-lancedb-b0-p2.json
# With the P1 report from 0a172dd, verify identical inputs, anchors and parameters:
python scripts/bench_loader.py --compare /tmp/lerobot-lancedb-b0-p1-0a172dd.json \
    /tmp/lerobot-lancedb-b0-p2.json --markdown /tmp/lerobot-lancedb-b0-p2.compare.md
python scripts/bench_projection.py --repeats 6 --batches 64
```

The projection run timed warm, single-process local reads with batch size eight
and one Torch thread. It alternated read order across six repetitions and checked
every selected value against the full reader on the same 512 sampled indices.
The wide fixture added an 8,192-element float32 vector to the three-camera input.
Its projected reader requested one camera, state and action. PushT requested all
features and the task label to measure the cost of specifying an equivalent projection.

| Local workload | Full samples/s | Projected samples/s | Projected / full |
|---|---:|---:|---:|
| Three cameras and wide vector → one camera, state and action | 807.4 | 840.9 | 1.04× |
| PushT, all output keys in both runs | 4,082.3 | 3,901.1 | 0.96× |

These are median throughputs. Both paths returned exactly matching selected tensors.
The small throughput differences are specific to this warm, local CPU workload.

## What goes in, what comes out

```
INPUT  (LeRobot v3.0, on disk or Hub)        OUTPUT  (./pusht-lance/)
────────────────────────────────────         ─────────────────────────────
meta/                                         meta/            # stamped LeRobot metadata
  info.json  stats.json  tasks.parquet        meta.lance       # meta/ files as (path, bytes)
  episodes/chunk-000/file-000.parquet         frames.lance     # 1 row / frame, tabular only
data/                                         videos.lance     # 1 row / mp4, bytes + byte index
  chunk-000/file-000.parquet ...
videos/
  observation.images.top/chunk-000/file-000.mp4 ...
```

`meta/` is copied twice on purpose: once as a plain directory (so the output is
still a normal LeRobot dataset you can inspect), and once into `meta.lance` as
rows of `(path, bytes)`. The second copy is the transport for remote roots: when
the loader points at `s3://.../pusht-lance`, there is no local `meta/` to read, so
it materializes one from `meta.lance`. See "the meta table" below.

## Step 1 - the meta table (the transport)

```python
# in:  every file under meta/            out: meta.lance, 10 rows
meta_files = sorted(f for f in (src_root / "meta").rglob("*") if f.is_file())
db.create_table(META_TABLE, pa.Table.from_pylist(
    [{"path": str(f.relative_to(src_root/"meta")), "data": f.read_bytes()} for f in meta_files],
    schema=pa.schema([("path", pa.string()), ("data", pa.large_binary())])), mode="overwrite")
```

```
path                                   data (bytes)
info.json                              b'{"codebase_version": "v3.0", ...}'
stats.json                             b'{"observation.state": {"mean": ...}}'
tasks.parquet                          b'PAR1...'
episodes/chunk-000/file-000.parquet    b'PAR1...'
...                                    (10 rows)
```

Why a table and not just the copied directory: object stores don't have a "read
this directory" primitive the way a filesystem does, and `LeRobotDatasetMetadata`
(kept unforked on purpose) reads files from a directory. So the table is the wire
format, and the loader turns it back into a local `meta/` cache on first open,
byte-for-byte. The files are the API; the table is the transport.

## Step 2 - the frames table (streamed, order preserved)

One row per frame, tabular features only (no pixels). Three things happen to the
columns, then the whole thing is streamed to one commit.

```python
data_files, video_files = _referenced_files(meta, src_root)  # only files meta/episodes points at
db.create_table(FRAMES_TABLE, _frames_reader(data_files), mode="overwrite")
```

**The file list comes from `meta/episodes`, not a `data/` glob.** Public
datasets carry stray shards: DROID 1.0.1 ships 70 unreferenced parquet files
(12.4M rows) whose frame indices are not where their filenames sort, so a
filename-sorted glob reads them out of order and trips the check below. The
converter skips them with a warning that counts the files and their rows (read
from parquet footers, so an unreadable orphan cannot fail the conversion). The
same applies to videos. A file that *is* referenced and missing fails the
conversion before any output is written; only unreferenced files are skipped.

**Column transforms** (computed once from the first file's schema, applied per
batch):

```
source column            ->  lance column           transform
observation.state        ->  observation_state      list<float>  -> fixed_size_list<float>[14]
action                   ->  action                 list<float>  -> fixed_size_list<float>[14]
timestamp                ->  timestamp              (as-is)
index                    ->  index                  (as-is)
episode_index            ->  episode_index          (as-is)
task_index               ->  task_index             (as-is)
language_instruction*    ->  language_instruction*  list<struct> -> keep, strip JSON extension type
```

- **Dots to underscores.** Feature keys like `observation.state` are valid LeRobot
  names but not valid Lance column names, so every field is renamed with
  `to_lance_column` (`.` -> `_`). The loader maps them back on read. This is half
  the schema contract: the loader must rename identically, which is why both sides
  share `to_lance_column` (see "the schema contract").
- **Vectors become fixed-size lists.** A 2-dim state is stored in parquet as a
  variable-length `list<float>`; Lance stores it as `fixed_size_list<float>[2]`, so
  it reads back as a real `(n, 2)` array with no per-row length. The width is read
  once from the first row.
- **Language columns keep their shape.** The `list<struct>` message rows
  (lerobot#3467) stay nested; only Arrow extension types Lance can't store (the
  JSON type inside `tool_calls`) are swapped for their storage type (plain string),
  which is also what the loader expects.

**Streaming, and why order is load-bearing.** The naive version was
`pa.concat_tables([pq.read_table(f) for f in files]).sort_by("index")` - read every
parquet into memory, concatenate (another copy), sort (another). Fine at 9 MB,
but it materializes the entire tabular frames table, which OOMs at DROID/AgiBot
scale. Instead `_frames_reader` yields one record batch at a time through a single
`create_table(RecordBatchReader)`: bounded producer memory, one commit.

The catch is the sort. The loader's core invariant is **row N == absolute frame
N** (that is what lets it skip a lookup and index the frames table directly). A
streaming write can't do a global sort without holding everything, so instead of
sorting it *verifies*: the `index` column must stay monotonically non-decreasing
across the whole stream, and an out-of-order file fails loudly rather than
silently producing a mis-ordered table. LeRobot v3.0 writes `data/` in ascending
index order, so valid datasets never trip it; a re-chunked or hand-edited one that
would have been silently "fixed" by the old sort now gets a clear error.

```python
# out: frames.lance, 15,000 rows, row i has index == i
#   observation_state  fixed_size_list<float>[14]
#   action             fixed_size_list<float>[14]
#   timestamp float64 | index int64 | episode_index int64 | task_index int64
```

## Step 3 - the videos table (bytes plus a byte index)

One row per source mp4. In LeRobot v3.0 episodes share video files, so this is far
fewer rows than the frames table (one row is many episodes' worth of frames).

```python
schema = pa.schema([
    ("video_key", pa.string()), ("chunk_index", pa.int64()), ("file_index", pa.int64()),
    ("file_size", pa.int64()), ("moov_offset", pa.int64()), ("moov_size", pa.int64()),
    ("kf_indices", pa.list_(pa.int64())), ("kf_positions", pa.list_(pa.int64())),
    lancedb.blob(VIDEO_BLOB_COLUMN),   # video_bytes, a blob-v2 column
])
```

**The bytes.** External mode writes the original clip's URI, position and size.
Managed mode copies the MP4 bytes into a Lance blob. Both modes share a byte index
so the reader can fetch the spans covering a batch's frame windows.

**The byte index** is the other half of that. For each file, `build_video_byte_index`
walks the container once (with pyav) and records, per keyframe, the frame index and
its byte offset, plus where the mp4 `moov` box lives:

```python
# build_video_byte_index("observation.images.top/chunk-000/file-000.mp4"):
{
  "file_size":   4_910_233,
  "moov_offset": 4_872_101,        # this file's moov is at the END (not faststart)
  "moov_size":     38_040,
  "kf_indices":   [0, 12, 24, 36, ...],       # frame index of every keyframe
  "kf_positions": [48, 22_310, 41_002, ...],  # and its byte offset in the file
}
```

At read time the loader turns "I need frames 610-617" into "the keyframe at or
before 610 starts at byte P, the next keyframe after 617 ends at byte Q" and
fetches `[P, Q)`. The keyframe columns work for any container pyav can demux and
assume constant frame rate (the same assumption the loader's timestamp->index
conversion makes); the `moov` columns are mp4-specific and matter because a
non-faststart mp4 keeps its index at the *end* of the file, so the loader has to
know where it is to open the container in one ranged read instead of a walk.

**Streamed to one commit, and the blob-v2 gotcha.** Like the frames table, videos
are streamed: batches of ~512 MB are yielded lazily through a single
`videos_table.add(RecordBatchReader)`, so producer memory stays bounded but the
whole table is one commit. The gotcha is that pyarrow can't build the
`lance.blob.v2` extension array from raw `bytes` (`from_pylist` rejects it), so the
streamed batches carry the blob column as its **storage type** (`large_binary`);
Lance encodes that into the managed blob column on write, matched by name. This is
exactly what the old list-of-dicts `add()` did internally, just made explicit for
the reader.

```python
# out: videos.lance, one row per mp4 (episodes share files, so e.g. 8 rows here,
#      vs 15,000 in frames), ONE version
```

## Step 4 - scalar indexes (for humans, not the loader)

```python
frames_table.create_scalar_index("episode_index")                  # BTree
frames_table.create_scalar_index("task_index", index_type="BITMAP")
videos_table.create_scalar_index("video_key", index_type="BITMAP")
```

The training loader never uses these: it reads by `_rowid`. They exist so a human
poking at the dataset can `WHERE episode_index = 5` or `WHERE task_index = 3`
quickly. They are the last step, and safe to skip on a very large table (the index
build sorts the column, which is memory-heavy at hundreds of millions of rows).

## Why the streaming matters (the fragmentation story)

Both the frames and videos writes used to commit incrementally: the videos build
did `videos_table.add(pending)` once per ~512 MB accumulated. Every `add()` is a
separate Lance commit, so a large dataset produced a version and a fragment per
512 MB:

```
                     versions   fragments      per training-worker open, that's:
DROID videos (old)      ~394        ~393        394 manifests + 393 fragment scans
DROID videos (new)         1     Lance-sized    1 commit, a handful of fragments
```

A fragmented table isn't wrong, but every worker pays for it: more manifest
reads, more descriptor takes, more object opens on the hot path. The streaming
rewrite keeps producer memory bounded (one batch in flight) while Lance keeps a
single writer open across the batches, so the whole table is one commit. Same
result for the frames table, where the win is memory as much as fragment count.

(If you have an *already*-fragmented table from the old converter, you don't
re-run the conversion: `cleanup_old_versions()` prunes the version history cheaply
with no blob rewrite. Full `optimize()` on a blob table would re-write every
managed blob - terabytes - so it's the wrong tool there.)

## The schema contract

The converter imports the table names and byte-index contract from the companion
reader. LeRobot 0.7.0 selects its native reader from the `storage_format` stamp
and ignores the optional `lance_external_bases` extension.

## The doctor - audit before (or after) you convert

`lerobot-lance-doctor --root /path/to/dataset` runs five read-only checks (no
decode, container metadata only). It exists because every large public dataset we
converted shipped at least one silent defect:

```
check        what it catches                                     seen in
1 META       metadata loads at all                               -
2 BOUNDARIES episode ranges tile [0, total_frames) exactly       agibot (empty episodes)
3 DATA       every referenced parquet exists; rows==total_frames droid (44% orphan rows)
             orphan parquet under data/
4 VIDEOS     every referenced video exists and is non-empty      agibot (zero-byte videos)
5 SUPPLY     per file: frames the moov declares >= frames the    berkeley (truncated tails)
             episodes need
```

Exit code is the number of failed checks, so it drops into CI. The converter and
the doctor share the view that "it converted without error" is not the same as
"it's correct": the loader has integrity checks at open (row count == total_frames,
episode tiling), and the doctor is the same discipline pointed at the *source* so
you find the defect before it's baked into the Lance copy.

## Reference: the three tables

```
frames.lance   1 row / frame        row N == absolute frame N (sorted by index)
               tabular features only; dots->underscores; vectors as fixed_size_list
videos.lance   1 row / source mp4    video_key, chunk_index, file_index,
               file_size, moov_offset, moov_size, kf_indices, kf_positions,
               video_bytes (blob v2)
meta.lance     1 row / meta file     (path, bytes) - transport for the standard meta/ dir
```

## Reference: the CLI

```
lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance   # local tables, remote MP4s
lerobot-lance-convert --root ./my-dataset      --out ./my-lance     # local, nothing downloaded
lerobot-lance-doctor  --root ./my-dataset                           # audit; exit = # failures

# point the training loader at the converted root:
lerobot-train --dataset.root ./pusht-lance ...
lerobot-train --dataset.root s3://bucket/pusht-lance ...
```
