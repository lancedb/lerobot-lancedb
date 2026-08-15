# Converting a LeRobot dataset to Lance - one dataset, end to end

This follows one dataset through `lerobot-lance-convert`, showing what each step
reads and writes. The source is a standard LeRobot v3.0 dataset - the same
illustrative one the loader walkthrough uses: 100 episodes of 150 frames (15,000
total), 30 fps, one camera (`observation.images.top`), a 14-dim
`observation.state` and 14-dim `action`. v2.0 / v2.1 sources (one parquet and
one mp4 per episode, jsonl metadata) go through the same pipeline after
`meta/` is rewritten to v3.0; the tabular frames and video bytes are not
re-chunked or re-encoded. The output is the three-table Lance
layout that `LanceDBDataset` reads (the companion loader PR). Values below are
illustrative; the CLI examples are exact.

The whole job is a format transform. Nothing is re-encoded, no pixels are
touched: the mp4 bytes are copied verbatim, the tabular columns are copied with
their names normalized, and the standard `meta/` directory is carried along so
the result is still a self-describing LeRobot dataset. The two things that make
it a *Lance* dataset rather than a tarball are the byte-index columns on the
videos (so the loader can fetch one frame's bytes without the whole file) and the
fact that all three tables are written as single commits (so a training worker
opens a handful of files, not thousands).

```
lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance
#   or:  --root /path/to/local/dataset   (nothing downloaded)
```

## Benchmarks

Why convert at all: the Lance layout is map-style random access over object
storage, so training does a true global shuffle straight from S3, fetching only
the byte ranges each batch needs. Upstream's only remote mode is an iterable
streamer (windowed reservoir shuffle, one worker per shard, OOMs on large
frames). Batch 32, 8 workers, steady-state samples/s.

**Broad picture - lance local/S3 (global shuffle) vs upstream (iterable):**

| dataset | lance local | lance S3 | upstream local | upstream Hub stream |
|---|---:|---:|---:|---:|
| pusht | 4,160 | 2,508 | 2,142 | 430 |
| aloha | 113 | 107 | 109 | 9.7 (1-worker cap) |
| koch | 189 | 182 | 120 | 11.7 |
| soarm | 79 | 74 | 69 | crashes |
| berkeley | 108 | 92 | 61 | 6.1 |
| droid (386 GB) | 227 | 136 | 132 | OOM |

**Same HF-bucket backend, controlling for storage AND access** (2026-08-01). Lance
is shown in both access modes, so columns 2-vs-3 are the true apples-to-apples
against upstream's iterable streaming; column 1 is lance's training-correct global
shuffle, which upstream structurally cannot match:

| dataset | lance bucket (global shuffle) | lance bucket (iterable) | upstream bucket (iterable) | upstream hub (iterable) |
|---|---:|---:|---:|---:|
| pusht | 775 | 991 | 436 | 437 |
| aloha | 101 | 102 | OOM | OOM |
| koch | 142 | 117 | 12 | 12 |
| soarm | 63 | 57 | crash | crash |
| berkeley | 64 | 56 | 5 | 6 |

Columns 2-vs-3 are airtight: same bucket, same iterable+windowed-shuffle access,
only the loader/format differ, and lance is 2.3x (pusht), 9.8x (koch), 11x
(berkeley), and simply runs where upstream crashes (aloha/soarm). Column 1 is what
upstream can never offer: global-shuffled training straight from object storage.
`upstream bucket ≈ upstream hub` (436≈437, 12≈12) confirms the storage backend is
irrelevant to a sequential streamer - the loader is the whole variable. (Lance
from S3 is faster than from an HF bucket; the bucket gateway adds per-request
latency that taxes lance's many small ranged reads. S3 is the deployment target;
buckets are convenient but rate-limited, and their absolute numbers move with
gateway load.)

**Why upstream OOMs/crashes.** `StreamingLeRobotDataset` holds a reservoir buffer
of up to ~1000 decoded frames per worker to approximate shuffling; buffer memory
scales with workers x buffer x frame-size, so on large or multi-camera frames
(aloha, droid) it exhausts RAM. soarm's failure is a torchcodec frame-index bug
that crashes the decode worker regardless of memory. Lance never hits either: it
fetches only the exact byte ranges a batch needs and caps its per-worker decoder
cache at 2 GiB by construction, so memory is flat and access-pattern-independent.

## What goes in, what comes out

```
INPUT  (LeRobot v3.0, on disk or Hub)        OUTPUT  (./pusht-lance/)
────────────────────────────────────         ─────────────────────────────
meta/                                         meta/            # verbatim copy
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
files = sorted((src_root / "data").rglob("*.parquet"))      # index-ordered by layout
db.create_table(FRAMES_TABLE, _frames_reader(files), mode="overwrite")
```

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
v2.0 / v2.1 sources keep one mp4 per episode, so the videos table has one row per
episode per camera; `from_timestamp` in the rewritten metadata is 0.

```python
schema = pa.schema([
    ("video_key", pa.string()), ("chunk_index", pa.int64()), ("file_index", pa.int64()),
    ("file_size", pa.int64()), ("moov_offset", pa.int64()), ("moov_size", pa.int64()),
    ("kf_indices", pa.list_(pa.int64())), ("kf_positions", pa.list_(pa.int64())),
    lancedb.blob(VIDEO_BLOB_COLUMN),   # video_bytes, a managed blob-v2 column
])
```

**The bytes.** `video_bytes` is the mp4 read verbatim into a Lance **blob v2**
column. Blob v2 is what makes the loader's fast path possible: the bytes live in
their own object, addressable by byte range, so the loader can `fetch_blob_ranges`
exactly the bytes one batch's frames need instead of pulling the whole file.

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

The converter writes exactly what the loader reads: same table names, same column
names, same byte-index layout, same `to_lance_column` mapping. Today the loader
isn't on PyPI yet, so the converter can't `from lerobot.datasets.lancedb_dataset
import ...` without breaking every user on released lerobot. So a small set of
definitions (`FRAMES_TABLE`, `to_lance_column`, `build_video_byte_index`, ...) is
vendored into `vendored_schema.py`, to be deleted once the loader PR merges.

A vendored copy invites drift, so a test asserts the vendored constants and the
byte-index builders match the loader in logic (AST comparison with docstrings
stripped, so cosmetic edits don't false-positive; it skips when the loader isn't
importable). It has already earned its keep once, catching an `import` that moved
in the loader but not the copy.

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
lerobot-lance-convert --repo-id lerobot/pusht --out ./pusht-lance   # download + convert
lerobot-lance-convert --root ./my-dataset      --out ./my-lance     # local, nothing downloaded
lerobot-lance-doctor  --root ./my-dataset                           # audit; exit = # failures

# then train straight off it (loader PR), local or object storage:
lerobot-train --dataset.root ./pusht-lance ...
lerobot-train --dataset.root s3://bucket/pusht-lance ...
```
