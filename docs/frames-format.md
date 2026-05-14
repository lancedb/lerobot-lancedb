# Frames format (`LeRobotLanceDataset`)

One row per frame. Each row holds tabular fields (state, action, timestamps, episode / frame indices) plus the JPEG-encoded image bytes.

At read time the bytes are decoded with:

- **torchvision** (NVJPEG when CUDA is available — `decode_device="auto"`).
- **PIL** as a fallback if torchvision can't be imported.

## Schema

```
<table>.lance/
  episode_index:     int32
  frame_index:       int32
  index:             int64
  timestamp:         float32
  task_index:        int32
  observation.image: binary             # JPEG bytes
  observation.state: list<float32>[D]
  action:            list<float32>[A]
  ...other tabular features
```

No separate videos table — every frame stores its own JPEG.

## Reader

```python
from lerobot_lancedb import LeRobotLanceDataset

ds = LeRobotLanceDataset(root="./pusht_lance")
sample = ds[0]
# sample has: episode_index, frame_index, timestamp,
#             observation.image (C,H,W), observation.state, action, ...
```

The dataset subclasses `LeRobotDataset`, so it plugs into:

- the upstream training factory
- `EpisodeAwareSampler`
- any code that does `isinstance(ds, LeRobotDataset)`

## Sources

Three constructor entry points:

- **Local directory**
  ```python
  LeRobotLanceDataset(root="./pusht_lance")
  ```
- **Hugging Face Hub** (uses your `HF_TOKEN` if set):
  ```python
  LeRobotLanceDataset(repo_id="me/pusht_lance")
  ```
- **Cloud URI** (S3 / GCS / HF Buckets):
  ```python
  LeRobotLanceDataset(
      uri="s3://bucket/path/pusht.lance",
      meta_root="./pusht_lance",   # local meta/ sidecar
  )
  ```

## GPU NVJPEG decode

`decode_device` picks where the JPEG decode happens:

- `"auto"` (default) — `"cuda"` if available, else `"cpu"`.
- `"cuda"` — explicit GPU decode. NVJPEG is typically ~10× faster than libjpeg-turbo and tensors land on the GPU directly (no H2D copy).
- `"cpu"` — explicit CPU decode. Useful for apples-to-apples comparisons or when GPU memory is tight.

```python
LeRobotLanceDataset(root="./pusht_lance", decode_device="cuda")
LeRobotLanceDataset(root="./pusht_lance", decode_device="cpu")
```

## `delta_timestamps`

Same API as upstream. Multiple timestamps per camera are batched into a single decode:

```python
ds = LeRobotLanceDataset(
    root="./pusht_lance",
    delta_timestamps={
        "observation.image": [-0.1, -0.05, 0.0],
        "observation.state": [-0.1, -0.05, 0.0],
        "action":            [0.0, 0.05, 0.1, 0.15, 0.2],
    },
)
```

## Quality knobs at conversion time

JPEG is lossy. Two knobs let you trade size for fidelity:

- `--jpeg-quality` (default 95). Higher = larger files, fewer artifacts.
- `--jpeg-subsampling` (default 2 = 4:2:0). Set to 0 for 4:4:4 chroma (no subsampling, near-lossless when combined with `--jpeg-quality=100`).

See [Conversion](conversion.md#frames-format) for the CLI flags, and [Benchmarks](benchmarks.md) for what each setting costs and buys.

If you need bit-exact pixels on a `dtype=video` source, prefer the [video format](video-format.md) — JPEG quality settings get close, but the video format actually matches upstream bit-for-bit.

## Cloud auth

The reader picks up credentials from the standard environment:

- **S3** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- **GCS** — `GOOGLE_APPLICATION_CREDENTIALS`
- **HF Hub** — `HF_TOKEN` (or `huggingface-cli login`)

Lance does byte-range fetches on demand — no full-dataset download.

## Spawn-mode workers

Lance forces `multiprocessing.set_start_method("spawn")` on import (necessary for safe fork-mode behavior).

What this means in practice:

- Launch your training script from a real file, not `python -c` or a REPL.
- DataLoader `num_workers > 0` with `persistent_workers=True` works as expected.
