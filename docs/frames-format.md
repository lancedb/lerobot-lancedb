# Frames format (`LeRobotLanceDataset`)

One row per frame. Each row holds the tabular fields (state, action, timestamps, episode/frame indices) plus the encoded image bytes — JPEG by default, PNG with `--lossless`. At read time the bytes are decoded with torchvision (NVJPEG if available) or PIL (PNG fallback) and returned as tensors.

## Schema

```
<table>.lance/
  episode_index:     int32
  frame_index:       int32
  index:             int64
  timestamp:         float32
  task_index:        int32
  observation.image: binary             # JPEG or PNG bytes
  observation.state: list<float32>[D]
  action:            list<float32>[A]
  ...other tabular features
```

No separate videos table — every frame stores its own image bytes.

## Reader

```python
from lerobot_lancedb import LeRobotLanceDataset

ds = LeRobotLanceDataset(root="./pusht_lance")
print(len(ds))           # number of frames
sample = ds[0]
print(sample.keys())     # episode_index, frame_index, timestamp,
                         # observation.image (C,H,W), observation.state, action, ...
```

The dataset subclasses `LeRobotDataset`, so it plugs into the upstream training factory, `EpisodeAwareSampler`, and any third-party code that does `isinstance(ds, LeRobotDataset)`.

### Sources

```python
# Local directory
LeRobotLanceDataset(root="./pusht_lance")

# Hugging Face Hub
LeRobotLanceDataset(repo_id="me/pusht_lance")

# Cloud URI (S3 / GCS / HF Buckets). meta_root points at the local meta/ sidecar.
LeRobotLanceDataset(uri="s3://bucket/path/pusht.lance", meta_root="./pusht_lance")
```

### GPU NVJPEG decode

`decode_device="auto"` (the default) picks `"cuda"` when `torch.cuda.is_available()`, else `"cpu"`. On NVIDIA GPUs torchvision's `decode_jpeg` uses NVJPEG and returns tensors on the GPU directly (saves the H2D copy).

```python
LeRobotLanceDataset(root="./pusht_lance", decode_device="cuda")   # explicit
LeRobotLanceDataset(root="./pusht_lance", decode_device="cpu")    # force CPU
```

!!! note
    NVJPEG only handles JPEG bytes. With `--lossless` (PNG) the reader falls back to PIL on CPU regardless of `decode_device`.

### `delta_timestamps`

Works the same as upstream:

```python
ds = LeRobotLanceDataset(
    root="./pusht_lance",
    delta_timestamps={
        "observation.image": [-0.1, -0.05, 0.0],
        "observation.state": [-0.1, -0.05, 0.0],
        "action": [0.0, 0.05, 0.1, 0.15, 0.2],
    },
)
```

## Format trade-offs at a glance

| flag | size on disk | speed | pixel fidelity | when to use |
|---|---|---|---|---|
| default (JPEG-95) | smallest | fastest | ~6 % pixels visibly differ on pusht, 1.4 % on ALOHA, 13.5 % on Koch | size-bound, inference-only |
| `--jpeg-quality=100 --jpeg-subsampling=0` | ~1.8× larger than default | slightly slower | ~10 – 100× fewer visible artifacts than default; still not bit-exact at sharp edges | "best JPEG" — keeps NVJPEG path |
| `--lossless` (PNG) | bigger, sometimes much bigger on natural images | slower decode (PIL on CPU) | bit-exact | `dtype=image` sources, anything where you need byte-exact pixels and can afford the size |

See [Benchmarks](benchmarks.md) for the actual numbers.

## Cloud reads

Auth picks up from the standard env vars:

| Backend | Vars |
|---|---|
| S3 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| GCS | `GOOGLE_APPLICATION_CREDENTIALS` |
| HF Hub | `HF_TOKEN` (or `huggingface-cli login`) |

Lance reads byte ranges from the object store on demand — no full-dataset download.

```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."

ds = LeRobotLanceDataset(
    uri="s3://my-bucket/aloha_cups_open.lance",
    meta_root="./aloha_cups_open_meta",  # local meta/ sidecar
)
```

## Worker-mode caveat

Lance's spawn-mode safety code forces `multiprocessing.set_start_method("spawn")` on import. PyTorch's DataLoader workers re-import the main script in spawn mode, so launch this from an actual script file (not `python -c` or a REPL).
