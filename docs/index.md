# lerobot-lancedb

Lance-backed datasets for [LeRobot](https://github.com/huggingface/lerobot).

Two storage layouts, both subclasses of `LeRobotDataset`:

- **[Frames format](frames-format.md)** — per-frame JPEG bytes (`LeRobotLanceDataset`).
- **[Video format](video-format.md)** — per-file mp4 bytes via Lance blob v2 (`LeRobotLanceVideoDataset`).

Both readers expose the same API. Pick by source `dtype` (see [Conversion](conversion.md)).

## Install

Until the first PyPI release, install from GitHub:

```bash
pip install git+https://github.com/lancedb/lerobot-lancedb.git
```

For local development:

```bash
git clone https://github.com/lancedb/lerobot-lancedb.git
cd lerobot-lancedb
pip install -e '.[dev]'
```

Either path pulls in:

- `lerobot[dataset]`
- `lancedb` / `pylance`
- `torchcodec` (used by the video format)

## 30-second tour

Convert a video-stored dataset to the recommended (bit-exact) layout:

```bash
lerobot-convert-to-lance-video \
    --repo-id=lerobot/aloha_static_cups_open \
    --output=./aloha_cups_open_lance_video \
    --overwrite
```

Use it as a regular `LeRobotDataset`:

```python
from lerobot_lancedb import LeRobotLanceVideoDataset

ds = LeRobotLanceVideoDataset(root="./aloha_cups_open_lance_video")
```

Plug it into any code that expects a `LeRobotDataset`:

- the upstream training factory
- `EpisodeAwareSampler`
- third-party trainers that do `isinstance(ds, LeRobotDataset)`

## Headline benchmark

Realistic training read pattern (`delta_timestamps`, 8 frames per sample, `batch_size=32`, `num_workers=4`, CPU decode, H100) on `lerobot/aloha_static_cups_open` (480×640, 4-cam bimanual):

| format | size MB | fps | speedup | bit-exact? |
|---|---:|---:|---:|:---:|
| upstream parquet+mp4 | 485.6 | 18.7 | 1.00× | ✓ |
| `convert_to_lance` (JPEG-95) | 3 626 | 46.0 | 2.46× | ✗ |
| `convert_to_lance --jpeg-quality=100 --jpeg-subsampling=0` | 8 735 | 32.5 | 1.74× | ✗ |
| **`convert_to_lance_video`** | **487.4** | **45.6** | **2.44×** | **✓** |

Full numbers across three datasets (pusht, ALOHA, Koch): [Benchmarks](benchmarks.md).

## Next

- [Conversion](conversion.md) — both CLIs + Python API
- [Frames format](frames-format.md) — `LeRobotLanceDataset` reference
- [Video format](video-format.md) — `LeRobotLanceVideoDataset` reference
- [Examples](examples.md) — training + benchmark scripts
- [Benchmarks](benchmarks.md) — size × throughput × accuracy tables
