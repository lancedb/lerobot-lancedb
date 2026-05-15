# Contributing

Issues and PRs welcome. The package is small and focused — feel free to
file a quick issue describing what you want before sending a large patch.

## Dev setup

```bash
git clone https://github.com/lancedb/lerobot-lancedb.git
cd lerobot-lancedb
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

`.[dev]` brings in:

- `pytest` + `pytest-timeout` (test runners)
- `ruff` (lint)
- `mkdocs` + `mkdocs-material` + `pymdown-extensions` (docs)

## Run the tests

```bash
pytest -v
```

Expected: 17 passing, 0 failing. CI runs the same on every PR.

## Lint

```bash
ruff check .
ruff format --check .
```

## Build the docs locally

```bash
mkdocs serve
# open http://127.0.0.1:8000/
```

`mkdocs build --strict` (no warnings allowed) is what the deploy workflow runs.

## Repo layout

```
src/lerobot_lancedb/
  dataset.py             # JPEG-per-frame reader (LeRobotLanceDataset)
  lance_video_dataset.py # mp4-blob reader (LeRobotLanceVideoDataset)
  writer.py              # both converters (convert_to_lance / convert_to_lance_video)
  benchmark.py           # shared throughput-benchmark utilities
  auto.py                # make_lerobot_dataset auto-detection
  _spawn_compat.py       # mp.set_start_method("spawn") helper
  scripts/               # CLI entry points (lerobot-convert-to-lance{,-video})

examples/
  conversion.py          # batch converter + the legacy GPU throughput benchmark
  benchmark_formats.py   # the size/throughput/fidelity matrix used in docs
  train_and_eval_lance.py
  train_with_lance.py
  aloha_loader_parity.py # ACT head-to-head between Lance and upstream

docs/                    # MkDocs Material source (deployed to gh-pages by .github/workflows/docs.yml)
tests/                   # pytest
```

## What landed in v0

- Two storage layouts: per-frame JPEG (`LeRobotLanceDataset`) and per-file mp4 blob (`LeRobotLanceVideoDataset`).
- Two CLIs: `lerobot-convert-to-lance` (with `--jpeg-quality` / `--jpeg-subsampling`) and `lerobot-convert-to-lance-video`.
- Bit-exact pixel verification + training-accuracy parity tests (see [`examples/aloha_loader_parity.py`](examples/aloha_loader_parity.py)).
- GPU NVJPEG decode for the JPEG layout (`decode_device="auto"`).
- Cloud reads: `s3://`, `gs://`, `hf://datasets/...`, `hf://buckets/...`.
- Spawn-mode worker safety (lancedb is fork-unsafe).

## Likely next things

- NVDEC for the video-blob layout (needs torchcodec built against the NVIDIA Video Codec SDK; not blocking).
- Cloud-coherent sequential reads via `PermutationBuilder.shuffle(clump_size=...)` for big remote datasets.
- More worked examples (Koch + SO-100 training scripts).

## Code style

- Type hints where useful, not religiously.
- Docstrings on the public API; one-line summary + an explanation of why-not-just-what.
- Tests for new readers / writers. Pixel-level verification against upstream for any new storage layout (see [`examples/benchmark_formats.py`](examples/benchmark_formats.py) for the pattern).
