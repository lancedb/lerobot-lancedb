# Contributing

Issues and PRs welcome. The package is small and focused — feel free to
file a quick issue describing what you want before sending a large patch.

## Dev setup

```bash
git clone https://github.com/lancedb/lerobot-lancedb.git
cd lerobot-lancedb
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://pypi.fury.io/lancedb/ -e '.[dev]'
```

The extra index is where lancedb publishes beta wheels (the pinned
`lancedb>=0.37.1b0` is a beta at the time of writing).

## Lint

```bash
ruff check src/
```

## Repo layout

```
src/lerobot_lancedb/
  convert.py           # lerobot-lance-convert: LeRobot v3.0 -> three-table Lance layout
  doctor.py            # lerobot-lance-doctor: dataset defect auditor (5 read-only checks)
  vendored_schema.py   # TEMPORARY vendored schema contract (see module docstring)
```

`vendored_schema.py` is a verbatim copy of the schema contract from
lerobot's `lerobot.datasets.lancedb_dataset` (pending upstream PR). Do not
change it independently — when the upstream PR merges, the whole module gets
deleted and the converter imports the contract from lerobot directly.

## Testing changes

There is no pytest suite right now; the acceptance gate is the smoke flow
from the README quickstart: convert `lerobot/pusht`, open it with lerobot's
`LanceDBDataset`, and compare items bit-exact against upstream
`LeRobotDataset`. Run `lerobot-lance-doctor` on the source first — a defect
the doctor flags is a source problem, not a converter bug.

## Code style

- Type hints where useful, not religiously.
- Docstrings on the public API; one-line summary + an explanation of why-not-just-what.
