"""Guard against the vendored schema drifting from lerobot's LanceDBDataset loader.

vendored_schema.py is a TEMPORARY copy of the loader's schema contract (released
lerobot doesn't ship the loader yet). The converter writes exactly what the loader
reads, so the two must stay identical. Skips when the loader isn't importable
(released lerobot), runs in dev where the loader PR is on the path.
"""

import ast
import inspect

import pytest

from lerobot_lancedb import vendored_schema as v


def _logic(fn) -> str:
    """AST of a function with its docstring stripped: compares logic, ignoring
    docstrings/comments/formatting so cosmetic edits don't fail the guard."""
    func = ast.parse(inspect.getsource(fn)).body[0]
    if (
        func.body
        and isinstance(func.body[0], ast.Expr)
        and isinstance(getattr(func.body[0], "value", None), ast.Constant)
        and isinstance(func.body[0].value.value, str)
    ):
        func.body = func.body[1:]
    return ast.dump(func)


def test_vendored_schema_matches_loader():
    loader = pytest.importorskip("lerobot.datasets.lancedb_dataset")

    # table/column names + index columns: exact value match
    assert v.FRAMES_TABLE == loader.FRAMES_TABLE
    assert v.VIDEOS_TABLE == loader.VIDEOS_TABLE
    assert v.META_TABLE == loader.META_TABLE
    assert v.VIDEO_BLOB_COLUMN == loader.VIDEO_BLOB_COLUMN
    assert v.VIDEO_INDEX_COLUMNS == loader.VIDEO_INDEX_COLUMNS
    assert v.to_lance_column("observation.images.top") == loader.to_lance_column("observation.images.top")

    # byte-index builders must match in logic (the converter writes what the loader reads)
    assert _logic(v.build_video_byte_index) == _logic(loader.build_video_byte_index)
    assert _logic(v._find_moov) == _logic(loader._find_moov)
