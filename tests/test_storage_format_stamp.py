"""Converted datasets must declare storage_format=lance in meta/info.json."""

import json

import pyarrow as pa

from lerobot_lancedb.convert import _stamp_storage_format, _videos_schema


def test_optional_index_appended_to_upstream_video_schema():
    schema = _videos_schema()
    assert schema.names == [
        "video_key",
        "chunk_index",
        "file_index",
        "file_size",
        "moov_offset",
        "moov_size",
        "kf_indices",
        "kf_positions",
        "video_bytes",
        "sample_index",
    ]
    assert schema.field(-1) == pa.field("sample_index", pa.binary(), nullable=True)


def _info(meta_dir):
    return json.loads((meta_dir / "info.json").read_text())


def test_stamp_adds_field(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 10}))

    _stamp_storage_format(meta)

    info = _info(meta)
    assert info["storage_format"] == "lance"
    assert info["codebase_version"] == "v3.0"  # everything else untouched
    assert info["fps"] == 10


def test_stamp_is_idempotent(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))

    _stamp_storage_format(meta)
    first = (meta / "info.json").read_text()
    _stamp_storage_format(meta)

    assert (meta / "info.json").read_text() == first
