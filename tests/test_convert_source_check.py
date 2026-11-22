"""The converter audits the source first: damaged referenced videos are refused.

The fixture is the ``berkeley_autolab_ur5`` defect in miniature - the aggregated
mp4 holds fewer frames than ``meta/`` says the episodes need. The old converter
byte-copied it and produced a dataset whose reads fail past the last real frame,
so the doctor's SUPPLY check has to gate the conversion.
"""

import sys
from pathlib import Path

import av
import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from lerobot_lancedb.convert import (
    FRAMES_TABLE,
    VIDEOS_TABLE,
    DamagedSourceError,
    _reject_damaged_source,
    convert,
    main,
)
from lerobot_lancedb.doctor import audit

FPS = 10
FRAMES = 20
VIDEO_KEY = "observation.images.cam"


def _encode_video(path: Path, frames: int) -> None:
    """Write an aggregated per-camera mp4, like LeRobot stores one file per chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_dataset(root: Path, video_frames: int) -> Path:
    """One-episode LeRobot v3.0 dataset whose video holds ``video_frames`` frames.

    ``meta/`` always implies ``FRAMES`` frames, so ``video_frames < FRAMES`` is a
    truncated video and ``video_frames == FRAMES`` is a sound dataset.
    """
    meta = LeRobotDatasetMetadata.create(
        repo_id="local/convert-source-check",
        fps=FPS,
        features={
            VIDEO_KEY: {"dtype": "video", "shape": (16, 16, 3), "names": ["height", "width", "channels"]}
        },
        root=root,
    )
    meta.save_episode_tasks(["pick up the cube"])
    meta.save_episode(
        episode_index=0,
        episode_length=FRAMES,
        episode_tasks=["pick up the cube"],
        episode_stats={},
        episode_metadata={
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": FRAMES,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": 0,
            f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
            f"videos/{VIDEO_KEY}/to_timestamp": FRAMES / FPS,
        },
    )
    meta.finalize()

    data_file = root / meta.get_data_file_path(0)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "index": pa.array(range(FRAMES), pa.int64()),
                "episode_index": pa.array([0] * FRAMES, pa.int64()),
                "task_index": pa.array([0] * FRAMES, pa.int64()),
                "timestamp": pa.array([i / FPS for i in range(FRAMES)], pa.float32()),
                "frame_index": pa.array(range(FRAMES), pa.int64()),
                "observation.state": pa.array([[float(i)] * 4 for i in range(FRAMES)], pa.list_(pa.float32())),
            }
        ),
        data_file,
    )
    _encode_video(root / meta.get_video_file_path(0, VIDEO_KEY), video_frames)
    return root


def test_damaged_source_is_refused_before_any_output(tmp_path):
    src = _make_dataset(tmp_path / "src", video_frames=14)
    out = tmp_path / "out"

    with pytest.raises(DamagedSourceError, match="short by 6"):
        convert(out, root=src)

    # refused before the output directory exists: nothing half-converted is left behind
    assert not out.exists()


def test_cli_refuses_by_default_and_needs_the_flag_to_continue(tmp_path, monkeypatch, capsys):
    src = _make_dataset(tmp_path / "src", video_frames=14)
    out = tmp_path / "out"
    argv = ["lerobot-lance-convert", "--root", str(src), "--out", str(out)]

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code == 1
    assert "refusing to convert" in capsys.readouterr().err
    assert not out.exists()

    monkeypatch.setattr(sys, "argv", [*argv, "--allow-defects"])
    main()

    assert lancedb.connect(str(out)).open_table(VIDEOS_TABLE).count_rows() == 1


def test_allow_defects_warns_about_what_it_copies(tmp_path, capsys):
    src = _make_dataset(tmp_path / "src", video_frames=14)
    out = tmp_path / "out"

    convert(out, root=src, allow_defects=True)

    warning = capsys.readouterr().err
    assert "--allow-defects" in warning
    assert "short by 6" in warning
    assert lancedb.connect(str(out)).open_table(FRAMES_TABLE).count_rows() == FRAMES


def test_sound_source_converts_without_a_warning(tmp_path, capsys):
    src = _make_dataset(tmp_path / "src", video_frames=FRAMES)
    out = tmp_path / "out"

    convert(out, root=src)

    assert "--allow-defects" not in capsys.readouterr().err
    assert lancedb.connect(str(out)).open_table(FRAMES_TABLE).count_rows() == FRAMES


def test_orphan_files_are_reported_but_do_not_block_conversion(tmp_path):
    """Only defects in referenced files refuse a conversion.

    An orphan parquet is not part of the dataset a correct file selection copies,
    so the doctor reports it as a failed check while the converter still proceeds.
    """
    src = _make_dataset(tmp_path / "src", video_frames=FRAMES)
    orphan = src / "data/chunk-000/file-999.parquet"
    pq.write_table(pa.table({"index": pa.array([20, 21, 22], pa.int64())}), orphan)

    result = audit(src)
    assert any("orphan parquet" in problem.message for check in result.checks for problem in check.problems)
    assert result.blocking_problems == []

    _reject_damaged_source(src, None, allow_defects=False)
