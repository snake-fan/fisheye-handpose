from pathlib import Path

import pytest

from fisheye_handpose.errors import DiscoveryError
from fisheye_handpose.session import discover_session


def touch(path: Path) -> None:
    path.write_bytes(b"fixture")


def test_discovery_is_structural_and_part_sorted(tmp_path):
    prefix = "capture_with_left_in_name"
    touch(tmp_path / f"{prefix}_calibration_camera.yaml")
    for part in (2, 1):
        for side in ("left", "right"):
            touch(tmp_path / f"{prefix}_camera_{side}_part{part:04d}.mp4")
            touch(tmp_path / f"{prefix}_camera_{side}_part{part:04d}_pts.csv")
    session = discover_session(tmp_path)
    assert [part.part_number for part in session.parts] == [1, 2]
    assert session.parts[0].right_video.name.endswith("camera_right_part0001.mp4")


def test_discovery_rejects_incomplete_part(tmp_path):
    prefix = "capture"
    touch(tmp_path / f"{prefix}_calibration_camera.yaml")
    touch(tmp_path / f"{prefix}_camera_left_part0001.mp4")
    touch(tmp_path / f"{prefix}_camera_left_part0001_pts.csv")
    with pytest.raises(DiscoveryError, match="incomplete"):
        discover_session(tmp_path)


def test_discovery_rejects_calibration_from_a_different_prefix(tmp_path):
    prefix = "capture"
    touch(tmp_path / "other_rig_calibration_camera.yaml")
    for side in ("left", "right"):
        touch(tmp_path / f"{prefix}_camera_{side}_part0001.mp4")
        touch(tmp_path / f"{prefix}_camera_{side}_part0001_pts.csv")
    with pytest.raises(DiscoveryError, match="calibration prefix"):
        discover_session(tmp_path)


def test_discovery_rejects_non_contiguous_parts(tmp_path):
    prefix = "capture"
    touch(tmp_path / f"{prefix}_calibration_camera.yaml")
    for part in (1, 3):
        for side in ("left", "right"):
            touch(tmp_path / f"{prefix}_camera_{side}_part{part:04d}.mp4")
            touch(tmp_path / f"{prefix}_camera_{side}_part{part:04d}_pts.csv")
    with pytest.raises(DiscoveryError, match="contiguous"):
        discover_session(tmp_path)


def test_discovery_rejects_named_artifact_that_is_not_a_file(tmp_path):
    prefix = "capture"
    touch(tmp_path / f"{prefix}_calibration_camera.yaml")
    for side in ("left", "right"):
        video = tmp_path / f"{prefix}_camera_{side}_part0001.mp4"
        if side == "left":
            video.mkdir()
        else:
            touch(video)
        touch(tmp_path / f"{prefix}_camera_{side}_part0001_pts.csv")
    with pytest.raises(DiscoveryError, match="not a file"):
        discover_session(tmp_path)
