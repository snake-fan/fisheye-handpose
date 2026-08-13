from pathlib import Path

import pytest
import yaml  # noqa: F401

from fisheye_handpose.calibration import load_orbbec_stereo
from fisheye_handpose.errors import CalibrationError

YAML = """
calibration_info:
  reference_camera: cam_0
cameras:
  - id: cam_0
    name: IR_L
    distortion_model: KB
    image_width: 1600
    image_height: 1300
    intrinsics: {fx: 506.6, fy: 506.4, cx: 811.2, cy: 623.4}
    distortion: {k1: 0.07, k2: -0.004, k3: -0.006, k4: 0.002, k5: 0, k6: 0, p1: 0, p2: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [0, 0, 0]
  - id: cam_1
    name: IR_R
    distortion_model: KB
    image_width: 1600
    image_height: 1300
    intrinsics: {fx: 502.8, fy: 502.7, cx: 790.1, cy: 647.6}
    distortion: {k1: 0.072, k2: -0.013, k3: 0.0025, k4: -0.0004, k5: 0, k6: 0, p1: 0, p2: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [-119.889, -1.133, -0.256]
"""


def write_yaml(path: Path, text: str = YAML) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_orbbec_calibration_normalizes_mm_and_names_direction(tmp_path):
    calibration = load_orbbec_stereo(
        write_yaml(tmp_path / "calibration.yaml"),
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="mm",
        extrinsics_convention="reference_to_camera",
    )
    assert calibration.right_from_left.from_frame == "cam_0"
    assert calibration.right_from_left.to_frame == "cam_1"
    assert calibration.right_from_left.baseline_m == pytest.approx(0.1198946, rel=1e-5)
    assert calibration.right_from_left.translation_m[0] == pytest.approx(-0.119889)


def test_wrong_translation_unit_fails_baseline_gate(tmp_path):
    with pytest.raises(CalibrationError, match="baseline"):
        load_orbbec_stereo(
            write_yaml(tmp_path / "calibration.yaml"),
            left_id="cam_0",
            right_id="cam_1",
            translation_unit="m",
            extrinsics_convention="reference_to_camera",
        )


def test_nonzero_unsupported_coefficient_is_rejected(tmp_path):
    bad = YAML.replace("k5: 0,", "k5: 0.1,")
    with pytest.raises(CalibrationError, match="unsupported"):
        load_orbbec_stereo(
            write_yaml(tmp_path / "bad.yaml", bad),
            left_id="cam_0",
            right_id="cam_1",
            translation_unit="mm",
            extrinsics_convention="reference_to_camera",
        )


def test_camera_to_reference_convention_is_inverted_explicitly(tmp_path):
    camera_to_reference = YAML.replace(
        "translation: [-119.889, -1.133, -0.256]",
        "translation: [119.889, 1.133, 0.256]",
    )
    calibration = load_orbbec_stereo(
        write_yaml(tmp_path / "calibration.yaml", camera_to_reference),
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="mm",
        extrinsics_convention="camera_to_reference",
    )
    assert calibration.right_from_left.translation_m == pytest.approx(
        (-0.119889, -0.001133, -0.000256)
    )


def test_reference_camera_must_have_identity_self_extrinsics(tmp_path):
    bad = YAML.replace("translation: [0, 0, 0]", "translation: [1, 0, 0]", 1)
    with pytest.raises(CalibrationError, match="identity self-extrinsics"):
        load_orbbec_stereo(
            write_yaml(tmp_path / "bad.yaml", bad),
            left_id="cam_0",
            right_id="cam_1",
            translation_unit="mm",
            extrinsics_convention="reference_to_camera",
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("image_width: 1600", "image_width: 1600.5", "positive integer"),
        ("fx: 506.6", "fx: true", "real number"),
    ],
)
def test_numeric_fields_are_not_silently_coerced(tmp_path, old, new, message):
    with pytest.raises(CalibrationError, match=message):
        load_orbbec_stereo(
            write_yaml(tmp_path / "bad.yaml", YAML.replace(old, new, 1)),
            left_id="cam_0",
            right_id="cam_1",
            translation_unit="mm",
            extrinsics_convention="reference_to_camera",
        )


def test_kb_mapping_must_be_invertible_over_the_full_image(tmp_path):
    folded = YAML.replace("k1: 0.07", "k1: -1.0", 1)
    with pytest.raises(CalibrationError, match="folds|monotonic"):
        load_orbbec_stereo(
            write_yaml(tmp_path / "folded.yaml", folded),
            left_id="cam_0",
            right_id="cam_1",
            translation_unit="mm",
            extrinsics_convention="reference_to_camera",
        )


def test_baseline_safety_range_is_configurable(tmp_path):
    calibration = load_orbbec_stereo(
        write_yaml(tmp_path / "calibration.yaml"),
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="m",
        extrinsics_convention="reference_to_camera",
        baseline_range_m=(100.0, 130.0),
    )
    assert calibration.right_from_left.baseline_m == pytest.approx(119.8946, rel=1e-5)
    assert calibration.to_dict()["baseline_range_m"] == [100.0, 130.0]


def test_non_reference_cameras_are_composed_through_reference(tmp_path):
    three_camera_yaml = """
calibration_info:
  reference_camera: rig
cameras:
  - id: rig
    name: rig
    distortion_model: KB
    image_width: 1600
    image_height: 1300
    intrinsics: {fx: 500, fy: 500, cx: 800, cy: 650}
    distortion: {k1: 0, k2: 0, k3: 0, k4: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [0, 0, 0]
  - id: left
    name: left
    distortion_model: KB
    image_width: 1600
    image_height: 1300
    intrinsics: {fx: 500, fy: 500, cx: 800, cy: 650}
    distortion: {k1: 0, k2: 0, k3: 0, k4: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [10, 0, 0]
  - id: right
    name: right
    distortion_model: KB
    image_width: 1600
    image_height: 1300
    intrinsics: {fx: 500, fy: 500, cx: 800, cy: 650}
    distortion: {k1: 0, k2: 0, k3: 0, k4: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [-110, 0, 0]
"""
    calibration = load_orbbec_stereo(
        write_yaml(tmp_path / "calibration.yaml", three_camera_yaml),
        left_id="left",
        right_id="right",
        translation_unit="mm",
        extrinsics_convention="reference_to_camera",
    )
    assert calibration.right_from_left.translation_m == pytest.approx((-0.12, 0.0, 0.0))
