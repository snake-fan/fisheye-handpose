import math

import cv2
import numpy as np
import pytest
import yaml  # noqa: F401

from fisheye_handpose.calibration import load_orbbec_stereo
from fisheye_handpose.geometry import RectificationConfig, StereoRectifier
from tests.test_calibration import YAML, write_yaml


def calibration_with_non_identity_rotation() -> str:
    identity = "rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    rotation = (
        "rotation: [[0.99999412, 0.00222554, -0.00261015], "
        "[-0.00222226, 0.99999674, 0.00125856], "
        "[0.00261294, -0.00125276, 0.99999580]]"
    )
    prefix, separator, suffix = YAML.rpartition(identity)
    assert separator
    return prefix + rotation + suffix


def test_rectified_synthetic_correspondences_have_matching_rows(tmp_path):
    calibration = load_orbbec_stereo(
        write_yaml(tmp_path / "calibration.yaml", calibration_with_non_identity_rotation()),
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="mm",
        extrinsics_convention="reference_to_camera",
    )
    rectifier = StereoRectifier.build(
        calibration, RectificationConfig(output_size=(1600, 1300), balance=0.5)
    )
    points_left = np.array(
        [[-0.15, -0.1, 0.5], [0.0, 0.0, 0.7], [0.2, 0.1, 1.0]], dtype=np.float64
    ).reshape(-1, 1, 3)
    rotation = np.asarray(calibration.right_from_left.rotation)
    translation = np.asarray(calibration.right_from_left.translation_m)
    points_right = (points_left.reshape(-1, 3) @ rotation.T + translation).reshape(-1, 1, 3)

    def project(camera, points):
        image, _ = cv2.fisheye.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            np.asarray(camera.intrinsics),
            np.asarray(camera.distortion),
        )
        return image

    left_uv = project(calibration.left, points_left)
    right_uv = project(calibration.right, points_right)
    left_rect = cv2.fisheye.undistortPoints(
        left_uv,
        np.asarray(calibration.left.intrinsics),
        np.asarray(calibration.left.distortion),
        R=rectifier.r1,
        P=rectifier.p1,
    )
    right_rect = cv2.fisheye.undistortPoints(
        right_uv,
        np.asarray(calibration.right.intrinsics),
        np.asarray(calibration.right.distortion),
        R=rectifier.r2,
        P=rectifier.p2,
    )
    assert np.max(np.abs(left_rect[..., 1] - right_rect[..., 1])) < 1e-5

    homogeneous = cv2.triangulatePoints(
        rectifier.p1,
        rectifier.p2,
        left_rect.reshape(-1, 2).T,
        right_rect.reshape(-1, 2).T,
    )
    triangulated = (homogeneous[:3] / homogeneous[3]).T
    expected_rectified_left = points_left.reshape(-1, 3) @ np.asarray(rectifier.r1).T
    assert np.all(triangulated[:, 2] > 0)
    assert np.allclose(triangulated, expected_rectified_left, rtol=1e-7, atol=1e-8)

    disparities = left_rect[..., 0] - right_rect[..., 0]
    q_input = np.column_stack(
        (
            left_rect[..., 0],
            left_rect[..., 1],
            disparities,
            np.ones(len(disparities)),
        )
    )
    q_homogeneous = (rectifier.q @ q_input.T).T
    q_points = q_homogeneous[:, :3] / q_homogeneous[:, 3, None]
    assert np.allclose(q_points, expected_rectified_left, rtol=1e-7, atol=1e-8)

    rectified_tx = rectifier.p2[0, 3] / rectifier.p2[0, 0] - rectifier.p1[0, 3] / rectifier.p1[0, 0]
    assert abs(rectified_tx) == pytest.approx(calibration.right_from_left.baseline_m)
    assert abs(1.0 / rectifier.q[3, 2]) == pytest.approx(calibration.right_from_left.baseline_m)
    assert math.isfinite(rectifier.common_valid_fraction)
    assert rectifier.common_valid_fraction > 0.1


def test_effective_fov_uses_off_center_principal_point(tmp_path):
    calibration = load_orbbec_stereo(
        write_yaml(tmp_path / "calibration.yaml", YAML),
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="mm",
        extrinsics_convention="reference_to_camera",
    )
    rectifier = StereoRectifier.build(
        calibration, RectificationConfig(output_size=(1600, 1300), balance=0.5)
    )
    rectifier.p1 = rectifier.p1.copy()
    rectifier.p1[0, 2] = 300.0
    rectifier.p1[1, 2] = 400.0
    horizontal, vertical = rectifier.effective_fov_degrees()
    fx, fy = rectifier.p1[0, 0], rectifier.p1[1, 1]
    assert horizontal == pytest.approx(math.degrees(math.atan2(300.0, fx) + math.atan2(1299.0, fx)))
    assert vertical == pytest.approx(math.degrees(math.atan2(400.0, fy) + math.atan2(899.0, fy)))


def test_linear_remap_valid_mask_excludes_last_source_row_and_column(tmp_path):
    calibration = load_orbbec_stereo(
        write_yaml(tmp_path / "calibration.yaml", YAML),
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="mm",
        extrinsics_convention="reference_to_camera",
    )
    rectifier = StereoRectifier.build(
        calibration, RectificationConfig(output_size=(1600, 1300), balance=0.5)
    )
    map_x, map_y = rectifier.left_maps
    expected = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0)
        & (map_x < 1599)
        & (map_y >= 0)
        & (map_y < 1299)
    )
    assert np.array_equal(rectifier.left_valid_mask, expected)
