from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.calibration import RectifiedStereo  # noqa: E402
from fisheye_h20_worker.crop import VirtualPerspectiveCropper  # noqa: E402


def _stereo(
    *,
    left_k: np.ndarray,
    right_k: np.ndarray | None = None,
    left_d: np.ndarray | None = None,
    right_d: np.ndarray | None = None,
    right_from_left_rotation: np.ndarray | None = None,
    right_from_left_translation_m: np.ndarray | None = None,
) -> RectifiedStereo:
    right_k = left_k.copy() if right_k is None else right_k
    zero_d = np.zeros((4, 1), dtype=np.float64)
    left_d = zero_d.copy() if left_d is None else left_d
    right_d = zero_d.copy() if right_d is None else right_d
    identity = np.eye(3, dtype=np.float64)
    right_from_left_rotation = (
        identity.copy() if right_from_left_rotation is None else right_from_left_rotation
    )
    right_from_left_translation_m = (
        np.array([-0.1, 0.0, 0.0])
        if right_from_left_translation_m is None
        else right_from_left_translation_m
    )
    projection = np.column_stack((left_k, np.zeros(3, dtype=np.float64)))
    right_projection = np.column_stack((right_k, np.zeros(3, dtype=np.float64)))
    dummy_map = np.zeros((200, 200), dtype=np.float32)
    return RectifiedStereo(
        calibration_id="fixture-calibration",
        image_size=(200, 200),
        output_size=(200, 200),
        left_k=left_k,
        left_d=left_d,
        right_k=right_k,
        right_d=right_d,
        right_from_left_rotation=right_from_left_rotation,
        right_from_left_translation_m=right_from_left_translation_m,
        r1=identity.copy(),
        r2=identity.copy(),
        p1=projection,
        p2=right_projection,
        q=np.eye(4, dtype=np.float64),
        left_undistort_maps=(dummy_map, dummy_map),
        right_undistort_maps=(dummy_map, dummy_map),
        left_rectify_maps=(dummy_map, dummy_map),
        right_rectify_maps=(dummy_map, dummy_map),
    )


def test_zero_distortion_is_still_equidistant_kb4_and_preserves_center_ray() -> None:
    source_k = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    crop = VirtualPerspectiveCropper(output_size=(65, 65), bbox_scale=1.0).create(
        np.zeros((200, 200, 3), dtype=np.uint8),
        "left",
        (60.0, 60.0, 140.0, 140.0),
        _stereo(left_k=source_k),
    )

    np.testing.assert_allclose(crop.R_source_from_virtual[:, 2], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        crop.crop_uv_to_source_uv(np.array([[32.0, 32.0]])),
        [[100.0, 100.0]],
        atol=1e-10,
    )

    virtual_focal = float(crop.K_virtual[0, 0])
    crop_point = np.array([[32.0 + 0.25 * virtual_focal, 32.0]])
    mapped = crop.crop_uv_to_source_uv(crop_point)
    expected_u = 100.0 + 100.0 * np.arctan(0.25)
    np.testing.assert_allclose(mapped, [[expected_u, 100.0]], atol=1e-8)


def test_source_and_crop_coordinates_round_trip_with_nonzero_kb4() -> None:
    source_k = np.array(
        [[112.0, 0.0, 96.0], [0.0, 108.0, 103.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array([[0.025], [-0.006], [0.001], [-0.0001]], dtype=np.float64)
    crop = VirtualPerspectiveCropper(output_size=(96, 80), bbox_scale=1.25).create(
        np.zeros((200, 200, 3), dtype=np.uint8),
        "left",
        (74.0, 68.0, 154.0, 158.0),
        _stereo(left_k=source_k, left_d=distortion),
    )
    source_points = np.array(
        [[114.0, 113.0], [90.0, 92.0], [139.0, 139.0], [120.5, 83.25]],
        dtype=np.float64,
    )

    crop_points = crop.source_uv_to_crop_uv(source_points)
    recovered = crop.crop_uv_to_source_uv(crop_points)

    np.testing.assert_allclose(recovered, source_points, atol=1e-7)


def test_crop_near_source_boundary_marks_only_sampleable_pixels_valid() -> None:
    source_k = np.array(
        [[92.0, 0.0, 100.0], [0.0, 92.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frame = np.full((200, 200, 3), 173, dtype=np.uint8)
    crop = VirtualPerspectiveCropper(output_size=(80, 64), bbox_scale=1.4).create(
        frame,
        "left",
        (-20.0, -18.0, 72.0, 78.0),
        _stereo(left_k=source_k),
    )

    grid_u, grid_v = np.meshgrid(np.arange(80), np.arange(64))
    source_uv = crop.crop_uv_to_source_uv(np.stack((grid_u, grid_v), axis=-1))
    expected_mask = (
        (source_uv[..., 0] >= 0.0)
        & (source_uv[..., 0] <= 199.0)
        & (source_uv[..., 1] >= 0.0)
        & (source_uv[..., 1] <= 199.0)
    )

    assert crop.valid_mask.dtype == np.bool_
    assert crop.valid_mask.shape == (64, 80)
    assert crop.valid_mask.any()
    assert not crop.valid_mask.all()
    np.testing.assert_array_equal(crop.valid_mask, expected_mask)
    assert np.all(crop.image[~crop.valid_mask] == 0)
    assert np.all(crop.image[crop.valid_mask] == 173)


def test_camera_side_selects_its_own_kb4_intrinsics() -> None:
    left_k = np.array(
        [[90.0, 0.0, 100.0], [0.0, 90.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    right_k = np.array(
        [[150.0, 0.0, 100.0], [0.0, 150.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    stereo = _stereo(left_k=left_k, right_k=right_k)
    cropper = VirtualPerspectiveCropper(output_size=(65, 65), bbox_scale=1.0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    left = cropper.create(frame, "left", (60.0, 60.0, 140.0, 140.0), stereo)
    right = cropper.create(frame, "right", (60.0, 60.0, 140.0, 140.0), stereo)
    left_probe = np.array([[32.0 + 0.2 * left.K_virtual[0, 0], 32.0]])
    right_probe = np.array([[32.0 + 0.2 * right.K_virtual[0, 0], 32.0]])

    np.testing.assert_allclose(
        left.crop_uv_to_source_uv(left_probe),
        [[100.0 + 90.0 * np.arctan(0.2), 100.0]],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        right.crop_uv_to_source_uv(right_probe),
        [[100.0 + 150.0 * np.arctan(0.2), 100.0]],
        atol=1e-8,
    )


def test_right_virtual_camera_pose_is_expressed_in_left_camera_rig_frame() -> None:
    source_k = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    angle = np.deg2rad(17.0)
    right_from_left_rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    right_from_left_translation = np.array([-0.09, 0.012, 0.018], dtype=np.float64)
    stereo = _stereo(
        left_k=source_k,
        right_from_left_rotation=right_from_left_rotation,
        right_from_left_translation_m=right_from_left_translation,
    )

    right = VirtualPerspectiveCropper(output_size=(65, 65), bbox_scale=1.0).create(
        np.zeros((200, 200, 3), dtype=np.uint8),
        "right",
        (60.0, 60.0, 140.0, 140.0),
        stereo,
    )
    left = VirtualPerspectiveCropper(output_size=(65, 65), bbox_scale=1.0).create(
        np.zeros((200, 200, 3), dtype=np.uint8),
        "left",
        (60.0, 60.0, 140.0, 140.0),
        stereo,
    )

    left_from_right_rotation = right_from_left_rotation.T
    expected_origin_left = -left_from_right_rotation @ right_from_left_translation
    expected_axis_left = left_from_right_rotation @ right.R_source_from_virtual[:, 2]
    np.testing.assert_allclose(right.T_rig_from_virtual[:3, 3], expected_origin_left, atol=1e-12)
    np.testing.assert_allclose(
        right.T_rig_from_virtual[:3, :3] @ np.array([0.0, 0.0, 1.0]),
        expected_axis_left,
        atol=1e-12,
    )
    np.testing.assert_allclose(right.T_rig_from_virtual[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(left.T_rig_from_virtual[:3, :3], left.R_source_from_virtual)
    np.testing.assert_allclose(left.T_rig_from_virtual[:3, 3], [0.0, 0.0, 0.0])


def test_crop_identity_is_deterministic_geometry_not_frame_content() -> None:
    source_k = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    stereo = _stereo(left_k=source_k)
    cropper = VirtualPerspectiveCropper(output_size=(48, 40), bbox_scale=1.2)
    bbox = (50.0, 62.0, 137.0, 151.0)

    dark = cropper.create(np.zeros((200, 200, 3), dtype=np.uint8), "left", bbox, stereo)
    bright = cropper.create(np.full((200, 200, 3), 255, dtype=np.uint8), "left", bbox, stereo)

    assert dark.crop_id == bright.crop_id
    assert dark.policy_id == bright.policy_id
    assert dark.crop_id != dark.policy_id
    assert dark.side == "left"
    assert dark.bbox == bbox
    assert dark.valid_fraction == float(np.mean(dark.valid_mask))


@pytest.mark.parametrize(
    "bbox",
    [
        (20.0, 20.0, 20.0, 60.0),
        (60.0, 20.0, 20.0, 60.0),
        (20.0, float("nan"), 60.0, 80.0),
        (True, 20.0, 60.0, 80.0),
        (220.0, 20.0, 260.0, 80.0),
    ],
)
def test_invalid_bbox_is_rejected(bbox: tuple[object, object, object, object]) -> None:
    source_k = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="bbox"):
        VirtualPerspectiveCropper().create(
            np.zeros((200, 200, 3), dtype=np.uint8),
            "left",
            bbox,
            _stereo(left_k=source_k),
        )


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((200, 200), dtype=np.uint8),
        np.zeros((200, 200, 4), dtype=np.uint8),
        np.zeros((200, 200, 3), dtype=np.float32),
        np.zeros((199, 200, 3), dtype=np.uint8),
    ],
)
def test_model_input_frame_must_be_calibrated_uint8_bgr(frame: np.ndarray) -> None:
    source_k = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="frame"):
        VirtualPerspectiveCropper().create(
            frame,
            "left",
            (60.0, 60.0, 140.0, 140.0),
            _stereo(left_k=source_k),
        )


def test_unknown_camera_side_is_rejected() -> None:
    source_k = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="camera side"):
        VirtualPerspectiveCropper().create(
            np.zeros((200, 200, 3), dtype=np.uint8),
            "centre",
            (60.0, 60.0, 140.0, 140.0),
            _stereo(left_k=source_k),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_size": (1, 256)},
        {"output_size": (256.0, 256)},
        {"output_size": None},
        {"bbox_scale": 0.0},
        {"bbox_scale": float("nan")},
        {"bbox_scale": True},
        {"bbox_scale": "1.5"},
    ],
)
def test_invalid_crop_policy_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        VirtualPerspectiveCropper(**kwargs)  # type: ignore[arg-type]
