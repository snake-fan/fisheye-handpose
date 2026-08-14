from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.fusion import (  # noqa: E402
    JointObservation,
    RobustStereoFusion,
    StereoFusionConfig,
)

LEFT_PROJECTION = np.array(
    [
        [800.0, 0.0, 320.0, 0.0],
        [0.0, 800.0, 240.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
RIGHT_PROJECTION = np.array(
    [
        [800.0, 0.0, 320.0, -80.0],
        [0.0, 800.0, 240.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _observation(u: float, v: float, *, score: float = 1.0) -> JointObservation:
    return JointObservation(
        uv=(u, v),
        score=score,
        covariance_px2=((1.0, 0.0), (0.0, 1.0)),
    )


def test_joint_fusion_recovers_a_known_metric_point() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(min_ray_angle_deg=0.1),
    )

    # Independent worked projection for XYZ=(0.04, -0.02, 0.8) m:
    # left=(360, 220), right=(260, 220) for f=800 px and baseline=0.1 m.
    result = fusion.fuse_joint(_observation(360.0, 220.0), _observation(260.0, 220.0))

    assert result.validity == "VALID"
    assert result.reason is None
    np.testing.assert_allclose(result.point_xyz_m, (0.04, -0.02, 0.8), atol=1e-9)
    assert result.left_reprojection_error_px is not None
    assert result.left_reprojection_error_px < 1e-9
    assert result.right_reprojection_error_px is not None
    assert result.right_reprojection_error_px < 1e-9
    assert result.left_depth_m == pytest.approx(0.8)
    assert result.right_depth_m == pytest.approx(0.8)
    assert result.ray_angle_deg is not None and result.ray_angle_deg > 7.0
    assert result.covariance_status == "HEURISTIC_UNCALIBRATED"
    assert result.covariance_m2 is not None


def test_low_score_observation_has_less_influence_on_robust_fusion() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(
            min_keypoint_score=0.2,
            max_epipolar_error_px=10.0,
            max_reprojection_error_px=10.0,
            min_ray_angle_deg=0.1,
            huber_delta=2.0,
        ),
    )

    # The right v coordinate is an 8 px outlier but remains above the score gate.
    # A score of 0.25 gives it 1/16 of the nominal image-space information.
    result = fusion.fuse_joint(
        _observation(360.0, 220.0),
        _observation(260.0, 228.0, score=0.25),
    )

    assert result.validity == "VALID"
    assert result.left_reprojection_error_px is not None
    assert result.right_reprojection_error_px is not None
    assert result.left_reprojection_error_px < 1.0
    assert result.right_reprojection_error_px > 7.0


def test_epipolar_gate_rejects_inconsistent_rectified_observations() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(max_epipolar_error_px=3.0),
    )

    result = fusion.fuse_joint(_observation(360.0, 220.0), _observation(260.0, 224.0))

    assert result.validity == "INVALID"
    assert result.reason == "EPIPOLAR_ERROR"
    assert result.point_xyz_m is None
    assert result.epipolar_error_px == 4.0
    assert result.support_view_count == 2


def test_small_ray_angle_is_rejected_before_uncertain_depth_is_accepted() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(min_ray_angle_deg=0.5),
    )

    # A 0.1 px disparity implies an almost parallel ray pair (about 0.007 degrees).
    result = fusion.fuse_joint(_observation(360.0, 220.0), _observation(359.9, 220.0))

    assert result.validity == "INVALID"
    assert result.reason == "RAY_ANGLE_TOO_SMALL"
    assert result.ray_angle_deg is not None
    assert result.ray_angle_deg < 0.01
    assert result.covariance_m2 is None


def test_cheirality_gate_rejects_a_point_behind_both_cameras() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(min_ray_angle_deg=0.1),
    )

    # Negative disparity is the exact projection of XYZ=(-0.04, 0.02, -0.8) m.
    result = fusion.fuse_joint(_observation(360.0, 220.0), _observation(460.0, 220.0))

    assert result.validity == "INVALID"
    assert result.reason == "BEHIND_CAMERA"
    assert result.left_depth_m == pytest.approx(-0.8)
    assert result.right_depth_m == pytest.approx(-0.8)
    assert result.point_xyz_m is None


def test_working_depth_gate_rejects_a_geometric_point_outside_range() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(
            min_ray_angle_deg=0.1,
            min_depth_m=0.15,
            max_depth_m=1.5,
        ),
    )

    # Independent projection of XYZ=(0.04, -0.02, 2.0) m is L=(336,232), R=(296,232).
    result = fusion.fuse_joint(_observation(336.0, 232.0), _observation(296.0, 232.0))

    assert result.validity == "INVALID"
    assert result.reason == "DEPTH_OUT_OF_RANGE"
    assert result.left_depth_m == pytest.approx(2.0)
    assert result.right_depth_m == pytest.approx(2.0)


def test_covariance_is_psd_and_grows_as_disparity_shrinks() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(min_ray_angle_deg=0.1, max_depth_m=2.0),
    )

    # XYZ=(0,0,0.5) gives disparity 160 px; XYZ=(0,0,1.5) gives 53 1/3 px.
    near = fusion.fuse_joint(_observation(320.0, 240.0), _observation(160.0, 240.0))
    far = fusion.fuse_joint(
        _observation(320.0, 240.0),
        _observation(266.6666666666667, 240.0),
    )

    for result in (near, far):
        assert result.validity == "VALID"
        assert result.support_view_count == 2
        assert result.covariance_status == "HEURISTIC_UNCALIBRATED"
        covariance = np.asarray(result.covariance_m2)
        assert covariance.shape == (3, 3)
        assert np.all(np.isfinite(covariance))
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-15

    near_covariance = np.asarray(near.covariance_m2)
    far_covariance = np.asarray(far.covariance_m2)
    assert far_covariance[2, 2] > 50.0 * near_covariance[2, 2]


def test_hand_fusion_requires_minimum_current_palm_support() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(min_ray_angle_deg=0.1, min_palm_support=3),
    )
    left = [_observation(360.0, 220.0) for _ in range(21)]
    right = [_observation(260.0, 220.0) for _ in range(21)]
    for joint_index in (5, 9, 13):
        left[joint_index] = JointObservation(
            uv=(360.0, 220.0),
            score=1.0,
            covariance_px2=((1.0, 0.0), (0.0, 1.0)),
            visible=False,
        )

    result = fusion.fuse_hand(left, right)

    assert result.validity == "INVALID"
    assert result.reason == "INSUFFICIENT_PALM_SUPPORT"
    assert result.palm_support_count == 2
    assert result.minimum_palm_support == 3
    assert result.valid_joint_count == 18
    assert len(result.joints) == 21
    assert result.joints[0].validity == "VALID"
    assert result.joints[5].reason == "NOT_VISIBLE"


def test_reprojection_gate_invalidates_an_unrepaired_outlier() -> None:
    fusion = RobustStereoFusion(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        config=StereoFusionConfig(
            min_keypoint_score=0.2,
            max_epipolar_error_px=10.0,
            max_reprojection_error_px=2.0,
            min_ray_angle_deg=0.1,
        ),
    )

    result = fusion.fuse_joint(
        _observation(360.0, 220.0),
        _observation(260.0, 228.0, score=0.25),
    )

    assert result.validity == "INVALID"
    assert result.reason == "REPROJECTION_ERROR"
    assert result.left_reprojection_error_px is not None
    assert result.right_reprojection_error_px is not None
    assert result.right_reprojection_error_px > 7.0
    assert result.covariance_m2 is None


def test_non_finite_observation_fails_closed_for_only_that_joint() -> None:
    fusion = RobustStereoFusion(LEFT_PROJECTION, RIGHT_PROJECTION)
    invalid_left = JointObservation(
        uv=(float("nan"), 220.0),
        score=1.0,
        covariance_px2=((1.0, 0.0), (0.0, 1.0)),
    )

    result = fusion.fuse_joint(invalid_left, _observation(260.0, 220.0))

    assert result.validity == "INVALID"
    assert result.reason == "NON_FINITE_OBSERVATION"
    assert result.point_xyz_m is None


def test_non_positive_definite_image_covariance_fails_closed() -> None:
    fusion = RobustStereoFusion(LEFT_PROJECTION, RIGHT_PROJECTION)
    invalid_left = JointObservation(
        uv=(360.0, 220.0),
        score=1.0,
        covariance_px2=((1.0, 2.0), (2.0, 1.0)),
    )

    result = fusion.fuse_joint(invalid_left, _observation(260.0, 220.0))

    assert result.validity == "INVALID"
    assert result.reason == "INVALID_OBSERVATION_COVARIANCE"
    assert result.covariance_m2 is None


def test_fusion_rejects_unbounded_quality_weight_before_covariance_scaling() -> None:
    fusion = RobustStereoFusion(LEFT_PROJECTION, RIGHT_PROJECTION)

    result = fusion.fuse_joint(
        _observation(360.0, 220.0, score=1.0953675508499146),
        _observation(260.0, 220.0),
    )

    assert result.validity == "INVALID"
    assert result.reason == "INVALID_QUALITY_WEIGHT"
    assert result.covariance_m2 is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_keypoint_score": -0.1},
        {"max_epipolar_error_px": 0.0},
        {"min_ray_angle_deg": -0.1},
        {"min_depth_m": 1.0, "max_depth_m": 0.5},
        {"huber_delta": 0.0},
        {"max_iterations": 0},
        {"min_palm_support": 6},
        {"palm_indices": (0, 5, 5, 13, 17)},
    ],
)
def test_invalid_fusion_configuration_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StereoFusionConfig(**overrides)


def test_fusion_rejects_projection_pair_without_a_stereo_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        RobustStereoFusion(LEFT_PROJECTION, LEFT_PROJECTION)
