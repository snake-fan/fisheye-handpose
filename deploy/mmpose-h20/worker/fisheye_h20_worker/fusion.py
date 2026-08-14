"""Uncertainty-weighted metric stereo fusion for rectified keypoints.

This module deliberately performs geometric fusion only.  Two-view reprojection cannot
detect an epipolar-consistent cross-hand mismatch, and no anatomical or bone-length prior
is allowed to manufacture a Raw 3D measurement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

COVARIANCE_STATUS = "HEURISTIC_UNCALIBRATED"
FUSION_METHOD_ID = "robust_stereo_huber_irls_v1"


@dataclass(frozen=True)
class StereoFusionConfig:
    min_keypoint_score: float = 0.2
    max_epipolar_error_px: float = 5.0
    max_reprojection_error_px: float = 3.0
    min_ray_angle_deg: float = 0.5
    min_depth_m: float = 0.1
    max_depth_m: float = 2.0
    huber_delta: float = 2.0
    max_iterations: int = 20
    min_palm_support: int = 3
    palm_indices: tuple[int, ...] = (0, 5, 9, 13, 17)

    def __post_init__(self) -> None:
        numeric = (
            self.min_keypoint_score,
            self.max_epipolar_error_px,
            self.max_reprojection_error_px,
            self.min_ray_angle_deg,
            self.min_depth_m,
            self.max_depth_m,
            self.huber_delta,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("fusion thresholds must be finite")
        if not 0.0 <= self.min_keypoint_score <= 1.0:
            raise ValueError("min_keypoint_score must be in [0, 1]")
        if self.max_epipolar_error_px <= 0.0 or self.max_reprojection_error_px <= 0.0:
            raise ValueError("pixel gates must be positive")
        if self.min_ray_angle_deg <= 0.0:
            raise ValueError("min_ray_angle_deg must be positive")
        if self.min_depth_m <= 0.0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("depth range must be positive and increasing")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        if (
            not self.palm_indices
            or len(set(self.palm_indices)) != len(self.palm_indices)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 21
                for index in self.palm_indices
            )
        ):
            raise ValueError("palm_indices must be unique FHP21 indices")
        if (
            isinstance(self.min_palm_support, bool)
            or not isinstance(self.min_palm_support, int)
            or not 1 <= self.min_palm_support <= len(self.palm_indices)
        ):
            raise ValueError("min_palm_support must fit the configured palm indices")


@dataclass(frozen=True)
class JointObservation:
    uv: tuple[float, float]
    score: float
    covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    visible: bool = True
    valid_mask: bool = True


@dataclass(frozen=True)
class JointFusionResult:
    point_xyz_m: tuple[float, float, float] | None
    validity: Literal["VALID", "INVALID"]
    reason: str | None
    support_view_count: int
    epipolar_error_px: float | None
    left_reprojection_error_px: float | None
    right_reprojection_error_px: float | None
    ray_angle_deg: float | None
    left_depth_m: float | None
    right_depth_m: float | None
    covariance_m2: tuple[tuple[float, float, float], ...] | None
    covariance_status: str

    @property
    def valid(self) -> bool:
        return self.validity == "VALID"


@dataclass(frozen=True)
class HandFusionResult:
    joints: tuple[JointFusionResult, ...]
    validity: Literal["VALID", "INVALID"]
    reason: str | None
    valid_joint_count: int
    palm_support_count: int
    minimum_palm_support: int

    @property
    def valid(self) -> bool:
        return self.validity == "VALID"


class RobustStereoFusion:
    def __init__(
        self,
        left_projection: np.ndarray,
        right_projection: np.ndarray,
        *,
        config: StereoFusionConfig | None = None,
    ) -> None:
        self._left_projection = _projection(left_projection, "left_projection")
        self._right_projection = _projection(right_projection, "right_projection")
        self.config = config or StereoFusionConfig()
        self._left_center, self._left_inverse = _camera_geometry(self._left_projection)
        self._right_center, self._right_inverse = _camera_geometry(self._right_projection)
        self.baseline_m = float(np.linalg.norm(self._right_center - self._left_center))
        if self.baseline_m <= 1e-9:
            raise ValueError("stereo projections must have a non-zero baseline")

    def fuse_joint(
        self,
        left: JointObservation,
        right: JointObservation,
    ) -> JointFusionResult:
        if not _observation_is_finite(left) or not _observation_is_finite(right):
            return _invalid_result("NON_FINITE_OBSERVATION")
        if not _observation_covariance_is_valid(left) or not _observation_covariance_is_valid(
            right
        ):
            return _invalid_result("INVALID_OBSERVATION_COVARIANCE")
        eligible = (
            bool(left.valid_mask and left.visible and left.score >= self.config.min_keypoint_score),
            bool(
                right.valid_mask and right.visible and right.score >= self.config.min_keypoint_score
            ),
        )
        support_view_count = sum(eligible)
        if not left.valid_mask or not right.valid_mask:
            return _invalid_result("INVALID_MASK", support_view_count=support_view_count)
        if not left.visible or not right.visible:
            return _invalid_result("NOT_VISIBLE", support_view_count=support_view_count)
        if (
            left.score < self.config.min_keypoint_score
            or right.score < self.config.min_keypoint_score
        ):
            return _invalid_result("LOW_KEYPOINT_SCORE", support_view_count=support_view_count)
        epipolar_error_px = abs(float(left.uv[1]) - float(right.uv[1]))
        if epipolar_error_px > self.config.max_epipolar_error_px:
            return _invalid_result(
                "EPIPOLAR_ERROR",
                support_view_count=support_view_count,
                epipolar_error_px=epipolar_error_px,
            )
        left_ray = self._left_inverse @ np.asarray([*left.uv, 1.0], dtype=np.float64)
        right_ray = self._right_inverse @ np.asarray([*right.uv, 1.0], dtype=np.float64)
        left_ray /= np.linalg.norm(left_ray)
        right_ray /= np.linalg.norm(right_ray)
        ray_angle_deg = math.degrees(
            math.acos(float(np.clip(np.dot(left_ray, right_ray), -1.0, 1.0)))
        )
        if ray_angle_deg < self.config.min_ray_angle_deg:
            return _invalid_result(
                "RAY_ANGLE_TOO_SMALL",
                support_view_count=support_view_count,
                epipolar_error_px=epipolar_error_px,
                ray_angle_deg=ray_angle_deg,
            )
        point = _linear_triangulation(
            self._left_projection,
            self._right_projection,
            left.uv,
            right.uv,
        )
        point = _optimize_point(
            point,
            self._left_projection,
            self._right_projection,
            left,
            right,
            self.config,
        )
        left_depth, left_projected, left_jacobian = _project_with_jacobian(
            self._left_projection, point
        )
        right_depth, right_projected, right_jacobian = _project_with_jacobian(
            self._right_projection, point
        )
        left_error = float(np.linalg.norm(left_projected - np.asarray(left.uv)))
        right_error = float(np.linalg.norm(right_projected - np.asarray(right.uv)))
        if left_depth <= 0.0 or right_depth <= 0.0:
            return _invalid_result(
                "BEHIND_CAMERA",
                support_view_count=support_view_count,
                epipolar_error_px=epipolar_error_px,
                left_reprojection_error_px=left_error,
                right_reprojection_error_px=right_error,
                ray_angle_deg=ray_angle_deg,
                left_depth_m=left_depth,
                right_depth_m=right_depth,
            )
        if (
            min(left_depth, right_depth) < self.config.min_depth_m
            or max(left_depth, right_depth) > self.config.max_depth_m
        ):
            return _invalid_result(
                "DEPTH_OUT_OF_RANGE",
                support_view_count=support_view_count,
                epipolar_error_px=epipolar_error_px,
                left_reprojection_error_px=left_error,
                right_reprojection_error_px=right_error,
                ray_angle_deg=ray_angle_deg,
                left_depth_m=left_depth,
                right_depth_m=right_depth,
            )
        if max(left_error, right_error) > self.config.max_reprojection_error_px:
            return _invalid_result(
                "REPROJECTION_ERROR",
                support_view_count=support_view_count,
                epipolar_error_px=epipolar_error_px,
                left_reprojection_error_px=left_error,
                right_reprojection_error_px=right_error,
                ray_angle_deg=ray_angle_deg,
                left_depth_m=left_depth,
                right_depth_m=right_depth,
            )
        covariance = _covariance(
            left_jacobian,
            right_jacobian,
            left,
            right,
        )
        return JointFusionResult(
            point_xyz_m=tuple(float(value) for value in point),
            validity="VALID",
            reason=None,
            support_view_count=support_view_count,
            epipolar_error_px=epipolar_error_px,
            left_reprojection_error_px=left_error,
            right_reprojection_error_px=right_error,
            ray_angle_deg=ray_angle_deg,
            left_depth_m=left_depth,
            right_depth_m=right_depth,
            covariance_m2=tuple(tuple(float(value) for value in row) for row in covariance),
            covariance_status=COVARIANCE_STATUS,
        )

    def fuse_hand(
        self,
        left: Sequence[JointObservation],
        right: Sequence[JointObservation],
    ) -> HandFusionResult:
        if len(left) != 21 or len(right) != 21:
            raise ValueError("hand fusion requires exactly 21 observations per view")
        joints = tuple(
            self.fuse_joint(left_observation, right_observation)
            for left_observation, right_observation in zip(left, right, strict=True)
        )
        valid_joint_count = sum(joint.valid for joint in joints)
        palm_support_count = sum(joints[index].valid for index in self.config.palm_indices)
        valid = palm_support_count >= self.config.min_palm_support
        return HandFusionResult(
            joints=joints,
            validity="VALID" if valid else "INVALID",
            reason=None if valid else "INSUFFICIENT_PALM_SUPPORT",
            valid_joint_count=valid_joint_count,
            palm_support_count=palm_support_count,
            minimum_palm_support=self.config.min_palm_support,
        )


def _invalid_result(
    reason: str,
    *,
    support_view_count: int = 0,
    epipolar_error_px: float | None = None,
    left_reprojection_error_px: float | None = None,
    right_reprojection_error_px: float | None = None,
    ray_angle_deg: float | None = None,
    left_depth_m: float | None = None,
    right_depth_m: float | None = None,
) -> JointFusionResult:
    return JointFusionResult(
        point_xyz_m=None,
        validity="INVALID",
        reason=reason,
        support_view_count=support_view_count,
        epipolar_error_px=epipolar_error_px,
        left_reprojection_error_px=left_reprojection_error_px,
        right_reprojection_error_px=right_reprojection_error_px,
        ray_angle_deg=ray_angle_deg,
        left_depth_m=left_depth_m,
        right_depth_m=right_depth_m,
        covariance_m2=None,
        covariance_status="NOT_AVAILABLE",
    )


def _projection(value: np.ndarray, label: str) -> np.ndarray:
    projection = np.asarray(value, dtype=np.float64)
    if projection.shape != (3, 4) or not np.all(np.isfinite(projection)):
        raise ValueError(f"{label} must be a finite 3x4 matrix")
    if abs(float(np.linalg.det(projection[:, :3]))) <= 1e-12:
        raise ValueError(f"{label} must have an invertible left 3x3 block")
    return projection.copy()


def _camera_geometry(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(projection[:, :3])
    center = -inverse @ projection[:, 3]
    return center, inverse


def _linear_triangulation(
    left_projection: np.ndarray,
    right_projection: np.ndarray,
    left_uv: tuple[float, float],
    right_uv: tuple[float, float],
) -> np.ndarray:
    rows = np.stack(
        (
            left_uv[0] * left_projection[2] - left_projection[0],
            left_uv[1] * left_projection[2] - left_projection[1],
            right_uv[0] * right_projection[2] - right_projection[0],
            right_uv[1] * right_projection[2] - right_projection[1],
        )
    )
    _, _, vh = np.linalg.svd(rows)
    homogeneous = vh[-1]
    return homogeneous[:3] / homogeneous[3]


def _project_with_jacobian(
    projection: np.ndarray, point: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    homogeneous = projection @ np.append(point, 1.0)
    depth = float(homogeneous[2])
    uv = homogeneous[:2] / depth
    jacobian = np.stack(
        (
            (projection[0, :3] * depth - homogeneous[0] * projection[2, :3]) / depth**2,
            (projection[1, :3] * depth - homogeneous[1] * projection[2, :3]) / depth**2,
        )
    )
    return depth, uv, jacobian


def _effective_covariance(observation: JointObservation) -> np.ndarray:
    return np.asarray(observation.covariance_px2, dtype=np.float64) / max(
        float(observation.score) ** 2, 1e-12
    )


def _observation_is_finite(observation: JointObservation) -> bool:
    try:
        uv = np.asarray(observation.uv, dtype=np.float64)
        covariance = np.asarray(observation.covariance_px2, dtype=np.float64)
        score = float(observation.score)
    except (TypeError, ValueError):
        return False
    return bool(
        uv.shape == (2,)
        and covariance.shape == (2, 2)
        and np.all(np.isfinite(uv))
        and np.all(np.isfinite(covariance))
        and math.isfinite(score)
    )


def _observation_covariance_is_valid(observation: JointObservation) -> bool:
    covariance = np.asarray(observation.covariance_px2, dtype=np.float64)
    return bool(
        np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
        and np.linalg.eigvalsh(covariance).min() > 0.0
    )


def _optimize_point(
    initial_point: np.ndarray,
    left_projection: np.ndarray,
    right_projection: np.ndarray,
    left: JointObservation,
    right: JointObservation,
    config: StereoFusionConfig,
) -> np.ndarray:
    point = initial_point.copy()
    observations = (
        (left_projection, left, np.linalg.inv(_effective_covariance(left))),
        (right_projection, right, np.linalg.inv(_effective_covariance(right))),
    )
    for _ in range(config.max_iterations):
        information = np.zeros((3, 3), dtype=np.float64)
        gradient = np.zeros(3, dtype=np.float64)
        for projection, observation, precision in observations:
            _, projected, jacobian = _project_with_jacobian(projection, point)
            residual = projected - np.asarray(observation.uv, dtype=np.float64)
            norm = float(np.sqrt(max(residual @ precision @ residual, 0.0)))
            robust_weight = 1.0 if norm <= config.huber_delta else config.huber_delta / norm
            weighted_precision = robust_weight * precision
            information += jacobian.T @ weighted_precision @ jacobian
            gradient += jacobian.T @ weighted_precision @ residual
        try:
            step = np.linalg.solve(information, -gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        point += step
        if float(np.linalg.norm(step)) <= 1e-10:
            break
    return point


def _covariance(
    left_jacobian: np.ndarray,
    right_jacobian: np.ndarray,
    left: JointObservation,
    right: JointObservation,
) -> np.ndarray:
    left_covariance = _effective_covariance(left)
    right_covariance = _effective_covariance(right)
    information = (
        left_jacobian.T @ np.linalg.inv(left_covariance) @ left_jacobian
        + right_jacobian.T @ np.linalg.inv(right_covariance) @ right_jacobian
    )
    covariance = np.linalg.inv(information)
    return (covariance + covariance.T) * 0.5


__all__ = [
    "COVARIANCE_STATUS",
    "FUSION_METHOD_ID",
    "HandFusionResult",
    "JointFusionResult",
    "JointObservation",
    "RobustStereoFusion",
    "StereoFusionConfig",
]
