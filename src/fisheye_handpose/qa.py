"""Empirical checks for stereo rectification and metric depth orientation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .errors import GeometryError
from .geometry import StereoRectifier
from .video import StereoFramePair

QaStatus = Literal["PASS", "FAIL", "INCONCLUSIVE"]


@dataclass(frozen=True, slots=True)
class EpipolarQaConfig:
    sample_pairs: int = 12
    max_features_per_image: int = 2500
    ratio_test: float = 0.65
    ransac_threshold_px: float = 0.5
    min_total_inliers: int = 60
    max_median_vertical_error_px: float = 0.75
    max_p95_vertical_error_px: float = 2.0
    min_expected_disparity_fraction: float = 0.80
    min_positive_depth_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.sample_pairs <= 0 or self.max_features_per_image < 32:
            raise GeometryError("epipolar QA sample/features settings must be positive")
        if not 0.0 < self.ratio_test < 1.0:
            raise GeometryError("epipolar QA ratio_test must be in (0, 1)")
        if self.ransac_threshold_px <= 0:
            raise GeometryError("epipolar QA RANSAC threshold must be positive")
        if self.min_total_inliers < 8:
            raise GeometryError("epipolar QA requires at least 8 total inliers")
        for label, value in (
            ("min_expected_disparity_fraction", self.min_expected_disparity_fraction),
            ("min_positive_depth_fraction", self.min_positive_depth_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise GeometryError(f"{label} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EpipolarQaReport:
    status: QaStatus
    sampled_pair_indices: tuple[int, ...]
    attempted_pair_count: int
    usable_pair_count: int
    inlier_count: int
    median_vertical_error_px: float | None
    p95_vertical_error_px: float | None
    expected_disparity_sign: int | None
    expected_disparity_fraction: float | None
    positive_depth_fraction: float | None
    failures: tuple[str, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sampled_pair_indices": list(self.sampled_pair_indices),
            "attempted_pair_count": self.attempted_pair_count,
            "usable_pair_count": self.usable_pair_count,
            "inlier_count": self.inlier_count,
            "median_vertical_error_px": self.median_vertical_error_px,
            "p95_vertical_error_px": self.p95_vertical_error_px,
            "expected_disparity_sign": self.expected_disparity_sign,
            "expected_disparity_fraction": self.expected_disparity_fraction,
            "positive_depth_fraction": self.positive_depth_fraction,
            "failures": list(self.failures),
            "reason": self.reason,
        }


def _sample_indices(pair_count: int, sample_count: int) -> tuple[int, ...]:
    if pair_count <= 0:
        return ()
    if sample_count >= pair_count:
        return tuple(range(pair_count))
    if sample_count == 1:
        return (pair_count // 2,)
    indices = {
        round(index * (pair_count - 1) / (sample_count - 1)) for index in range(sample_count)
    }
    return tuple(sorted(indices))


def summarize_rectified_correspondences(
    p1: Any,
    p2: Any,
    left_points: Any,
    right_points: Any,
    *,
    config: EpipolarQaConfig,
    sampled_pair_indices: tuple[int, ...] = (),
    attempted_pair_count: int = 0,
    usable_pair_count: int = 0,
) -> EpipolarQaReport:
    """Evaluate already-filtered rectified correspondences.

    This is separate from feature extraction so geometry thresholds can be tested with
    deterministic synthetic points.
    """

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise GeometryError("NumPy and OpenCV are required for epipolar QA") from exc

    left = np.asarray(left_points, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(right_points, dtype=np.float64).reshape(-1, 2)
    if left.shape != right.shape:
        raise GeometryError("left/right QA correspondences must have the same shape")
    finite = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    left = left[finite]
    right = right[finite]
    inlier_count = int(left.shape[0])
    if inlier_count < config.min_total_inliers:
        return EpipolarQaReport(
            status="INCONCLUSIVE",
            sampled_pair_indices=sampled_pair_indices,
            attempted_pair_count=attempted_pair_count,
            usable_pair_count=usable_pair_count,
            inlier_count=inlier_count,
            median_vertical_error_px=None,
            p95_vertical_error_px=None,
            expected_disparity_sign=None,
            expected_disparity_fraction=None,
            positive_depth_fraction=None,
            failures=(),
            reason=(
                f"only {inlier_count} filtered correspondences; "
                f"at least {config.min_total_inliers} are required"
            ),
        )

    vertical_error = np.abs(left[:, 1] - right[:, 1])
    median_vertical = float(np.median(vertical_error))
    p95_vertical = float(np.quantile(vertical_error, 0.95))

    projection_left = np.asarray(p1, dtype=np.float64)
    projection_right = np.asarray(p2, dtype=np.float64)
    if projection_left.shape != (3, 4) or projection_right.shape != (3, 4):
        raise GeometryError("P1/P2 must both be 3x4 projection matrices")
    disparity_scale = float(projection_left[0, 3] - projection_right[0, 3])
    expected_sign = 0 if abs(disparity_scale) < 1e-12 else (1 if disparity_scale > 0 else -1)
    if expected_sign == 0:
        expected_disparity_fraction = 0.0
    else:
        disparities = left[:, 0] - right[:, 0]
        expected_disparity_fraction = float(np.mean(disparities * expected_sign > 0))

    homogeneous = cv2.triangulatePoints(
        projection_left,
        projection_right,
        left.T,
        right.T,
    )
    valid_w = np.abs(homogeneous[3]) > 1e-12
    depths = np.full(inlier_count, np.nan, dtype=np.float64)
    depths[valid_w] = homogeneous[2, valid_w] / homogeneous[3, valid_w]
    positive_depth_fraction = float(np.mean(np.isfinite(depths) & (depths > 0)))

    failures: list[str] = []
    if median_vertical > config.max_median_vertical_error_px:
        failures.append(
            f"median |dy| {median_vertical:.3f}px exceeds "
            f"{config.max_median_vertical_error_px:.3f}px"
        )
    if p95_vertical > config.max_p95_vertical_error_px:
        failures.append(
            f"p95 |dy| {p95_vertical:.3f}px exceeds {config.max_p95_vertical_error_px:.3f}px"
        )
    if expected_disparity_fraction < config.min_expected_disparity_fraction:
        failures.append(
            "disparity sign agreement "
            f"{expected_disparity_fraction:.3f} is below "
            f"{config.min_expected_disparity_fraction:.3f}"
        )
    if positive_depth_fraction < config.min_positive_depth_fraction:
        failures.append(
            f"positive-depth fraction {positive_depth_fraction:.3f} is below "
            f"{config.min_positive_depth_fraction:.3f}"
        )
    return EpipolarQaReport(
        status="FAIL" if failures else "PASS",
        sampled_pair_indices=sampled_pair_indices,
        attempted_pair_count=attempted_pair_count,
        usable_pair_count=usable_pair_count,
        inlier_count=inlier_count,
        median_vertical_error_px=median_vertical,
        p95_vertical_error_px=p95_vertical,
        expected_disparity_sign=expected_sign,
        expected_disparity_fraction=expected_disparity_fraction,
        positive_depth_fraction=positive_depth_fraction,
        failures=tuple(failures),
    )


def evaluate_epipolar_qa(
    rectifier: StereoRectifier,
    pairs: Iterable[StereoFramePair],
    *,
    pair_count: int,
    config: EpipolarQaConfig,
) -> EpipolarQaReport:
    """Detect/match features on selected rectified stereo pairs and apply QA gates."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise GeometryError("NumPy and OpenCV are required for epipolar QA") from exc

    selected = _sample_indices(pair_count, config.sample_pairs)
    selected_set = set(selected)
    detector = cv2.ORB_create(nfeatures=config.max_features_per_image)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    all_left: list[Any] = []
    all_right: list[Any] = []
    usable = 0

    left_mask = rectifier.left_valid_mask.astype(np.uint8) * 255
    right_mask = rectifier.right_valid_mask.astype(np.uint8) * 255
    for pair in pairs:
        if pair.pair_index not in selected_set:
            if pair.pair_index > selected[-1]:
                break
            continue
        left_rectified, right_rectified = rectifier.apply(pair.left_bgr, pair.right_bgr)
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        left_keypoints, left_descriptors = detector.detectAndCompute(left_gray, left_mask)
        right_keypoints, right_descriptors = detector.detectAndCompute(right_gray, right_mask)
        if left_descriptors is None or right_descriptors is None:
            continue
        left_to_right = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        right_to_left = matcher.knnMatch(right_descriptors, left_descriptors, k=2)
        forward = [
            first
            for neighbors in left_to_right
            if len(neighbors) == 2
            for first, second in [neighbors]
            if first.distance < config.ratio_test * second.distance
        ]
        reverse = [
            first
            for neighbors in right_to_left
            if len(neighbors) == 2
            for first, second in [neighbors]
            if first.distance < config.ratio_test * second.distance
        ]
        reverse_lookup = {match.queryIdx: match.trainIdx for match in reverse}
        # Bidirectional ratio agreement removes repeated-texture descriptor matches
        # that otherwise inflate the epipolar tail despite a correct calibration.
        good = [match for match in forward if reverse_lookup.get(match.trainIdx) == match.queryIdx]
        if len(good) < 8:
            continue
        left_points = np.float32([left_keypoints[item.queryIdx].pt for item in good])
        right_points = np.float32([right_keypoints[item.trainIdx].pt for item in good])
        _, inlier_mask = cv2.findFundamentalMat(
            left_points,
            right_points,
            cv2.FM_RANSAC,
            config.ransac_threshold_px,
            0.999,
        )
        if inlier_mask is None:
            continue
        keep = inlier_mask.reshape(-1).astype(bool)
        if int(keep.sum()) < 8:
            continue
        all_left.append(left_points[keep])
        all_right.append(right_points[keep])
        usable += 1

    if all_left:
        left_combined = np.concatenate(all_left, axis=0)
        right_combined = np.concatenate(all_right, axis=0)
    else:
        left_combined = np.empty((0, 2), dtype=np.float32)
        right_combined = np.empty((0, 2), dtype=np.float32)
    return summarize_rectified_correspondences(
        rectifier.p1,
        rectifier.p2,
        left_combined,
        right_combined,
        config=config,
        sampled_pair_indices=selected,
        attempted_pair_count=len(selected),
        usable_pair_count=usable,
    )


__all__ = [
    "EpipolarQaConfig",
    "EpipolarQaReport",
    "evaluate_epipolar_qa",
    "summarize_rectified_correspondences",
]
