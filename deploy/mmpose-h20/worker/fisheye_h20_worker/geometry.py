"""Deterministic epipolar association and metric rectified stereo triangulation."""

from __future__ import annotations

import math
import statistics
from itertools import combinations
from typing import Any

from ._generated_project_contract import FHP21_SCHEMA_ID
from .calibration import RectifiedStereo
from .contracts import ThresholdRequest, WorkerError
from .fusion import (
    FUSION_METHOD_ID,
    JointObservation,
    RobustStereoFusion,
    StereoFusionConfig,
)
from .scores import (
    MODEL_SCORE_SEMANTICS,
    QUALITY_WEIGHT_METHOD,
    QUALITY_WEIGHT_STATUS,
    quality_weight,
)


def normalize_instances(
    instances: Any,
    *,
    side: str,
    rectification: RectifiedStereo,
) -> list[dict[str, Any]]:
    if not isinstance(instances, list):
        raise WorkerError("runtime inference must return a list")
    normalized: list[dict[str, Any]] = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise WorkerError("runtime pose instance must be an object")
        bbox = instance.get("bbox_xyxy")
        points = instance.get("keypoints_uv")
        scores = instance.get("keypoint_scores")
        model_scores = instance.get("model_keypoint_scores", scores)
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise WorkerError("runtime bbox_xyxy must contain four values")
        if not isinstance(points, list) or len(points) != 21:
            raise WorkerError("runtime keypoints_uv must contain 21 points")
        if not isinstance(scores, list) or len(scores) != 21:
            raise WorkerError("runtime keypoint_scores must contain 21 values")
        if not isinstance(model_scores, list) or len(model_scores) != 21:
            raise WorkerError("runtime model_keypoint_scores must contain 21 values")
        if any(not isinstance(point, list) or len(point) != 2 for point in points):
            raise WorkerError("each runtime keypoint must contain u and v")
        bbox_score = instance.get("bbox_score")
        label = instance.get("label")
        if isinstance(label, bool) or not isinstance(label, int):
            raise WorkerError("runtime detection label must be an integer")
        numeric = [
            *bbox,
            *scores,
            *model_scores,
            bbox_score,
            *(value for point in points for value in point),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise WorkerError("runtime inference contains a non-finite numeric value")
        candidate_id = instance.get("candidate_id", f"{side}-{index}")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise WorkerError("runtime candidate_id must be a non-empty string")
        normalized_instance = {
            "candidate_id": candidate_id,
            "bbox_xyxy": [float(value) for value in bbox],
            "bbox_score": float(bbox_score),
            "label": label,
            "keypoints_uv": [[float(u), float(v)] for u, v in points],
            "keypoints_uv_rectified": rectification.rectify_points(side, points),
            "keypoint_scores": [quality_weight(value) for value in scores],
            "model_keypoint_scores": [float(value) for value in model_scores],
            "keypoint_score_semantics": MODEL_SCORE_SEMANTICS,
            "keypoint_quality_weight_method": QUALITY_WEIGHT_METHOD,
            "keypoint_quality_weight_status": QUALITY_WEIGHT_STATUS,
        }
        for optional in (
            "source_index",
            "classification",
            "reason",
            "eligible_for_association",
            "final_selection",
            "keypoints_uv_crop",
            "keypoint_valid_mask",
            "model_input_space",
            "crop_policy_id",
            "virtual_camera_id",
        ):
            if optional in instance:
                normalized_instance[optional] = instance[optional]
        normalized.append(normalized_instance)
    return normalized


def _association_cost(
    left: dict[str, Any],
    right: dict[str, Any],
    keypoint_threshold: float,
) -> tuple[float, int]:
    residuals = [
        abs(left_point[1] - right_point[1])
        for left_point, right_point, left_score, right_score in zip(
            left["keypoints_uv_rectified"],
            right["keypoints_uv_rectified"],
            left["keypoint_scores"],
            right["keypoint_scores"],
            strict=True,
        )
        if left_score >= keypoint_threshold and right_score >= keypoint_threshold
    ]
    return (float(statistics.median(residuals)), len(residuals)) if residuals else (math.inf, 0)


def associate(
    left_instances: list[dict[str, Any]],
    right_instances: list[dict[str, Any]],
    thresholds: ThresholdRequest,
) -> dict[str, Any]:
    if len(left_instances) > 4 or len(right_instances) > 4:
        raise WorkerError("association supports at most four candidates per view")
    candidates: list[tuple[float, int, int, int]] = []
    for left_index, left in enumerate(left_instances):
        for right_index, right in enumerate(right_instances):
            cost, support = _association_cost(left, right, thresholds.keypoint_score)
            if support and cost <= thresholds.association_epipolar_px:
                candidates.append((cost, -support, left_index, right_index))
    candidates.sort(key=lambda value: (value[2], value[3], value[0], value[1]))
    assignments: list[tuple[tuple[float, int, int, int], ...]] = [()]
    for size in range(1, min(2, len(left_instances), len(right_instances)) + 1):
        for subset in combinations(candidates, size):
            left_indices = {candidate[2] for candidate in subset}
            right_indices = {candidate[3] for candidate in subset}
            if len(left_indices) == size and len(right_indices) == size:
                assignments.append(subset)
    selected = min(
        assignments,
        key=lambda subset: (
            -len(subset),
            sum(candidate[0] for candidate in subset),
            tuple((candidate[2], candidate[3]) for candidate in subset),
        ),
    )
    selected = tuple(sorted(selected, key=lambda value: (value[2], value[3])))
    used_left = {candidate[2] for candidate in selected}
    used_right = {candidate[3] for candidate in selected}
    matches: list[dict[str, Any]] = []
    for cost, negative_support, left_index, right_index in selected:
        matches.append(
            {
                "match_id": f"match-{len(matches)}",
                "left_index": left_index,
                "right_index": right_index,
                "left_candidate_id": left_instances[left_index]["candidate_id"],
                "right_candidate_id": right_instances[right_index]["candidate_id"],
                "median_epipolar_error_px": cost,
                "supporting_keypoint_count": -negative_support,
            }
        )
    return {
        "matches": matches,
        "unmatched_left_indices": [
            index for index in range(len(left_instances)) if index not in used_left
        ],
        "unmatched_right_indices": [
            index for index in range(len(right_instances)) if index not in used_right
        ],
    }


def _camera_center_and_inverse(projection: Any) -> tuple[Any, Any]:
    import numpy as np

    matrix = projection[:, :3]
    inverse = np.linalg.inv(matrix)
    center = -inverse @ projection[:, 3]
    return center, inverse


def triangulate_match(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    rectification: RectifiedStereo,
    thresholds: ThresholdRequest,
) -> dict[str, Any]:
    def observations(instance: dict[str, Any]) -> list[JointObservation]:
        masks = instance.get("keypoint_valid_mask", [True] * 21)
        covariances = instance.get(
            "keypoint_covariance_px2",
            [[[1.0, 0.0], [0.0, 1.0]] for _ in range(21)],
        )
        if not isinstance(masks, list) or len(masks) != 21:
            raise WorkerError("keypoint_valid_mask must contain 21 booleans")
        if any(not isinstance(value, bool) for value in masks):
            raise WorkerError("keypoint_valid_mask must contain booleans")
        if not isinstance(covariances, list) or len(covariances) != 21:
            raise WorkerError("keypoint_covariance_px2 must contain 21 matrices")
        return [
            JointObservation(
                uv=(float(uv[0]), float(uv[1])),
                score=float(score),
                covariance_px2=(
                    (float(covariance[0][0]), float(covariance[0][1])),
                    (float(covariance[1][0]), float(covariance[1][1])),
                ),
                valid_mask=mask,
            )
            for uv, score, covariance, mask in zip(
                instance["keypoints_uv_rectified"],
                instance["keypoint_scores"],
                covariances,
                masks,
                strict=True,
            )
        ]

    fusion = RobustStereoFusion(
        rectification.p1,
        rectification.p2,
        config=StereoFusionConfig(
            min_keypoint_score=thresholds.keypoint_score,
            max_epipolar_error_px=thresholds.association_epipolar_px,
            max_reprojection_error_px=thresholds.max_reprojection_error_px,
            min_ray_angle_deg=thresholds.min_ray_angle_deg,
            min_depth_m=thresholds.min_depth_m,
            max_depth_m=thresholds.max_depth_m,
        ),
    )
    hand = fusion.fuse_hand(observations(left), observations(right))
    landmarks = [
        None if joint.point_xyz_m is None else list(joint.point_xyz_m) for joint in hand.joints
    ]
    validity = ["VALID" if joint.valid else (joint.reason or "INVALID") for joint in hand.joints]
    covariance_m2 = [
        None if joint.covariance_m2 is None else [list(row) for row in joint.covariance_m2]
        for joint in hand.joints
    ]
    covariance_status = [
        joint.covariance_status if joint.covariance_m2 is not None else "NOT_ESTIMATED"
        for joint in hand.joints
    ]
    metrics = [
        {
            "joint_index": joint_index,
            "epipolar_error_px": joint.epipolar_error_px,
            "left_score": float(left["keypoint_scores"][joint_index]),
            "right_score": float(right["keypoint_scores"][joint_index]),
            "left_reprojection_error_px": joint.left_reprojection_error_px,
            "right_reprojection_error_px": joint.right_reprojection_error_px,
            "ray_angle_deg": joint.ray_angle_deg,
            "left_depth_m": joint.left_depth_m,
            "right_depth_m": joint.right_depth_m,
            "support_view_count": joint.support_view_count,
            "covariance_status": joint.covariance_status,
        }
        for joint_index, joint in enumerate(hand.joints)
    ]
    return {
        "fusion_method": FUSION_METHOD_ID,
        "coordinate_frame": "rectified_left_camera",
        "length_unit": "m",
        "landmark_schema": FHP21_SCHEMA_ID,
        "landmarks_xyz_m": landmarks,
        "validity": validity,
        "covariance_m2": covariance_m2,
        "covariance_status": covariance_status,
        "metrics": metrics,
        "valid_landmark_count": sum(value == "VALID" for value in validity),
        "hand_validity": hand.validity,
        "hand_reason": hand.reason,
        "palm_support_count": hand.palm_support_count,
        "minimum_palm_support": hand.minimum_palm_support,
    }


__all__ = ["associate", "normalize_instances", "triangulate_match"]
