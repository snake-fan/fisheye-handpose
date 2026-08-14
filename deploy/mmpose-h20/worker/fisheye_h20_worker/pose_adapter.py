"""RTMPose adapter for hand-centred physical virtual-perspective crops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import RectifiedStereo
from .contracts import WorkerError
from .crop import PerspectiveCrop, VirtualPerspectiveCropper
from .scores import (
    MODEL_SCORE_SEMANTICS,
    QUALITY_WEIGHT_METHOD,
    QUALITY_WEIGHT_STATUS,
    quality_weight,
)


def _detection(value: Any, *, side: str, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerError("detector candidate must be an object")
    bbox = value.get("bbox_xyxy")
    score = value.get("bbox_score")
    label = value.get("label")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise WorkerError("detector bbox_xyxy must contain four values")
    if isinstance(label, bool) or not isinstance(label, int):
        raise WorkerError("detector label must be an integer")
    numeric = [*bbox, score]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in numeric
    ):
        raise WorkerError("detector candidate contains a non-finite numeric value")
    candidate_id = value.get("candidate_id", f"{side}-{index}")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise WorkerError("detector candidate_id must be a non-empty string")
    detection = {
        "candidate_id": candidate_id,
        "bbox_xyxy": [float(item) for item in bbox],
        "bbox_score": float(score),
        "label": label,
    }
    for field in (
        "source_index",
        "classification",
        "reason",
        "eligible_for_association",
        "final_selection",
    ):
        if field in value:
            detection[field] = value[field]
    return detection


def _pose(value: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, dict):
        raise WorkerError("pose inference result must be an object")
    points = np.asarray(value.get("keypoints_uv"), dtype=np.float64)
    scores = np.asarray(value.get("keypoint_scores"), dtype=np.float64)
    if points.shape != (21, 2) or scores.shape != (21,):
        raise WorkerError("pose inference must return exactly 21 keypoints and 21 keypoint scores")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(scores)):
        raise WorkerError("pose inference contains non-finite evidence")
    return points, scores


@dataclass(frozen=True)
class CropPoseResult:
    """One detector candidate and its produced or rejected crop evidence."""

    candidate_id: str
    detection: dict[str, Any]
    status: str
    reason: str | None
    crop: PerspectiveCrop | None
    instance: dict[str, Any] | None

    def trace_payload(self) -> dict[str, Any]:
        crop_payload: dict[str, Any] | None = None
        if self.crop is not None:
            crop_payload = {
                "virtual_camera_id": self.crop.crop_id,
                "crop_policy_id": self.crop.policy_id,
                "side": self.crop.side,
                "source_bbox_xyxy": list(self.crop.bbox),
                "output_size": [
                    int(self.crop.image.shape[1]),
                    int(self.crop.image.shape[0]),
                ],
                "K_virtual": self.crop.K_virtual.astype(float).tolist(),
                "R_source_from_virtual": (self.crop.R_source_from_virtual.astype(float).tolist()),
                "T_rig_from_virtual": self.crop.T_rig_from_virtual.astype(float).tolist(),
                "valid_fraction": self.crop.valid_fraction,
            }
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "detection": self.detection,
            "output_status": self.status,
            "reason": self.reason,
            "model_input_space": "virtual_pinhole",
            "virtual_camera": crop_payload,
        }
        if self.instance is not None:
            payload.update(
                {
                    "keypoints_uv_crop": self.instance["keypoints_uv_crop"],
                    "keypoints_uv_native": self.instance["keypoints_uv"],
                    "keypoint_scores": self.instance["keypoint_scores"],
                    "model_keypoint_scores": self.instance["model_keypoint_scores"],
                    "keypoint_score_semantics": self.instance["keypoint_score_semantics"],
                    "keypoint_quality_weight_method": self.instance[
                        "keypoint_quality_weight_method"
                    ],
                    "keypoint_quality_weight_status": self.instance[
                        "keypoint_quality_weight_status"
                    ],
                    "keypoint_valid_mask": self.instance["keypoint_valid_mask"],
                }
            )
        return payload


@dataclass(frozen=True)
class CropPoseBatch:
    """All candidate outcomes for one physical source view."""

    results: tuple[CropPoseResult, ...]

    @property
    def produced_instances(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            result.instance
            for result in self.results
            if result.status == "PRODUCED" and result.instance is not None
        )


class VirtualCropPoseAdapter:
    """Run pose on virtual crops and return evidence in native distorted pixels."""

    def __init__(
        self,
        *,
        cropper: VirtualPerspectiveCropper,
        min_valid_fraction: float,
    ) -> None:
        if (
            isinstance(min_valid_fraction, bool)
            or not isinstance(min_valid_fraction, (int, float))
            or not math.isfinite(float(min_valid_fraction))
            or not 0.0 <= float(min_valid_fraction) <= 1.0
        ):
            raise ValueError("min_valid_fraction must be finite and in [0, 1]")
        self.cropper = cropper
        self.min_valid_fraction = float(min_valid_fraction)

    def infer(
        self,
        *,
        runtime: Any,
        models: Any,
        frame: Any,
        side: str,
        detections: Any,
        rectification: RectifiedStereo,
    ) -> CropPoseBatch:
        if not isinstance(detections, list):
            raise WorkerError("runtime detection must return a list")
        outcomes: list[CropPoseResult] = []
        for index, value in enumerate(detections):
            detection = _detection(value, side=side, index=index)
            try:
                crop = self.cropper.create(
                    frame,
                    side,
                    detection["bbox_xyxy"],
                    rectification,
                )
            except ValueError as exc:
                raise WorkerError(f"cannot construct virtual crop: {exc}") from exc
            if crop.valid_fraction < self.min_valid_fraction:
                outcomes.append(
                    CropPoseResult(
                        candidate_id=detection["candidate_id"],
                        detection=detection,
                        status="NOT_PRODUCED",
                        reason="CROP_VALID_FRACTION_BELOW_THRESHOLD",
                        crop=crop,
                        instance=None,
                    )
                )
                continue

            height, width = crop.image.shape[:2]
            pose_results = runtime.infer_pose(
                models,
                crop.image,
                bboxes=[[0.0, 0.0, float(width - 1), float(height - 1)]],
            )
            if not isinstance(pose_results, list) or len(pose_results) != 1:
                raise WorkerError("one virtual crop must produce exactly one pose instance")
            crop_points, model_scores = _pose(pose_results[0])
            inside = (
                (crop_points[:, 0] >= 0.0)
                & (crop_points[:, 0] <= width - 1.0)
                & (crop_points[:, 1] >= 0.0)
                & (crop_points[:, 1] <= height - 1.0)
            )
            sampled_valid = np.zeros(21, dtype=bool)
            if np.any(inside):
                rounded = np.rint(crop_points[inside]).astype(int)
                sampled_valid[inside] = crop.valid_mask[rounded[:, 1], rounded[:, 0]]
            source_points = crop.crop_uv_to_source_uv(crop_points)
            if source_points.shape != (21, 2) or not np.all(np.isfinite(source_points)):
                raise WorkerError("virtual crop mapping produced non-finite native keypoints")
            quality_scores = np.asarray(
                [quality_weight(value) for value in model_scores],
                dtype=np.float64,
            )
            effective_scores = np.where(sampled_valid, quality_scores, 0.0)
            instance = {
                **detection,
                "keypoints_uv": source_points.astype(float).tolist(),
                "keypoint_scores": effective_scores.astype(float).tolist(),
                "keypoints_uv_crop": crop_points.astype(float).tolist(),
                "model_keypoint_scores": model_scores.astype(float).tolist(),
                "keypoint_score_semantics": MODEL_SCORE_SEMANTICS,
                "keypoint_quality_weight_method": QUALITY_WEIGHT_METHOD,
                "keypoint_quality_weight_status": QUALITY_WEIGHT_STATUS,
                "keypoint_valid_mask": sampled_valid.tolist(),
                "model_input_space": "virtual_pinhole",
                "crop_policy_id": crop.policy_id,
                "virtual_camera_id": crop.crop_id,
            }
            outcomes.append(
                CropPoseResult(
                    candidate_id=detection["candidate_id"],
                    detection=detection,
                    status="PRODUCED",
                    reason=None,
                    crop=crop,
                    instance=instance,
                )
            )
        return CropPoseBatch(tuple(outcomes))


__all__ = ["CropPoseBatch", "CropPoseResult", "VirtualCropPoseAdapter"]
