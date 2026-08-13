"""Build and validate the operational ``fhp21/v1`` JSONL producer contract.

This module intentionally uses only the Python standard library so the Python 3.11
core can import the worker bridge without importing CUDA, OpenCV, NumPy, or Torch.
"""

from __future__ import annotations

import math
from typing import Any

from .artifacts import FHP21_OUTPUT_SCHEMA
from .contracts import WorkerError

RTMPOSE_FHP21_MAPPING_ID = "rtmpose-hand5-native21-to-fhp21/v1"
OUTPUT_AXIS_CONVENTION = "OPENCV_X_RIGHT_Y_DOWN_Z_FORWARD"


def build_pose_estimate(
    *,
    sequence_id: str,
    estimate_id: str,
    frame_id: str,
    frame_index: int,
    timestamp_ns: int,
    track_id: str,
    source_observation_id: str,
    calibration_id: str,
    raw: dict[str, Any],
    mano: dict[str, Any] | None,
    temporal: dict[str, Any],
    keypoint_score_threshold: float,
    backend_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Create one complete pose estimate without inventing uncertainty.

    The current triangulator and temporal baseline do not estimate metric covariance,
    calibrated visibility, or calibrated confidence. Those dimensions are therefore
    present but explicitly null, paired with ``covariance_status=NOT_ESTIMATED``.
    """

    points = temporal["landmarks_xyz_m"]
    temporal_validity = temporal["validity"]
    metrics = raw["metrics"]
    validity: list[str] = []
    invalid_reason: list[str | None] = []
    evidence_source: list[str] = []
    support_view_ids: list[list[str]] = []
    residuals: list[dict[str, float | None]] = []
    kind: list[str] = []
    for index, (point, flag, metric, temporal_refined) in enumerate(
        zip(
            points,
            temporal_validity,
            metrics,
            temporal["refinement_applied"],
            strict=True,
        )
    ):
        valid = flag == "VALID" and point is not None
        validity.append("VALID" if valid else "INVALID")
        invalid_reason.append(None if valid else str(flag))
        support = [
            side
            for side in ("left", "right")
            if float(metric[f"{side}_score"]) >= keypoint_score_threshold
        ]
        support_view_ids.append(support)
        evidence_source.append(
            "MULTIVIEW" if len(support) == 2 else "MONOCULAR" if support else "NONE"
        )
        residuals.append(
            {
                "left": metric["left_reprojection_error_px"],
                "right": metric["right_reprojection_error_px"],
            }
        )
        mano_refined = (
            mano is not None
            and mano["validity"][index] == "VALID"
            and mano["landmarks_xyz_m"][index] is not None
        )
        kind.append("REFINED" if mano_refined or temporal_refined else "MEASURED")

    handedness = {"left": 0.0, "right": 0.0, "unknown": 1.0}
    mapping_ids = [RTMPOSE_FHP21_MAPPING_ID]
    if mano is not None:
        side = mano["side"]
        handedness = {
            "left": 1.0 if side == "left" else 0.0,
            "right": 1.0 if side == "right" else 0.0,
            "unknown": 0.0,
        }
        mapping_ids.append(mano["mapping_id"])

    return {
        "record_type": "PoseEstimate",
        "sequence_id": sequence_id,
        "estimate_id": estimate_id,
        "frame_id": frame_id,
        "frame_index": frame_index,
        "timestamp_ns": timestamp_ns,
        "track_id": track_id,
        "source_observation_ids": [source_observation_id],
        "calibration_id": calibration_id,
        "output_status": "PRODUCED",
        "output_frame": {
            "frame_id": "rectified_left_camera",
            "kind": "CAMERA",
            "axis_convention": OUTPUT_AXIS_CONVENTION,
            "length_unit": "m",
        },
        # Compatibility aliases consumed by the current trace inspector.
        "coordinate_frame": "rectified_left_camera",
        "length_unit": "m",
        "landmark_schema": "fhp21/v1",
        "handedness_probabilities": handedness,
        "stage": "TEMPORAL_REFINEMENT",
        "selected_output_stage": "TEMPORAL_REFINEMENT",
        "kind": kind,
        "landmarks_xyz_m": points,
        "covariance_m2": [None] * 21,
        "covariance_status": ["NOT_ESTIMATED"] * 21,
        "validity": validity,
        "invalid_reason": invalid_reason,
        "evidence_source": evidence_source,
        "visibility_probability": [None] * 21,
        "visibility_status": ["NOT_ESTIMATED"] * 21,
        "confidence_probability": [None] * 21,
        "confidence_status": "NOT_CALIBRATED",
        "confidence_radius_m": None,
        "support_view_ids": support_view_ids,
        "reprojection_residuals_px": residuals,
        "mapping_ids": mapping_ids,
        "backend_provenance": backend_provenance,
        "raw": raw,
        "mano": mano,
        "temporal": temporal,
    }


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _finite_number(value: Any, label: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise WorkerError(f"{label} must be a finite JSON number")
    return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerError(f"{label} must be a non-empty string")
    return value


def _array21(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != 21:
        raise WorkerError(f"{label} must contain exactly 21 entries")
    return value


def _finite_tree(value: Any, label: str = "fhp21 record") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkerError(f"{label} contains a non-finite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")


def _validate_raw_observation(raw: Any, label: str) -> None:
    if not isinstance(raw, dict):
        raise WorkerError(f"{label} raw source observation is invalid")
    required = {
        "coordinate_frame",
        "length_unit",
        "landmark_schema",
        "landmarks_xyz_m",
        "validity",
        "metrics",
        "valid_landmark_count",
    }
    if set(raw) != required:
        raise WorkerError(f"{label} raw source observation fields are invalid")
    if (
        raw["coordinate_frame"] != "rectified_left_camera"
        or raw["length_unit"] != "m"
        or raw["landmark_schema"] != "fhp21/v1"
    ):
        raise WorkerError(f"{label} raw source observation frame/schema is invalid")
    points = _array21(raw["landmarks_xyz_m"], f"{label}.raw.landmarks_xyz_m")
    validity = _array21(raw["validity"], f"{label}.raw.validity")
    metrics = _array21(raw["metrics"], f"{label}.raw.metrics")
    valid_count = 0
    metric_fields = {
        "joint_index",
        "epipolar_error_px",
        "left_score",
        "right_score",
        "left_reprojection_error_px",
        "right_reprojection_error_px",
        "ray_angle_deg",
    }
    for index, (point, flag, metric) in enumerate(zip(points, validity, metrics, strict=True)):
        item_label = f"{label}.raw.landmark[{index}]"
        if flag == "VALID":
            if not isinstance(point, list) or len(point) != 3:
                raise WorkerError(f"{item_label} valid coordinate must contain xyz")
            for number in point:
                _finite_number(number, f"{item_label}.xyz")
            valid_count += 1
        elif point is not None or not isinstance(flag, str) or not flag:
            raise WorkerError(f"{item_label} invalid coordinate/reason is malformed")
        if not isinstance(metric, dict) or set(metric) != metric_fields:
            raise WorkerError(f"{item_label} metric fields are invalid")
        if metric["joint_index"] != index:
            raise WorkerError(f"{item_label} metric joint_index is invalid")
        for field in ("epipolar_error_px", "left_score", "right_score"):
            if _finite_number(metric[field], f"{item_label}.{field}") < 0.0:
                raise WorkerError(f"{item_label}.{field} cannot be negative")
        if metric["left_score"] > 1.0 or metric["right_score"] > 1.0:
            raise WorkerError(f"{item_label} keypoint scores cannot exceed one")
        for field in (
            "left_reprojection_error_px",
            "right_reprojection_error_px",
            "ray_angle_deg",
        ):
            number = metric[field]
            if number is not None and _finite_number(number, f"{item_label}.{field}") < 0.0:
                raise WorkerError(f"{item_label}.{field} cannot be negative")
    _integer(raw["valid_landmark_count"], f"{label}.raw.valid_landmark_count")
    if raw["valid_landmark_count"] != valid_count:
        raise WorkerError(f"{label}.raw.valid_landmark_count is inconsistent")


def _validate_temporal_payload(temporal: Any, label: str) -> None:
    if not isinstance(temporal, dict):
        raise WorkerError(f"{label} temporal refinement payload is invalid")
    required = {
        "method",
        "timestamp_ns",
        "landmarks_xyz_m",
        "validity",
        "valid_landmark_count",
        "reset_reason",
        "alpha",
        "refinement_applied",
    }
    if set(temporal) != required:
        raise WorkerError(f"{label} temporal refinement payload fields are invalid")
    _text(temporal["method"], f"{label}.temporal.method")
    _integer(temporal["timestamp_ns"], f"{label}.temporal.timestamp_ns")
    points = _array21(temporal["landmarks_xyz_m"], f"{label}.temporal.landmarks_xyz_m")
    validity = _array21(temporal["validity"], f"{label}.temporal.validity")
    refinement_applied = _array21(
        temporal["refinement_applied"],
        f"{label}.temporal.refinement_applied",
    )
    valid_count = 0
    for index, (point, flag) in enumerate(zip(points, validity, strict=True)):
        item_label = f"{label}.temporal.landmark[{index}]"
        if flag == "VALID":
            if not isinstance(point, list) or len(point) != 3:
                raise WorkerError(f"{item_label} valid coordinate must contain xyz")
            for number in point:
                _finite_number(number, f"{item_label}.xyz")
            valid_count += 1
        elif point is not None or not isinstance(flag, str) or not flag:
            raise WorkerError(f"{item_label} invalid coordinate/reason is malformed")
    _integer(temporal["valid_landmark_count"], f"{label}.temporal.valid_landmark_count")
    if temporal["valid_landmark_count"] != valid_count:
        raise WorkerError(f"{label}.temporal.valid_landmark_count is inconsistent")
    if any(not isinstance(item, bool) for item in refinement_applied):
        raise WorkerError(f"{label}.temporal.refinement_applied must contain booleans")
    alpha = temporal["alpha"]
    if alpha is not None:
        alpha_value = _finite_number(alpha, f"{label}.temporal.alpha")
        if not 0.0 <= alpha_value <= 1.0:
            raise WorkerError(f"{label}.temporal.alpha is outside [0, 1]")
    reset_reason = temporal["reset_reason"]
    if reset_reason is not None and (not isinstance(reset_reason, str) or not reset_reason):
        raise WorkerError(f"{label}.temporal.reset_reason is invalid")
    if reset_reason is not None and any(refinement_applied):
        raise WorkerError(f"{label}.temporal reset cannot claim EMA refinement")


def _validate_mano_payload(mano: Any, label: str) -> None:
    if not isinstance(mano, dict):
        raise WorkerError(f"{label} MANO refinement payload is invalid")
    required = {
        "side",
        "handedness",
        "mapping_id",
        "pose",
        "global_orient",
        "transl",
        "beta",
        "rmse_m",
        "loss",
        "landmarks_xyz_m",
        "validity",
    }
    if set(mano) != required:
        raise WorkerError(f"{label} MANO refinement payload fields are invalid")
    if mano["side"] not in {"left", "right"} or mano["handedness"] != mano["side"]:
        raise WorkerError(f"{label} MANO handedness is invalid")
    for field in ("mapping_id",):
        _text(mano[field], f"{label}.mano.{field}")
    for field, length in (("pose", 45), ("global_orient", 3), ("transl", 3), ("beta", 10)):
        values = mano[field]
        if not isinstance(values, list) or len(values) != length:
            raise WorkerError(f"{label}.mano.{field} must contain {length} values")
        for value in values:
            _finite_number(value, f"{label}.mano.{field}")
    rmse = _finite_number(mano["rmse_m"], f"{label}.mano.rmse_m")
    if rmse < 0.0:
        raise WorkerError(f"{label}.mano.rmse_m cannot be negative")
    loss = mano["loss"]
    if not isinstance(loss, dict) or set(loss) != {"metric", "value"}:
        raise WorkerError(f"{label}.mano.loss fields are invalid")
    if (
        loss["metric"] != "RMSE_M"
        or _finite_number(loss["value"], f"{label}.mano.loss.value") != rmse
    ):
        raise WorkerError(f"{label}.mano.loss is inconsistent")
    points = _array21(mano["landmarks_xyz_m"], f"{label}.mano.landmarks_xyz_m")
    validity = _array21(mano["validity"], f"{label}.mano.validity")
    for index, (point, flag) in enumerate(zip(points, validity, strict=True)):
        if flag == "VALID":
            if not isinstance(point, list) or len(point) != 3:
                raise WorkerError(f"{label}.mano.landmark[{index}] valid point must contain xyz")
            for number in point:
                _finite_number(number, f"{label}.mano.landmark[{index}].xyz")
        elif point is not None or not isinstance(flag, str) or not flag:
            raise WorkerError(f"{label}.mano.landmark[{index}] is malformed")


def validate_pose_estimate(value: Any, *, line_number: int) -> dict[str, Any]:
    """Fail closed unless one decoded JSON line satisfies the complete v1 contract."""

    label = f"fhp21 line {line_number}"
    if not isinstance(value, dict):
        raise WorkerError(f"{label} must be an object")
    _finite_tree(value, label)
    required = {
        "schema_version",
        "record_type",
        "sequence_id",
        "estimate_id",
        "frame_id",
        "frame_index",
        "timestamp_ns",
        "track_id",
        "source_observation_ids",
        "calibration_id",
        "output_status",
        "output_frame",
        "coordinate_frame",
        "length_unit",
        "landmark_schema",
        "handedness_probabilities",
        "stage",
        "selected_output_stage",
        "kind",
        "landmarks_xyz_m",
        "covariance_m2",
        "covariance_status",
        "validity",
        "invalid_reason",
        "evidence_source",
        "visibility_probability",
        "visibility_status",
        "confidence_probability",
        "confidence_status",
        "confidence_radius_m",
        "support_view_ids",
        "reprojection_residuals_px",
        "mapping_ids",
        "backend_provenance",
        "raw",
        "mano",
        "temporal",
    }
    missing = sorted(required - set(value))
    if missing:
        raise WorkerError(f"{label} is missing required fields: {missing}")
    unknown = sorted(set(value) - required)
    if unknown:
        raise WorkerError(f"{label} contains unknown fields: {unknown}")
    if value["schema_version"] != FHP21_OUTPUT_SCHEMA:
        raise WorkerError(f"{label} has an unexpected schema_version")
    if value["record_type"] != "PoseEstimate" or value["output_status"] != "PRODUCED":
        raise WorkerError(f"{label} is not a produced PoseEstimate")
    for field in (
        "sequence_id",
        "estimate_id",
        "frame_id",
        "track_id",
        "calibration_id",
    ):
        _text(value[field], f"{label}.{field}")
    _integer(value["frame_index"], f"{label}.frame_index")
    _integer(value["timestamp_ns"], f"{label}.timestamp_ns")
    sources = value["source_observation_ids"]
    if (
        not isinstance(sources, list)
        or not sources
        or len(sources) != len(set(sources))
        or any(not isinstance(item, str) or not item for item in sources)
    ):
        raise WorkerError(f"{label}.source_observation_ids is invalid")

    frame = value["output_frame"]
    if not isinstance(frame, dict) or set(frame) != {
        "frame_id",
        "kind",
        "axis_convention",
        "length_unit",
    }:
        raise WorkerError(f"{label}.output_frame fields are invalid")
    if (
        frame["kind"] not in {"CAMERA", "RIG"}
        or frame["length_unit"] != "m"
        or value["length_unit"] != "m"
        or value["coordinate_frame"] != frame["frame_id"]
    ):
        raise WorkerError(f"{label}.output_frame semantics are invalid")
    _text(frame["frame_id"], f"{label}.output_frame.frame_id")
    _text(frame["axis_convention"], f"{label}.output_frame.axis_convention")
    if value["landmark_schema"] != "fhp21/v1":
        raise WorkerError(f"{label}.landmark_schema is invalid")

    handedness = value["handedness_probabilities"]
    if not isinstance(handedness, dict) or set(handedness) != {"left", "right", "unknown"}:
        raise WorkerError(f"{label}.handedness_probabilities fields are invalid")
    probabilities = [
        _finite_number(handedness[key], f"{label}.handedness_probabilities.{key}")
        for key in ("left", "right", "unknown")
    ]
    outside_probability_range = any(
        probability < 0.0 or probability > 1.0 for probability in probabilities
    )
    if outside_probability_range or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise WorkerError(f"{label}.handedness_probabilities must sum to one")
    if value["stage"] != "TEMPORAL_REFINEMENT" or value["selected_output_stage"] != value["stage"]:
        raise WorkerError(f"{label}.stage is invalid")

    points = _array21(value["landmarks_xyz_m"], f"{label}.landmarks_xyz_m")
    validity = _array21(value["validity"], f"{label}.validity")
    reasons = _array21(value["invalid_reason"], f"{label}.invalid_reason")
    kinds = _array21(value["kind"], f"{label}.kind")
    evidence = _array21(value["evidence_source"], f"{label}.evidence_source")
    covariance = _array21(value["covariance_m2"], f"{label}.covariance_m2")
    covariance_status = _array21(value["covariance_status"], f"{label}.covariance_status")
    visibility = _array21(value["visibility_probability"], f"{label}.visibility_probability")
    visibility_status = _array21(value["visibility_status"], f"{label}.visibility_status")
    confidence = _array21(value["confidence_probability"], f"{label}.confidence_probability")
    supports = _array21(value["support_view_ids"], f"{label}.support_view_ids")
    residuals = _array21(value["reprojection_residuals_px"], f"{label}.reprojection_residuals_px")
    for index in range(21):
        item_label = f"{label}.landmark[{index}]"
        if validity[index] not in {"VALID", "INVALID"}:
            raise WorkerError(f"{item_label}.validity is invalid")
        point = points[index]
        if validity[index] == "VALID":
            if not isinstance(point, list) or len(point) != 3:
                raise WorkerError(f"{item_label} valid coordinate must contain xyz")
            for axis, number in enumerate(point):
                _finite_number(number, f"{item_label}.xyz[{axis}]")
            if reasons[index] is not None:
                raise WorkerError(f"{item_label} valid coordinate cannot have invalid_reason")
        elif point is not None or not isinstance(reasons[index], str) or not reasons[index]:
            raise WorkerError(f"{item_label} invalid coordinate must be null with a reason")
        if kinds[index] not in {"MEASURED", "REFINED", "PREDICTED"}:
            raise WorkerError(f"{item_label}.kind is invalid")
        if evidence[index] not in {"MULTIVIEW", "MONOCULAR", "NONE"}:
            raise WorkerError(f"{item_label}.evidence_source is invalid")

        matrix = covariance[index]
        status = covariance_status[index]
        if matrix is None:
            if status != "NOT_ESTIMATED":
                raise WorkerError(f"{item_label} null covariance must be NOT_ESTIMATED")
        else:
            if status != "ESTIMATED" or not isinstance(matrix, list) or len(matrix) != 3:
                raise WorkerError(f"{item_label}.covariance_m2 is invalid")
            for row in matrix:
                if not isinstance(row, list) or len(row) != 3:
                    raise WorkerError(f"{item_label}.covariance_m2 must be 3x3")
                for number in row:
                    _finite_number(number, f"{item_label}.covariance_m2")
        for field_name, probability in (
            ("visibility_probability", visibility[index]),
            ("confidence_probability", confidence[index]),
        ):
            if probability is not None:
                number = _finite_number(probability, f"{item_label}.{field_name}")
                if not 0.0 <= number <= 1.0:
                    raise WorkerError(f"{item_label}.{field_name} is outside [0, 1]")
        if visibility[index] is None:
            if visibility_status[index] != "NOT_ESTIMATED":
                raise WorkerError(f"{item_label} null visibility must be NOT_ESTIMATED")
        elif visibility_status[index] != "ESTIMATED":
            raise WorkerError(f"{item_label} visibility status is inconsistent")
        support = supports[index]
        if (
            not isinstance(support, list)
            or len(support) != len(set(support))
            or any(view not in {"left", "right"} for view in support)
        ):
            raise WorkerError(f"{item_label}.support_view_ids is invalid")
        expected_evidence = "MULTIVIEW" if len(support) == 2 else "MONOCULAR" if support else "NONE"
        if evidence[index] != expected_evidence:
            raise WorkerError(f"{item_label}.evidence_source is inconsistent with support_view_ids")
        residual = residuals[index]
        if not isinstance(residual, dict) or set(residual) != {"left", "right"}:
            raise WorkerError(f"{item_label}.reprojection_residuals_px is invalid")
        for view, number in residual.items():
            if number is not None:
                if _finite_number(number, f"{item_label}.residual.{view}") < 0.0:
                    raise WorkerError(f"{item_label}.residual.{view} cannot be negative")

    radius = value["confidence_radius_m"]
    if value["confidence_status"] not in {"NOT_CALIBRATED", "CALIBRATED"}:
        raise WorkerError(f"{label}.confidence_status is invalid")
    if radius is None:
        if value["confidence_status"] != "NOT_CALIBRATED" or any(
            probability is not None for probability in confidence
        ):
            raise WorkerError(f"{label} confidence requires confidence_radius_m")
    else:
        if (
            value["confidence_status"] != "CALIBRATED"
            or _finite_number(radius, f"{label}.confidence_radius_m") <= 0.0
            or any(probability is None for probability in confidence)
        ):
            raise WorkerError(f"{label} calibrated confidence contract is invalid")

    mapping_ids = value["mapping_ids"]
    if (
        not isinstance(mapping_ids, list)
        or not mapping_ids
        or len(mapping_ids) != len(set(mapping_ids))
        or any(not isinstance(item, str) or not item for item in mapping_ids)
    ):
        raise WorkerError(f"{label}.mapping_ids is invalid")
    provenance = value["backend_provenance"]
    provenance_fields = {
        "producer",
        "producer_version",
        "worker_request_sha256",
        "model_manifest_sha256",
        "mmpose_commit",
        "detector",
        "pose",
        "fusion_method",
        "kinematic_method",
        "temporal_method",
    }
    if not isinstance(provenance, dict) or not provenance_fields <= set(provenance):
        raise WorkerError(f"{label}.backend_provenance is incomplete")
    for field in provenance_fields - {"detector", "pose"}:
        _text(provenance[field], f"{label}.backend_provenance.{field}")
    for model_field in ("detector", "pose"):
        model = provenance[model_field]
        if not isinstance(model, dict) or not {"id", "sha256", "config"} <= set(model):
            raise WorkerError(f"{label}.backend_provenance.{model_field} is incomplete")
        for field in ("id", "sha256", "config"):
            _text(model[field], f"{label}.backend_provenance.{model_field}.{field}")
    _validate_raw_observation(value["raw"], label)
    if value["mano"] is not None:
        _validate_mano_payload(value["mano"], label)
    temporal = value["temporal"]
    _validate_temporal_payload(temporal, label)
    if temporal.get("landmarks_xyz_m") != points:
        raise WorkerError(f"{label} selected landmarks disagree with temporal source")
    if temporal.get("timestamp_ns") != value["timestamp_ns"]:
        raise WorkerError(f"{label} temporal source timestamp is inconsistent")
    expected_kinds = [
        (
            "REFINED"
            if (
                refined
                or (
                    value["mano"] is not None
                    and value["mano"]["validity"][index] == "VALID"
                    and value["mano"]["landmarks_xyz_m"][index] is not None
                )
            )
            else "MEASURED"
        )
        for index, refined in enumerate(temporal["refinement_applied"])
    ]
    if kinds != expected_kinds:
        raise WorkerError(f"{label}.kind is inconsistent with MANO/temporal provenance")
    return value


__all__ = [
    "OUTPUT_AXIS_CONVENTION",
    "RTMPOSE_FHP21_MAPPING_ID",
    "build_pose_estimate",
    "validate_pose_estimate",
]
