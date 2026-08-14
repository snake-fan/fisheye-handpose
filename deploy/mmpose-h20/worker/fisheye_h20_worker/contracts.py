"""Strict JSON request contract for the process-isolated H20 worker."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "fisheye-handpose/h20-worker-request/v1"


class WorkerError(RuntimeError):
    """A worker input, runtime, or output contract failed."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerError(f"{label} must be a JSON object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError(f"{label} must be a non-empty string")
    return value.strip()


def _path(value: Any, label: str) -> Path:
    return Path(_text(value, label)).expanduser().resolve()


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise WorkerError(f"{label} must be finite and in [{minimum}, {maximum}]")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkerError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class SessionRequest:
    path: Path
    timestamp_column: str
    timestamp_unit: str
    max_skew_ns: int
    clock_offset_ns: int
    max_pairs: int


@dataclass(frozen=True)
class CalibrationRequest:
    path: Path
    left_camera_id: str
    right_camera_id: str
    translation_unit: str
    extrinsics_convention: str
    output_size: tuple[int, int]
    balance: float
    fov_scale: float


@dataclass(frozen=True)
class ThresholdRequest:
    bbox_score: float
    keypoint_score: float
    association_epipolar_px: float
    max_reprojection_error_px: float
    min_ray_angle_deg: float


@dataclass(frozen=True)
class ModelRequest:
    manifest: Path
    model_dir: Path
    mmpose_source: Path
    device: str
    detector_category_id: int
    license_risk_acknowledged: bool


@dataclass(frozen=True)
class ArtifactRequest:
    source_frames: str
    sample_every: int
    image_format: str
    overlay_video: bool


@dataclass(frozen=True)
class TrackingRequest:
    max_root_distance_m: float
    max_gap_ms: float


@dataclass(frozen=True)
class ManoRequest:
    model_root: Path
    manifest: Path
    min_valid_landmarks: int
    max_fit_rmse_m: float
    iterations: int
    learning_rate: float


@dataclass(frozen=True)
class TemporalRequest:
    method: str
    time_constant_ms: float
    gap_reset_ms: float


@dataclass(frozen=True)
class WorkerRequest:
    source_path: Path
    source_sha256: str
    session: SessionRequest
    calibration: CalibrationRequest
    thresholds: ThresholdRequest
    models: ModelRequest
    artifacts: ArtifactRequest
    tracking: TrackingRequest
    mano: ManoRequest | None
    temporal: TemporalRequest


def load_request(path: str | Path) -> WorkerRequest:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise WorkerError(f"request JSON is not a file: {source}")
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"cannot parse request JSON: {exc}") from exc
    root = _mapping(document, "request")
    if root.get("schema_version") != REQUEST_SCHEMA:
        raise WorkerError(f"schema_version must be {REQUEST_SCHEMA!r}")

    session = _mapping(root.get("session"), "session")
    timestamp_unit = _text(session.get("timestamp_unit"), "session.timestamp_unit")
    if timestamp_unit not in {"ns", "us", "ms"}:
        raise WorkerError("session.timestamp_unit must be ns, us, or ms")
    max_skew_us = _positive_int(session.get("max_skew_us"), "session.max_skew_us")
    clock_offset_ns = session.get("clock_offset_ns", 0)
    if isinstance(clock_offset_ns, bool) or not isinstance(clock_offset_ns, int):
        raise WorkerError("session.clock_offset_ns must be an integer")
    session_request = SessionRequest(
        path=_path(session.get("path"), "session.path"),
        timestamp_column=_text(session.get("timestamp_column"), "session.timestamp_column"),
        timestamp_unit=timestamp_unit,
        max_skew_ns=max_skew_us * 1_000,
        clock_offset_ns=clock_offset_ns,
        max_pairs=_positive_int(session.get("max_pairs"), "session.max_pairs"),
    )

    calibration = _mapping(root.get("calibration"), "calibration")
    output_size = calibration.get("output_size")
    if (
        not isinstance(output_size, list)
        or len(output_size) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in output_size
        )
    ):
        raise WorkerError("calibration.output_size must contain two positive integers")
    translation_unit = _text(calibration.get("translation_unit"), "calibration.translation_unit")
    if translation_unit not in {"mm", "m"}:
        raise WorkerError("calibration.translation_unit must be mm or m")
    convention = _text(
        calibration.get("extrinsics_convention"),
        "calibration.extrinsics_convention",
    )
    if convention not in {"reference_to_camera", "camera_to_reference"}:
        raise WorkerError("unsupported calibration.extrinsics_convention")
    calibration_request = CalibrationRequest(
        path=_path(calibration.get("path"), "calibration.path"),
        left_camera_id=_text(calibration.get("left_camera_id"), "calibration.left_camera_id"),
        right_camera_id=_text(calibration.get("right_camera_id"), "calibration.right_camera_id"),
        translation_unit=translation_unit,
        extrinsics_convention=convention,
        output_size=(output_size[0], output_size[1]),
        balance=_number(
            calibration.get("balance"), "calibration.balance", minimum=0.0, maximum=1.0
        ),
        fov_scale=_number(
            calibration.get("fov_scale"),
            "calibration.fov_scale",
            minimum=0.01,
            maximum=100.0,
        ),
    )
    if calibration_request.left_camera_id == calibration_request.right_camera_id:
        raise WorkerError("left and right camera IDs must differ")

    thresholds = _mapping(root.get("thresholds"), "thresholds")
    threshold_request = ThresholdRequest(
        bbox_score=_number(
            thresholds.get("bbox_score"), "thresholds.bbox_score", minimum=0.0, maximum=1.0
        ),
        keypoint_score=_number(
            thresholds.get("keypoint_score"),
            "thresholds.keypoint_score",
            minimum=0.0,
            maximum=1.0,
        ),
        association_epipolar_px=_number(
            thresholds.get("association_epipolar_px"),
            "thresholds.association_epipolar_px",
            minimum=0.0,
            maximum=10_000.0,
        ),
        max_reprojection_error_px=_number(
            thresholds.get("max_reprojection_error_px"),
            "thresholds.max_reprojection_error_px",
            minimum=0.0,
            maximum=10_000.0,
        ),
        min_ray_angle_deg=_number(
            thresholds.get("min_ray_angle_deg"),
            "thresholds.min_ray_angle_deg",
            minimum=0.0,
            maximum=90.0,
        ),
    )

    models = _mapping(root.get("models"), "models")
    device = _text(models.get("device"), "models.device")
    if re.fullmatch(r"cuda(?::\d+)?", device) is None:
        raise WorkerError("models.device must name a CUDA device")
    category = models.get("detector_category_id")
    if isinstance(category, bool) or not isinstance(category, int) or category < 0:
        raise WorkerError("models.detector_category_id must be a non-negative integer")
    acknowledgement = models.get("license_risk_acknowledged")
    if not isinstance(acknowledgement, bool):
        raise WorkerError("models.license_risk_acknowledged must be boolean")
    model_request = ModelRequest(
        manifest=_path(models.get("manifest"), "models.manifest"),
        model_dir=_path(models.get("model_dir"), "models.model_dir"),
        mmpose_source=_path(models.get("mmpose_source"), "models.mmpose_source"),
        device=device,
        detector_category_id=category,
        license_risk_acknowledged=acknowledgement,
    )

    artifacts = _mapping(root.get("artifacts"), "artifacts")
    source_frames = _text(artifacts.get("source_frames"), "artifacts.source_frames").upper()
    if source_frames not in {"NONE", "ALL", "SAMPLED"}:
        raise WorkerError("artifacts.source_frames must be NONE, ALL, or SAMPLED")
    image_format = str(artifacts.get("image_format", "jpg")).lower()
    if image_format not in {"jpg", "png"}:
        raise WorkerError("artifacts.image_format must be jpg or png")
    overlay_video = artifacts.get("overlay_video", False)
    if not isinstance(overlay_video, bool):
        raise WorkerError("artifacts.overlay_video must be boolean")
    artifact_request = ArtifactRequest(
        source_frames=source_frames,
        sample_every=_positive_int(artifacts.get("sample_every"), "artifacts.sample_every"),
        image_format=image_format,
        overlay_video=overlay_video,
    )

    raw_tracking = root.get("tracking")
    tracking = {} if raw_tracking is None else _mapping(raw_tracking, "tracking")
    tracking_request = TrackingRequest(
        max_root_distance_m=_number(
            tracking.get("max_root_distance_m", 0.15),
            "tracking.max_root_distance_m",
            minimum=0.000001,
            maximum=10.0,
        ),
        max_gap_ms=_number(
            tracking.get("max_gap_ms", 250.0),
            "tracking.max_gap_ms",
            minimum=0.001,
            maximum=3_600_000.0,
        ),
    )

    raw_mano = root.get("mano")
    if raw_mano is None:
        mano_request = None
    else:
        mano = _mapping(raw_mano, "mano")
        min_valid = _positive_int(mano.get("min_valid_landmarks"), "mano.min_valid_landmarks")
        if min_valid > 21:
            raise WorkerError("mano.min_valid_landmarks must be at most 21")
        iterations = _positive_int(mano.get("iterations"), "mano.iterations")
        if iterations > 10_000:
            raise WorkerError("mano.iterations must be at most 10000")
        mano_request = ManoRequest(
            model_root=_path(mano.get("model_root"), "mano.model_root"),
            manifest=_path(mano.get("manifest"), "mano.manifest"),
            min_valid_landmarks=min_valid,
            max_fit_rmse_m=_number(
                mano.get("max_fit_rmse_m"),
                "mano.max_fit_rmse_m",
                minimum=0.000001,
                maximum=10.0,
            ),
            iterations=iterations,
            learning_rate=_number(
                mano.get("learning_rate"),
                "mano.learning_rate",
                minimum=0.00000001,
                maximum=10.0,
            ),
        )

    raw_temporal = root.get("temporal")
    temporal = {} if raw_temporal is None else _mapping(raw_temporal, "temporal")
    temporal_method = _text(temporal.get("method", "causal_time_ema_v1"), "temporal.method")
    if temporal_method != "causal_time_ema_v1":
        raise WorkerError("temporal.method must be causal_time_ema_v1")
    temporal_request = TemporalRequest(
        method=temporal_method,
        time_constant_ms=_number(
            temporal.get("time_constant_ms", 80.0),
            "temporal.time_constant_ms",
            minimum=0.001,
            maximum=3_600_000.0,
        ),
        gap_reset_ms=_number(
            temporal.get("gap_reset_ms", 250.0),
            "temporal.gap_reset_ms",
            minimum=0.001,
            maximum=3_600_000.0,
        ),
    )

    import hashlib

    return WorkerRequest(
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        session=session_request,
        calibration=calibration_request,
        thresholds=threshold_request,
        models=model_request,
        artifacts=artifact_request,
        tracking=tracking_request,
        mano=mano_request,
        temporal=temporal_request,
    )


__all__ = [
    "ArtifactRequest",
    "CalibrationRequest",
    "ModelRequest",
    "ManoRequest",
    "REQUEST_SCHEMA",
    "SessionRequest",
    "ThresholdRequest",
    "TemporalRequest",
    "TrackingRequest",
    "WorkerError",
    "WorkerRequest",
    "load_request",
]
