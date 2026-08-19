"""End-to-end preflight for calibrated stereo capture sessions."""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ._generated_project_contract import AUDIT_SCHEMA
from .calibration import load_orbbec_stereo
from .errors import FisheyeHandposeError
from .geometry import RectificationConfig, StereoRectifier
from .joints import FHP21
from .qa import EpipolarQaConfig, evaluate_epipolar_qa
from .session import discover_session
from .sync import TimestampSeries, match_timestamps, read_timestamp_csv
from .video import StereoPairReader, audit_video


class AuditConfigurationError(FisheyeHandposeError):
    """An audit gate or declared input convention is invalid."""


@dataclass(frozen=True, slots=True)
class AuditConfig:
    left_id: str
    right_id: str
    translation_unit: str
    extrinsics_convention: str
    timestamp_column: str = "timestamp_us"
    timestamp_unit: str = "us"
    max_skew_ns: int = 1_000_000
    clock_offset_ns: int = 0
    min_pair_count: int = 20
    min_overlap_duration_ns: int = 750_000_000
    min_overlap_match_rate: float = 0.0
    min_timestamp_fps: float = 29.5
    max_timestamp_fps: float = 30.5
    max_timestamp_fps_relative_difference: float = 0.001
    max_p99_skew_ns: int = 250_000
    max_observed_skew_ns: int = 500_000
    max_overlap_unmatched_fraction: float = 0.005
    overlap_unmatched_floor: int = 2
    max_gap_periods: float = 2.5
    max_missing_frame_fraction: float = 0.005
    missing_frame_floor: int = 1
    baseline_range_m: tuple[float, float] = (0.02, 0.30)
    output_size: tuple[int, int] = (1600, 1300)
    balance: float = 0.8
    fov_scale: float = 1.0
    min_common_valid_fraction: float = 0.80
    min_per_camera_valid_fraction: float = 0.82
    min_hfov_deg: float = 150.0
    min_vfov_deg: float = 145.0
    run_epipolar_qa: bool = True
    epipolar: EpipolarQaConfig = EpipolarQaConfig()

    def __post_init__(self) -> None:
        if not self.left_id or not self.right_id or self.left_id == self.right_id:
            raise AuditConfigurationError("left/right camera IDs must be non-empty and distinct")
        if self.translation_unit not in ("mm", "m"):
            raise AuditConfigurationError("translation_unit must be 'mm' or 'm'")
        if self.extrinsics_convention not in (
            "reference_to_camera",
            "camera_to_reference",
        ):
            raise AuditConfigurationError("unsupported extrinsics convention")
        if self.timestamp_unit not in ("ns", "us", "ms") or not self.timestamp_column:
            raise AuditConfigurationError("timestamp unit/column is invalid")
        for label, value in (
            ("max_skew_ns", self.max_skew_ns),
            ("min_pair_count", self.min_pair_count),
            ("min_overlap_duration_ns", self.min_overlap_duration_ns),
            ("max_p99_skew_ns", self.max_p99_skew_ns),
            ("max_observed_skew_ns", self.max_observed_skew_ns),
            ("overlap_unmatched_floor", self.overlap_unmatched_floor),
            ("missing_frame_floor", self.missing_frame_floor),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AuditConfigurationError(f"{label} must be a positive integer")
        for label, value in (
            ("min_overlap_match_rate", self.min_overlap_match_rate),
            ("min_common_valid_fraction", self.min_common_valid_fraction),
            ("max_overlap_unmatched_fraction", self.max_overlap_unmatched_fraction),
            ("max_missing_frame_fraction", self.max_missing_frame_fraction),
            (
                "max_timestamp_fps_relative_difference",
                self.max_timestamp_fps_relative_difference,
            ),
            ("min_per_camera_valid_fraction", self.min_per_camera_valid_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise AuditConfigurationError(f"{label} must be in [0, 1]")
        if not 0 < self.min_timestamp_fps <= self.max_timestamp_fps:
            raise AuditConfigurationError("timestamp FPS range must be positive and ordered")
        if (
            len(self.baseline_range_m) != 2
            or not 0 < self.baseline_range_m[0] < self.baseline_range_m[1]
        ):
            raise AuditConfigurationError("baseline range must be positive and ordered")
        if self.min_hfov_deg <= 0 or self.min_vfov_deg <= 0:
            raise AuditConfigurationError("minimum FOV gates must be positive")
        if self.max_gap_periods <= 1.0:
            raise AuditConfigurationError("max_gap_periods must be greater than one")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline_range_m"] = list(self.baseline_range_m)
        payload["output_size"] = list(self.output_size)
        return payload


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _new_report(session_path: Path, config: AuditConfig) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "FAIL",
        "input_session": str(session_path),
        "config": config.to_dict(),
        "software": {
            "python": platform.python_version(),
            "fisheye-handpose": _package_version("fisheye-handpose"),
            "numpy": _package_version("numpy"),
            "opencv-python-headless": _package_version("opencv-python-headless"),
            "PyYAML": _package_version("PyYAML"),
            "av": _package_version("av"),
        },
        "session": None,
        "calibration": None,
        "rectification": None,
        "parts": [],
        "fhp21_schema": FHP21.to_dict(),
        "errors": [],
        "hard_failures": [],
        "warnings": [],
        "stages": {},
    }


def _error(
    report: dict[str, Any],
    *,
    stage: str,
    code: str,
    message: str,
    part_number: int | None = None,
    metric: float | int | None = None,
    threshold: float | int | None = None,
) -> None:
    item: dict[str, Any] = {"stage": stage, "code": code, "message": message}
    if part_number is not None:
        item["part_number"] = part_number
    if metric is not None:
        item["metric"] = metric
    if threshold is not None:
        item["threshold"] = threshold
    report["errors"].append(item)
    report["hard_failures"].append(item)
    report["stages"][stage] = "FAIL"


def _warning(
    report: dict[str, Any],
    *,
    stage: str,
    code: str,
    message: str,
    part_number: int | None = None,
) -> None:
    item: dict[str, Any] = {"stage": stage, "code": code, "message": message}
    if part_number is not None:
        item["part_number"] = part_number
    report["warnings"].append(item)
    report["stages"].setdefault(stage, "WARN")


def _capture_stage_error(
    report: dict[str, Any],
    *,
    stage: str,
    exc: FisheyeHandposeError,
    part_number: int | None = None,
) -> None:
    _error(
        report,
        stage=stage,
        code=type(exc).__name__,
        message=str(exc),
        part_number=part_number,
    )


def _fps(series: TimestampSeries) -> float | None:
    period = series.nominal_period_ns
    return None if period is None or period <= 0 else 1_000_000_000.0 / period


def _apply_sync_gates(
    report: dict[str, Any],
    *,
    part_number: int,
    left: TimestampSeries,
    right: TimestampSeries,
    sync_report: dict[str, Any],
    config: AuditConfig,
) -> None:
    if sync_report["pair_count"] < config.min_pair_count:
        _error(
            report,
            stage="sync_gate",
            code="PAIR_COUNT_TOO_LOW",
            message="matched pair count is below the configured minimum",
            part_number=part_number,
            metric=sync_report["pair_count"],
            threshold=config.min_pair_count,
        )
    if sync_report["overlap_duration_ns"] < config.min_overlap_duration_ns:
        _error(
            report,
            stage="sync_gate",
            code="OVERLAP_TOO_SHORT",
            message="strict common timestamp interval is too short",
            part_number=part_number,
            metric=sync_report["overlap_duration_ns"],
            threshold=config.min_overlap_duration_ns,
        )
    for side in ("left", "right"):
        rate = sync_report[f"{side}_overlap_match_rate"]
        if rate < config.min_overlap_match_rate:
            _error(
                report,
                stage="sync_gate",
                code=f"{side.upper()}_OVERLAP_MATCH_RATE_TOO_LOW",
                message=f"{side} strict-overlap match rate is below the gate",
                part_number=part_number,
                metric=rate,
                threshold=config.min_overlap_match_rate,
            )
    p99 = sync_report["p99_abs_skew_ns"]
    if p99 is None or p99 > config.max_p99_skew_ns:
        _error(
            report,
            stage="sync_gate",
            code="P99_SKEW_TOO_HIGH",
            message="p99 absolute corrected timestamp skew exceeds the gate",
            part_number=part_number,
            metric=p99,
            threshold=config.max_p99_skew_ns,
        )
    observed_max = sync_report["max_abs_skew_ns"]
    if observed_max is None or observed_max > config.max_observed_skew_ns:
        _error(
            report,
            stage="sync_gate",
            code="MAX_SKEW_TOO_HIGH",
            message="maximum absolute corrected timestamp skew exceeds the gate",
            part_number=part_number,
            metric=observed_max,
            threshold=config.max_observed_skew_ns,
        )
    measured_fps_by_side = {"left": _fps(left), "right": _fps(right)}
    left_fps = measured_fps_by_side["left"]
    right_fps = measured_fps_by_side["right"]
    if left_fps is not None and right_fps is not None:
        relative_difference = abs(left_fps - right_fps) / max(left_fps, right_fps)
        if relative_difference > config.max_timestamp_fps_relative_difference:
            _error(
                report,
                stage="timestamp_gate",
                code="STEREO_FPS_MISMATCH",
                message="left/right hardware timestamp rates differ beyond the gate",
                part_number=part_number,
                metric=relative_difference,
                threshold=config.max_timestamp_fps_relative_difference,
            )
    for side, series in (("left", left), ("right", right)):
        measured_fps = measured_fps_by_side[side]
        if (
            measured_fps is None
            or measured_fps < config.min_timestamp_fps
            or measured_fps > config.max_timestamp_fps
        ):
            _error(
                report,
                stage="timestamp_gate",
                code=f"{side.upper()}_FPS_OUT_OF_RANGE",
                message=f"{side} hardware timestamp rate is outside the gate",
                part_number=part_number,
                metric=measured_fps,
            )
        overlap_count = sync_report[f"{side}_overlap_frame_count"]
        unmatched_count = sync_report[f"overlap_unmatched_{side}_count"]
        unmatched_limit = max(
            config.overlap_unmatched_floor,
            math.ceil(config.max_overlap_unmatched_fraction * overlap_count),
        )
        if unmatched_count > unmatched_limit:
            _error(
                report,
                stage="sync_gate",
                code=f"{side.upper()}_OVERLAP_UNMATCHED_TOO_HIGH",
                message=f"{side} strict-overlap unmatched count exceeds the gate",
                part_number=part_number,
                metric=unmatched_count,
                threshold=unmatched_limit,
            )
        period = series.nominal_period_ns
        maximum_interval = series.maximum_interval_ns
        if (
            period is not None
            and maximum_interval is not None
            and maximum_interval >= config.max_gap_periods * period
        ):
            _error(
                report,
                stage="timestamp_gate",
                code=f"{side.upper()}_CONSECUTIVE_DROPS",
                message=f"{side} stream contains a gap too large for the configured gate",
                part_number=part_number,
                metric=maximum_interval,
                threshold=config.max_gap_periods * period,
            )
        estimated_missing = 0
        if period is not None:
            estimated_missing = sum(
                max(0, round(interval / period) - 1) for interval in series.intervals_ns
            )
        missing_limit = max(
            config.missing_frame_floor,
            math.ceil(config.max_missing_frame_fraction * max(0, len(series.values_ns) - 1)),
        )
        if estimated_missing > missing_limit:
            _error(
                report,
                stage="timestamp_gate",
                code=f"{side.upper()}_MISSING_FRAMES_TOO_HIGH",
                message=f"{side} estimated dropped-frame count exceeds the gate",
                part_number=part_number,
                metric=estimated_missing,
                threshold=missing_limit,
            )
        if series.gap_after_indices:
            _warning(
                report,
                stage="timestamp_gate",
                code=f"{side.upper()}_FRAME_GAPS",
                message=(
                    f"{len(series.gap_after_indices)} likely dropped-frame gap(s); "
                    "unmatched masks must be preserved"
                ),
                part_number=part_number,
            )


def audit_session(session_path: str | Path, config: AuditConfig) -> dict[str, Any]:
    """Run every safe preflight stage and always return a serializable report."""

    source = Path(session_path).expanduser().resolve()
    report = _new_report(source, config)
    try:
        session = discover_session(source)
    except FisheyeHandposeError as exc:
        _capture_stage_error(report, stage="discovery", exc=exc)
        return report
    report["session"] = session.to_dict()
    report["stages"]["discovery"] = "PASS"

    try:
        calibration = load_orbbec_stereo(
            session.calibration_path,
            left_id=config.left_id,
            right_id=config.right_id,
            translation_unit=config.translation_unit,
            extrinsics_convention=config.extrinsics_convention,
            baseline_range_m=config.baseline_range_m,
        )
    except FisheyeHandposeError as exc:
        _capture_stage_error(report, stage="calibration", exc=exc)
        return report
    report["calibration"] = calibration.to_dict()
    report["stages"]["calibration"] = "PASS"

    rectifier: StereoRectifier | None = None
    try:
        rectifier = StereoRectifier.build(
            calibration,
            RectificationConfig(
                output_size=config.output_size,
                balance=config.balance,
                fov_scale=config.fov_scale,
                map_type="float32",
            ),
        )
        report["rectification"] = rectifier.to_dict()
        report["stages"]["rectification"] = "PASS"
    except FisheyeHandposeError as exc:
        _capture_stage_error(report, stage="rectification", exc=exc)
    if rectifier is not None:
        if rectifier.common_valid_fraction < config.min_common_valid_fraction:
            _error(
                report,
                stage="rectification_gate",
                code="COMMON_VALID_FRACTION_TOO_LOW",
                message="rectification common valid fraction is below the gate",
                metric=rectifier.common_valid_fraction,
                threshold=config.min_common_valid_fraction,
            )
        for side, valid_fraction in (
            ("left", rectifier.left_valid_fraction),
            ("right", rectifier.right_valid_fraction),
        ):
            if valid_fraction < config.min_per_camera_valid_fraction:
                _error(
                    report,
                    stage="rectification_gate",
                    code=f"{side.upper()}_VALID_FRACTION_TOO_LOW",
                    message=f"{side} rectification valid fraction is below the gate",
                    metric=valid_fraction,
                    threshold=config.min_per_camera_valid_fraction,
                )
        hfov, vfov = rectifier.effective_fov_degrees()
        if hfov < config.min_hfov_deg:
            _error(
                report,
                stage="rectification_gate",
                code="HFOV_TOO_LOW",
                message="effective rectified horizontal field of view is below the gate",
                metric=hfov,
                threshold=config.min_hfov_deg,
            )
        if vfov < config.min_vfov_deg:
            _error(
                report,
                stage="rectification_gate",
                code="VFOV_TOO_LOW",
                message="effective rectified vertical field of view is below the gate",
                metric=vfov,
                threshold=config.min_vfov_deg,
            )

    for part in session.parts:
        part_report: dict[str, Any] = {
            "spec": part.to_dict(),
            "timestamps": {"left": None, "right": None},
            "sync": None,
            "video": {"left": None, "right": None},
            "epipolar_qa": None,
        }
        report["parts"].append(part_report)
        streams: list[TimestampSeries | None] = []
        for side, path in (
            ("left", part.left_timestamps),
            ("right", part.right_timestamps),
        ):
            try:
                series = read_timestamp_csv(
                    path,
                    column=config.timestamp_column,
                    unit=config.timestamp_unit,
                )
                part_report["timestamps"][side] = series.to_dict()
                streams.append(series)
            except FisheyeHandposeError as exc:
                _capture_stage_error(
                    report,
                    stage=f"{side}_timestamps",
                    exc=exc,
                    part_number=part.part_number,
                )
                streams.append(None)
        left, right = streams
        if left is None or right is None:
            continue
        report["stages"].setdefault("timestamps", "PASS")
        try:
            sync = match_timestamps(
                left,
                right,
                max_skew_ns=config.max_skew_ns,
                clock_offset_ns=config.clock_offset_ns,
            )
            sync_report = sync.to_dict()
            part_report["sync"] = sync_report
            _apply_sync_gates(
                report,
                part_number=part.part_number,
                left=left,
                right=right,
                sync_report=sync_report,
                config=config,
            )
            report["stages"].setdefault("sync", "PASS")
        except FisheyeHandposeError as exc:
            _capture_stage_error(
                report,
                stage="sync",
                exc=exc,
                part_number=part.part_number,
            )
            continue

        try:
            left_video = audit_video(
                part.left_video,
                calibration.left.image_size,
                len(left.values_ns),
            )
            right_video = audit_video(
                part.right_video,
                calibration.right.image_size,
                len(right.values_ns),
            )
        except FisheyeHandposeError as exc:
            _capture_stage_error(
                report,
                stage="video_runtime",
                exc=exc,
                part_number=part.part_number,
            )
            continue
        part_report["video"]["left"] = left_video.to_dict()
        part_report["video"]["right"] = right_video.to_dict()
        if left_video.passed and right_video.passed:
            report["stages"].setdefault("video", "PASS")
        for side, video_report in (("left", left_video), ("right", right_video)):
            for message in video_report.hard_failures:
                _error(
                    report,
                    stage=f"{side}_video",
                    code="VIDEO_AUDIT_FAILED",
                    message=message,
                    part_number=part.part_number,
                )
            for message in video_report.warnings:
                _warning(
                    report,
                    stage=f"{side}_video",
                    code="CONTAINER_PTS_DIAGNOSTIC",
                    message=message,
                    part_number=part.part_number,
                )

        if not config.run_epipolar_qa:
            part_report["epipolar_qa"] = {
                "status": "SKIPPED",
                "reason": "explicitly disabled by audit configuration",
            }
            _warning(
                report,
                stage="epipolar_qa",
                code="EPIPOLAR_QA_SKIPPED",
                message="empirical epipolar/cheirality QA was explicitly skipped",
                part_number=part.part_number,
            )
        elif rectifier is not None and left_video.passed and right_video.passed:
            try:
                reader = StereoPairReader(
                    part.left_video,
                    part.right_video,
                    left_video,
                    right_video,
                    sync,
                )
                qa_report = evaluate_epipolar_qa(
                    rectifier,
                    reader,
                    pair_count=len(sync.matches),
                    config=config.epipolar,
                )
                part_report["epipolar_qa"] = qa_report.to_dict()
                if qa_report.status != "PASS":
                    _error(
                        report,
                        stage="epipolar_qa",
                        code=f"EPIPOLAR_QA_{qa_report.status}",
                        message=(qa_report.reason or "; ".join(qa_report.failures)),
                        part_number=part.part_number,
                    )
            except FisheyeHandposeError as exc:
                _capture_stage_error(
                    report,
                    stage="epipolar_qa",
                    exc=exc,
                    part_number=part.part_number,
                )

    if report["errors"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"
    return report


__all__ = ["AuditConfig", "AuditConfigurationError", "audit_session"]
