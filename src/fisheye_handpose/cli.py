"""Command-line interface for capture preflight and geometry artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .audit import AuditConfig, AuditConfigurationError, audit_session
from .calibration import load_orbbec_stereo
from .errors import FisheyeHandposeError
from .joints import FHP21
from .qa import EpipolarQaConfig
from .session import discover_sessions
from .sync import match_timestamps, read_timestamp_csv


def _finite_float(value: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _port(value: str) -> int:
    number = _positive_int(value)
    if number > 65_535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return number


def _loopback_host(value: str) -> str:
    if value not in {"127.0.0.1", "localhost"}:
        raise argparse.ArgumentTypeError("host must be 127.0.0.1 or localhost")
    return value


def _nonempty_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must be non-empty")
    return value


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return number


def _positive_float(value: str) -> float:
    number = _finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _unit_interval(value: str) -> float:
    number = _finite_float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return number


def _write_json(payload: Any, output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if not output:
        sys.stdout.write(text)
        return
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--left-id", required=True)
    parser.add_argument("--right-id", required=True)
    parser.add_argument("--translation-unit", required=True, choices=("mm", "m"))
    parser.add_argument(
        "--extrinsics-convention",
        required=True,
        choices=("reference_to_camera", "camera_to_reference"),
    )
    parser.add_argument("--min-baseline-m", type=float, default=0.02)
    parser.add_argument("--max-baseline-m", type=float, default=0.30)


def _baseline_range(args: argparse.Namespace) -> tuple[float, float]:
    baseline = (args.min_baseline_m, args.max_baseline_m)
    if not 0 < baseline[0] < baseline[1]:
        raise AuditConfigurationError(
            "--min-baseline-m and --max-baseline-m must be positive and ordered"
        )
    return baseline


def _load_calibration(args: argparse.Namespace, path: str | Path):
    return load_orbbec_stereo(
        path,
        left_id=args.left_id,
        right_id=args.right_id,
        translation_unit=args.translation_unit,
        extrinsics_convention=args.extrinsics_convention,
        baseline_range_m=_baseline_range(args),
    )


def _command_discover(args: argparse.Namespace) -> None:
    sessions = discover_sessions(args.root)
    _write_json({"sessions": [session.to_dict() for session in sessions]}, args.output)


def _command_inspect_calibration(args: argparse.Namespace) -> None:
    calibration = _load_calibration(args, args.calibration)
    _write_json(calibration.to_dict(), args.output)


def _command_pair_pts(args: argparse.Namespace) -> None:
    left = read_timestamp_csv(args.left, column=args.column, unit=args.timestamp_unit)
    right = read_timestamp_csv(args.right, column=args.column, unit=args.timestamp_unit)
    result = match_timestamps(
        left,
        right,
        max_skew_ns=args.max_skew_us * 1_000,
        clock_offset_ns=args.clock_offset_us * 1_000,
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(result.matches[0].to_dict())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(match.to_dict() for match in result.matches)
    _write_json(
        {
            "left": left.to_dict(),
            "right": right.to_dict(),
            "sync": result.to_dict(include_matches=not bool(args.output)),
            "pairs_output": str(Path(args.output).resolve()) if args.output else None,
        },
        None,
    )


def _command_schema(args: argparse.Namespace) -> None:
    _write_json(FHP21.to_dict(), args.output)


def _command_trace_init(args: argparse.Namespace) -> None:
    from .trace import RunArtifactWriter

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_id = args.run_id or run_dir.name
    writer = RunArtifactWriter.create(
        run_dir,
        run_id=run_id,
        pipeline_version=args.pipeline_version,
    )
    writer.close()
    _write_json(
        {
            "ok": True,
            "run_dir": str(run_dir),
            "run_id": run_id,
            "status": "ACTIVE",
        },
        None,
    )


def _demo_keypoints(frame_index: int, view_id: str) -> list[list[float]]:
    shift_x = frame_index * 12 + (8 if view_id == "right" else 0)
    points = [
        (320, 390),
        (275, 355),
        (235, 325),
        (200, 295),
        (170, 270),
        (285, 325),
        (280, 270),
        (275, 215),
        (270, 165),
        (320, 315),
        (320, 250),
        (320, 185),
        (320, 125),
        (355, 325),
        (360, 270),
        (365, 215),
        (370, 170),
        (385, 345),
        (400, 300),
        (415, 260),
        (430, 225),
    ]
    return [[float(x + shift_x), float(y)] for x, y in points]


def _demo_svg(frame_index: int, view_id: str, points: list[list[float]]) -> bytes:
    color = "#6ee7ff" if view_id == "left" else "#ffcb6b"
    edges = "".join(
        f'<line x1="{points[parent][0]:.1f}" y1="{points[parent][1]:.1f}" '
        f'x2="{points[child][0]:.1f}" y2="{points[child][1]:.1f}" />'
        for parent, child in FHP21.edges
    )
    joints = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" />' for x, y in points)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" '
        'viewBox="0 0 640 480">'
        '<rect width="640" height="480" fill="#101723"/>'
        '<path d="M0 360 Q160 300 320 350 T640 330 V480 H0Z" fill="#182638"/>'
        f'<text x="24" y="38" fill="#e8f1ff" font-family="sans-serif" font-size="20">'
        f"FHP trace demo · {view_id} · frame {frame_index:06d}</text>"
        f'<g stroke="{color}" stroke-width="4" stroke-linecap="round" '
        f'fill="{color}">{edges}{joints}</g>'
        '<rect x="140" y="90" width="340" height="330" fill="none" '
        'stroke="#ff5c8a" stroke-width="2" stroke-dasharray="8 5"/>'
        "</svg>"
    ).encode()


def _command_trace_demo(args: argparse.Namespace) -> None:
    from .trace import (
        RunArtifactWriter,
        RunStatus,
        TraceStage,
        TraceStatus,
    )

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_id = args.run_id or "trace-demo"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id=run_id,
        pipeline_version=args.pipeline_version,
        config={"demo": True, "frame_count": 3, "landmark_schema": FHP21.version},
        inputs=[{"kind": "synthetic_stereo", "view_ids": ["left", "right"]}],
        metadata={"purpose": "front-end and traceability inspection"},
    )
    record_ids: list[str] = []

    def append(**kwargs: Any):
        record = writer.append(**kwargs)
        record_ids.append(record.record_id)
        return record

    try:
        system = append(
            record_id="demo:system",
            stage=TraceStage.SYSTEM,
            status=TraceStatus.SUCCEEDED,
            event="run_started",
            payload={"mode": "deterministic_demo"},
        )
        discovery = append(
            record_id="demo:discovery",
            stage=TraceStage.DISCOVERY,
            status=TraceStatus.SUCCEEDED,
            event="session_discovered",
            payload={"sequence_id": "demo-sequence", "part_count": 1},
            parent_ids=(system.record_id,),
        )
        calibration = append(
            record_id="demo:calibration",
            stage=TraceStage.CALIBRATION,
            status=TraceStatus.SUCCEEDED,
            event="calibration_loaded",
            payload={
                "calibration_id": "demo-calibration/v1",
                "camera_model": "KB4",
                "baseline_m": 0.12,
                "view_ids": ["left", "right"],
            },
            parent_ids=(discovery.record_id,),
        )
        rectification = append(
            record_id="demo:rectification",
            stage=TraceStage.RECTIFICATION,
            status=TraceStatus.SUCCEEDED,
            event="rectification_geometry_built",
            payload={"common_valid_fraction": 0.86, "qa_only": True},
            parent_ids=(calibration.record_id,),
        )

        previous_temporal_id: str | None = None
        for frame_index in range(3):
            frame_id = f"frame/{frame_index:06d}"
            timestamp_ns = 1_000_000_000 + frame_index * 33_333_333
            decoded: dict[str, Any] = {}
            poses: dict[str, Any] = {}
            for view_id in ("left", "right"):
                points = _demo_keypoints(frame_index, view_id)
                source_blob = writer.put_blob(
                    _demo_svg(frame_index, view_id, points),
                    role=f"source_{view_id}",
                    media_type="image/svg+xml",
                    suffix=".svg",
                )
                decoded[view_id] = append(
                    record_id=f"demo:{frame_index:06d}:decode:{view_id}",
                    stage=TraceStage.DECODE,
                    status=TraceStatus.SUCCEEDED,
                    event="source_frame_decoded",
                    payload={
                        "frame_id": frame_id,
                        "frame_index": frame_index,
                        "timestamp_ns": timestamp_ns,
                        "view_id": view_id,
                        "image_size": [640, 480],
                    },
                    parent_ids=(discovery.record_id,),
                    blobs=(source_blob,),
                )
            sync = append(
                record_id=f"demo:{frame_index:06d}:sync",
                stage=TraceStage.SYNCHRONIZATION,
                status=TraceStatus.SUCCEEDED,
                event="stereo_pair_matched",
                payload={
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "left_frame_index": frame_index,
                    "right_frame_index": frame_index,
                    "corrected_skew_ns": 24_000 + frame_index * 1_000,
                },
                parent_ids=tuple(record.record_id for record in decoded.values()),
            )
            for view_id in ("left", "right"):
                points = _demo_keypoints(frame_index, view_id)
                detection = append(
                    record_id=f"demo:{frame_index:06d}:detection:{view_id}",
                    stage=TraceStage.DETECTION,
                    status=TraceStatus.SUCCEEDED,
                    event="hand_detected",
                    payload={
                        "frame_id": frame_id,
                        "timestamp_ns": timestamp_ns,
                        "view_id": view_id,
                        "track_id": "hand-0",
                        "detections": [
                            {
                                "bbox_xyxy": [140.0, 90.0, 480.0, 420.0],
                                "score": 0.96 - frame_index * 0.01,
                                "label": "hand",
                            }
                        ],
                    },
                    parent_ids=(decoded[view_id].record_id, sync.record_id),
                )
                crop = append(
                    record_id=f"demo:{frame_index:06d}:crop:{view_id}",
                    stage=TraceStage.POSE_2D,
                    status=TraceStatus.SUCCEEDED,
                    event="virtual_perspective_crop_generated",
                    payload={
                        "frame_id": frame_id,
                        "timestamp_ns": timestamp_ns,
                        "view_id": view_id,
                        "track_id": "hand-0",
                        "crop_size": [256, 256],
                        "source_bbox_xyxy": [140.0, 90.0, 480.0, 420.0],
                    },
                    parent_ids=(detection.record_id, calibration.record_id),
                )
                poses[view_id] = append(
                    record_id=f"demo:{frame_index:06d}:pose2d:{view_id}",
                    stage=TraceStage.POSE_2D,
                    status=TraceStatus.SUCCEEDED,
                    event="view_keypoints_inferred",
                    payload={
                        "frame_id": frame_id,
                        "timestamp_ns": timestamp_ns,
                        "view_id": view_id,
                        "track_id": "hand-0",
                        "landmark_schema": FHP21.version,
                        "keypoints_uv": points,
                        "keypoint_scores": [round(0.94 - index * 0.004, 3) for index in range(21)],
                    },
                    parent_ids=(crop.record_id,),
                )

            association = append(
                record_id=f"demo:{frame_index:06d}:association",
                stage=TraceStage.CROSS_VIEW_ASSOCIATION,
                status=TraceStatus.SUCCEEDED,
                event="cross_view_track_associated",
                payload={
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "track_id": "hand-0",
                    "view_ids": ["left", "right"],
                    "epipolar_cost_px": round(0.24 + frame_index * 0.03, 3),
                },
                parent_ids=tuple(record.record_id for record in poses.values()),
            )
            points_3d = [
                [
                    round((point[0] - 320.0) * 0.0008, 5),
                    round((point[1] - 300.0) * 0.0008, 5),
                    round(0.55 + index * 0.0004 + frame_index * 0.001, 5),
                ]
                for index, point in enumerate(_demo_keypoints(frame_index, "left"))
            ]
            fusion = append(
                record_id=f"demo:{frame_index:06d}:fusion",
                stage=TraceStage.RAW_FUSION,
                status=TraceStatus.SUCCEEDED,
                event="raw_landmarks_fused",
                payload={
                    "observation_id": f"demo-observation-{frame_index:06d}",
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "track_id": "hand-0",
                    "schema_version": FHP21.version,
                    "landmarks_xyz_m": points_3d,
                    "validity": ["VALID"] * 21,
                    "mean_reprojection_error_px": round(0.42 + frame_index * 0.02, 3),
                },
                parent_ids=(association.record_id, calibration.record_id),
            )
            kinematic = append(
                record_id=f"demo:{frame_index:06d}:kinematic",
                stage=TraceStage.KINEMATIC_REFINEMENT,
                status=TraceStatus.SUCCEEDED,
                event="kinematic_pose_refined",
                payload={
                    "estimate_id": f"demo-kinematic-{frame_index:06d}",
                    "source_observation_ids": [f"demo-observation-{frame_index:06d}"],
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "track_id": "hand-0",
                    "landmarks_xyz_m": points_3d,
                    "optimizer_iterations": 8,
                },
                parent_ids=(fusion.record_id,),
            )
            temporal_parents = [kinematic.record_id]
            if previous_temporal_id is not None:
                temporal_parents.append(previous_temporal_id)
            temporal = append(
                record_id=f"demo:{frame_index:06d}:temporal",
                stage=TraceStage.TEMPORAL_REFINEMENT,
                status=TraceStatus.SUCCEEDED,
                event="temporal_pose_refined",
                payload={
                    "estimate_id": f"demo-temporal-{frame_index:06d}",
                    "source_estimate_id": f"demo-kinematic-{frame_index:06d}",
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "track_id": "hand-0",
                    "landmarks_xyz_m": points_3d,
                    "state": "MEASURED",
                },
                parent_ids=tuple(temporal_parents),
            )
            previous_temporal_id = temporal.record_id
            qa = append(
                record_id=f"demo:{frame_index:06d}:qa",
                stage=TraceStage.QA,
                status=TraceStatus.SUCCEEDED,
                event="frame_quality_evaluated",
                payload={
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "track_id": "hand-0",
                    "valid_landmark_count": 21,
                    "reprojection_p95_px": round(0.71 + frame_index * 0.02, 3),
                },
                parent_ids=(temporal.record_id, rectification.record_id),
            )
            append(
                record_id=f"demo:{frame_index:06d}:export",
                stage=TraceStage.EXPORT,
                status=TraceStatus.SUCCEEDED,
                event="fhp21_pose_exported",
                payload={
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "track_id": "hand-0",
                    "schema_version": FHP21.version,
                    "estimate_id": f"demo-temporal-{frame_index:06d}",
                },
                parent_ids=(temporal.record_id, qa.record_id),
            )

        writer.finalize(
            status=RunStatus.COMPLETED,
            summary={
                "frame_count": 3,
                "track_count": 1,
                "record_count": len(record_ids),
                "landmark_schema": FHP21.version,
                "result": "synthetic_demo_completed",
            },
        )
    except BaseException:
        try:
            writer.finalize(
                status=RunStatus.FAILED,
                summary={"frame_count": 0, "result": "demo_generation_failed"},
            )
        except BaseException:
            writer.close()
        raise
    _write_json(
        {
            "ok": True,
            "run_dir": str(run_dir),
            "run_id": run_id,
            "status": "COMPLETED",
            "frame_count": 3,
            "record_count": len(record_ids),
        },
        None,
    )


def _validation_payload(report: Any, run_dir: Path) -> dict[str, Any]:
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    else:
        payload = {
            "ok": bool(report.ok),
            "errors": list(report.errors),
            "warnings": list(report.warnings),
        }
    payload.setdefault("errors", [])
    payload.setdefault("warnings", [])
    return {"run_dir": str(run_dir), **payload}


def _command_trace_validate(args: argparse.Namespace) -> None:
    from .trace import RunArtifactReader, TraceValidationError

    run_dir = Path(args.run_dir).expanduser().resolve()
    try:
        reader = RunArtifactReader(run_dir)
        report = reader.validate(verify_blobs=not args.skip_blob_verification)
        payload = _validation_payload(report, run_dir)
    except (OSError, TraceValidationError) as exc:
        payload = {
            "run_dir": str(run_dir),
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
        }
    _write_json(payload, None)
    if not payload["ok"]:
        raise FisheyeHandposeError("trace validation failed")


def _command_trace_serve(args: argparse.Namespace) -> None:
    from .trace import RunArtifactReader, TraceValidationError
    from .viewer import serve_trace

    run_dir = Path(args.run_dir).expanduser().resolve()
    try:
        reader = RunArtifactReader(run_dir)
        reader.validate()
    except (OSError, TraceValidationError) as exc:
        raise FisheyeHandposeError(f"cannot serve invalid trace: {exc}") from exc
    _write_json(
        {
            "ok": True,
            "run_dir": str(run_dir),
            "url": f"http://{args.host}:{args.port}/",
        },
        None,
    )
    sys.stdout.flush()
    serve_trace(reader, host=args.host, port=args.port)


def _audit_config(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        left_id=args.left_id,
        right_id=args.right_id,
        translation_unit=args.translation_unit,
        extrinsics_convention=args.extrinsics_convention,
        timestamp_column=args.column,
        timestamp_unit=args.timestamp_unit,
        max_skew_ns=args.max_skew_us * 1_000,
        clock_offset_ns=args.clock_offset_us * 1_000,
        min_pair_count=1 if args.allow_short_session else args.min_pair_count,
        min_overlap_duration_ns=(
            1 if args.allow_short_session else round(args.min_overlap_duration_s * 1_000_000_000)
        ),
        min_overlap_match_rate=(0.0 if args.allow_short_session else args.min_overlap_match_rate),
        min_timestamp_fps=args.min_timestamp_fps,
        max_timestamp_fps=args.max_timestamp_fps,
        max_timestamp_fps_relative_difference=args.max_timestamp_fps_relative_difference,
        max_p99_skew_ns=args.max_p99_skew_us * 1_000,
        max_observed_skew_ns=args.max_observed_skew_us * 1_000,
        baseline_range_m=_baseline_range(args),
        output_size=(args.output_width, args.output_height),
        balance=args.balance,
        fov_scale=args.fov_scale,
        min_common_valid_fraction=args.min_common_valid_fraction,
        min_per_camera_valid_fraction=(
            0.0 if args.allow_short_session else args.min_camera_valid_fraction
        ),
        min_hfov_deg=1.0 if args.allow_short_session else args.min_hfov_deg,
        min_vfov_deg=1.0 if args.allow_short_session else args.min_vfov_deg,
        run_epipolar_qa=not args.skip_epipolar_qa,
        epipolar=EpipolarQaConfig(
            sample_pairs=args.epipolar_sample_pairs,
            min_total_inliers=args.epipolar_min_inliers,
            max_median_vertical_error_px=args.max_median_epipolar_error_px,
            max_p95_vertical_error_px=args.max_p95_epipolar_error_px,
        ),
    )


def _command_audit_session(args: argparse.Namespace) -> None:
    def persist_trace(report: dict[str, Any]) -> None:
        if args.trace_output:
            from .audit_trace import persist_audit_trace

            persist_audit_trace(
                report,
                args.trace_output,
                audit_report_path=args.output,
            )

    try:
        config = _audit_config(args)
    except FisheyeHandposeError as exc:
        report = {
            "schema_version": "fisheye-handpose/audit/v1",
            "status": "FAIL",
            "input_session": str(Path(args.session).expanduser().resolve()),
            "errors": [
                {
                    "stage": "configuration",
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            ],
            "hard_failures": [
                {
                    "stage": "configuration",
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            ],
            "warnings": [],
            "stages": {"configuration": "FAIL"},
        }
        _write_json(report, args.output)
        persist_trace(report)
        raise
    report = audit_session(args.session, config)
    _write_json(report, args.output)
    persist_trace(report)
    if report["status"] == "FAIL":
        raise FisheyeHandposeError("session failed one or more hard audit gates")


def _command_run_item(args: argparse.Namespace) -> None:
    from .h20_executor import H20ExecutorConfig, H20WorkerExecutor
    from .pipeline import PipelineRunRequest, run_data_item

    executor = None
    if args.h20_executor_config:
        executor = H20WorkerExecutor(H20ExecutorConfig.from_file(args.h20_executor_config))
    result = run_data_item(
        PipelineRunRequest(
            session_path=args.session,
            runs_root=args.runs_root,
            audit_config=_audit_config(args),
            item_id=args.item_id,
            run_id=args.run_id,
            pipeline_version=args.pipeline_version,
        ),
        backends=executor,
    )
    _write_json(result.to_dict(), None)
    if result.status.value == "FAILED":
        raise FisheyeHandposeError("data item failed one or more hard pipeline gates")


def _add_audit_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--column", default="timestamp_us")
    parser.add_argument("--timestamp-unit", choices=("ns", "us", "ms"), default="us")
    parser.add_argument("--max-skew-us", required=True, type=_positive_int)
    parser.add_argument("--clock-offset-us", type=int, default=0)
    parser.add_argument(
        "--min-video-bytes",
        type=_nonnegative_int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-pair-count", type=_positive_int, default=20)
    parser.add_argument("--min-overlap-duration-s", type=_positive_float, default=0.75)
    parser.add_argument("--min-overlap-match-rate", type=_unit_interval, default=0.0)
    parser.add_argument("--min-timestamp-fps", type=_positive_float, default=29.5)
    parser.add_argument("--max-timestamp-fps", type=_positive_float, default=30.5)
    parser.add_argument(
        "--max-timestamp-fps-relative-difference",
        type=_unit_interval,
        default=0.001,
    )
    parser.add_argument("--max-p99-skew-us", type=_positive_int, default=250)
    parser.add_argument("--max-observed-skew-us", type=_positive_int, default=500)
    parser.add_argument("--output-width", type=_positive_int, default=1600)
    parser.add_argument("--output-height", type=_positive_int, default=1300)
    parser.add_argument("--balance", type=_unit_interval, default=0.8)
    parser.add_argument("--fov-scale", type=_positive_float, default=1.0)
    parser.add_argument("--min-common-valid-fraction", type=_unit_interval, default=0.80)
    parser.add_argument("--min-camera-valid-fraction", type=_unit_interval, default=0.82)
    parser.add_argument("--min-hfov-deg", type=_positive_float, default=150.0)
    parser.add_argument("--min-vfov-deg", type=_positive_float, default=145.0)
    parser.add_argument("--allow-short-session", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-epipolar-qa", action="store_true")
    parser.add_argument("--epipolar-sample-pairs", type=int, default=12)
    parser.add_argument("--epipolar-min-inliers", type=int, default=60)
    parser.add_argument("--max-median-epipolar-error-px", type=float, default=0.75)
    parser.add_argument("--max-p95-epipolar-error-px", type=float, default=2.0)
    _add_calibration_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fisheye-handpose")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="discover complete stereo sessions")
    discover.add_argument("root")
    discover.add_argument("--output")
    discover.set_defaults(func=_command_discover)

    inspect = subparsers.add_parser("inspect-calibration", help="normalize an Orbbec KB YAML")
    inspect.add_argument("calibration")
    inspect.add_argument("--output")
    _add_calibration_arguments(inspect)
    inspect.set_defaults(func=_command_inspect_calibration)

    pair = subparsers.add_parser("pair-pts", help="pair hardware timestamp CSV streams")
    pair.add_argument("left")
    pair.add_argument("right")
    pair.add_argument("--column", default="timestamp_us")
    pair.add_argument("--timestamp-unit", choices=("ns", "us", "ms"), default="us")
    pair.add_argument("--max-skew-us", required=True, type=_positive_int)
    pair.add_argument("--clock-offset-us", type=int, default=0)
    pair.add_argument("--output", help="write matched pairs as CSV")
    pair.set_defaults(func=_command_pair_pts)

    schema = subparsers.add_parser("schema", help="print the canonical fhp21 contract")
    schema.add_argument("--output")
    schema.set_defaults(func=_command_schema)

    trace_init = subparsers.add_parser("trace-init", help="create an empty, resumable trace run")
    trace_init.add_argument("run_dir")
    trace_init.add_argument("--run-id", type=_nonempty_text)
    trace_init.add_argument("--pipeline-version", type=_nonempty_text, default=__version__)
    trace_init.set_defaults(func=_command_trace_init)

    trace_demo = subparsers.add_parser(
        "trace-demo", help="create a deterministic three-frame trace for viewer inspection"
    )
    trace_demo.add_argument("run_dir")
    trace_demo.add_argument("--run-id", type=_nonempty_text)
    trace_demo.add_argument("--pipeline-version", type=_nonempty_text, default=__version__)
    trace_demo.set_defaults(func=_command_trace_demo)

    trace_validate = subparsers.add_parser(
        "trace-validate", help="verify a trace hash chain and referenced blobs"
    )
    trace_validate.add_argument("run_dir")
    trace_validate.add_argument("--skip-blob-verification", action="store_true")
    trace_validate.set_defaults(func=_command_trace_validate)

    trace_serve = subparsers.add_parser(
        "trace-serve", help="serve the local read-only trace inspection UI"
    )
    trace_serve.add_argument("run_dir")
    trace_serve.add_argument("--host", type=_loopback_host, default="127.0.0.1")
    trace_serve.add_argument("--port", type=_port, default=8000)
    trace_serve.set_defaults(func=_command_trace_serve)

    audit = subparsers.add_parser("audit-session", help="fully preflight one stereo session")
    audit.add_argument("session")
    audit.add_argument("--output", required=True, help="atomic JSON audit report path")
    audit.add_argument(
        "--trace-output",
        help="create an immutable stage trace directory after writing the audit report",
    )
    _add_audit_execution_arguments(audit)
    audit.set_defaults(func=_command_audit_session)

    run_item = subparsers.add_parser(
        "run-item",
        help="run one stereo data item into an immutable catalog directory",
    )
    run_item.add_argument("session")
    run_item.add_argument("--runs-root", required=True)
    run_item.add_argument("--item-id", type=_nonempty_text)
    run_item.add_argument("--run-id", type=_nonempty_text)
    run_item.add_argument("--pipeline-version", type=_nonempty_text, default=__version__)
    run_item.add_argument(
        "--h20-executor-config",
        help="run model, MANO, temporal, and export stages in the isolated H20 worker",
    )
    _add_audit_execution_arguments(run_item)
    run_item.set_defaults(func=_command_run_item)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except FisheyeHandposeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: I/O failure: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
