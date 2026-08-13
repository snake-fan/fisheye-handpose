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
        raise
    report = audit_session(args.session, config)
    _write_json(report, args.output)
    if report["status"] == "FAIL":
        raise FisheyeHandposeError("session failed one or more hard audit gates")


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

    audit = subparsers.add_parser("audit-session", help="fully preflight one stereo session")
    audit.add_argument("session")
    audit.add_argument("--output", required=True, help="atomic JSON audit report path")
    audit.add_argument("--column", default="timestamp_us")
    audit.add_argument("--timestamp-unit", choices=("ns", "us", "ms"), default="us")
    audit.add_argument("--max-skew-us", required=True, type=_positive_int)
    audit.add_argument("--clock-offset-us", type=int, default=0)
    audit.add_argument(
        "--min-video-bytes",
        type=_nonnegative_int,
        default=0,
        help=argparse.SUPPRESS,
    )
    audit.add_argument("--min-pair-count", type=_positive_int, default=20)
    audit.add_argument("--min-overlap-duration-s", type=_positive_float, default=0.75)
    audit.add_argument("--min-overlap-match-rate", type=_unit_interval, default=0.0)
    audit.add_argument("--min-timestamp-fps", type=_positive_float, default=29.5)
    audit.add_argument("--max-timestamp-fps", type=_positive_float, default=30.5)
    audit.add_argument(
        "--max-timestamp-fps-relative-difference",
        type=_unit_interval,
        default=0.001,
    )
    audit.add_argument("--max-p99-skew-us", type=_positive_int, default=250)
    audit.add_argument("--max-observed-skew-us", type=_positive_int, default=500)
    audit.add_argument("--output-width", type=_positive_int, default=1600)
    audit.add_argument("--output-height", type=_positive_int, default=1300)
    audit.add_argument("--balance", type=_unit_interval, default=0.8)
    audit.add_argument("--fov-scale", type=_positive_float, default=1.0)
    audit.add_argument("--min-common-valid-fraction", type=_unit_interval, default=0.80)
    audit.add_argument("--min-camera-valid-fraction", type=_unit_interval, default=0.82)
    audit.add_argument("--min-hfov-deg", type=_positive_float, default=150.0)
    audit.add_argument("--min-vfov-deg", type=_positive_float, default=145.0)
    audit.add_argument("--allow-short-session", action="store_true", help=argparse.SUPPRESS)
    audit.add_argument("--skip-epipolar-qa", action="store_true")
    audit.add_argument("--epipolar-sample-pairs", type=int, default=12)
    audit.add_argument("--epipolar-min-inliers", type=int, default=60)
    audit.add_argument("--max-median-epipolar-error-px", type=float, default=0.75)
    audit.add_argument("--max-p95-epipolar-error-px", type=float, default=2.0)
    _add_calibration_arguments(audit)
    audit.set_defaults(func=_command_audit_session)
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
