#!/usr/bin/env python3
"""Export canonical fisheye-handpose overlay artifacts as named MP4 files.

The canonical run format stores videos as content-addressed blobs. This script
finds the exact successful EXPORT record in each completed run, verifies the
blob hash and video contract, and materializes one ``<item_id>.mp4`` per run in
a new output directory. Source runs are never modified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

from fisheye_handpose._generated_project_contract import MP4_EXPORT_SCHEMA
from fisheye_handpose.trace import (
    RunArtifactReader,
    RunStatus,
    TraceStage,
    TraceStatus,
    TraceValidationError,
)

EXPORT_EVENT = "raw_vs_stable_overlay_video_exported"
VIDEO_ROLE = "overlay_video_raw_vs_stable_stereo_rectified"
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
FFPROBE_RATIONAL = re.compile(r"^[0-9]+/[1-9][0-9]*$")
FFPROBE_INTEGER = re.compile(r"^[0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    return parser.parse_args()


def validated_video_record(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the canonical run and return its one successful video export."""

    try:
        reader = RunArtifactReader(run_dir)
        report = reader.validate(verify_blobs=True)
    except (OSError, TraceValidationError) as exc:
        raise ValueError(f"canonical run validation failed: {run_dir}: {exc}") from exc
    if report.status is not RunStatus.COMPLETED:
        raise ValueError(f"run is not COMPLETED: {run_dir}")
    summary = reader.summary
    if summary is None:
        raise ValueError(f"completed run has no summary: {run_dir}")
    records = reader.records(
        stage=TraceStage.EXPORT,
        status=TraceStatus.SUCCEEDED,
        event=EXPORT_EVENT,
    )
    matches = [
        (record, blob) for record in records for blob in record.blobs if blob.role == VIDEO_ROLE
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {VIDEO_ROLE!r} artifact, found {len(matches)}: {run_dir}")
    record, blob = matches[0]
    return summary, record.to_dict(), blob.to_dict()


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Return canonical runs without following item/run directory symlinks."""

    root = runs_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"runs root is not a directory: {root}")

    run_dirs: list[Path] = []
    for summary_path in sorted(root.glob("*/*/run_summary.json")):
        run_dir = summary_path.parent
        item_dir = run_dir.parent
        if item_dir.is_symlink():
            raise ValueError(f"symlinked item directory is not allowed: {item_dir}")
        if run_dir.is_symlink():
            raise ValueError(f"symlinked run directory is not allowed: {run_dir}")

        try:
            resolved_run = run_dir.resolve(strict=True)
            resolved_run.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"run directory escapes the runs root: {run_dir}") from exc

        manifest_path = run_dir / "run_manifest.json"
        trace_path = run_dir / "trace.jsonl"
        if summary_path.is_symlink() or manifest_path.is_symlink() or trace_path.is_symlink():
            raise ValueError(f"symlinked canonical metadata is not allowed: {run_dir}")
        if not trace_path.is_file():
            raise ValueError(f"canonical run has no trace.jsonl: {run_dir}")
        run_dirs.append(resolved_run)
    return run_dirs


def resolve_blob(run_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"invalid blob relative_path in {run_dir}")
    run_root = run_dir.resolve()
    blob_path = (run_root / relative_path).resolve(strict=True)
    try:
        blob_path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"blob escapes its run directory: {relative_path}") from exc
    if not blob_path.is_file():
        raise ValueError(f"blob is not a regular file: {blob_path}")
    return blob_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, *, label: str, path: Path) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid {label} reported by ffprobe: {value!r}: {path}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and FFPROBE_INTEGER.fullmatch(value) is not None:
        parsed = int(value)
    else:
        raise ValueError(f"invalid {label} reported by ffprobe: {value!r}: {path}")
    if parsed <= 0:
        raise ValueError(f"invalid {label} reported by ffprobe: {value!r}: {path}")
    return parsed


def _positive_rational(value: Any, *, label: str) -> Fraction:
    if isinstance(value, str):
        if FFPROBE_RATIONAL.fullmatch(value) is None:
            raise ValueError(f"invalid {label} rational: {value!r}")
        numerator_text, denominator_text = value.split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    elif isinstance(value, dict):
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator <= 0
        ):
            raise ValueError(f"invalid {label} rational: {value!r}")
    else:
        raise ValueError(f"invalid {label} rational: {value!r}")
    fraction = Fraction(numerator, denominator)
    if fraction <= 0:
        raise ValueError(f"invalid {label} rational: {value!r}")
    return fraction


def probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,"
                "r_frame_rate,time_base,nb_frames:format=format_name,duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"ffprobe returned a non-object payload for {path}")
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError(f"expected exactly one total media stream: {path}")
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        raise ValueError(f"expected the only media stream to be video: {path}")
    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        raise ValueError(f"ffprobe returned invalid format metadata for {path}")
    container = str(format_info.get("format_name", ""))
    frame_rate = _positive_rational(stream.get("avg_frame_rate"), label="frame_rate")
    nominal_frame_rate = _positive_rational(stream.get("r_frame_rate"), label="nominal frame_rate")
    if nominal_frame_rate != frame_rate:
        raise ValueError(
            f"expected a constant frame rate, found avg={frame_rate} "
            f"nominal={nominal_frame_rate}: {path}"
        )
    _positive_rational(stream.get("time_base"), label="time_base")
    duration_value = format_info.get("duration")
    try:
        duration_seconds = float(duration_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid duration reported by ffprobe: {duration_value!r}: {path}"
        ) from exc
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(f"invalid duration reported by ffprobe: {duration_value!r}: {path}")
    info = {
        "codec": str(stream.get("codec_name", "")),
        "pixel_format": str(stream.get("pix_fmt", "")),
        "width": _positive_int(stream.get("width"), label="width", path=path),
        "height": _positive_int(stream.get("height"), label="height", path=path),
        "frame_rate": str(stream["avg_frame_rate"]),
        "time_base": str(stream["time_base"]),
        "frame_count": _positive_int(stream.get("nb_frames"), label="frame_count", path=path),
        "duration_seconds": duration_seconds,
        "container": container,
    }
    if info["codec"] != "h264":
        raise ValueError(f"expected H.264, found {info['codec']!r}: {path}")
    if info["pixel_format"] != "yuv420p":
        raise ValueError(f"expected yuv420p, found {info['pixel_format']!r}: {path}")
    if "mp4" not in {name.strip() for name in container.split(",")}:
        raise ValueError(f"expected an MP4 container, found {container!r}: {path}")
    return info


def verify_contract(record: dict[str, Any], video: dict[str, Any], run_dir: Path) -> None:
    if record.get("status") != "SUCCEEDED":
        raise ValueError(f"video EXPORT record is not SUCCEEDED: {run_dir}")
    payload = record.get("payload")
    if not isinstance(payload, dict) or payload.get("output_status") != "PRODUCED":
        raise ValueError(f"video EXPORT payload is not PRODUCED: {run_dir}")
    expected = {
        "codec": "h264",
        "container": "mp4",
        "pixel_format": "yuv420p",
        "width": video["width"],
        "height": video["height"],
        "frame_count": video["frame_count"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"trace/video mismatch for {key}: trace={payload.get(key)!r}, "
                f"video={value!r}: {run_dir}"
            )
    for key in ("frame_rate", "time_base"):
        try:
            trace_value = _positive_rational(payload.get(key), label=f"trace {key}")
            video_value = _positive_rational(video.get(key), label=f"video {key}")
        except ValueError as exc:
            raise ValueError(f"invalid {key} contract in {run_dir}: {exc}") from exc
        if trace_value != video_value:
            raise ValueError(
                f"trace/video mismatch for {key}: trace={trace_value} "
                f"video={video_value}: {run_dir}"
            )


def materialize(source: Path, destination: Path) -> str:
    """Create an independent copy so consumers cannot mutate the canonical blob."""

    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
        destination.unlink(missing_ok=True)
        raise ValueError(f"materialized video failed integrity verification: {destination}")
    return "copy"


def export_one(run_dir: Path, temporary_dir: Path, ffprobe: str) -> dict[str, Any]:
    summary, record, blob = validated_video_record(run_dir)
    details = summary.get("summary")
    if not isinstance(details, dict):
        raise ValueError(f"run summary details are missing: {run_dir}")
    item_id = details.get("item_id")
    run_id = summary.get("run_id")
    if item_id != run_dir.parent.name or run_id != run_dir.name:
        raise ValueError(f"run summary identity does not match its directory: {run_dir}")
    if not isinstance(item_id, str) or not SAFE_NAME.fullmatch(item_id):
        raise ValueError(f"item_id is not safe as a filename: {item_id!r}")

    if blob.get("media_type") != "video/mp4":
        raise ValueError(f"unexpected video media type in {run_dir}: {blob.get('media_type')}")
    source = resolve_blob(run_dir, blob.get("relative_path"))
    actual_bytes = source.stat().st_size
    actual_sha256 = sha256(source)
    if blob.get("bytes") != actual_bytes or blob.get("sha256") != actual_sha256:
        raise ValueError(f"video blob size or SHA-256 mismatch: {source}")

    video = probe_video(source, ffprobe)
    verify_contract(record, video, run_dir)
    destination = temporary_dir / f"{item_id}.mp4"
    if destination.exists():
        raise ValueError(f"duplicate item_id would overwrite an export: {item_id}")
    method = materialize(source, destination)
    return {
        "item_id": item_id,
        "run_id": run_id,
        "run_output_status": details.get("output_status"),
        "source_run_dir": str(run_dir),
        "source_relative_path": blob["relative_path"],
        "output_filename": destination.name,
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "materialization": method,
        "video": video,
    }


def main() -> int:
    args = parse_args()
    try:
        runs_root = args.runs_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"runs root does not exist: {args.runs_root}") from exc
    output_dir = args.output_dir.expanduser().resolve()
    ffprobe = shutil.which(args.ffprobe)
    if not runs_root.is_dir():
        raise SystemExit(f"runs root is not a directory: {runs_root}")
    if ffprobe is None:
        raise SystemExit(f"ffprobe executable not found: {args.ffprobe}")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    if runs_root == output_dir or runs_root in output_dir.parents:
        raise SystemExit("output directory must not be inside the runs root")

    try:
        run_dirs = discover_run_dirs(runs_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not run_dirs:
        raise SystemExit(f"no canonical runs found below {runs_root}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.partial")
    temporary_dir.mkdir()
    entries: list[dict[str, Any]] = []
    try:
        for index, run_dir in enumerate(run_dirs, 1):
            entry = export_one(run_dir, temporary_dir, ffprobe)
            entries.append(entry)
            print(f"[{index}/{len(run_dirs)}] {entry['output_filename']}", flush=True)
        manifest = {
            "schema_version": MP4_EXPORT_SCHEMA,
            "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "runs_root": str(runs_root),
            "video_count": len(entries),
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "entries": entries,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "video_count": len(entries),
                "total_bytes": sum(entry["bytes"] for entry in entries),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
