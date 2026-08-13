"""Persist an ``audit-session`` report as a replayable stage trace."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .trace import RunArtifactWriter, RunStatus, TraceStage, TraceStatus


def _status(value: str | None) -> TraceStatus:
    normalized = (value or "").upper()
    if normalized == "PASS":
        return TraceStatus.SUCCEEDED
    if normalized in {"WARN", "WARNING", "INCONCLUSIVE"}:
        return TraceStatus.WARNING
    if normalized == "SKIPPED":
        return TraceStatus.SKIPPED
    return TraceStatus.FAILED


def _part_status(
    report: dict[str, Any], *, part_number: int, stage_fragments: tuple[str, ...]
) -> TraceStatus:
    matching_errors = [
        item
        for item in report.get("errors", [])
        if item.get("part_number") == part_number
        and any(fragment in str(item.get("stage", "")) for fragment in stage_fragments)
    ]
    if matching_errors:
        return TraceStatus.FAILED
    matching_warnings = [
        item
        for item in report.get("warnings", [])
        if item.get("part_number") == part_number
        and any(fragment in str(item.get("stage", "")) for fragment in stage_fragments)
    ]
    return TraceStatus.WARNING if matching_warnings else TraceStatus.SUCCEEDED


def _report_bytes(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_audit_trace(
    report: dict[str, Any],
    destination: Path,
    *,
    audit_report_path: str | Path | None = None,
    pipeline_version: str = __version__,
) -> Path:
    """Build one complete audit trace at a previously absent staging path.

    The audit already performs full video decoding. Preview extraction is deliberately
    omitted here so enabling tracing cannot trigger a second decode or change audit
    results. Later perception stages can attach their own source/crop overlays as blobs.
    """

    source = str(report.get("input_session") or "unknown-session")
    source_name = Path(source).name or "session"
    report_bytes = _report_bytes(report)
    writer = RunArtifactWriter.create(
        destination,
        run_id=f"audit-{source_name}",
        pipeline_version=pipeline_version,
        config=report.get("config"),
        inputs=[
            {
                "kind": "stereo_session",
                "path": source,
                "audit_report_path": (
                    str(Path(audit_report_path).expanduser().resolve())
                    if audit_report_path is not None
                    else None
                ),
            }
        ],
        metadata={
            "producer": "audit-session",
            "preview_images": {
                "status": "OMITTED",
                "reason": "avoid a second video decode during audit trace persistence",
            },
        },
    )
    record_ids: list[str] = []

    def append(**kwargs: Any) -> str:
        record = writer.append(**kwargs)
        record_ids.append(record.record_id)
        return record.record_id

    try:
        system_id = append(
            record_id="audit:system",
            stage=TraceStage.SYSTEM,
            status=TraceStatus.SUCCEEDED,
            event="audit_configured",
            payload={
                "input_session": source,
                "config": report.get("config"),
                "software": report.get("software"),
            },
        )

        session = report.get("session")
        discovery_id = append(
            record_id="audit:discovery",
            stage=TraceStage.DISCOVERY,
            status=(
                TraceStatus.SUCCEEDED
                if session is not None
                else _status(report.get("stages", {}).get("discovery"))
            ),
            event="session_discovered",
            payload={
                "session": session,
                "errors": [
                    item for item in report.get("errors", []) if item.get("stage") == "discovery"
                ],
            },
            parent_ids=(system_id,),
        )

        calibration_id: str | None = None
        if report.get("calibration") is not None:
            calibration_id = append(
                record_id="audit:calibration",
                stage=TraceStage.CALIBRATION,
                status=_status(report.get("stages", {}).get("calibration", "PASS")),
                event="calibration_normalized",
                payload={"calibration": report["calibration"]},
                parent_ids=(discovery_id,),
            )

        rectification_id: str | None = None
        if report.get("rectification") is not None:
            rectification_id = append(
                record_id="audit:rectification",
                stage=TraceStage.RECTIFICATION,
                status=_status(report.get("stages", {}).get("rectification", "PASS")),
                event="rectification_geometry_built",
                payload={"rectification": report["rectification"]},
                parent_ids=((calibration_id or discovery_id),),
            )

        for part in report.get("parts", []):
            spec = part.get("spec") or {}
            part_number = int(spec.get("part_number", len(record_ids)))
            timestamp_ids: list[str] = []
            for side in ("left", "right"):
                timestamps = (part.get("timestamps") or {}).get(side)
                if timestamps is None:
                    continue
                timestamp_ids.append(
                    append(
                        record_id=f"audit:part:{part_number}:timestamps:{side}",
                        stage=TraceStage.SYNCHRONIZATION,
                        status=_part_status(
                            report,
                            part_number=part_number,
                            stage_fragments=(f"{side}_timestamps", "timestamp_gate"),
                        ),
                        event="timestamp_stream_audited",
                        payload={
                            "part_number": part_number,
                            "view_id": side,
                            "source": spec.get(f"{side}_timestamps"),
                            "timestamps": timestamps,
                        },
                        parent_ids=(discovery_id,),
                    )
                )

            sync_id: str | None = None
            if part.get("sync") is not None:
                sync_id = append(
                    record_id=f"audit:part:{part_number}:sync",
                    stage=TraceStage.SYNCHRONIZATION,
                    status=_part_status(
                        report,
                        part_number=part_number,
                        stage_fragments=("sync", "timestamp_gate"),
                    ),
                    event="stereo_timestamps_paired",
                    payload={"part_number": part_number, "sync": part["sync"]},
                    parent_ids=tuple(timestamp_ids),
                )

            decode_ids: list[str] = []
            for side in ("left", "right"):
                video = (part.get("video") or {}).get(side)
                if video is None:
                    continue
                decode_ids.append(
                    append(
                        record_id=f"audit:part:{part_number}:decode:{side}",
                        stage=TraceStage.DECODE,
                        status=_status(video.get("status")),
                        event="video_stream_audited",
                        payload={
                            "part_number": part_number,
                            "view_id": side,
                            "source": spec.get(f"{side}_video"),
                            "video": video,
                        },
                        parent_ids=((sync_id or discovery_id),),
                    )
                )

            epipolar = part.get("epipolar_qa")
            if epipolar is not None:
                parents = [*decode_ids]
                if rectification_id is not None:
                    parents.append(rectification_id)
                append(
                    record_id=f"audit:part:{part_number}:epipolar",
                    stage=TraceStage.QA,
                    status=_status(epipolar.get("status")),
                    event="epipolar_geometry_evaluated",
                    payload={"part_number": part_number, "epipolar_qa": epipolar},
                    parent_ids=tuple(parents),
                )

        for index, item in enumerate(report.get("errors", [])):
            append(
                record_id=f"audit:gate:error:{index:04d}",
                stage=TraceStage.QA,
                status=TraceStatus.FAILED,
                event="audit_gate_result",
                payload={"severity": "ERROR", **item},
            )
        for index, item in enumerate(report.get("warnings", [])):
            append(
                record_id=f"audit:gate:warning:{index:04d}",
                stage=TraceStage.QA,
                status=TraceStatus.WARNING,
                event="audit_gate_result",
                payload={"severity": "WARNING", **item},
            )

        report_blob = writer.put_blob(
            report_bytes,
            role="audit_report",
            media_type="application/json",
            suffix=".json",
        )
        append(
            record_id="audit:report",
            stage=TraceStage.QA,
            status=_status(report.get("status")),
            event="audit_report_persisted",
            payload={
                "audit_status": report.get("status"),
                "hard_failure_count": len(report.get("hard_failures", [])),
                "warning_count": len(report.get("warnings", [])),
            },
            parent_ids=tuple(record_ids[-8:]),
            blobs=(report_blob,),
        )
        run_status = RunStatus.FAILED if report.get("status") == "FAIL" else RunStatus.COMPLETED
        writer.finalize(
            status=run_status,
            summary={
                "audit_status": report.get("status"),
                "record_count": len(record_ids),
                "part_count": len(report.get("parts", [])),
                "preview_images": "OMITTED",
            },
        )
    except BaseException:
        try:
            writer.finalize(
                status=RunStatus.FAILED,
                summary={"audit_status": report.get("status"), "persistence_failed": True},
            )
        except BaseException:
            writer.close()
        raise
    return destination


def persist_audit_trace(
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    audit_report_path: str | Path | None = None,
    pipeline_version: str = __version__,
) -> Path:
    """Atomically publish an immutable trace derived from a completed audit report."""

    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"trace output already exists: {destination}")
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    ) as temporary_directory:
        staged = Path(temporary_directory) / "run"
        _write_audit_trace(
            report,
            staged,
            audit_report_path=audit_report_path,
            pipeline_version=pipeline_version,
        )
        try:
            staged.rename(destination)
        except FileExistsError:
            raise FileExistsError(f"trace output already exists: {destination}") from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return destination


__all__ = ["persist_audit_trace"]
