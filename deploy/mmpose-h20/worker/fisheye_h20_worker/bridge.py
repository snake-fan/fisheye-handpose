"""Validate a worker package and expose records for the core trace writer.

This module is standard-library-only and imports neither the CUDA runtime nor the Python
3.11 core. A core ``PipelineStageExecutor`` can load the bundle, copy every ``CoreBlob``
through its existing ``RunArtifactWriter.put_blob``, then append each ``CoreImportRecord``
in order. This preserves the single-writer invariant instead of letting a subprocess
mutate the core run directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import EVENT_SCHEMA, MANIFEST_SCHEMA, SUMMARY_SCHEMA
from .contracts import WorkerError
from .output_contract import validate_pose_estimate

_STAGES = {
    "SYSTEM",
    "RECTIFICATION",
    "SYNCHRONIZATION",
    "DETECTION",
    "POSE_2D",
    "CROSS_VIEW_ASSOCIATION",
    "RAW_FUSION",
    "KINEMATIC_REFINEMENT",
    "TEMPORAL_REFINEMENT",
    "EXPORT",
}
_STATUSES = {"SUCCEEDED", "WARNING", "FAILED", "SKIPPED"}


def _reject_constant(value: str) -> None:
    raise WorkerError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"invalid {label}: {exc}") from exc


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


@dataclass(frozen=True)
class CoreBlob:
    source_path: Path
    relative_path: str
    role: str
    media_type: str
    bytes: int
    sha256: str
    suffix: str


@dataclass(frozen=True)
class CoreImportRecord:
    record_id: str
    stage: str
    status: str
    event: str
    payload: Any
    parent_ids: tuple[str, ...]
    blobs: tuple[CoreBlob, ...]


@dataclass(frozen=True)
class WorkerImportBundle:
    root: Path
    terminal_status: str
    output_status: str
    summary: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    blobs_by_event: tuple[tuple[CoreBlob, ...], ...]
    package_blobs: tuple[CoreBlob, ...]

    def core_records(self, *, external_parent_id: str) -> Iterator[CoreImportRecord]:
        if not isinstance(external_parent_id, str) or not external_parent_id:
            raise WorkerError("external_parent_id must be non-empty")
        for ordinal, (event, blobs) in enumerate(
            zip(self.events, self.blobs_by_event, strict=True)
        ):
            internal_parents = tuple(f"h20:{parent}" for parent in event["parent_event_ids"])
            parents = internal_parents or (external_parent_id,)
            payload = dict(event["payload"])
            payload["worker_provenance"] = {
                "result_dir": str(self.root),
                "event_id": event["event_id"],
                "ordinal": event["ordinal"],
            }
            yield CoreImportRecord(
                record_id=f"h20:{event['event_id']}",
                stage=event["stage"],
                status=event["status"],
                event=event["event"],
                payload=payload,
                parent_ids=parents,
                blobs=((*self.package_blobs, *blobs) if ordinal == 0 else blobs),
            )


def _blob(root: Path, value: Any) -> CoreBlob:
    if not isinstance(value, dict):
        raise WorkerError("worker blob reference must be an object")
    required = {"role", "media_type", "bytes", "sha256", "relative_path"}
    if set(value) != required:
        raise WorkerError("worker blob reference fields are invalid")
    digest = value["sha256"]
    byte_count = value["bytes"]
    relative = value["relative_path"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise WorkerError("worker blob SHA-256 is invalid")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise WorkerError("worker blob byte count is invalid")
    if not isinstance(relative, str) or "\\" in relative:
        raise WorkerError("worker blob relative path is invalid")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise WorkerError("worker blob path traversal is forbidden")
    expected_prefix = f"blobs/sha256/{digest[:2]}/{digest}"
    if not relative.startswith(expected_prefix):
        raise WorkerError("worker blob path is not content-addressed")
    suffix = relative[len(expected_prefix) :]
    if re.fullmatch(r"(?:\.[A-Za-z0-9_-]+)*", suffix) is None:
        raise WorkerError("worker blob suffix is invalid")
    source_path = root.joinpath(*relative_path.parts)
    try:
        source_path.resolve().relative_to(root)
    except ValueError as exc:
        raise WorkerError("worker blob escapes result root") from exc
    if not source_path.is_file():
        raise WorkerError(f"worker blob is missing: {relative}")
    actual_bytes, actual_hash = _sha256(source_path)
    if actual_bytes != byte_count or actual_hash != digest:
        raise WorkerError(f"worker blob identity mismatch: {relative}")
    role = value["role"]
    media_type = value["media_type"]
    if not isinstance(role, str) or not role or not isinstance(media_type, str) or not media_type:
        raise WorkerError("worker blob role/media_type must be non-empty")
    return CoreBlob(
        source_path=source_path,
        relative_path=relative,
        role=role,
        media_type=media_type,
        bytes=byte_count,
        sha256=digest,
        suffix=suffix,
    )


def _package_blob(root: Path, filename: str, role: str) -> CoreBlob:
    source_path = root / filename
    byte_count, digest = _sha256(source_path)
    suffix = source_path.suffix
    return CoreBlob(
        source_path=source_path,
        relative_path=filename,
        role=role,
        media_type=("application/x-ndjson" if suffix == ".jsonl" else "application/json"),
        bytes=byte_count,
        sha256=digest,
        suffix=suffix,
    )


def _validate_named_output(
    root: Path,
    artifact: Any,
    *,
    role: str,
    label: str,
) -> None:
    if not isinstance(artifact, dict):
        raise WorkerError(f"worker summary {label} is missing")
    if set(artifact) != {
        "role",
        "media_type",
        "bytes",
        "sha256",
        "relative_path",
    }:
        raise WorkerError(f"worker summary {label} fields are invalid")
    if (
        artifact.get("role") != role
        or artifact.get("media_type") != "application/x-ndjson"
        or artifact.get("relative_path") != "fhp21.jsonl"
    ):
        raise WorkerError(f"worker summary {label} contract is invalid")
    output_path = root / "fhp21.jsonl"
    if not output_path.is_file():
        raise WorkerError("worker final fhp21 output is missing")
    actual_bytes, actual_hash = _sha256(output_path)
    if artifact.get("bytes") != actual_bytes or artifact.get("sha256") != actual_hash:
        raise WorkerError("worker final fhp21 output identity mismatch")


def _validate_fhp21_records(root: Path, summary: dict[str, Any]) -> None:
    output_path = root / "fhp21.jsonl"
    try:
        data = output_path.read_bytes()
    except OSError as exc:
        raise WorkerError(f"cannot read worker fhp21 output: {exc}") from exc
    if not data or not data.endswith(b"\n"):
        raise WorkerError("worker fhp21 output must be non-empty newline-terminated JSONL")
    records: list[dict[str, Any]] = []
    estimate_ids: set[str] = set()
    previous_timestamp_by_track: dict[tuple[str, str], int] = {}
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            raise WorkerError(f"fhp21 line {line_number} is blank")
        record = validate_pose_estimate(
            _load_json(line, f"fhp21 line {line_number}"),
            line_number=line_number,
        )
        estimate_id = record["estimate_id"]
        if estimate_id in estimate_ids:
            raise WorkerError(f"fhp21 line {line_number} duplicates estimate_id")
        estimate_ids.add(estimate_id)
        state_key = (record["sequence_id"], record["track_id"])
        previous_timestamp = previous_timestamp_by_track.get(state_key)
        if previous_timestamp is not None and record["timestamp_ns"] <= previous_timestamp:
            raise WorkerError(f"fhp21 line {line_number} has a non-increasing track timestamp")
        previous_timestamp_by_track[state_key] = record["timestamp_ns"]
        records.append(record)
    export_count = summary.get("export_count")
    if (
        isinstance(export_count, bool)
        or not isinstance(export_count, int)
        or export_count != len(records)
    ):
        raise WorkerError("worker summary export_count does not match fhp21.jsonl")


def load_import_bundle(result_dir: str | Path) -> WorkerImportBundle:
    root = Path(result_dir).expanduser().resolve()
    if not root.is_dir():
        raise WorkerError(f"worker result directory does not exist: {root}")
    try:
        manifest = _load_json((root / "manifest.json").read_bytes(), "worker manifest")
        summary = _load_json((root / "summary.json").read_bytes(), "worker summary")
        trace_bytes = (root / "events.jsonl").read_bytes()
    except OSError as exc:
        raise WorkerError(f"worker result package is incomplete: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise WorkerError("unexpected worker manifest schema")
    if manifest.get("status") != "ACTIVE":
        raise WorkerError("worker manifest must retain immutable ACTIVE status")
    if not isinstance(summary, dict) or summary.get("schema_version") != SUMMARY_SCHEMA:
        raise WorkerError("unexpected worker summary schema")
    terminal_status = summary.get("status")
    if terminal_status not in {"COMPLETED", "FAILED"}:
        raise WorkerError("worker summary terminal status is invalid")
    output_status = summary.get("output_status")
    if output_status not in {"PRODUCED", "NOT_PRODUCED"}:
        raise WorkerError("worker summary output_status is invalid")
    output_file = summary.get("output_file")
    if output_status == "PRODUCED" and output_file != "fhp21.jsonl":
        raise WorkerError("worker produced status requires fhp21.jsonl")
    if terminal_status == "FAILED" and output_status != "NOT_PRODUCED":
        raise WorkerError("failed worker package cannot claim produced output")
    if (
        terminal_status == "COMPLETED"
        and output_status == "NOT_PRODUCED"
        and output_file is not None
    ):
        raise WorkerError("worker NOT_PRODUCED status cannot declare an output file")
    if output_file is not None:
        if output_file != "fhp21.jsonl":
            raise WorkerError("worker summary output_file is invalid")
        _validate_named_output(
            root,
            summary.get("output_artifact"),
            role="fhp21_output",
            label="output_artifact",
        )
    partial_artifact = summary.get("partial_output_artifact")
    if partial_artifact is not None:
        if terminal_status != "FAILED" or output_file is not None:
            raise WorkerError("partial fhp21 output is only valid for a failed package")
        _validate_named_output(
            root,
            partial_artifact,
            role="partial_fhp21_output",
            label="partial_output_artifact",
        )
    elif output_file is None and (root / "fhp21.jsonl").exists():
        raise WorkerError("undeclared worker fhp21 output is forbidden")
    if (root / "fhp21.jsonl").is_file():
        _validate_fhp21_records(root, summary)

    events: list[dict[str, Any]] = []
    blobs_by_event: list[tuple[CoreBlob, ...]] = []
    known_ids: set[str] = set()
    for ordinal, line in enumerate(trace_bytes.splitlines()):
        if not line.strip():
            raise WorkerError("worker events contain a blank line")
        event = _load_json(line, f"worker event line {ordinal + 1}")
        if not isinstance(event, dict) or event.get("schema_version") != EVENT_SCHEMA:
            raise WorkerError("unexpected worker event schema")
        if event.get("ordinal") != ordinal:
            raise WorkerError("worker event ordinal sequence is invalid")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in known_ids:
            raise WorkerError("worker event ID is invalid or duplicated")
        if event.get("stage") not in _STAGES or event.get("status") not in _STATUSES:
            raise WorkerError("worker event stage/status is invalid")
        if not isinstance(event.get("event"), str) or not event["event"]:
            raise WorkerError("worker event name is invalid")
        parents = event.get("parent_event_ids")
        if not isinstance(parents, list) or len(set(parents)) != len(parents):
            raise WorkerError("worker parent_event_ids are invalid")
        if any(parent not in known_ids for parent in parents):
            raise WorkerError("worker event references an unknown or forward parent")
        if not isinstance(event.get("payload"), dict):
            raise WorkerError("worker event payload must be an object")
        raw_blobs = event.get("blobs")
        if not isinstance(raw_blobs, list):
            raise WorkerError("worker event blobs must be a list")
        blobs_by_event.append(tuple(_blob(root, value) for value in raw_blobs))
        events.append(event)
        known_ids.add(event_id)
    if summary.get("event_count") != len(events):
        raise WorkerError("worker summary event_count does not match events.jsonl")
    package_blobs = [
        _package_blob(root, "manifest.json", "worker_manifest"),
        _package_blob(root, "events.jsonl", "worker_events"),
        _package_blob(root, "summary.json", "worker_summary"),
    ]
    if (root / "fhp21.jsonl").is_file():
        package_blobs.append(
            _package_blob(
                root,
                "fhp21.jsonl",
                (
                    "worker_fhp21_output"
                    if output_file is not None
                    else "worker_partial_fhp21_output"
                ),
            )
        )
    return WorkerImportBundle(
        root=root,
        terminal_status=terminal_status,
        output_status=output_status,
        summary=summary,
        events=tuple(events),
        blobs_by_event=tuple(blobs_by_event),
        package_blobs=tuple(package_blobs),
    )


__all__ = [
    "CoreBlob",
    "CoreImportRecord",
    "WorkerImportBundle",
    "load_import_bundle",
]
