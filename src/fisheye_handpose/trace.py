"""Append-only, tamper-evident artifacts for inspecting pipeline stages.

The trace is intentionally a small filesystem protocol.  JSONL records form a SHA-256
chain, large artifacts live in content-addressed files, and a separate immutable summary
closes a run.  An unfinished run can be safely reopened by another process after the
previous writer releases its advisory lock.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

RUN_MANIFEST_SCHEMA = "fisheye-handpose/run-manifest/v1"
TRACE_RECORD_SCHEMA = "fisheye-handpose/trace-record/v1"
RUN_SUMMARY_SCHEMA = "fisheye-handpose/run-summary/v1"
HASH_ALGORITHM = "sha256"


class TraceStage(StrEnum):
    SYSTEM = "SYSTEM"
    DISCOVERY = "DISCOVERY"
    CALIBRATION = "CALIBRATION"
    DECODE = "DECODE"
    SYNCHRONIZATION = "SYNCHRONIZATION"
    RECTIFICATION = "RECTIFICATION"
    DETECTION = "DETECTION"
    POSE_2D = "POSE_2D"
    CROSS_VIEW_ASSOCIATION = "CROSS_VIEW_ASSOCIATION"
    RAW_FUSION = "RAW_FUSION"
    KINEMATIC_REFINEMENT = "KINEMATIC_REFINEMENT"
    TEMPORAL_REFINEMENT = "TEMPORAL_REFINEMENT"
    QA = "QA"
    EXPORT = "EXPORT"


class TraceStatus(StrEnum):
    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


class RunStatus(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TraceValidationError(ValueError):
    """The on-disk trace violates the v1 filesystem protocol."""


@dataclass(frozen=True, slots=True)
class BlobRef:
    sha256: str
    bytes: int
    role: str
    media_type: str
    relative_path: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("blob sha256 must be 64 lowercase hexadecimal characters")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise ValueError("blob bytes must be a non-negative integer")
        _require_text(self.role, "blob role")
        _require_text(self.media_type, "blob media_type")
        _validate_relative_path(self.relative_path)
        expected_prefix = f"blobs/sha256/{self.sha256[:2]}/{self.sha256}"
        if not self.relative_path.startswith(expected_prefix):
            raise ValueError("blob path is not content-addressed by its sha256")
        suffix = self.relative_path[len(expected_prefix) :]
        _validate_suffix(suffix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "bytes": self.bytes,
            "role": self.role,
            "media_type": self.media_type,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, value: Any) -> BlobRef:
        if not isinstance(value, dict) or set(value) != {
            "sha256",
            "bytes",
            "role",
            "media_type",
            "relative_path",
        }:
            raise TraceValidationError("invalid blob reference")
        try:
            return cls(
                sha256=value["sha256"],
                bytes=value["bytes"],
                role=value["role"],
                media_type=value["media_type"],
                relative_path=value["relative_path"],
            )
        except (TypeError, ValueError) as error:
            raise TraceValidationError(f"invalid blob reference: {error}") from error


@dataclass(frozen=True, slots=True)
class TraceRecord:
    schema_version: str
    ordinal: int
    record_id: str
    timestamp_utc: str
    stage: TraceStage
    status: TraceStatus
    event: str
    parent_ids: tuple[str, ...]
    blobs: tuple[BlobRef, ...]
    payload: Any
    previous_hash: str | None
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "record_id": self.record_id,
            "timestamp_utc": self.timestamp_utc,
            "stage": self.stage.value,
            "status": self.status.value,
            "event": self.event,
            "parent_ids": list(self.parent_ids),
            "blobs": [blob.to_dict() for blob in self.blobs],
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    ok: bool
    run_id: str
    status: RunStatus
    record_count: int
    blob_count: int
    last_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "run_id": self.run_id,
            "status": self.status.value,
            "record_count": self.record_count,
            "blob_count": self.blob_count,
            "last_hash": self.last_hash,
        }


def _require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _reject_constant(value: str) -> None:
    raise TraceValidationError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"invalid {label}: {error}") from error


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite JSON number")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string JSON object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_record_body(body: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _validate_relative_path(value: Any) -> None:
    _require_text(value, "relative_path")
    if "\\" in value:
        raise ValueError("relative_path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("path traversal is forbidden")


def _validate_suffix(suffix: Any) -> None:
    if not isinstance(suffix, str) or not re.fullmatch(r"(?:\.[A-Za-z0-9_-]+)*", suffix):
        raise ValueError("blob suffix must be empty or dot-prefixed alphanumeric components")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes(path: Path, data: bytes) -> None:
    """Atomically publish bytes without ever replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"artifact already exists: {path}") from None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Any) -> None:
    _write_new_bytes(path, _canonical_json(value) + b"\n")


def _safe_artifact_path(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    resolved_root = root.resolve()
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise TraceValidationError(f"artifact escapes run root: {relative_path}") from error
    return candidate


class RunArtifactWriter:
    """Exclusive append-only writer for one run directory."""

    def __init__(self) -> None:
        raise TypeError("use RunArtifactWriter.create() or RunArtifactWriter.open()")

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        run_id: str,
        pipeline_version: str,
        config: Any = None,
        inputs: Any = None,
        metadata: Any = None,
    ) -> RunArtifactWriter:
        _require_text(run_id, "run_id")
        _require_text(pipeline_version, "pipeline_version")
        config = {} if config is None else config
        inputs = [] if inputs is None else inputs
        metadata = {} if metadata is None else metadata
        for value in (config, inputs, metadata):
            _validate_json_value(value)

        run_root = Path(root)
        run_root.mkdir(parents=True, exist_ok=False)
        instance = cls.__new__(cls)
        instance._initialize(run_root)
        try:
            instance._acquire_lock()
            manifest_body = {
                "schema_version": RUN_MANIFEST_SCHEMA,
                "run_id": run_id,
                "status": RunStatus.ACTIVE.value,
                "pipeline_version": pipeline_version,
                "created_at_utc": _utc_now(),
                "config": config,
                "inputs": inputs,
                "metadata": metadata,
                "artifacts": {
                    "trace": "trace.jsonl",
                    "summary": "run_summary.json",
                    "blob_root": "blobs/sha256",
                },
                "trace_schema_version": TRACE_RECORD_SCHEMA,
                "hash_algorithm": HASH_ALGORITHM,
            }
            manifest = {
                **manifest_body,
                "manifest_hash": _hash_record_body(manifest_body),
            }
            _write_new_json(instance._manifest_path, manifest)
            with instance._trace_path.open("xb") as trace_file:
                trace_file.flush()
                os.fsync(trace_file.fileno())
            _fsync_directory(instance.root)
            instance._manifest = manifest
            return instance
        except BaseException:
            instance.close()
            raise

    @classmethod
    def open(cls, root: str | Path) -> RunArtifactWriter:
        instance = cls.__new__(cls)
        instance._initialize(Path(root))
        instance._acquire_lock()
        try:
            reader = RunArtifactReader(instance.root)
            if reader.summary is not None:
                raise RuntimeError("a finalized run cannot be reopened")
            if reader.manifest.get("status") != RunStatus.ACTIVE.value:
                raise TraceValidationError("only an ACTIVE run can be reopened")
            reader.validate()
            records = reader.records()
            instance._manifest = reader.manifest
            instance._record_ids = {record.record_id for record in records}
            instance._next_ordinal = len(records)
            instance._previous_hash = records[-1].record_hash if records else None
            return instance
        except BaseException:
            instance.close()
            raise

    def _initialize(self, root: Path) -> None:
        self.root = root.resolve()
        self._manifest_path = self.root / "run_manifest.json"
        self._trace_path = self.root / "trace.jsonl"
        self._summary_path = self.root / "run_summary.json"
        self._lock_path = self.root / ".writer.lock"
        self._lock_file: BinaryIO | None = None
        self._closed = False
        self._finalized = False
        self._record_ids: set[str] = set()
        self._next_ordinal = 0
        self._previous_hash: str | None = None
        self._manifest: dict[str, Any] = {}

    def _acquire_lock(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {self.root}")
        lock_file = self._lock_path.open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            raise RuntimeError(f"run already has an active writer: {self.root}") from None
        self._lock_file = lock_file

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def append(
        self,
        *,
        record_id: str,
        stage: TraceStage,
        status: TraceStatus,
        event: str,
        payload: Any = None,
        parent_ids: Iterable[str] = (),
        blobs: Iterable[BlobRef] = (),
    ) -> TraceRecord:
        self._require_active()
        _require_text(record_id, "record_id")
        _require_text(event, "event")
        if not isinstance(stage, TraceStage):
            raise TypeError("stage must be a TraceStage")
        if not isinstance(status, TraceStatus):
            raise TypeError("status must be a TraceStatus")
        if record_id in self._record_ids:
            raise ValueError(f"duplicate trace record ID: {record_id}")
        parents = tuple(parent_ids)
        if any(not isinstance(parent, str) or not parent.strip() for parent in parents):
            raise ValueError("parent IDs must be non-empty strings")
        if len(set(parents)) != len(parents):
            raise ValueError("parent IDs must be unique")
        unknown_parents = [parent for parent in parents if parent not in self._record_ids]
        if unknown_parents:
            raise ValueError(f"unknown parent record IDs: {unknown_parents}")
        blob_refs = tuple(blobs)
        if any(not isinstance(blob, BlobRef) for blob in blob_refs):
            raise TypeError("blobs must contain BlobRef values")
        for blob in blob_refs:
            self._verify_blob(blob)
        _validate_json_value(payload)

        body = {
            "schema_version": TRACE_RECORD_SCHEMA,
            "ordinal": self._next_ordinal,
            "record_id": record_id,
            "timestamp_utc": _utc_now(),
            "stage": stage.value,
            "status": status.value,
            "event": event,
            "parent_ids": list(parents),
            "blobs": [blob.to_dict() for blob in blob_refs],
            "payload": payload,
            "previous_hash": self._previous_hash,
        }
        record_hash = _hash_record_body(body)
        encoded = _canonical_json({**body, "record_hash": record_hash}) + b"\n"
        with self._trace_path.open("ab") as trace_file:
            trace_file.write(encoded)
            trace_file.flush()
            os.fsync(trace_file.fileno())
        record = _record_from_dict({**body, "record_hash": record_hash})
        self._record_ids.add(record_id)
        self._next_ordinal += 1
        self._previous_hash = record_hash
        return record

    def put_blob(
        self,
        data: bytes,
        *,
        role: str,
        media_type: str,
        suffix: str = "",
    ) -> BlobRef:
        self._require_active()
        if not isinstance(data, bytes):
            raise TypeError("blob data must be bytes")
        _require_text(role, "role")
        _require_text(media_type, "media_type")
        _validate_suffix(suffix)
        digest = hashlib.sha256(data).hexdigest()
        relative_path = f"blobs/sha256/{digest[:2]}/{digest}{suffix}"
        path = _safe_artifact_path(self.root, relative_path)
        reference = BlobRef(digest, len(data), role, media_type, relative_path)
        if path.exists():
            self._verify_blob(reference)
            return reference
        try:
            _write_new_bytes(path, data)
        except FileExistsError:
            self._verify_blob(reference)
        return reference

    def put_blob_file(
        self,
        source_path: str | Path,
        *,
        role: str,
        media_type: str,
        suffix: str = "",
    ) -> BlobRef:
        """Stream a file into the content-addressed store without loading it in memory."""

        self._require_active()
        _require_text(role, "role")
        _require_text(media_type, "media_type")
        _validate_suffix(suffix)
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"blob source is not a file: {source}")

        incoming_root = self.root / "blobs" / "sha256"
        incoming_root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                source.open("rb") as source_file,
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=".incoming-",
                    suffix=".tmp",
                    dir=incoming_root,
                    delete=False,
                ) as destination,
            ):
                temporary = Path(destination.name)
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())

            sha256 = digest.hexdigest()
            relative_path = f"blobs/sha256/{sha256[:2]}/{sha256}{suffix}"
            path = _safe_artifact_path(self.root, relative_path)
            reference = BlobRef(sha256, size, role, media_type, relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(temporary, path)
                _fsync_directory(path.parent)
            except FileExistsError:
                self._verify_blob(reference)
            return reference
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _verify_blob(self, blob: BlobRef) -> None:
        path = _safe_artifact_path(self.root, blob.relative_path)
        if not path.is_file():
            raise TraceValidationError(f"missing blob: {blob.relative_path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as blob_file:
            for chunk in iter(lambda: blob_file.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != blob.bytes:
            raise TraceValidationError(f"blob size mismatch: {blob.relative_path}")
        if digest.hexdigest() != blob.sha256:
            raise TraceValidationError(f"blob hash mismatch: {blob.relative_path}")

    def finalize(
        self,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        summary: Any = None,
    ) -> dict[str, Any]:
        self._require_active()
        if not isinstance(status, RunStatus) or status is RunStatus.ACTIVE:
            raise ValueError("final status must be RunStatus.COMPLETED or RunStatus.FAILED")
        summary = {} if summary is None else summary
        _validate_json_value(summary)
        body = {
            "schema_version": RUN_SUMMARY_SCHEMA,
            "run_id": self._manifest["run_id"],
            "status": status.value,
            "finalized_at_utc": _utc_now(),
            "record_count": self._next_ordinal,
            "last_record_hash": self._previous_hash,
            "summary": summary,
        }
        value = {**body, "summary_hash": _hash_record_body(body)}
        _write_new_json(self._summary_path, value)
        self._finalized = True
        self.close()
        return value

    def _require_active(self) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._finalized:
            raise RuntimeError("run is finalized")

    def close(self) -> None:
        if self._closed:
            return
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
        self._closed = True

    def __enter__(self) -> RunArtifactWriter:
        self._require_active()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if self._closed:
            return False
        if exc is None:
            self.finalize(status=RunStatus.COMPLETED)
        else:
            self.finalize(
                status=RunStatus.FAILED,
                summary={"error": {"type": type(exc).__name__, "message": str(exc)}},
            )
        return False


class RunArtifactReader:
    """Read and verify a run without mutating it."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {self.root}")
        self._manifest_path = self.root / "run_manifest.json"
        self._trace_path = self.root / "trace.jsonl"
        self._summary_path = self.root / "run_summary.json"
        self._manifest = self._load_json_file(self._manifest_path, "run manifest")
        self._validate_manifest()

    @staticmethod
    def _load_json_file(path: Path, label: str) -> Any:
        try:
            return _load_json_bytes(path.read_bytes(), label)
        except FileNotFoundError:
            raise TraceValidationError(f"missing {label}: {path.name}") from None

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    @property
    def summary(self) -> dict[str, Any] | None:
        if not self._summary_path.exists():
            return None
        value = self._load_json_file(self._summary_path, "run summary")
        if not isinstance(value, dict):
            raise TraceValidationError("run summary must be a JSON object")
        return value

    def records(
        self,
        *,
        stage: TraceStage | None = None,
        status: TraceStatus | None = None,
        event: str | None = None,
    ) -> tuple[TraceRecord, ...]:
        if stage is not None and not isinstance(stage, TraceStage):
            raise TypeError("stage filter must be a TraceStage")
        if status is not None and not isinstance(status, TraceStatus):
            raise TypeError("status filter must be a TraceStatus")
        records = tuple(record for record, _ in self._read_record_entries())
        return tuple(
            record
            for record in records
            if (stage is None or record.stage is stage)
            and (status is None or record.status is status)
            and (event is None or record.event == event)
        )

    def get(self, record_id: str) -> TraceRecord:
        for record in self.records():
            if record.record_id == record_id:
                return record
        raise KeyError(record_id)

    def validate(self, *, verify_blobs: bool = True) -> ValidationReport:
        entries = self._read_record_entries()
        record_ids: set[str] = set()
        previous_hash: str | None = None
        blobs: dict[tuple[str, str], BlobRef] = {}
        for expected_ordinal, (record, raw) in enumerate(entries):
            if record.ordinal != expected_ordinal:
                raise TraceValidationError(f"trace ordinal mismatch at line {expected_ordinal + 1}")
            if record.record_id in record_ids:
                raise TraceValidationError(f"duplicate trace record ID: {record.record_id}")
            if record.previous_hash != previous_hash:
                raise TraceValidationError(f"broken hash chain at record {record.record_id}")
            unknown_parents = [parent for parent in record.parent_ids if parent not in record_ids]
            if unknown_parents:
                raise TraceValidationError(
                    f"record {record.record_id} has unknown/forward parent IDs: {unknown_parents}"
                )
            raw_body = dict(raw)
            stored_hash = raw_body.pop("record_hash")
            if _hash_record_body(raw_body) != stored_hash:
                raise TraceValidationError(f"record hash mismatch: {record.record_id}")
            record_ids.add(record.record_id)
            previous_hash = record.record_hash
            for blob in record.blobs:
                blobs[(blob.sha256, blob.relative_path)] = blob

        if verify_blobs:
            for blob in blobs.values():
                self._verify_blob(blob)

        summary = self.summary
        if summary is None:
            run_status = RunStatus.ACTIVE
        else:
            run_status = self._validate_summary(summary, len(entries), previous_hash)
        return ValidationReport(
            ok=True,
            run_id=self._manifest["run_id"],
            status=run_status,
            record_count=len(entries),
            blob_count=len(blobs),
            last_hash=previous_hash,
        )

    def _validate_manifest(self) -> None:
        manifest = self._manifest
        if not isinstance(manifest, dict):
            raise TraceValidationError("run manifest must be a JSON object")
        required = {
            "schema_version",
            "run_id",
            "status",
            "pipeline_version",
            "created_at_utc",
            "config",
            "inputs",
            "metadata",
            "artifacts",
            "trace_schema_version",
            "hash_algorithm",
            "manifest_hash",
        }
        if set(manifest) != required:
            raise TraceValidationError("run manifest has missing or unknown fields")
        if manifest["schema_version"] != RUN_MANIFEST_SCHEMA:
            raise TraceValidationError("unsupported run manifest schema")
        for field in ("run_id", "pipeline_version", "created_at_utc"):
            try:
                _require_text(manifest[field], field)
            except ValueError as error:
                raise TraceValidationError(str(error)) from error
        if manifest["status"] != RunStatus.ACTIVE.value:
            raise TraceValidationError("v1 run manifest status must be ACTIVE")
        if manifest["trace_schema_version"] != TRACE_RECORD_SCHEMA:
            raise TraceValidationError("unsupported trace record schema")
        if manifest["hash_algorithm"] != HASH_ALGORITHM:
            raise TraceValidationError("unsupported trace hash algorithm")
        manifest_body = dict(manifest)
        stored_manifest_hash = manifest_body.pop("manifest_hash")
        if not isinstance(stored_manifest_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", stored_manifest_hash
        ):
            raise TraceValidationError("manifest_hash must be a sha256")
        if _hash_record_body(manifest_body) != stored_manifest_hash:
            raise TraceValidationError("run manifest hash mismatch")
        if manifest["artifacts"] != {
            "trace": "trace.jsonl",
            "summary": "run_summary.json",
            "blob_root": "blobs/sha256",
        }:
            raise TraceValidationError("manifest artifact paths do not match the v1 protocol")
        for field in ("config", "inputs", "metadata"):
            try:
                _validate_json_value(manifest[field])
            except (TypeError, ValueError) as error:
                raise TraceValidationError(f"invalid manifest {field}: {error}") from error

    def _read_record_entries(self) -> tuple[tuple[TraceRecord, dict[str, Any]], ...]:
        try:
            data = self._trace_path.read_bytes()
        except FileNotFoundError:
            raise TraceValidationError("missing trace: trace.jsonl") from None
        if data and not data.endswith(b"\n"):
            raise TraceValidationError("trace.jsonl has a truncated tail (missing newline)")
        entries: list[tuple[TraceRecord, dict[str, Any]]] = []
        for line_number, line in enumerate(data.splitlines(), start=1):
            if not line:
                raise TraceValidationError(f"blank trace line at {line_number}")
            raw = _load_json_bytes(line, f"trace line {line_number}")
            try:
                record = _record_from_dict(raw)
            except TraceValidationError as error:
                raise TraceValidationError(f"invalid trace line {line_number}: {error}") from error
            entries.append((record, raw))
        return tuple(entries)

    def _verify_blob(self, blob: BlobRef) -> None:
        path = _safe_artifact_path(self.root, blob.relative_path)
        if not path.is_file():
            raise TraceValidationError(f"missing blob: {blob.relative_path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as blob_file:
            for chunk in iter(lambda: blob_file.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != blob.bytes:
            raise TraceValidationError(f"blob size mismatch: {blob.relative_path}")
        if digest.hexdigest() != blob.sha256:
            raise TraceValidationError(f"blob hash mismatch: {blob.relative_path}")

    def _validate_summary(
        self, summary: dict[str, Any], record_count: int, last_hash: str | None
    ) -> RunStatus:
        required = {
            "schema_version",
            "run_id",
            "status",
            "finalized_at_utc",
            "record_count",
            "last_record_hash",
            "summary",
            "summary_hash",
        }
        if set(summary) != required:
            raise TraceValidationError("run summary has missing or unknown fields")
        if summary["schema_version"] != RUN_SUMMARY_SCHEMA:
            raise TraceValidationError("unsupported run summary schema")
        summary_body = dict(summary)
        stored_summary_hash = summary_body.pop("summary_hash")
        if not isinstance(stored_summary_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", stored_summary_hash
        ):
            raise TraceValidationError("summary_hash must be a sha256")
        if _hash_record_body(summary_body) != stored_summary_hash:
            raise TraceValidationError("run summary hash mismatch")
        if summary["run_id"] != self._manifest["run_id"]:
            raise TraceValidationError("run summary ID does not match manifest")
        try:
            status = RunStatus(summary["status"])
        except (TypeError, ValueError) as error:
            raise TraceValidationError("invalid final run status") from error
        if status is RunStatus.ACTIVE:
            raise TraceValidationError("final run summary cannot be ACTIVE")
        if summary["record_count"] != record_count:
            raise TraceValidationError("run summary record count mismatch")
        if summary["last_record_hash"] != last_hash:
            raise TraceValidationError("run summary last hash mismatch")
        try:
            _require_text(summary["finalized_at_utc"], "finalized_at_utc")
            _validate_json_value(summary["summary"])
        except (TypeError, ValueError) as error:
            raise TraceValidationError(f"invalid run summary: {error}") from error
        return status


def _record_from_dict(value: Any) -> TraceRecord:
    required = {
        "schema_version",
        "ordinal",
        "record_id",
        "timestamp_utc",
        "stage",
        "status",
        "event",
        "parent_ids",
        "blobs",
        "payload",
        "previous_hash",
        "record_hash",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise TraceValidationError("trace record has missing or unknown fields")
    if value["schema_version"] != TRACE_RECORD_SCHEMA:
        raise TraceValidationError("unsupported trace record schema")
    if (
        isinstance(value["ordinal"], bool)
        or not isinstance(value["ordinal"], int)
        or value["ordinal"] < 0
    ):
        raise TraceValidationError("record ordinal must be a non-negative integer")
    try:
        _require_text(value["record_id"], "record_id")
        _require_text(value["timestamp_utc"], "timestamp_utc")
        _require_text(value["event"], "event")
        stage = TraceStage(value["stage"])
        status = TraceStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise TraceValidationError(f"invalid typed record field: {error}") from error
    parents = value["parent_ids"]
    if not isinstance(parents, list) or any(
        not isinstance(parent, str) or not parent.strip() for parent in parents
    ):
        raise TraceValidationError("parent_ids must be an array of non-empty strings")
    if len(set(parents)) != len(parents):
        raise TraceValidationError("parent_ids must be unique")
    if not isinstance(value["blobs"], list):
        raise TraceValidationError("blobs must be an array")
    blobs = tuple(BlobRef.from_dict(blob) for blob in value["blobs"])
    previous_hash = value["previous_hash"]
    if previous_hash is not None and not (
        isinstance(previous_hash, str) and re.fullmatch(r"[0-9a-f]{64}", previous_hash)
    ):
        raise TraceValidationError("previous_hash must be null or a sha256")
    record_hash = value["record_hash"]
    if not isinstance(record_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", record_hash):
        raise TraceValidationError("record_hash must be a sha256")
    try:
        _validate_json_value(value["payload"])
    except (TypeError, ValueError) as error:
        raise TraceValidationError(f"invalid record payload: {error}") from error
    return TraceRecord(
        schema_version=value["schema_version"],
        ordinal=value["ordinal"],
        record_id=value["record_id"],
        timestamp_utc=value["timestamp_utc"],
        stage=stage,
        status=status,
        event=value["event"],
        parent_ids=tuple(parents),
        blobs=blobs,
        payload=value["payload"],
        previous_hash=previous_hash,
        record_hash=record_hash,
    )


__all__ = [
    "BlobRef",
    "RunArtifactReader",
    "RunArtifactWriter",
    "RunStatus",
    "TraceRecord",
    "TraceStage",
    "TraceStatus",
    "TraceValidationError",
    "ValidationReport",
]
