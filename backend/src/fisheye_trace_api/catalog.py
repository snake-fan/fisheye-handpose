"""Filesystem-backed, read-only catalog for pipeline trace runs."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from fisheye_handpose.trace import (
    RUN_MANIFEST_SCHEMA,
    RunArtifactReader,
    TraceStage,
    TraceValidationError,
)

_MANIFEST_NAMES = ("run_manifest.json", "trace_manifest.json", "manifest.json")
_TRACE_NAMES = ("trace.jsonl", "records.jsonl")
_SUMMARY_NAMES = ("run_summary.json", "summary.json")
DEFAULT_DISCOVERY_CACHE_TTL_SECONDS = 30.0
DEFAULT_VALIDATION_CACHE_TTL_SECONDS = 300.0
_TRACE_STAGE_RANK = {stage.value: index for index, stage in enumerate(TraceStage)}


class ArtifactNotFoundError(LookupError):
    """The requested path is not a safe, referenced artifact of the run."""


class ArtifactIntegrityError(ValueError):
    """The artifact bytes no longer match their trace reference."""


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    path: Path
    media_type: str
    size: int
    sha256: str
    content_addressed: bool


_FileFingerprint = tuple[int, int, int, int, int]
_RunFingerprint = tuple[tuple[str, _FileFingerprint] | None, ...]
_ValidationFingerprint = _RunFingerprint
_ArtifactFingerprint = tuple[tuple[str, str | None, _FileFingerprint | None], ...]


@dataclass(frozen=True, slots=True)
class _RunIndex:
    fingerprint: _RunFingerprint
    manifest: dict[str, Any]
    summary: dict[str, Any] | None
    records: tuple[dict[str, Any], ...]
    finalized: bool
    frames: tuple[dict[str, Any], ...]
    frames_by_key: dict[str, dict[str, Any]]
    records_by_frame: dict[str, tuple[dict[str, Any], ...]]
    records_by_id: dict[str, dict[str, Any]]
    artifacts_by_path: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _VerifiedArtifact:
    fingerprint: _FileFingerprint
    size: int


@dataclass(frozen=True, slots=True)
class _CachedValidation:
    result: dict[str, Any]
    expires_at: float
    artifact_fingerprint: _ArtifactFingerprint | None


class TraceCatalog:
    """Query trace folders under one configured catalog root."""

    def __init__(
        self,
        root: str | Path,
        *,
        discovery_cache_ttl_seconds: float = DEFAULT_DISCOVERY_CACHE_TTL_SECONDS,
        validation_cache_ttl_seconds: float = DEFAULT_VALIDATION_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if discovery_cache_ttl_seconds < 0:
            raise ValueError("discovery_cache_ttl_seconds must be non-negative")
        if validation_cache_ttl_seconds < 0:
            raise ValueError("validation_cache_ttl_seconds must be non-negative")
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self._discovery_cache_ttl_seconds = discovery_cache_ttl_seconds
        self._validation_cache_ttl_seconds = validation_cache_ttl_seconds
        self._clock = clock
        self._cache_lock = threading.RLock()
        self._run_indexes: dict[Path, _RunIndex] = {}
        self._run_index_locks: dict[Path, threading.Lock] = {}
        self._validation_cache: dict[
            tuple[Path, _ValidationFingerprint, bool], _CachedValidation
        ] = {}
        self._validation_locks: dict[tuple[Path, _ValidationFingerprint, bool], threading.Lock] = {}
        self._artifact_cache: dict[tuple[Path, str, int | None], _VerifiedArtifact] = {}
        self._artifact_locks: dict[tuple[Path, str, int | None], threading.Lock] = {}
        self._known_runs: dict[str, Path] = {}
        self._discovered_runs: tuple[Path, ...] | None = None
        self._discovery_marker: tuple[tuple[str, int, int], ...] | None = None
        self._discovery_deadline = 0.0

    def list_runs(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        paths = self._run_directories()
        needle = q.casefold() if q is not None and q.strip() else None
        if needle is not None or (status is not None and status != "INVALID"):
            filtered_paths: list[Path] = []
            for path in paths:
                run_id, item_id, cheap_status = self._cheap_run_metadata(path)
                if needle is not None and not any(
                    needle in value.casefold() for value in (run_id, item_id)
                ):
                    continue
                if (
                    status is not None
                    and status != "INVALID"
                    and cheap_status is not None
                    and cheap_status != status
                ):
                    continue
                filtered_paths.append(path)
            paths = tuple(filtered_paths)
        items = [self._summarize_safe(path) for path in paths]
        if status is not None:
            items = [item for item in items if item["status"] == status]
        if q is not None and q.strip():
            needle = q.casefold()
            items = [
                item
                for item in items
                if needle in item["run_id"].casefold()
                or (
                    isinstance(item["data_item_id"], str)
                    and needle in item["data_item_id"].casefold()
                )
            ]
        items.sort(key=lambda item: (item["created_at_utc"] or "", item["run_id"]), reverse=True)
        return {
            "items": items[offset : offset + limit],
            "offset": offset,
            "limit": limit,
            "total": len(items),
        }

    def _cheap_run_metadata(self, run: Path) -> tuple[str, str, str | None]:
        manifest = _load_json_safe(_first_existing(run, _MANIFEST_NAMES)) or {}
        summary = _load_json_safe(_first_existing(run, _SUMMARY_NAMES))
        metadata = manifest.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        relative_parts = run.relative_to(self.root).parts
        fallback_item_id = relative_parts[-2] if len(relative_parts) > 1 else run.name
        item_id = next(
            (
                value
                for value in (
                    metadata.get("item_id"),
                    metadata.get("data_item_id"),
                    metadata.get("source_item_id"),
                    manifest.get("item_id"),
                    manifest.get("data_item_id"),
                )
                if isinstance(value, str) and value
            ),
            fallback_item_id,
        )
        run_id = manifest.get("run_id")
        cheap_status = (summary or manifest).get("status")
        return (
            run_id if isinstance(run_id, str) and run_id else run.name,
            item_id,
            cheap_status if isinstance(cheap_status, str) else None,
        )

    def get_run(self, run_key: str) -> dict[str, Any]:
        run = self._find_run(run_key)
        run_summary = self._summarize_safe(run)
        try:
            index = self._index(run)
            manifest = index.manifest
            summary = index.summary
            records = index.records
        except Exception as error:
            manifest = _load_json_safe(_first_existing(run, _MANIFEST_NAMES)) or {}
            summary = _load_json_safe(_first_existing(run, _SUMMARY_NAMES))
            return {
                "run": run_summary,
                "manifest": manifest,
                "summary": summary,
                "validation": {
                    "ok": False,
                    "mode": "CATALOG_ERROR",
                    "errors": [f"{type(error).__name__}: {error}"],
                    "warnings": [],
                },
                "provenance": {},
                "stages": [],
                "track_ids": [],
                "view_ids": [],
                "global_records": [],
            }
        payloads = [_metadata(record) for record in records]
        return {
            "run": run_summary,
            "manifest": manifest,
            "summary": summary,
            "validation": self._validate_run(run, index, verify_blobs=True),
            "provenance": _run_provenance(manifest, records),
            "stages": _frame_filter_stages(records),
            "track_ids": sorted(
                {str(payload["track_id"]) for payload in payloads if payload.get("track_id")}
            ),
            "view_ids": sorted(
                {str(payload["view_id"]) for payload in payloads if payload.get("view_id")}
            ),
            "global_records": [
                record
                for record in records
                if not isinstance(_metadata(record).get("frame_id"), str)
                or not _metadata(record)["frame_id"]
            ],
        }

    def list_frames(
        self,
        run_key: str,
        *,
        offset: int = 0,
        limit: int = 100,
        stage: str | None = None,
        track_id: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        index = self._index(self._find_run(run_key))
        records = index.records
        if stage is None and track_id is None and status is None:
            items = list(index.frames)
            return {
                "items": items[offset : offset + limit],
                "offset": offset,
                "limit": limit,
                "total": len(items),
            }
        frames: dict[str, dict[str, Any]] = {}
        for record in records:
            payload = _metadata(record)
            if stage is not None and record.get("stage") != stage:
                continue
            if status is not None and record.get("status") != status:
                continue
            if track_id is not None and payload.get("track_id") != track_id:
                continue
            frame_id = payload.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                continue
            item = frames.setdefault(
                frame_id,
                {
                    "frame_key": _frame_key(frame_id),
                    "frame_id": frame_id,
                    "frame_index": _frame_index(payload, frame_id),
                    "timestamp_ns": _frame_timestamp_ns(payload),
                    "record_ids": [],
                    "stages": [],
                    "statuses": [],
                    "track_ids": [],
                    "view_ids": [],
                },
            )
            if item["frame_index"] is None:
                item["frame_index"] = _frame_index(payload, frame_id)
            if item["timestamp_ns"] is None:
                item["timestamp_ns"] = _frame_timestamp_ns(payload)
            _append_text(item["record_ids"], record.get("record_id", record.get("id")))
            _append_text(item["stages"], record.get("stage"))
            _append_text(item["statuses"], record.get("status"))
            _append_text(item["track_ids"], payload.get("track_id"))
            _append_text(item["view_ids"], payload.get("view_id"))
        items = sorted(
            frames.values(),
            key=lambda item: (
                item["frame_index"] is None,
                item["frame_index"] if item["frame_index"] is not None else 0,
                item["timestamp_ns"] if isinstance(item["timestamp_ns"], int) else 0,
                item["frame_id"],
            ),
        )
        return {
            "items": items[offset : offset + limit],
            "offset": offset,
            "limit": limit,
            "total": len(items),
        }

    def get_frame(self, run_key: str, frame_key: str) -> dict[str, Any]:
        run = self._find_run(run_key)
        index = self._index(run)
        frame = index.frames_by_key.get(frame_key)
        if frame is None:
            raise KeyError((run_key, frame_key))
        return {
            "run_key": run_key,
            "run_id": str(index.manifest.get("run_id") or run.name),
            "frame": frame,
            "records": list(index.records_by_frame.get(frame["frame_id"], ())),
        }

    def get_records(self, run_key: str, stage: str, frame_key: str) -> dict[str, Any]:
        items = [
            record
            for record in self.get_frame(run_key, frame_key)["records"]
            if record.get("stage") == stage
        ]
        run = self._find_run(run_key)
        index = self._index(run)
        return {
            "run_key": run_key,
            "run_id": str(index.manifest.get("run_id") or run.name),
            "stage": stage,
            "frame_key": frame_key,
            "items": items,
            "total": len(items),
        }

    def get_record(self, run_key: str, record_id: str) -> dict[str, Any]:
        index = self._index(self._find_run(run_key))
        record = index.records_by_id.get(record_id)
        if record is None:
            raise KeyError((run_key, record_id))
        return record

    def resolve_artifact(self, run_key: str, relative_path: str) -> ResolvedArtifact:
        run = self._find_run(run_key)
        parts = PurePosixPath(relative_path)
        if (
            not relative_path
            or parts.is_absolute()
            or any(part in {"", ".", ".."} for part in parts.parts)
            or "\\" in relative_path
            or "\0" in relative_path
        ):
            raise ArtifactNotFoundError(relative_path)
        reference = self._index(run).artifacts_by_path.get(relative_path)
        if reference is None:
            raise ArtifactNotFoundError(relative_path)
        try:
            candidate = run.joinpath(*parts.parts).resolve(strict=True)
        except (OSError, RuntimeError):
            self._invalidate_validation_cache(run)
            raise ArtifactNotFoundError(relative_path) from None
        if not candidate.is_relative_to(run) or not candidate.is_file():
            self._invalidate_validation_cache(run)
            raise ArtifactNotFoundError(relative_path)
        expected_sha = reference.get("sha256")
        expected_size = reference.get("bytes", reference.get("size"))
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
            raise ArtifactIntegrityError("artifact reference has no valid SHA-256")
        cache_key = (
            candidate,
            expected_sha.lower(),
            expected_size
            if isinstance(expected_size, int) and not isinstance(expected_size, bool)
            else None,
        )
        artifact_lock = self._artifact_lock(cache_key)
        with artifact_lock:
            try:
                fingerprint = _file_fingerprint(candidate)
            except OSError:
                self._invalidate_validation_cache(run)
                raise ArtifactNotFoundError(relative_path) from None
            with self._cache_lock:
                cached_artifact = self._artifact_cache.get(cache_key)
            if cached_artifact is not None and cached_artifact.fingerprint == fingerprint:
                size = cached_artifact.size
            else:
                digest = hashlib.sha256()
                size = 0
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                try:
                    stable_fingerprint = _file_fingerprint(candidate)
                except OSError:
                    self._invalidate_validation_cache(run)
                    raise ArtifactNotFoundError(relative_path) from None
                if fingerprint != stable_fingerprint:
                    self._invalidate_validation_cache(run)
                    raise ArtifactIntegrityError(f"artifact changed while reading: {relative_path}")
                if digest.hexdigest() != expected_sha.lower() or (
                    isinstance(expected_size, int)
                    and not isinstance(expected_size, bool)
                    and size != expected_size
                ):
                    with self._cache_lock:
                        self._artifact_cache.pop(cache_key, None)
                    self._invalidate_validation_cache(run)
                    raise ArtifactIntegrityError(f"artifact integrity mismatch: {relative_path}")
                with self._cache_lock:
                    self._artifact_cache[cache_key] = _VerifiedArtifact(stable_fingerprint, size)
        media_type = reference.get("media_type", reference.get("mime_type"))
        if not isinstance(media_type, str) or not media_type:
            media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        normalized_sha = expected_sha.lower()
        content_addressed = bool(
            re.fullmatch(
                rf"blobs/sha256/{normalized_sha[:2]}/{normalized_sha}(?:\.[A-Za-z0-9_-]+)*",
                relative_path,
            )
        )
        return ResolvedArtifact(candidate, media_type, size, normalized_sha, content_addressed)

    def _find_run(self, run_key: str) -> Path:
        with self._cache_lock:
            known = self._known_runs.get(run_key)
        if known is not None and known.is_dir():
            return known
        for path in self._run_directories():
            if self._run_key(path) == run_key:
                return path
        raise KeyError(run_key)

    def _run_key(self, run: Path) -> str:
        relative_path = run.relative_to(self.root).as_posix()
        return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]

    def _run_directories(self) -> tuple[Path, ...]:
        marker = _directory_marker(self.root)
        now = self._clock()
        with self._cache_lock:
            if (
                self._discovered_runs is not None
                and marker == self._discovery_marker
                and now < self._discovery_deadline
            ):
                return self._discovered_runs
        manifests: set[Path] = set()
        for name in _MANIFEST_NAMES:
            manifests.update(self.root.rglob(name))
            if (self.root / name).is_file():
                manifests.add(self.root / name)
        directories: list[Path] = []
        for manifest in manifests:
            directory = manifest.parent.resolve()
            if directory.is_relative_to(self.root) and _first_existing(directory, _TRACE_NAMES):
                directories.append(directory)
        result = tuple(sorted(set(directories)))
        with self._cache_lock:
            self._discovered_runs = result
            self._discovery_marker = marker
            self._discovery_deadline = self._clock() + self._discovery_cache_ttl_seconds
            self._known_runs = {self._run_key(path): path for path in result}
        return result

    def _summarize(self, run: Path) -> dict[str, Any]:
        index = self._index(run)
        manifest = index.manifest
        summary = index.summary
        records = index.records
        if manifest.get("schema_version") == RUN_MANIFEST_SCHEMA:
            report = self._validate_run(run, index, verify_blobs=False)
            if not report["ok"]:
                raise TraceValidationError("; ".join(report["errors"]))
        stages = Counter(str(record.get("stage")) for record in records if record.get("stage"))
        frame_ids = {
            payload["frame_id"]
            for record in records
            if (payload := _metadata(record))
            and isinstance(payload.get("frame_id"), str)
            and payload["frame_id"]
        }
        statuses = Counter(str(record.get("status")) for record in records)
        metadata = manifest.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        item_id = next(
            (
                value
                for value in (
                    metadata.get("item_id"),
                    metadata.get("data_item_id"),
                    metadata.get("source_item_id"),
                    manifest.get("item_id"),
                    manifest.get("data_item_id"),
                )
                if isinstance(value, str) and value
            ),
            None,
        )
        if item_id is None:
            relative_parts = run.relative_to(self.root).parts
            item_id = relative_parts[-2] if len(relative_parts) > 1 else run.name
        return {
            "run_key": self._run_key(run),
            "item_id": item_id,
            "run_id": str(manifest.get("run_id") or run.name),
            "data_item_id": item_id,
            "status": (summary or manifest).get("status", "ACTIVE"),
            "created_at_utc": manifest.get("created_at_utc"),
            "finalized_at_utc": summary.get("finalized_at_utc") if summary else None,
            "pipeline_version": manifest.get("pipeline_version"),
            "record_count": len(records),
            "frame_count": len(frame_ids),
            "stage_counts": dict(stages),
            "warning_count": statuses["WARNING"],
            "failure_count": statuses["FAILED"],
        }

    def _summarize_safe(self, run: Path) -> dict[str, Any]:
        try:
            return self._summarize(run)
        except Exception as error:
            manifest: dict[str, Any] = {}
            manifest_path = _first_existing(run, _MANIFEST_NAMES)
            if manifest_path is not None:
                try:
                    manifest = _load_json(manifest_path)
                except Exception:
                    pass
            metadata = manifest.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            relative_parts = run.relative_to(self.root).parts
            fallback_item_id = relative_parts[-2] if len(relative_parts) > 1 else run.name
            item_id = next(
                (
                    value
                    for value in (
                        metadata.get("item_id"),
                        metadata.get("data_item_id"),
                        metadata.get("source_item_id"),
                        manifest.get("item_id"),
                        manifest.get("data_item_id"),
                    )
                    if isinstance(value, str) and value
                ),
                fallback_item_id,
            )
            run_id = manifest.get("run_id")
            return {
                "run_key": self._run_key(run),
                "item_id": item_id,
                "run_id": run_id if isinstance(run_id, str) and run_id else run.name,
                "data_item_id": item_id,
                "status": "INVALID",
                "created_at_utc": (
                    manifest.get("created_at_utc")
                    if isinstance(manifest.get("created_at_utc"), str)
                    else None
                ),
                "finalized_at_utc": None,
                "pipeline_version": (
                    manifest.get("pipeline_version")
                    if isinstance(manifest.get("pipeline_version"), str)
                    else None
                ),
                "record_count": 0,
                "frame_count": 0,
                "stage_counts": {},
                "warning_count": 0,
                "failure_count": 0,
                "catalog_error": {
                    "code": "corrupt_run",
                    "message": f"{type(error).__name__}: {error}",
                },
            }

    @staticmethod
    def _manifest(run: Path) -> dict[str, Any]:
        return _load_json(_required_existing(run, _MANIFEST_NAMES))

    @staticmethod
    def _summary(run: Path) -> dict[str, Any] | None:
        path = _first_existing(run, _SUMMARY_NAMES)
        return _load_json(path) if path else None

    @staticmethod
    def _records(run: Path) -> list[dict[str, Any]]:
        return _load_jsonl(_required_existing(run, _TRACE_NAMES))

    def _index(self, run: Path) -> _RunIndex:
        fingerprint = _run_fingerprint(run)
        with self._cache_lock:
            cached = self._run_indexes.get(run)
            if cached is not None and cached.finalized and cached.fingerprint == fingerprint:
                return cached
            run_lock = self._run_index_locks.setdefault(run, threading.Lock())
        with run_lock:
            fingerprint = _run_fingerprint(run)
            with self._cache_lock:
                cached = self._run_indexes.get(run)
                if cached is not None and cached.finalized and cached.fingerprint == fingerprint:
                    return cached
            manifest = self._manifest(run)
            summary = self._summary(run)
            records = tuple(self._records(run))
            index = _build_index(fingerprint, manifest, summary, records)
            with self._cache_lock:
                if index.finalized:
                    self._run_indexes[run] = index
                    stale = [
                        key
                        for key in self._validation_cache
                        if key[0] == run and key[1] != fingerprint
                    ]
                    for key in stale:
                        self._validation_cache.pop(key, None)
                else:
                    self._run_indexes.pop(run, None)
            return index

    def _validate_run(
        self,
        run: Path,
        index: _RunIndex,
        *,
        verify_blobs: bool,
    ) -> dict[str, Any]:
        if index.manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
            return _legacy_validation()
        if not index.finalized:
            return _validate_canonical(run, verify_blobs=verify_blobs)
        key = (run, index.fingerprint, verify_blobs)
        artifact_fingerprint = (
            _artifact_fingerprint(run, index.artifacts_by_path) if verify_blobs else None
        )
        now = self._clock()
        with self._cache_lock:
            cached = self._validation_cache.get(key)
            if (
                cached is not None
                and now < cached.expires_at
                and cached.artifact_fingerprint == artifact_fingerprint
            ):
                return cached.result
            validation_lock = self._validation_locks.setdefault(key, threading.Lock())
        with validation_lock:
            artifact_fingerprint = (
                _artifact_fingerprint(run, index.artifacts_by_path) if verify_blobs else None
            )
            now = self._clock()
            with self._cache_lock:
                cached = self._validation_cache.get(key)
                if (
                    cached is not None
                    and now < cached.expires_at
                    and cached.artifact_fingerprint == artifact_fingerprint
                ):
                    return cached.result
            result = _validate_canonical(run, verify_blobs=verify_blobs)
            stable_artifact_fingerprint = (
                _artifact_fingerprint(run, index.artifacts_by_path) if verify_blobs else None
            )
            if artifact_fingerprint == stable_artifact_fingerprint:
                with self._cache_lock:
                    self._validation_cache[key] = _CachedValidation(
                        result=result,
                        expires_at=self._clock() + self._validation_cache_ttl_seconds,
                        artifact_fingerprint=stable_artifact_fingerprint,
                    )
            return result

    def _invalidate_validation_cache(self, run: Path) -> None:
        with self._cache_lock:
            stale = [key for key in self._validation_cache if key[0] == run]
            for key in stale:
                self._validation_cache.pop(key, None)

    def _artifact_lock(self, key: tuple[Path, str, int | None]) -> threading.Lock:
        with self._cache_lock:
            return self._artifact_locks.setdefault(key, threading.Lock())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_json_safe(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        records.append(value)
    return records


def _file_fingerprint(path: Path) -> _FileFingerprint:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _artifact_fingerprint(
    run: Path,
    artifacts_by_path: dict[str, dict[str, Any]],
) -> _ArtifactFingerprint:
    entries: list[tuple[str, str | None, _FileFingerprint | None]] = []
    for relative_path in sorted(artifacts_by_path):
        resolved_path: str | None = None
        fingerprint: _FileFingerprint | None = None
        try:
            pure_path = PurePosixPath(relative_path)
            if pure_path.is_absolute() or ".." in pure_path.parts:
                raise ValueError(relative_path)
            candidate = run.joinpath(*pure_path.parts).resolve(strict=True)
            if not candidate.is_relative_to(run):
                raise ValueError(relative_path)
            resolved_path = candidate.as_posix()
            fingerprint = _file_fingerprint(candidate)
        except (OSError, ValueError):
            pass
        entries.append((relative_path, resolved_path, fingerprint))
    return tuple(entries)


def _run_fingerprint(run: Path) -> _RunFingerprint:
    paths = (
        _first_existing(run, _MANIFEST_NAMES),
        _first_existing(run, _TRACE_NAMES),
        _first_existing(run, _SUMMARY_NAMES),
    )
    return tuple(
        (path.name, _file_fingerprint(path)) if path is not None else None for path in paths
    )


def _directory_marker(root: Path) -> tuple[tuple[str, int, int], ...]:
    markers: list[tuple[str, int, int]] = []
    try:
        children = tuple(root.iterdir())
    except OSError:
        return ()
    for child in children:
        try:
            stat = child.stat()
        except OSError:
            continue
        markers.append((child.name, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(sorted(markers))


def _build_index(
    fingerprint: _RunFingerprint,
    manifest: dict[str, Any],
    summary: dict[str, Any] | None,
    records: tuple[dict[str, Any], ...],
) -> _RunIndex:
    frames: dict[str, dict[str, Any]] = {}
    records_by_frame: dict[str, list[dict[str, Any]]] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    artifacts_by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record.get("record_id", record.get("id"))
        if isinstance(record_id, str) and record_id:
            records_by_id[record_id] = record
        for reference in _artifact_references(record):
            path = reference.get("relative_path", reference.get("path"))
            if isinstance(path, str):
                artifacts_by_path.setdefault(path, reference)
        payload = _metadata(record)
        frame_id = payload.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        records_by_frame.setdefault(frame_id, []).append(record)
        item = frames.setdefault(
            frame_id,
            {
                "frame_key": _frame_key(frame_id),
                "frame_id": frame_id,
                "frame_index": _frame_index(payload, frame_id),
                "timestamp_ns": _frame_timestamp_ns(payload),
                "record_ids": [],
                "stages": [],
                "statuses": [],
                "track_ids": [],
                "view_ids": [],
            },
        )
        if item["frame_index"] is None:
            item["frame_index"] = _frame_index(payload, frame_id)
        if item["timestamp_ns"] is None:
            item["timestamp_ns"] = _frame_timestamp_ns(payload)
        _append_text(item["record_ids"], record_id)
        _append_text(item["stages"], record.get("stage"))
        _append_text(item["statuses"], record.get("status"))
        _append_text(item["track_ids"], payload.get("track_id"))
        _append_text(item["view_ids"], payload.get("view_id"))
    ordered_frames = tuple(
        sorted(
            frames.values(),
            key=lambda item: (
                item["frame_index"] is None,
                item["frame_index"] if item["frame_index"] is not None else 0,
                item["timestamp_ns"] if isinstance(item["timestamp_ns"], int) else 0,
                item["frame_id"],
            ),
        )
    )
    return _RunIndex(
        fingerprint=fingerprint,
        manifest=manifest,
        summary=summary,
        records=records,
        finalized=summary is not None and summary.get("status") in {"COMPLETED", "FAILED"},
        frames=ordered_frames,
        frames_by_key={item["frame_key"]: item for item in ordered_frames},
        records_by_frame={key: tuple(value) for key, value in records_by_frame.items()},
        records_by_id=records_by_id,
        artifacts_by_path=artifacts_by_path,
    )


def _frame_index(payload: dict[str, Any], frame_id: str) -> int | None:
    for key in ("frame_index", "frame_idx"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    match = re.search(r"(\d+)$", frame_id)
    return int(match.group(1)) if match else None


def _frame_timestamp_ns(payload: dict[str, Any]) -> int | None:
    for key in ("timestamp_ns", "pair_timestamp_ns"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _frame_key(frame_id: str) -> str:
    return hashlib.sha256(frame_id.encode("utf-8")).hexdigest()


def _frame_filter_stages(records: tuple[dict[str, Any], ...]) -> list[str]:
    """Return stages that can match the frame timeline, in pipeline order."""
    stages: set[str] = set()
    for record in records:
        payload = _metadata(record)
        frame_id = payload.get("frame_id")
        stage = record.get("stage")
        if isinstance(frame_id, str) and frame_id and isinstance(stage, str) and stage:
            stages.add(stage)
    return sorted(
        stages,
        key=lambda stage: (_TRACE_STAGE_RANK.get(stage, len(_TRACE_STAGE_RANK)), stage),
    )


def _append_text(items: list[str], value: Any) -> None:
    if isinstance(value, str) and value and value not in items:
        items.append(value)


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else record


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    return next((root / name for name in names if (root / name).is_file()), None)


def _required_existing(root: Path, names: tuple[str, ...]) -> Path:
    path = _first_existing(root, names)
    if path is None:
        raise FileNotFoundError(f"none of {names!r} exists in {root}")
    return path


def _artifact_references(value: Any) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    if isinstance(value, dict):
        path = value.get("relative_path", value.get("path"))
        if isinstance(path, str) and isinstance(value.get("sha256"), str):
            references.append(value)
        for child in value.values():
            references.extend(_artifact_references(child))
    elif isinstance(value, list):
        for child in value:
            references.extend(_artifact_references(child))
    return references


def _run_provenance(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        return {}
    provenance: dict[str, Any] = {}
    if "config" in manifest and "metadata" in manifest:
        provenance["manifest"] = {
            "config": manifest["config"],
            "metadata": manifest["metadata"],
        }
    for record in records:
        if (
            "worker_inputs" not in provenance
            and record.get("stage") == "SYSTEM"
            and record.get("event") == "worker_inputs_verified"
        ):
            facet = _record_provenance_facet(record)
            if facet is not None:
                provenance["worker_inputs"] = facet
        if (
            "calibration" not in provenance
            and record.get("stage") == "RECTIFICATION"
            and record.get("event") == "worker_rectification_loaded"
        ):
            facet = _record_provenance_facet(record)
            if facet is not None:
                provenance["calibration"] = facet
    return provenance


def _record_provenance_facet(record: dict[str, Any]) -> dict[str, Any] | None:
    record_id = record.get("record_id")
    payload = record.get("payload")
    if not isinstance(record_id, str) or not record_id or not isinstance(payload, dict):
        return None
    facet: dict[str, Any] = {
        "record_id": record_id,
        "payload": {key: value for key, value in payload.items() if key != "worker_provenance"},
    }
    worker_provenance = payload.get("worker_provenance")
    if isinstance(worker_provenance, dict):
        facet["worker_provenance"] = worker_provenance
    return facet


def _validate_run(
    run: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        return _legacy_validation()
    return _validate_canonical(run, verify_blobs=True)


def _legacy_validation() -> dict[str, Any]:
    return {
        "ok": False,
        "mode": "LEGACY_UNVERIFIED",
        "errors": ["legacy trace has no canonical v1 integrity proof"],
        "warnings": [],
    }


def _validate_canonical(run: Path, *, verify_blobs: bool) -> dict[str, Any]:
    try:
        report = RunArtifactReader(run).validate(verify_blobs=verify_blobs)
    except (TraceValidationError, OSError, UnicodeError, ValueError) as error:
        return {
            "ok": False,
            "mode": "CANONICAL_V1",
            "errors": [str(error)],
            "warnings": [],
        }
    return {
        **report.to_dict(),
        "mode": "CANONICAL_V1",
        "errors": [],
        "warnings": [],
    }
