"""Filesystem-backed, read-only catalog for pipeline trace runs."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from fisheye_handpose.trace import RunArtifactReader, TraceValidationError

_MANIFEST_NAMES = ("run_manifest.json", "trace_manifest.json", "manifest.json")
_TRACE_NAMES = ("trace.jsonl", "records.jsonl")
_SUMMARY_NAMES = ("run_summary.json", "summary.json")


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


class TraceCatalog:
    """Query trace folders under one configured catalog root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def list_runs(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        items = [self._summarize_safe(path) for path in self._run_directories()]
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

    def get_run(self, run_key: str) -> dict[str, Any]:
        run = self._find_run(run_key)
        run_summary = self._summarize_safe(run)
        try:
            manifest = self._manifest(run)
            summary = self._summary(run)
            records = self._records(run)
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
            "validation": _validate_run(run, manifest),
            "provenance": _run_provenance(manifest, records),
            "stages": sorted({str(record["stage"]) for record in records if record.get("stage")}),
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
        records = self._records(self._find_run(run_key))
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
        page = self.list_frames(run_key, offset=0, limit=2**31 - 1)
        frame = next(
            (item for item in page["items"] if item["frame_key"] == frame_key),
            None,
        )
        if frame is None:
            raise KeyError((run_key, frame_key))
        run = self._find_run(run_key)
        records = self._records(run)
        matches = []
        for record in records:
            payload = _metadata(record)
            if payload.get("frame_id") == frame["frame_id"]:
                matches.append(record)
        return {
            "run_key": run_key,
            "run_id": str(self._manifest(run).get("run_id") or run.name),
            "frame": frame,
            "records": matches,
        }

    def get_records(self, run_key: str, stage: str, frame_key: str) -> dict[str, Any]:
        items = [
            record
            for record in self.get_frame(run_key, frame_key)["records"]
            if record.get("stage") == stage
        ]
        run = self._find_run(run_key)
        return {
            "run_key": run_key,
            "run_id": str(self._manifest(run).get("run_id") or run.name),
            "stage": stage,
            "frame_key": frame_key,
            "items": items,
            "total": len(items),
        }

    def get_record(self, run_key: str, record_id: str) -> dict[str, Any]:
        record = next(
            (
                record
                for record in self._records(self._find_run(run_key))
                if record.get("record_id", record.get("id")) == record_id
            ),
            None,
        )
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
        references = []
        for record in self._records(run):
            references.extend(_artifact_references(record))
        reference = next(
            (
                value
                for value in references
                if value.get("relative_path", value.get("path")) == relative_path
            ),
            None,
        )
        if reference is None:
            raise ArtifactNotFoundError(relative_path)
        try:
            candidate = run.joinpath(*parts.parts).resolve(strict=True)
        except (OSError, RuntimeError):
            raise ArtifactNotFoundError(relative_path) from None
        if not candidate.is_relative_to(run) or not candidate.is_file():
            raise ArtifactNotFoundError(relative_path)
        expected_sha = reference.get("sha256")
        expected_size = reference.get("bytes", reference.get("size"))
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
            raise ArtifactIntegrityError("artifact reference has no valid SHA-256")
        digest = hashlib.sha256()
        size = 0
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != expected_sha.lower() or (
            isinstance(expected_size, int)
            and not isinstance(expected_size, bool)
            and size != expected_size
        ):
            raise ArtifactIntegrityError(f"artifact integrity mismatch: {relative_path}")
        media_type = reference.get("media_type", reference.get("mime_type"))
        if not isinstance(media_type, str) or not media_type:
            media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        return ResolvedArtifact(candidate, media_type, size, expected_sha.lower())

    def _find_run(self, run_key: str) -> Path:
        for path in self._run_directories():
            if self._run_key(path) == run_key:
                return path
        raise KeyError(run_key)

    def _run_key(self, run: Path) -> str:
        relative_path = run.relative_to(self.root).as_posix()
        return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]

    def _run_directories(self) -> tuple[Path, ...]:
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
        return tuple(sorted(set(directories)))

    def _summarize(self, run: Path) -> dict[str, Any]:
        manifest = self._manifest(run)
        summary = self._summary(run)
        records = self._records(run)
        if manifest.get("schema_version") == "fisheye-handpose/run-manifest/v1":
            RunArtifactReader(run).validate(verify_blobs=False)
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
    if manifest.get("schema_version") != "fisheye-handpose/run-manifest/v1":
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
    if manifest.get("schema_version") != "fisheye-handpose/run-manifest/v1":
        return {
            "ok": False,
            "mode": "LEGACY_UNVERIFIED",
            "errors": ["legacy trace has no canonical v1 integrity proof"],
            "warnings": [],
        }
    try:
        report = RunArtifactReader(run).validate()
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
