"""Append-only worker events and content-addressed source-frame blobs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import WorkerError

EVENT_SCHEMA = "fisheye-handpose/h20-worker-event/v1"
MANIFEST_SCHEMA = "fisheye-handpose/h20-worker-manifest/v1"
SUMMARY_SCHEMA = "fisheye-handpose/h20-worker-summary/v1"
FHP21_OUTPUT_SCHEMA = "fisheye-handpose/fhp21-output/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkerError(f"worker artifact is not strict JSON: {exc}") from exc


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _file_identity(path: Path, *, role: str) -> dict[str, Any]:
    byte_count, digest = _file_digest(path)
    return {
        "role": role,
        "media_type": "application/x-ndjson",
        "bytes": byte_count,
        "sha256": digest,
        "relative_path": path.name,
    }


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


class ResultWriter:
    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self._events = self.root / "events.jsonl"
        self._summary = self.root / "summary.json"
        self._fhp21 = self.root / "fhp21.jsonl"
        self._ordinal = 0
        self._finalized = False
        self._event_ids: set[str] = set()
        _write_new(
            self.root / "manifest.json",
            {
                "schema_version": MANIFEST_SCHEMA,
                "status": "ACTIVE",
                "created_at_utc": _now(),
                **manifest,
            },
        )
        with self._events.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    def put_blob(
        self,
        data: bytes,
        *,
        role: str,
        media_type: str,
        suffix: str,
    ) -> dict[str, Any]:
        if not isinstance(data, bytes):
            raise WorkerError("blob data must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("blobs") / "sha256" / digest[:2] / f"{digest}{suffix}"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            try:
                with path.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                pass
        if path.read_bytes() != data:
            raise WorkerError("content-addressed blob collision or corruption")
        return {
            "role": role,
            "media_type": media_type,
            "bytes": len(data),
            "sha256": digest,
            "relative_path": relative.as_posix(),
        }

    def put_blob_file(
        self,
        source: str | Path,
        *,
        role: str,
        media_type: str,
        suffix: str,
    ) -> dict[str, Any]:
        """Publish a file-backed blob without materializing it in process memory."""

        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise WorkerError(f"blob source is not a file: {source_path}")
        byte_count, digest = _file_digest(source_path)
        relative = Path("blobs") / "sha256" / digest[:2] / f"{digest}{suffix}"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{digest}.",
                    suffix=".tmp",
                    dir=destination.parent,
                    delete=False,
                ) as output:
                    temporary = Path(output.name)
                    with source_path.open("rb") as input_file:
                        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        actual_bytes, actual_digest = _file_digest(destination)
        if actual_bytes != byte_count or actual_digest != digest:
            raise WorkerError("content-addressed blob collision or corruption")
        return {
            "role": role,
            "media_type": media_type,
            "bytes": byte_count,
            "sha256": digest,
            "relative_path": relative.as_posix(),
        }

    def append(
        self,
        *,
        event_id: str,
        stage: str,
        status: str,
        event: str,
        payload: Any,
        blobs: list[dict[str, Any]] | None = None,
        parent_event_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if self._finalized:
            raise WorkerError("cannot append to finalized result")
        if not isinstance(event_id, str) or not event_id or event_id in self._event_ids:
            raise WorkerError(f"event_id must be new and non-empty: {event_id!r}")
        if not isinstance(event, str) or not event:
            raise WorkerError("event must be non-empty")
        if len(set(parent_event_ids)) != len(parent_event_ids):
            raise WorkerError("parent_event_ids must be unique")
        unknown = [parent for parent in parent_event_ids if parent not in self._event_ids]
        if unknown:
            raise WorkerError(f"unknown parent_event_ids: {unknown}")
        event = {
            "schema_version": EVENT_SCHEMA,
            "ordinal": self._ordinal,
            "event_id": event_id,
            "timestamp_utc": _now(),
            "stage": stage,
            "status": status,
            "event": event,
            "parent_event_ids": list(parent_event_ids),
            "payload": payload,
            "blobs": [] if blobs is None else blobs,
        }
        encoded = _json_bytes(event)
        with self._events.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._ordinal += 1
        self._event_ids.add(event_id)
        return event

    def append_fhp21(self, value: dict[str, Any]) -> None:
        if self._finalized:
            raise WorkerError("cannot append output to finalized result")
        encoded = _json_bytes({**value, "schema_version": FHP21_OUTPUT_SCHEMA})
        mode = "ab" if self._fhp21.exists() else "xb"
        with self._fhp21.open(mode) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def finalize(self, *, status: str, summary: dict[str, Any]) -> dict[str, Any]:
        if self._finalized:
            raise WorkerError("result is already finalized")
        value = {
            "schema_version": SUMMARY_SCHEMA,
            "status": status,
            "finalized_at_utc": _now(),
            "event_count": self._ordinal,
            **summary,
        }
        output_file = value.get("output_file")
        if output_file is not None:
            if output_file != self._fhp21.name or not self._fhp21.is_file():
                raise WorkerError("declared fhp21 output file is missing or invalid")
            value["output_artifact"] = _file_identity(self._fhp21, role="fhp21_output")
        elif self._fhp21.is_file():
            value["partial_output_artifact"] = _file_identity(
                self._fhp21, role="partial_fhp21_output"
            )
        _write_new(self._summary, value)
        self._finalized = True
        return value


__all__ = ["FHP21_OUTPUT_SCHEMA", "ResultWriter"]
