"""Read-only HTTP viewer for persisted hand-pose trace runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import mimetypes
import re
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def _jsonable(value: Any) -> Any:
    """Convert public trace values to JSON-compatible values without mutating them."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    return value


def _record_dict(record: Any) -> dict[str, Any]:
    converted = _jsonable(record)
    if not isinstance(converted, dict):
        raise TypeError("trace records must serialize to JSON objects")
    return converted


def _frame_summaries(reader: Any, stage: str | None, track_id: str | None) -> list[dict[str, Any]]:
    source = reader.records()
    frames: dict[str, dict[str, Any]] = {}
    for value in source:
        record = _record_dict(value)
        if stage is not None and record.get("stage") != stage:
            continue
        payload = record.get("payload")
        metadata = payload if isinstance(payload, dict) else record
        if track_id is not None and metadata.get("track_id") != track_id:
            continue
        frame_id = metadata.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        summary = frames.setdefault(
            frame_id,
            {
                "frame_id": frame_id,
                "timestamp_ns": metadata.get("timestamp_ns"),
                "track_ids": [],
                "stages": [],
                "record_ids": [],
            },
        )
        current_track = metadata.get("track_id")
        current_stage = record.get("stage")
        record_id = record.get("record_id", record.get("id"))
        if isinstance(current_track, str) and current_track not in summary["track_ids"]:
            summary["track_ids"].append(current_track)
        if isinstance(current_stage, str) and current_stage not in summary["stages"]:
            summary["stages"].append(current_stage)
        if isinstance(record_id, str):
            summary["record_ids"].append(record_id)
    return list(frames.values())


def _global_record_ids(reader: Any, stage: str | None, track_id: str | None) -> list[str]:
    source = reader.records()
    result: list[str] = []
    for value in source:
        record = _record_dict(value)
        if stage is not None and record.get("stage") != stage:
            continue
        payload = record.get("payload")
        metadata = payload if isinstance(payload, dict) else record
        if track_id is not None and metadata.get("track_id") != track_id:
            continue
        if metadata.get("frame_id") not in (None, ""):
            continue
        record_id = record.get("record_id", record.get("id"))
        if isinstance(record_id, str):
            result.append(record_id)
    return result


def _records_for_frame(reader: Any, frame_id: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for value in reader.records():
        record = _record_dict(value)
        payload = record.get("payload")
        metadata = payload if isinstance(payload, dict) else record
        if metadata.get("frame_id") == frame_id:
            matches.append(record)
    return matches


def _get_record(reader: Any, record_id: str) -> Any:
    getter = getattr(reader, "get", None)
    if not callable(getter):
        getter = getattr(reader, "get_record", None)
    if not callable(getter):
        raise LookupError(record_id)
    return getter(record_id)


def _artifact_refs(value: Any):
    converted = _jsonable(value)
    if isinstance(converted, dict):
        artifact_path = converted.get("path", converted.get("relative_path"))
        if isinstance(converted.get("sha256"), str) and isinstance(artifact_path, str):
            yield converted
        for child in converted.values():
            yield from _artifact_refs(child)
    elif isinstance(converted, list):
        for child in converted:
            yield from _artifact_refs(child)


def _find_artifact(reader: Any, digest: str) -> tuple[Path, str] | None:
    if _SHA256_RE.fullmatch(digest) is None:
        return None
    root = Path(reader.root).resolve()
    sources = [reader.manifest, *reader.records()]
    for source in sources:
        for reference in _artifact_refs(source):
            if reference["sha256"].lower() != digest:
                continue
            relative_path = Path(reference.get("path", reference.get("relative_path")))
            if relative_path.is_absolute():
                continue
            try:
                candidate = (root / relative_path).resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if not candidate.is_relative_to(root) or not candidate.is_file():
                continue
            media_type = reference.get("media_type", reference.get("mime_type"))
            if not isinstance(media_type, str) or not media_type:
                media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            return candidate, media_type
    return None


def _summary(reader: Any) -> Any:
    summary = getattr(reader, "summary", {})
    return summary() if callable(summary) else summary


def _handler_for(reader: Any) -> type[BaseHTTPRequestHandler]:
    class TraceViewerHandler(BaseHTTPRequestHandler):
        server_version = "FisheyeHandposeTraceViewer/1"

        def do_GET(self) -> None:  # noqa: N802
            target = urlsplit(self.path)
            path = target.path
            if path == "/api/run":
                self._send_json(
                    200,
                    {
                        "manifest": _jsonable(reader.manifest),
                        "summary": _jsonable(_summary(reader)),
                        "validation": _jsonable(reader.validate()),
                    },
                )
                return
            if path == "/api/frames":
                query = parse_qs(target.query)
                try:
                    offset = int(query.get("offset", ["0"])[0])
                    limit = int(query.get("limit", ["100"])[0])
                    if offset < 0 or not 1 <= limit <= 1000:
                        raise ValueError
                except ValueError:
                    self._send_json(
                        400,
                        {
                            "error": {
                                "code": "invalid_query",
                                "message": "offset must be non-negative and limit must be 1..1000",
                            }
                        },
                    )
                    return
                stage = query.get("stage", [None])[0]
                track_id = query.get("track_id", [None])[0]
                frames = _frame_summaries(reader, stage, track_id)
                self._send_json(
                    200,
                    {
                        "items": frames[offset : offset + limit],
                        "global_record_ids": _global_record_ids(reader, stage, track_id),
                        "offset": offset,
                        "limit": limit,
                        "total": len(frames),
                    },
                )
                return
            if path.startswith("/api/frames/"):
                frame_id = unquote(path.removeprefix("/api/frames/"))
                records = _records_for_frame(reader, frame_id)
                if not frame_id or not records:
                    self._send_json(
                        404,
                        {"error": {"code": "not_found", "message": "Frame not found"}},
                    )
                    return
                self._send_json(200, {"frame_id": frame_id, "records": records})
                return
            if path.startswith("/api/records/"):
                record_id = unquote(path.removeprefix("/api/records/"))
                try:
                    record = _get_record(reader, record_id)
                    if record is None:
                        raise LookupError(record_id)
                except (KeyError, LookupError, StopIteration):
                    self._send_json(
                        404,
                        {"error": {"code": "not_found", "message": "Record not found"}},
                    )
                    return
                self._send_json(200, record)
                return
            if path.startswith("/artifacts/"):
                digest = path.removeprefix("/artifacts/").lower()
                artifact = _find_artifact(reader, digest)
                if artifact is None:
                    self._send_json(
                        404,
                        {"error": {"code": "not_found", "message": "Artifact not found"}},
                    )
                    return
                file_path, media_type = artifact
                with file_path.open("rb") as handle:
                    actual_digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    if actual_digest != digest:
                        self._send_json(
                            409,
                            {
                                "error": {
                                    "code": "artifact_integrity_error",
                                    "message": (
                                        "Artifact SHA-256 does not match its trace reference"
                                    ),
                                }
                            },
                        )
                        return
                    handle.seek(0)
                    self.send_response(200)
                    self.send_header("Content-Type", media_type)
                    self.send_header("Content-Length", str(file_path.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    while chunk := handle.read(1024 * 1024):
                        self.wfile.write(chunk)
                return
            static = _STATIC_FILES.get(path)
            if static is not None:
                filename, media_type = static
                body = Path(__file__).with_name("static").joinpath(filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", media_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self'; media-src 'self'; "
                    "style-src 'self'; script-src 'self'; object-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_json(404, {"error": {"code": "not_found", "message": "Not found"}})

        def do_POST(self) -> None:  # noqa: N802
            self._reject_write()

        def do_PUT(self) -> None:  # noqa: N802
            self._reject_write()

        def do_PATCH(self) -> None:  # noqa: N802
            self._reject_write()

        def do_DELETE(self) -> None:  # noqa: N802
            self._reject_write()

        def _reject_write(self) -> None:
            self._send_json(
                405,
                {"error": {"code": "method_not_allowed", "message": "Read-only viewer"}},
            )

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(
                _jsonable(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return TraceViewerHandler


def create_server(reader: Any, *, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create, but do not start, a local read-only trace viewer server."""

    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("trace viewer host must be a loopback host")
    return ThreadingHTTPServer((host, port), _handler_for(reader))


def serve_trace(reader: Any, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve a trace until interrupted, binding only to loopback by default."""

    server = create_server(reader, host=host, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
