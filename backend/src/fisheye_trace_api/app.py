"""FastAPI adapter for the framework-independent trace catalog service."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .catalog import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ResolvedArtifact,
    TraceCatalog,
)

API_PREFIX = "/api/v1"


def create_app(
    catalog_root: str | Path,
    *,
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ),
) -> FastAPI:
    """Create a read-only application over one trace catalog root."""

    catalog = TraceCatalog(catalog_root)
    app = FastAPI(
        title="Fisheye Handpose Trace API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.state.catalog_root = str(catalog.root)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["Range", "Content-Type"],
        expose_headers=["Accept-Ranges", "Content-Length", "Content-Range"],
    )

    @app.middleware("http")
    async def read_only_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path.startswith(API_PREFIX) and request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            return _error(405, "method_not_allowed", "Read-only API")
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.get(f"{API_PREFIX}/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "read_only": True}

    @app.get(f"{API_PREFIX}/runs")
    def list_runs(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=1000),
        status: str | None = None,
        q: str | None = None,
    ) -> dict[str, object]:
        return catalog.list_runs(offset=offset, limit=limit, status=status, q=q)

    @app.get(f"{API_PREFIX}/runs/{{run_key}}")
    def get_run(run_key: str) -> Any:
        try:
            return catalog.get_run(run_key)
        except KeyError:
            return _error(404, "run_not_found", "Run not found")

    @app.get(f"{API_PREFIX}/runs/{{run_key}}/frames")
    def list_frames(
        run_key: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        stage: str | None = None,
        track_id: str | None = None,
        status: str | None = None,
    ) -> Any:
        try:
            return catalog.list_frames(
                run_key,
                offset=offset,
                limit=limit,
                stage=stage,
                track_id=track_id,
                status=status,
            )
        except KeyError:
            return _error(404, "run_not_found", "Run not found")

    @app.get(f"{API_PREFIX}/runs/{{run_key}}/frames/{{frame_key}}")
    def get_frame(run_key: str, frame_key: str) -> Any:
        try:
            return catalog.get_frame(run_key, frame_key)
        except KeyError:
            return _error(404, "frame_not_found", "Frame not found")

    @app.get(f"{API_PREFIX}/runs/{{run_key}}/records/{{record_id}}")
    def get_record(run_key: str, record_id: str) -> Any:
        try:
            return catalog.get_record(run_key, record_id)
        except KeyError:
            return _error(404, "record_not_found", "Record not found")

    @app.get(f"{API_PREFIX}/runs/{{run_key}}/record")
    def get_record_by_id(run_key: str, record_id: str = Query(..., min_length=1)) -> Any:
        """Resolve canonical record IDs even when they contain path separators."""
        try:
            return catalog.get_record(run_key, record_id)
        except KeyError:
            return _error(404, "record_not_found", "Record not found")

    @app.get(f"{API_PREFIX}/runs/{{run_key}}/records/{{stage}}/{{frame_key}}")
    def get_records(run_key: str, stage: str, frame_key: str) -> Any:
        try:
            result = catalog.get_records(run_key, stage, frame_key)
        except KeyError:
            return _error(404, "frame_not_found", "Frame not found")
        if result["total"] == 0:
            return _error(404, "record_not_found", "Stage record not found")
        return result

    artifact_route = f"{API_PREFIX}/runs/{{run_key}}/artifacts/{{artifact_path:path}}"

    @app.get(artifact_route)
    def get_artifact(
        run_key: str,
        artifact_path: str,
        request: Request,
    ) -> Response:
        return _serve_artifact(catalog, run_key, artifact_path, request.headers.get("range"), False)

    @app.head(artifact_route)
    def head_artifact(
        run_key: str,
        artifact_path: str,
        request: Request,
    ) -> Response:
        return _serve_artifact(catalog, run_key, artifact_path, request.headers.get("range"), True)

    return app


def _serve_artifact(
    catalog: TraceCatalog,
    run_key: str,
    artifact_path: str,
    range_header: str | None,
    head_only: bool,
) -> Response:
    try:
        artifact = catalog.resolve_artifact(run_key, artifact_path)
    except KeyError:
        return _error(404, "run_not_found", "Run not found")
    except ArtifactNotFoundError:
        return _error(404, "artifact_not_found", "Artifact not found")
    except ArtifactIntegrityError:
        return _error(409, "artifact_integrity_error", "Artifact integrity check failed")
    return _artifact_response(artifact, range_header, head_only)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _artifact_response(
    artifact: ResolvedArtifact,
    range_header: str | None,
    head_only: bool,
) -> Response:
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if range_header is None:
        headers = {**common_headers, "Content-Length": str(artifact.size)}
        if head_only:
            return Response(status_code=200, headers=headers, media_type=artifact.media_type)
        return StreamingResponse(
            _file_chunks(artifact.path, 0, artifact.size),
            status_code=200,
            headers=headers,
            media_type=artifact.media_type,
        )
    byte_range = _parse_range(range_header, artifact.size)
    if byte_range is None:
        return Response(
            status_code=416,
            headers={**common_headers, "Content-Range": f"bytes */{artifact.size}"},
        )
    start, end = byte_range
    length = end - start + 1
    headers = {
        **common_headers,
        "Content-Length": str(length),
        "Content-Range": f"bytes {start}-{end}/{artifact.size}",
    }
    if head_only:
        return Response(status_code=206, headers=headers, media_type=artifact.media_type)
    return StreamingResponse(
        _file_chunks(artifact.path, start, length),
        status_code=206,
        headers=headers,
        media_type=artifact.media_type,
    )


def _parse_range(value: str, size: int) -> tuple[int, int] | None:
    if not value.startswith("bytes=") or "," in value or size <= 0:
        return None
    spec = value.removeprefix("bytes=").strip()
    if spec.count("-") != 1:
        return None
    first, last = spec.split("-", 1)
    try:
        if first:
            start = int(first)
            end = int(last) if last else size - 1
            if start < 0 or end < start or start >= size:
                return None
            return start, min(end, size - 1)
        suffix_length = int(last)
        if suffix_length <= 0:
            return None
        return max(size - suffix_length, 0), size - 1
    except ValueError:
        return None


def _file_chunks(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
