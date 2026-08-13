from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fisheye_handpose.trace import (
    RunArtifactReader,
    RunArtifactWriter,
    RunStatus,
    TraceStage,
    TraceStatus,
)
from fisheye_handpose.viewer import create_server


@dataclass(frozen=True)
class FakeManifest:
    schema_version: str = "fhp-trace/v1"
    run_id: str = "run-test"


@dataclass(frozen=True)
class FakeTraceRecord:
    record_id: str
    stage: str
    payload: dict[str, Any]


class FakeReader:
    def __init__(self, root: Path, records: list[Any] | None = None) -> None:
        self.root = root
        self.manifest = FakeManifest()
        self.summary = {"record_count": len(records or []), "stage_counts": {}}
        self._records = records or []

    def records(
        self,
        *,
        stage: str | None = None,
        status: str | None = None,
        event: str | None = None,
    ) -> list[Any]:
        del status, event
        if stage is None:
            return self._records
        return [
            record
            for record in self._records
            if (record.get("stage") if isinstance(record, dict) else getattr(record, "stage", None))
            == stage
        ]

    def get(self, record_id: str) -> Any:
        return next(
            record
            for record in self._records
            if (
                record.get("record_id")
                if isinstance(record, dict)
                else getattr(record, "record_id", None)
            )
            == record_id
        )

    def validate(self) -> dict[str, Any]:
        return {"ok": True, "errors": [], "warnings": []}


@contextmanager
def running_server(reader: FakeReader):
    server = create_server(reader, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def get_json(base_url: str, path: str) -> tuple[int, dict[str, Any]]:
    with urllib.request.urlopen(f"{base_url}{path}") as response:
        return response.status, json.load(response)


def get_text(base_url: str, path: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(f"{base_url}{path}") as response:
        return response.status, response.headers["Content-Type"], response.read().decode("utf-8")


def test_run_endpoint_exposes_manifest_and_validation(tmp_path: Path) -> None:
    with running_server(FakeReader(tmp_path)) as base_url:
        status, payload = get_json(base_url, "/api/run")

    assert status == 200
    assert payload == {
        "manifest": {"run_id": "run-test", "schema_version": "fhp-trace/v1"},
        "summary": {"record_count": 0, "stage_counts": {}},
        "validation": {"errors": [], "ok": True, "warnings": []},
    }


def test_frames_endpoint_filters_then_aggregates_and_paginates(tmp_path: Path) -> None:
    records = [
        {
            "record_id": "rec-detect-1",
            "stage": "detection",
            "payload": {"frame_id": "frame/0001", "timestamp_ns": 100, "track_id": "hand-0"},
        },
        {
            "record_id": "rec-pose-1",
            "stage": "pose_2d",
            "payload": {"frame_id": "frame/0001", "timestamp_ns": 100, "track_id": "hand-0"},
        },
        {
            "record_id": "rec-detect-2",
            "stage": "detection",
            "payload": {"frame_id": "frame/0002", "timestamp_ns": 200, "track_id": "hand-1"},
        },
    ]

    with running_server(FakeReader(tmp_path, records)) as base_url:
        status, payload = get_json(
            base_url, "/api/frames?stage=detection&track_id=hand-0&offset=0&limit=1"
        )

    assert status == 200
    assert payload == {
        "items": [
            {
                "frame_id": "frame/0001",
                "record_ids": ["rec-detect-1"],
                "stages": ["detection"],
                "timestamp_ns": 100,
                "track_ids": ["hand-0"],
            }
        ],
        "global_record_ids": [],
        "limit": 1,
        "offset": 0,
        "total": 1,
    }


def test_frame_and_record_endpoints_return_original_serialized_records(tmp_path: Path) -> None:
    record = FakeTraceRecord(
        record_id="pose record/1",
        stage="pose_2d",
        payload={
            "frame_id": "frame/0001",
            "timestamp_ns": 100,
            "track_id": "hand-0",
            "keypoints_uv": [[10.0, 20.0]] * 21,
        },
    )
    reader = FakeReader(tmp_path, [record])
    frame_path = urllib.parse.quote("frame/0001", safe="")
    record_path = urllib.parse.quote("pose record/1", safe="")

    with running_server(reader) as base_url:
        frame_status, frame_payload = get_json(base_url, f"/api/frames/{frame_path}")
        record_status, record_payload = get_json(base_url, f"/api/records/{record_path}")

    expected = {
        "payload": {
            "frame_id": "frame/0001",
            "keypoints_uv": [[10.0, 20.0]] * 21,
            "timestamp_ns": 100,
            "track_id": "hand-0",
        },
        "record_id": "pose record/1",
        "stage": "pose_2d",
    }
    assert frame_status == 200
    assert frame_payload == {"frame_id": "frame/0001", "records": [expected]}
    assert record_status == 200
    assert record_payload == expected


def test_artifact_endpoint_serves_only_hash_addressed_files_inside_run_root(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifacts" / "overlay.png"
    artifact.parent.mkdir()
    artifact.write_bytes(b"not-a-real-png-but-known")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    outside = tmp_path.parent / "viewer-secret.txt"
    outside.write_bytes(b"do not serve")
    outside_digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    records = [
        {
            "record_id": "rec-artifacts",
            "stage": "visualization",
            "payload": {"frame_id": "frame-1"},
            "artifacts": [
                {
                    "role": "overlay",
                    "path": "artifacts/overlay.png",
                    "sha256": digest,
                    "media_type": "image/png",
                },
                {
                    "role": "source_left",
                    "path": "../viewer-secret.txt",
                    "sha256": outside_digest,
                },
            ],
        }
    ]

    with running_server(FakeReader(tmp_path, records)) as base_url:
        with urllib.request.urlopen(f"{base_url}/artifacts/{digest}") as response:
            body = response.read()
            content_type = response.headers["Content-Type"]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{base_url}/artifacts/{outside_digest}")

    assert body == b"not-a-real-png-but-known"
    assert content_type == "image/png"
    assert error.value.code == 404
    assert json.load(error.value) == {
        "error": {"code": "not_found", "message": "Artifact not found"}
    }


def test_static_viewer_is_self_contained_and_server_rejects_writes(tmp_path: Path) -> None:
    with running_server(FakeReader(tmp_path)) as base_url:
        status, content_type, html = get_text(base_url, "/")
        js_status, js_type, javascript = get_text(base_url, "/app.js")
        request = urllib.request.Request(f"{base_url}/api/run", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)

    assert status == 200
    assert content_type.startswith("text/html")
    assert all(
        identifier in html
        for identifier in (
            'id="run-summary"',
            'id="stage-filter"',
            'id="frame-list"',
            'id="record-list"',
            'id="skeleton-canvas"',
        )
    )
    assert "https://" not in html and "http://" not in html
    assert js_status == 200
    assert "javascript" in js_type
    assert all(
        field in javascript
        for field in (
            "detections",
            "bbox_xyxy",
            "keypoints_uv",
            "keypoint_scores",
            "landmarks_xyz_m",
            "validity",
            "source_left",
            "overlay",
            "filterFrameRecords",
            "contextRecords",
        )
    )
    assert error.value.code == 405
    assert json.load(error.value) == {
        "error": {"code": "method_not_allowed", "message": "Read-only viewer"}
    }


def test_server_rejects_non_loopback_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_server(FakeReader(tmp_path), host="0.0.0.0")


def test_real_trace_reader_filters_enum_stage_and_serves_blob_ref(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    writer = RunArtifactWriter.create(
        run_root,
        run_id="real-reader",
        pipeline_version="test",
    )
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
    blob = writer.put_blob(svg, role="source_left", media_type="image/svg+xml", suffix=".svg")
    writer.append(
        record_id="detection-1",
        stage=TraceStage.DETECTION,
        status=TraceStatus.SUCCEEDED,
        event="detected",
        payload={"frame_id": "f-1", "track_id": "hand-0"},
        blobs=(blob,),
    )
    writer.finalize(status=RunStatus.COMPLETED, summary={"stage_counts": {"DETECTION": 1}})

    with running_server(RunArtifactReader(run_root)) as base_url:
        _, frames = get_json(base_url, "/api/frames?stage=DETECTION")
        _, run = get_json(base_url, "/api/run")
        with urllib.request.urlopen(f"{base_url}/artifacts/{blob.sha256}") as response:
            blob_body = response.read()
            blob_type = response.headers["Content-Type"]

    assert frames["items"][0]["record_ids"] == ["detection-1"]
    assert run["summary"]["status"] == "COMPLETED"
    assert run["validation"]["status"] == "COMPLETED"
    assert blob_body == svg
    assert blob_type == "image/svg+xml"
