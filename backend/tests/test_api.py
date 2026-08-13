from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from fisheye_trace_api.app import create_app


def _catalog_fixture(root: Path) -> tuple[Path, str, bytes]:
    run = root / "one"
    run.mkdir()
    manifest = {
        "run_id": "run-one",
        "status": "ACTIVE",
        "pipeline_version": "test",
        "created_at_utc": "2026-08-13T00:00:00Z",
        "metadata": {"data_item_id": "item-one"},
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    body = b"0123456789"
    digest = hashlib.sha256(body).hexdigest()
    relative_path = f"blobs/sha256/{digest[:2]}/{digest}.bin"
    artifact = run / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(body)
    record = {
        "record_id": "decode-4",
        "stage": "DECODE",
        "status": "SUCCEEDED",
        "event": "decoded",
        "payload": {"frame_id": "frame/000004", "frame_index": 4, "view_id": "left"},
        "blobs": [
            {
                "sha256": digest,
                "bytes": len(body),
                "role": "source_left",
                "media_type": "application/octet-stream",
                "relative_path": relative_path,
            }
        ],
    }
    (run / "trace.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return run, relative_path, body


def test_api_exposes_catalog_health_runs_frames_and_stage_records(tmp_path: Path) -> None:
    _catalog_fixture(tmp_path)
    client = TestClient(create_app(tmp_path))

    assert client.get("/api/v1/health").json() == {"status": "ok", "read_only": True}
    run = client.get("/api/v1/runs").json()["items"][0]
    run_key = run["run_key"]
    frame_key = "3ddca2a9387542857508c054b5b136db8af7462b5d243a6e51dcffb4b335eef6"
    assert (run["item_id"], run["run_id"]) == ("item-one", "run-one")
    assert client.get(f"/api/v1/runs/{run_key}").json()["run"]["item_id"] == "item-one"
    assert client.get(f"/api/v1/runs/{run_key}/frames").json()["items"][0]["frame_index"] == 4
    assert (
        client.get(f"/api/v1/runs/{run_key}/frames/{frame_key}").json()["records"][0]["record_id"]
        == "decode-4"
    )
    assert (
        client.get(f"/api/v1/runs/{run_key}/records/DECODE/{frame_key}").json()["items"][0][
            "record_id"
        ]
        == "decode-4"
    )
    assert client.get(f"/api/v1/runs/{run_key}/records/decode-4").json()["record_id"] == "decode-4"


def test_record_query_endpoint_resolves_h20_ids_that_contain_slashes(tmp_path: Path) -> None:
    run, _, _ = _catalog_fixture(tmp_path)
    record = json.loads((run / "trace.jsonl").read_text(encoding="utf-8"))
    record["record_id"] = "h20:part0001/pair000000:raw:track-0000"
    (run / "trace.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    run_key = client.get("/api/v1/runs").json()["items"][0]["run_key"]

    response = client.get(
        f"/api/v1/runs/{run_key}/record",
        params={"record_id": record["record_id"]},
    )

    assert response.status_code == 200
    assert response.json()["record_id"] == record["record_id"]


def test_artifact_endpoint_supports_single_byte_ranges(tmp_path: Path) -> None:
    _, relative_path, body = _catalog_fixture(tmp_path)
    client = TestClient(create_app(tmp_path))
    run_key = client.get("/api/v1/runs").json()["items"][0]["run_key"]
    url = f"/api/v1/runs/{run_key}/artifacts/{relative_path}"

    full = client.get(url)
    partial = client.get(url, headers={"Range": "bytes=2-5"})
    suffix = client.get(url, headers={"Range": "bytes=-3"})
    unsatisfiable = client.get(url, headers={"Range": "bytes=90-99"})

    assert (full.status_code, full.content) == (200, body)
    assert full.headers["accept-ranges"] == "bytes"
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert suffix.content == b"789"
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == "bytes */10"


def test_api_is_read_only_and_uses_structured_not_found_errors(tmp_path: Path) -> None:
    _catalog_fixture(tmp_path)
    client = TestClient(create_app(tmp_path))

    missing = client.get("/api/v1/runs/not-here")
    write = client.post("/api/v1/runs", json={})

    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "run_not_found", "message": "Run not found"}}
    assert write.status_code == 405
    assert write.json() == {"error": {"code": "method_not_allowed", "message": "Read-only API"}}


def test_checked_in_openapi_contract_covers_every_runtime_api_path(tmp_path: Path) -> None:
    _catalog_fixture(tmp_path)
    runtime = create_app(tmp_path).openapi()
    contract_path = Path(__file__).parents[2] / "contracts" / "trace-api-v1.openapi.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    runtime_paths = {path for path in runtime["paths"] if path.startswith("/api/v1/")}
    assert set(contract["paths"]) == runtime_paths
    for path in runtime_paths:
        assert set(contract["paths"][path]) == set(runtime["paths"][path])
    assert contract["info"]["version"] == "1.0.0"
    assert "FrameIndex" not in contract["components"]["parameters"]
    assert contract["components"]["parameters"]["FrameKey"]["schema"] == {
        "type": "string",
        "pattern": "^[0-9a-f]{64}$",
    }
    assert "frame_key" in contract["components"]["schemas"]["Frame"]["required"]
    assert "provenance" in contract["components"]["schemas"]["RunDetail"]["required"]
    provenance_schema = contract["components"]["schemas"]["RunProvenance"]
    assert provenance_schema["additionalProperties"] is False
    assert set(provenance_schema["properties"]) == {
        "manifest",
        "worker_inputs",
        "calibration",
    }


def test_api_lists_and_explains_a_corrupt_active_run_without_hiding_healthy_runs(
    tmp_path: Path,
) -> None:
    _catalog_fixture(tmp_path)
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "corrupt-active",
                "status": "ACTIVE",
                "created_at_utc": "2026-08-13T01:00:00Z",
                "metadata": {"item_id": "corrupt-item"},
            }
        ),
        encoding="utf-8",
    )
    (corrupt / "trace.jsonl").write_text('{"record_id":"partial', encoding="utf-8")
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/v1/runs")

    assert response.status_code == 200
    assert response.json()["total"] == 2
    invalid = next(item for item in response.json()["items"] if item["status"] == "INVALID")
    detail = client.get(f"/api/v1/runs/{invalid['run_key']}")
    assert detail.status_code == 200
    assert detail.json()["validation"]["mode"] == "CATALOG_ERROR"
