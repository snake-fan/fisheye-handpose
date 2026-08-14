from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fisheye_handpose.trace import (
    RunArtifactReader,
    RunArtifactWriter,
    RunStatus,
    TraceStage,
    TraceStatus,
)

import fisheye_trace_api.catalog as catalog_module
from fisheye_trace_api.catalog import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    TraceCatalog,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_run(root: Path, *, run_id: str = "capture-001") -> Path:
    run = root / run_id
    run.mkdir(parents=True)
    _write_json(
        run / "run_manifest.json",
        {
            "schema_version": "fhp-trace/v1",
            "run_id": run_id,
            "status": "ACTIVE",
            "pipeline_version": "0.1.0",
            "created_at_utc": "2026-08-13T01:02:03Z",
            "metadata": {"data_item_id": "session-42"},
        },
    )
    _write_json(
        run / "run_summary.json",
        {
            "status": "COMPLETED",
            "finalized_at_utc": "2026-08-13T01:03:03Z",
            "record_count": 2,
            "summary": {"stage_counts": {"DECODE": 1, "POSE_2D": 1}},
        },
    )
    records = [
        {
            "record_id": "decode:left:7",
            "ordinal": 0,
            "stage": "DECODE",
            "status": "SUCCEEDED",
            "event": "decoded",
            "payload": {
                "frame_id": "frame/000007",
                "frame_index": 7,
                "timestamp_ns": 70,
                "view_id": "left",
            },
            "blobs": [],
        },
        {
            "record_id": "pose:left:7",
            "ordinal": 1,
            "stage": "POSE_2D",
            "status": "WARNING",
            "event": "pose",
            "payload": {
                "frame_id": "frame/000007",
                "frame_index": 7,
                "timestamp_ns": 70,
                "track_id": "hand-0",
                "view_id": "left",
            },
            "blobs": [],
        },
    ]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run


def _key(catalog: TraceCatalog, run_id: str) -> str:
    return next(
        item["run_key"] for item in catalog.list_runs()["items"] if item["run_id"] == run_id
    )


def _sha256_json(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_canonical_active_run(
    root: Path,
    *,
    run_id: str,
    record_extra: dict[str, object] | None = None,
) -> Path:
    run = root / run_id
    run.mkdir(parents=True)
    manifest_body: dict[str, object] = {
        "schema_version": "fisheye-handpose/run-manifest/v1",
        "run_id": run_id,
        "status": "ACTIVE",
        "pipeline_version": "test",
        "created_at_utc": "2026-08-13T00:00:00Z",
        "config": {},
        "inputs": [],
        "metadata": {"item_id": "canonical-item"},
        "artifacts": {
            "trace": "trace.jsonl",
            "summary": "run_summary.json",
            "blob_root": "blobs/sha256",
        },
        "trace_schema_version": "fisheye-handpose/trace-record/v1",
        "hash_algorithm": "sha256",
    }
    _write_json(
        run / "run_manifest.json",
        {**manifest_body, "manifest_hash": _sha256_json(manifest_body)},
    )
    record_body: dict[str, object] = {
        "schema_version": "fisheye-handpose/trace-record/v1",
        "ordinal": 0,
        "record_id": "canonical:frame:0",
        "timestamp_utc": "2026-08-13T00:00:01Z",
        "stage": "DECODE",
        "status": "SUCCEEDED",
        "event": "decoded",
        "parent_ids": [],
        "blobs": [],
        "payload": {"frame_id": "frame/zero", "timestamp_ns": 1},
        "previous_hash": None,
    }
    if record_extra:
        record_body.update(record_extra)
    record = {**record_body, "record_hash": _sha256_json(record_body)}
    (run / "trace.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return run


def test_catalog_lists_completed_runs_with_derived_counts(tmp_path: Path) -> None:
    _write_run(tmp_path)

    page = TraceCatalog(tmp_path).list_runs(offset=0, limit=20)

    assert page == {
        "items": [
            {
                "run_key": "4bedf626347c6219",
                "item_id": "session-42",
                "run_id": "capture-001",
                "data_item_id": "session-42",
                "status": "COMPLETED",
                "created_at_utc": "2026-08-13T01:02:03Z",
                "finalized_at_utc": "2026-08-13T01:03:03Z",
                "pipeline_version": "0.1.0",
                "record_count": 2,
                "frame_count": 1,
                "stage_counts": {"DECODE": 1, "POSE_2D": 1},
                "warning_count": 1,
                "failure_count": 0,
            }
        ],
        "offset": 0,
        "limit": 20,
        "total": 1,
    }


def test_catalog_run_detail_keeps_source_documents_and_filter_vocabularies(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)

    catalog = TraceCatalog(tmp_path)
    detail = catalog.get_run(_key(catalog, "capture-001"))

    assert detail == {
        "run": {
            "run_key": "4bedf626347c6219",
            "item_id": "session-42",
            "run_id": "capture-001",
            "data_item_id": "session-42",
            "status": "COMPLETED",
            "created_at_utc": "2026-08-13T01:02:03Z",
            "finalized_at_utc": "2026-08-13T01:03:03Z",
            "pipeline_version": "0.1.0",
            "record_count": 2,
            "frame_count": 1,
            "stage_counts": {"DECODE": 1, "POSE_2D": 1},
            "warning_count": 1,
            "failure_count": 0,
        },
        "manifest": {
            "schema_version": "fhp-trace/v1",
            "run_id": "capture-001",
            "status": "ACTIVE",
            "pipeline_version": "0.1.0",
            "created_at_utc": "2026-08-13T01:02:03Z",
            "metadata": {"data_item_id": "session-42"},
        },
        "summary": {
            "status": "COMPLETED",
            "finalized_at_utc": "2026-08-13T01:03:03Z",
            "record_count": 2,
            "summary": {"stage_counts": {"DECODE": 1, "POSE_2D": 1}},
        },
        "validation": {
            "ok": False,
            "mode": "LEGACY_UNVERIFIED",
            "errors": ["legacy trace has no canonical v1 integrity proof"],
            "warnings": [],
        },
        "provenance": {},
        "stages": ["DECODE", "POSE_2D"],
        "track_ids": ["hand-0"],
        "view_ids": ["left"],
        "global_records": [],
    }


def test_frame_page_filters_records_before_aggregating_frames(tmp_path: Path) -> None:
    _write_run(tmp_path)

    catalog = TraceCatalog(tmp_path)
    page = catalog.list_frames(
        _key(catalog, "capture-001"),
        offset=0,
        limit=10,
        stage="POSE_2D",
        track_id="hand-0",
        status="WARNING",
    )

    assert page == {
        "items": [
            {
                "frame_key": "968794e0bebc6459d9cd1071dbe6570d1de13a64dc85b31c51fb5ef1ca65b7e5",
                "frame_id": "frame/000007",
                "frame_index": 7,
                "timestamp_ns": 70,
                "record_ids": ["pose:left:7"],
                "stages": ["POSE_2D"],
                "statuses": ["WARNING"],
                "track_ids": ["hand-0"],
                "view_ids": ["left"],
            }
        ],
        "offset": 0,
        "limit": 10,
        "total": 1,
    }


def test_frame_query_returns_all_original_stage_records(tmp_path: Path) -> None:
    _write_run(tmp_path)

    catalog = TraceCatalog(tmp_path)
    result = catalog.get_frame(
        _key(catalog, "capture-001"),
        "968794e0bebc6459d9cd1071dbe6570d1de13a64dc85b31c51fb5ef1ca65b7e5",
    )

    assert result["run_key"] == "4bedf626347c6219"
    assert result["run_id"] == "capture-001"
    assert result["frame"] == {
        "frame_key": "968794e0bebc6459d9cd1071dbe6570d1de13a64dc85b31c51fb5ef1ca65b7e5",
        "frame_id": "frame/000007",
        "frame_index": 7,
        "timestamp_ns": 70,
        "record_ids": ["decode:left:7", "pose:left:7"],
        "stages": ["DECODE", "POSE_2D"],
        "statuses": ["SUCCEEDED", "WARNING"],
        "track_ids": ["hand-0"],
        "view_ids": ["left"],
    }
    assert [record["record_id"] for record in result["records"]] == [
        "decode:left:7",
        "pose:left:7",
    ]


def test_record_query_selects_one_stage_and_frame_index(tmp_path: Path) -> None:
    _write_run(tmp_path)

    catalog = TraceCatalog(tmp_path)
    result = catalog.get_records(
        _key(catalog, "capture-001"),
        "POSE_2D",
        "968794e0bebc6459d9cd1071dbe6570d1de13a64dc85b31c51fb5ef1ca65b7e5",
    )

    assert result == {
        "run_key": "4bedf626347c6219",
        "run_id": "capture-001",
        "stage": "POSE_2D",
        "frame_key": "968794e0bebc6459d9cd1071dbe6570d1de13a64dc85b31c51fb5ef1ca65b7e5",
        "items": [
            {
                "record_id": "pose:left:7",
                "ordinal": 1,
                "stage": "POSE_2D",
                "status": "WARNING",
                "event": "pose",
                "payload": {
                    "frame_id": "frame/000007",
                    "frame_index": 7,
                    "timestamp_ns": 70,
                    "track_id": "hand-0",
                    "view_id": "left",
                },
                "blobs": [],
            }
        ],
        "total": 1,
    }


def test_record_id_query_returns_the_original_record(tmp_path: Path) -> None:
    _write_run(tmp_path)

    catalog = TraceCatalog(tmp_path)
    record = catalog.get_record(_key(catalog, "capture-001"), "pose:left:7")

    assert record["record_id"] == "pose:left:7"
    assert record["payload"]["track_id"] == "hand-0"


def test_artifact_resolution_requires_an_integrity_checked_trace_reference(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path)
    body = b"0123456789"
    digest = hashlib.sha256(body).hexdigest()
    artifact_path = run / "blobs" / "sha256" / digest[:2] / f"{digest}.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(body)
    records = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
    records[0]["blobs"] = [
        {
            "sha256": digest,
            "bytes": 10,
            "role": "source_left",
            "media_type": "application/octet-stream",
            "relative_path": artifact_path.relative_to(run).as_posix(),
        }
    ]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    catalog = TraceCatalog(tmp_path)
    artifact = catalog.resolve_artifact(
        _key(catalog, "capture-001"), artifact_path.relative_to(run).as_posix()
    )

    assert artifact.path == artifact_path.resolve()
    assert artifact.media_type == "application/octet-stream"
    assert artifact.size == 10
    assert artifact.sha256 == digest


def test_artifact_resolution_rejects_invalid_filesystem_names(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    records = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
    records[0]["artifacts"] = [
        {
            "sha256": "0" * 64,
            "bytes": 0,
            "path": "bad\0name.bin",
        }
    ]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    catalog = TraceCatalog(tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        catalog.resolve_artifact(_key(catalog, "capture-001"), "bad\0name.bin")


def test_catalog_reads_legacy_manifest_record_names_and_flat_frame_metadata(
    tmp_path: Path,
) -> None:
    run = tmp_path / "old-folder"
    run.mkdir()
    _write_json(
        run / "manifest.json",
        {
            "schema_version": "fhp-trace/v1",
            "run_id": "legacy-01",
            "status": "COMPLETED",
            "pipeline_version": "legacy",
            "created_at_utc": "2025-01-01T00:00:00Z",
            "data_item_id": "old-session",
        },
    )
    (run / "records.jsonl").write_text(
        json.dumps(
            {
                "id": "old-pose-3",
                "stage": "POSE_2D",
                "status": "SUCCEEDED",
                "frame_id": "old-frame-3",
                "frame_idx": 3,
                "timestamp_ns": 300,
                "track_id": "old-hand",
                "view_id": "right",
                "artifacts": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    catalog = TraceCatalog(tmp_path)

    assert catalog.list_runs()["items"][0]["run_id"] == "legacy-01"
    assert catalog.list_runs()["items"][0]["data_item_id"] == "old-session"
    assert catalog.list_frames(_key(catalog, "legacy-01"))["items"] == [
        {
            "frame_key": "2369d3cc68657b8f0fe454501f6538ff8a8a912af4aa0cb38e790ec33c5e13b2",
            "frame_id": "old-frame-3",
            "frame_index": 3,
            "timestamp_ns": 300,
            "record_ids": ["old-pose-3"],
            "stages": ["POSE_2D"],
            "statuses": ["SUCCEEDED"],
            "track_ids": ["old-hand"],
            "view_ids": ["right"],
        }
    ]


def test_run_summary_uses_canonical_item_id_metadata(tmp_path: Path) -> None:
    run = _write_run(tmp_path, run_id="canonical-item")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["metadata"] = {"item_id": "capture-item", "source_item_id": "raw-item"}
    _write_json(run / "run_manifest.json", manifest)

    item = TraceCatalog(tmp_path).list_runs()["items"][0]

    assert item["data_item_id"] == "capture-item"


def test_run_detail_reports_a_broken_canonical_record_hash(tmp_path: Path) -> None:
    run = tmp_path / "strict"
    run.mkdir()
    manifest_body = {
        "schema_version": "fisheye-handpose/run-manifest/v1",
        "run_id": "strict-run",
        "status": "ACTIVE",
        "pipeline_version": "test",
        "created_at_utc": "2026-08-13T00:00:00Z",
        "config": {},
        "inputs": [],
        "metadata": {},
        "artifacts": {
            "trace": "trace.jsonl",
            "summary": "run_summary.json",
            "blob_root": "blobs/sha256",
        },
        "trace_schema_version": "fisheye-handpose/trace-record/v1",
        "hash_algorithm": "sha256",
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(run / "run_manifest.json", {**manifest_body, "manifest_hash": manifest_hash})
    record = {
        "schema_version": "fisheye-handpose/trace-record/v1",
        "ordinal": 0,
        "record_id": "strict:decode:0",
        "timestamp_utc": "2026-08-13T00:00:01Z",
        "stage": "DECODE",
        "status": "SUCCEEDED",
        "event": "decoded",
        "parent_ids": [],
        "blobs": [],
        "payload": {"frame_id": "frame/000000", "frame_index": 0},
        "previous_hash": None,
        "record_hash": "0" * 64,
    }
    (run / "trace.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    catalog = TraceCatalog(tmp_path)
    validation = catalog.get_run(_key(catalog, "strict-run"))["validation"]

    assert validation["ok"] is False
    assert validation["errors"] == ["record hash mismatch: strict:decode:0"]


def test_run_detail_exposes_stage_records_without_a_frame_id(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    with (run / "trace.jsonl").open("a", encoding="utf-8") as trace:
        trace.write(
            json.dumps(
                {
                    "record_id": "pipeline:skipped:mano",
                    "ordinal": 2,
                    "stage": "KINEMATIC_REFINEMENT",
                    "status": "SKIPPED",
                    "event": "stage_output_not_produced",
                    "payload": {"output_status": "NOT_PRODUCED", "reason": "no backend"},
                    "blobs": [],
                }
            )
            + "\n"
        )

    catalog = TraceCatalog(tmp_path)
    global_records = catalog.get_run(_key(catalog, "capture-001"))["global_records"]

    assert [record["record_id"] for record in global_records] == ["pipeline:skipped:mano"]


def test_run_key_disambiguates_equal_run_ids_in_different_data_items(tmp_path: Path) -> None:
    _write_run(tmp_path / "item-a", run_id="shared-run")
    second = _write_run(tmp_path / "item-b", run_id="shared-run")
    manifest = json.loads((second / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["metadata"] = {"item_id": "item-b"}
    _write_json(second / "run_manifest.json", manifest)

    catalog = TraceCatalog(tmp_path)
    items = catalog.list_runs()["items"]

    assert len({item["run_key"] for item in items}) == 2
    item_b = next(item for item in items if item["item_id"] == "item-b")
    assert item_b["run_id"] == "shared-run"
    assert catalog.get_run(item_b["run_key"])["run"] == item_b


def test_item_id_falls_back_to_parent_folder_for_old_traces(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "legacy-item", run_id="old-run")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["metadata"] = {}
    _write_json(run / "run_manifest.json", manifest)

    item = TraceCatalog(tmp_path).list_runs()["items"][0]

    assert item["item_id"] == "legacy-item"


def test_catalog_keeps_healthy_runs_visible_when_one_active_run_is_corrupt(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, run_id="healthy-run")
    corrupt = tmp_path / "corrupt-active"
    corrupt.mkdir()
    _write_json(
        corrupt / "run_manifest.json",
        {
            "schema_version": "fisheye-handpose/run-manifest/v1",
            "run_id": "corrupt-active",
            "status": "ACTIVE",
            "created_at_utc": "2026-08-13T02:00:00Z",
            "metadata": {"item_id": "capture-corrupt"},
        },
    )
    (corrupt / "trace.jsonl").write_text('{"record_id":"partial', encoding="utf-8")

    page = TraceCatalog(tmp_path).list_runs()

    assert page["total"] == 2
    healthy = next(item for item in page["items"] if item["run_id"] == "healthy-run")
    invalid = next(item for item in page["items"] if item["run_id"] == "corrupt-active")
    assert healthy["status"] == "COMPLETED"
    assert invalid["status"] == "INVALID"
    assert invalid["catalog_error"]["code"] == "corrupt_run"
    detail = TraceCatalog(tmp_path).get_run(invalid["run_key"])
    assert detail["run"] == invalid
    assert detail["validation"]["ok"] is False
    assert detail["validation"]["mode"] == "CATALOG_ERROR"
    assert detail["provenance"] == {}
    assert detail["global_records"] == []


def test_canonical_validation_rejects_hash_consistent_unknown_record_fields(
    tmp_path: Path,
) -> None:
    _write_canonical_active_run(
        tmp_path,
        run_id="schema-invalid",
        record_extra={"unexpected": True},
    )
    catalog = TraceCatalog(tmp_path)

    validation = catalog.get_run(_key(catalog, "schema-invalid"))["validation"]

    assert catalog.list_runs()["items"][0]["status"] == "INVALID"
    assert validation["ok"] is False
    assert any("missing or unknown fields" in error for error in validation["errors"])


def test_frame_keys_address_duplicate_indexes_and_non_numeric_frame_ids(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path, run_id="mixed-frame-ids")
    frames = (
        ("part0001/pair000000", 0),
        ("part0002/pair000000", 0),
        ("phase-alpha", None),
    )
    records = []
    for ordinal, (frame_id, frame_index) in enumerate(frames):
        payload: dict[str, object] = {"frame_id": frame_id, "timestamp_ns": ordinal + 1}
        if frame_index is not None:
            payload["frame_index"] = frame_index
        records.append(
            {
                "record_id": f"decode:{ordinal}",
                "ordinal": ordinal,
                "stage": "DECODE",
                "status": "SUCCEEDED",
                "event": "decoded",
                "payload": payload,
                "blobs": [],
            }
        )
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    catalog = TraceCatalog(tmp_path)
    run_key = _key(catalog, "mixed-frame-ids")

    page = catalog.list_frames(run_key)

    assert {item["frame_key"] for item in page["items"]} == {
        "626836bca19cce71ff9851e5ed6db00e3a94611b01c40179ec9b31fba4e7759b",
        "536cb045b049a64a6ebae3e059035cee9e1ccd3e0d7d851a8ba64e9e532baa1b",
        "dfa2c2091e8fec48b04d3262b993467d92f62514146288951f53185cd6ff7cb9",
    }
    alpha = catalog.get_frame(
        run_key,
        "dfa2c2091e8fec48b04d3262b993467d92f62514146288951f53185cd6ff7cb9",
    )
    assert alpha["frame"]["frame_id"] == "phase-alpha"
    assert alpha["frame"]["frame_index"] is None
    assert [record["record_id"] for record in alpha["records"]] == ["decode:2"]


def test_frame_summary_uses_worker_pair_time_and_backfills_later_timestamp(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path, run_id="worker-times")
    records = [
        {
            "record_id": "sync:pair-time",
            "stage": "SYNCHRONIZATION",
            "status": "SUCCEEDED",
            "payload": {
                "frame_id": "part0001/pair000000",
                "frame_index": 0,
                "pair_timestamp_ns": 1_250_000,
            },
        },
        {
            "record_id": "sync:no-time",
            "stage": "SYNCHRONIZATION",
            "status": "SUCCEEDED",
            "payload": {"frame_id": "part0001/pair000001"},
        },
        {
            "record_id": "pose:later-time",
            "stage": "POSE_2D",
            "status": "SUCCEEDED",
            "payload": {
                "frame_id": "part0001/pair000001",
                "frame_index": 1,
                "timestamp_ns": 2_500_000,
            },
        },
    ]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    catalog = TraceCatalog(tmp_path)

    frames = catalog.list_frames(_key(catalog, "worker-times"))["items"]

    assert [(frame["frame_index"], frame["timestamp_ns"]) for frame in frames] == [
        (0, 1_250_000),
        (1, 2_500_000),
    ]


def test_run_detail_extracts_only_evidence_backed_provenance_facets(tmp_path: Path) -> None:
    run = tmp_path / "canonical-provenance"
    writer = RunArtifactWriter.create(
        run,
        run_id="canonical-provenance",
        pipeline_version="test-revision",
        config={"audit": {"max_skew_ns": 1_000_000}},
        metadata={
            "item_id": "capture-17",
            "source_item_id": "raw capture 17",
            "producer": "run-item",
        },
    )
    system = writer.append(
        record_id="h20:system:verified",
        stage=TraceStage.SYSTEM,
        status=TraceStatus.SUCCEEDED,
        event="worker_inputs_verified",
        payload={
            "request_sha256": "1" * 64,
            "model_manifest_sha256": "2" * 64,
            "mmpose_commit": "0123456789abcdef",
            "worker_provenance": {"event_id": "system:verified", "ordinal": 0},
        },
    )
    writer.append(
        record_id="h20:calibration:rectification",
        stage=TraceStage.RECTIFICATION,
        status=TraceStatus.SUCCEEDED,
        event="worker_rectification_loaded",
        payload={
            "calibration_id": f"sha256:{'3' * 64}",
            "image_size": [1920, 1080],
            "output_size": [1024, 1024],
            "length_unit": "m",
            "coordinate_frame": "rectified_left_camera",
            "baseline_m": 0.064,
            "worker_provenance": {"event_id": "calibration:rectification", "ordinal": 1},
        },
        parent_ids=(system.record_id,),
    )
    writer.finalize(status=RunStatus.COMPLETED, summary={"output_status": "PRODUCED"})
    catalog = TraceCatalog(tmp_path)

    provenance = catalog.get_run(_key(catalog, "canonical-provenance"))["provenance"]

    assert provenance == {
        "manifest": {
            "config": {"audit": {"max_skew_ns": 1_000_000}},
            "metadata": {
                "item_id": "capture-17",
                "source_item_id": "raw capture 17",
                "producer": "run-item",
            },
        },
        "worker_inputs": {
            "record_id": "h20:system:verified",
            "payload": {
                "request_sha256": "1" * 64,
                "model_manifest_sha256": "2" * 64,
                "mmpose_commit": "0123456789abcdef",
            },
            "worker_provenance": {"event_id": "system:verified", "ordinal": 0},
        },
        "calibration": {
            "record_id": "h20:calibration:rectification",
            "payload": {
                "calibration_id": f"sha256:{'3' * 64}",
                "image_size": [1920, 1080],
                "output_size": [1024, 1024],
                "length_unit": "m",
                "coordinate_frame": "rectified_left_camera",
                "baseline_m": 0.064,
            },
            "worker_provenance": {"event_id": "calibration:rectification", "ordinal": 1},
        },
    }

    no_worker = _write_canonical_active_run(tmp_path, run_id="no-worker-provenance")
    assert no_worker.is_dir()
    no_worker_catalog = TraceCatalog(tmp_path)
    no_worker_detail = no_worker_catalog.get_run(_key(no_worker_catalog, "no-worker-provenance"))
    assert set(no_worker_detail["provenance"]) == {"manifest"}


def test_completed_run_reuses_one_parsed_index_across_catalog_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, run_id="stable-completed")
    parse_count = 0
    original_load_jsonl = catalog_module._load_jsonl

    def counted_load_jsonl(path: Path) -> list[dict[str, object]]:
        nonlocal parse_count
        parse_count += 1
        return original_load_jsonl(path)

    monkeypatch.setattr(catalog_module, "_load_jsonl", counted_load_jsonl)
    catalog = TraceCatalog(tmp_path)

    run = catalog.list_runs()["items"][0]
    run_key = run["run_key"]
    frame = catalog.list_frames(run_key)["items"][0]
    catalog.get_run(run_key)
    catalog.get_frame(run_key, frame["frame_key"])
    catalog.get_record(run_key, "pose:left:7")
    catalog.list_runs()

    assert parse_count == 1


def test_completed_run_index_is_rebuilt_when_trace_fingerprint_changes(tmp_path: Path) -> None:
    run = _write_run(tmp_path, run_id="changed-completed")
    catalog = TraceCatalog(tmp_path)
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.list_frames(run_key)["total"] == 1
    with (run / "trace.jsonl").open("a", encoding="utf-8") as trace:
        trace.write(
            json.dumps(
                {
                    "record_id": "decode:right:8",
                    "ordinal": 2,
                    "stage": "DECODE",
                    "status": "SUCCEEDED",
                    "event": "decoded",
                    "payload": {"frame_id": "frame/000008", "frame_index": 8},
                    "blobs": [],
                }
            )
            + "\n"
        )

    assert catalog.list_frames(run_key)["total"] == 2


def test_active_run_is_never_frozen_in_the_completed_index(tmp_path: Path) -> None:
    run = _write_run(tmp_path, run_id="still-writing")
    (run / "run_summary.json").unlink()
    catalog = TraceCatalog(tmp_path)
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.list_frames(run_key)["total"] == 1
    with (run / "trace.jsonl").open("a", encoding="utf-8") as trace:
        trace.write(
            json.dumps(
                {
                    "record_id": "decode:right:8",
                    "ordinal": 2,
                    "stage": "DECODE",
                    "status": "SUCCEEDED",
                    "event": "decoded",
                    "payload": {"frame_id": "frame/000008", "frame_index": 8},
                    "blobs": [],
                }
            )
            + "\n"
        )

    assert catalog.list_frames(run_key)["total"] == 2


def test_completed_canonical_run_performs_full_validation_once_within_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "validated-once"
    writer = RunArtifactWriter.create(
        run,
        run_id="validated-once",
        pipeline_version="test",
    )
    writer.append(
        record_id="decode:0",
        stage=TraceStage.DECODE,
        status=TraceStatus.SUCCEEDED,
        event="decoded",
        payload={"frame_id": "frame/000000", "frame_index": 0},
    )
    writer.finalize(status=RunStatus.COMPLETED)
    validation_modes: list[bool] = []
    original_validate = catalog_module.RunArtifactReader.validate

    def counted_validate(
        reader: RunArtifactReader,
        *,
        verify_blobs: bool = True,
    ) -> object:
        validation_modes.append(verify_blobs)
        return original_validate(reader, verify_blobs=verify_blobs)

    monkeypatch.setattr(catalog_module.RunArtifactReader, "validate", counted_validate)
    catalog = TraceCatalog(tmp_path)
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.get_run(run_key)["validation"]["ok"] is True
    assert catalog.get_run(run_key)["validation"]["ok"] is True

    assert validation_modes == [False, True]


def test_completed_run_detail_reuses_blob_validation_only_within_the_short_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "validated-blob-snapshot"
    writer = RunArtifactWriter.create(
        run,
        run_id="validated-blob-snapshot",
        pipeline_version="test",
    )
    blob = writer.put_blob(
        b"immutable artifact",
        role="source_left",
        media_type="application/octet-stream",
        suffix=".bin",
    )
    writer.append(
        record_id="decode:0",
        stage=TraceStage.DECODE,
        status=TraceStatus.SUCCEEDED,
        event="decoded",
        payload={"frame_id": "frame/000000", "frame_index": 0},
        blobs=(blob,),
    )
    writer.finalize(status=RunStatus.COMPLETED)
    artifact_path = (run / blob.relative_path).resolve()
    artifact_stats = 0
    original_stat = Path.stat

    def counted_stat(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal artifact_stats
        if path == artifact_path:
            artifact_stats += 1
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counted_stat)
    current_time = 100.0

    def clock() -> float:
        return current_time

    catalog = TraceCatalog(tmp_path, validation_cache_ttl_seconds=1.0, clock=clock)
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.get_run(run_key)["validation"]["ok"] is True
    first_request_stats = artifact_stats
    assert first_request_stats > 0

    assert catalog.get_run(run_key)["validation"]["ok"] is True
    assert artifact_stats == first_request_stats

    current_time = 101.0
    assert catalog.get_run(run_key)["validation"]["ok"] is True
    assert artifact_stats > first_request_stats


def test_artifact_hash_result_is_reused_until_the_file_stat_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_run(tmp_path, run_id="artifact-cache")
    body = b"0123456789"
    digest = hashlib.sha256(body).hexdigest()
    relative_path = f"blobs/sha256/{digest[:2]}/{digest}.bin"
    artifact_path = run / relative_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(body)
    records = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
    records[0]["blobs"] = [
        {
            "sha256": digest,
            "bytes": len(body),
            "role": "source_left",
            "media_type": "application/octet-stream",
            "relative_path": relative_path,
        }
    ]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    artifact_reads = 0
    original_open = Path.open

    def counted_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal artifact_reads
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == artifact_path and mode == "rb":
            artifact_reads += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    catalog = TraceCatalog(tmp_path)
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.resolve_artifact(run_key, relative_path).size == len(body)
    assert catalog.resolve_artifact(run_key, relative_path).size == len(body)
    assert artifact_reads == 1

    artifact_path.write_bytes(b"abcdefghij")
    with pytest.raises(ArtifactIntegrityError):
        catalog.resolve_artifact(run_key, relative_path)


def test_completed_run_detects_unrequested_blob_tamper_after_validation_ttl(
    tmp_path: Path,
) -> None:
    run = tmp_path / "validated-blob"
    writer = RunArtifactWriter.create(
        run,
        run_id="validated-blob",
        pipeline_version="test",
    )
    blob = writer.put_blob(
        b"trusted artifact",
        role="source_left",
        media_type="application/octet-stream",
        suffix=".bin",
    )
    writer.append(
        record_id="decode:0",
        stage=TraceStage.DECODE,
        status=TraceStatus.SUCCEEDED,
        event="decoded",
        payload={"frame_id": "frame/000000", "frame_index": 0},
        blobs=(blob,),
    )
    writer.finalize(status=RunStatus.COMPLETED)
    current_time = 100.0

    def clock() -> float:
        return current_time

    catalog = TraceCatalog(tmp_path, validation_cache_ttl_seconds=1.0, clock=clock)
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.get_run(run_key)["validation"]["ok"] is True

    (run / blob.relative_path).write_bytes(b"tampered artifact")

    assert catalog.get_run(run_key)["validation"]["ok"] is True

    current_time = 101.0
    validation = catalog.get_run(run_key)["validation"]
    assert validation["ok"] is False
    assert validation["mode"] == "CANONICAL_V1"


def test_artifact_integrity_failure_invalidates_completed_run_validation_within_ttl(
    tmp_path: Path,
) -> None:
    run = tmp_path / "artifact-read-invalidates"
    writer = RunArtifactWriter.create(
        run,
        run_id="artifact-read-invalidates",
        pipeline_version="test",
    )
    blob = writer.put_blob(
        b"trusted artifact",
        role="source_left",
        media_type="application/octet-stream",
        suffix=".bin",
    )
    writer.append(
        record_id="decode:0",
        stage=TraceStage.DECODE,
        status=TraceStatus.SUCCEEDED,
        event="decoded",
        payload={"frame_id": "frame/000000", "frame_index": 0},
        blobs=(blob,),
    )
    writer.finalize(status=RunStatus.COMPLETED)
    catalog = TraceCatalog(
        tmp_path,
        validation_cache_ttl_seconds=1.0,
        clock=lambda: 100.0,
    )
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.get_run(run_key)["validation"]["ok"] is True

    (run / blob.relative_path).write_bytes(b"tampered artifact")

    with pytest.raises(ArtifactIntegrityError):
        catalog.resolve_artifact(run_key, blob.relative_path)
    validation = catalog.get_run(run_key)["validation"]
    assert validation["ok"] is False
    assert validation["mode"] == "CANONICAL_V1"


def test_active_run_revalidates_referenced_blobs_without_waiting_for_the_ttl(
    tmp_path: Path,
) -> None:
    run = tmp_path / "active-blob"
    writer = RunArtifactWriter.create(
        run,
        run_id="active-blob",
        pipeline_version="test",
    )
    blob = writer.put_blob(
        b"trusted artifact",
        role="source_left",
        media_type="application/octet-stream",
        suffix=".bin",
    )
    writer.append(
        record_id="decode:0",
        stage=TraceStage.DECODE,
        status=TraceStatus.SUCCEEDED,
        event="decoded",
        payload={"frame_id": "frame/000000", "frame_index": 0},
        blobs=(blob,),
    )
    catalog = TraceCatalog(
        tmp_path,
        validation_cache_ttl_seconds=1.0,
        clock=lambda: 100.0,
    )
    run_key = catalog.list_runs()["items"][0]["run_key"]

    assert catalog.get_run(run_key)["validation"]["ok"] is True

    (run / blob.relative_path).write_bytes(b"tampered artifact")

    validation = catalog.get_run(run_key)["validation"]
    assert validation["ok"] is False
    assert validation["mode"] == "CANONICAL_V1"
