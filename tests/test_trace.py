from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye_handpose.trace import (
    RunArtifactReader,
    RunArtifactWriter,
    RunStatus,
    TraceStage,
    TraceStatus,
    TraceValidationError,
)


def test_writer_creates_manifest_and_reader_replays_one_record(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"

    writer = RunArtifactWriter.create(
        run_dir,
        run_id="run-001",
        pipeline_version="test-revision",
        config={"threshold": 0.25},
        inputs=[{"session_id": "capture-001"}],
    )
    record = writer.append(
        record_id="decode:000001:left",
        stage=TraceStage.DECODE,
        status=TraceStatus.SUCCEEDED,
        event="frame_decoded",
        payload={"frame_index": 1},
    )
    writer.close()

    reader = RunArtifactReader(run_dir)
    assert reader.manifest["schema_version"] == "fisheye-handpose/run-manifest/v1"
    assert reader.manifest["run_id"] == "run-001"
    assert reader.manifest["config"] == {"threshold": 0.25}
    assert len(reader.manifest["manifest_hash"]) == 64
    assert record.ordinal == 0
    assert record.previous_hash is None
    assert len(record.record_hash) == 64
    assert reader.records() == (record,)
    assert reader.get("decode:000001:left") == record

    on_disk = json.loads((run_dir / "trace.jsonl").read_text().strip())
    assert on_disk["record_hash"] == record.record_hash


def test_trace_enums_are_pipeline_specific_and_stable() -> None:
    assert [stage.value for stage in TraceStage] == [
        "SYSTEM",
        "DISCOVERY",
        "CALIBRATION",
        "DECODE",
        "SYNCHRONIZATION",
        "RECTIFICATION",
        "DETECTION",
        "POSE_2D",
        "CROSS_VIEW_ASSOCIATION",
        "RAW_FUSION",
        "KINEMATIC_REFINEMENT",
        "TEMPORAL_REFINEMENT",
        "QA",
        "EXPORT",
    ]
    assert [status.value for status in TraceStatus] == [
        "STARTED",
        "SUCCEEDED",
        "FAILED",
        "WARNING",
        "SKIPPED",
    ]
    assert [status.value for status in RunStatus] == ["ACTIVE", "COMPLETED", "FAILED"]


def test_create_refuses_an_existing_output_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()

    with pytest.raises(FileExistsError):
        RunArtifactWriter.create(
            run_dir,
            run_id="run-existing",
            pipeline_version="test-revision",
        )


def test_records_form_a_parent_checked_hash_chain_and_can_be_filtered(tmp_path: Path) -> None:
    writer = RunArtifactWriter.create(
        tmp_path / "run",
        run_id="run-chain",
        pipeline_version="test-revision",
    )
    first = writer.append(
        record_id="detect:1",
        stage=TraceStage.DETECTION,
        status=TraceStatus.SUCCEEDED,
        event="hands_detected",
    )
    second = writer.append(
        record_id="pose:1",
        stage=TraceStage.POSE_2D,
        status=TraceStatus.WARNING,
        event="keypoints_estimated",
        parent_ids=(first.record_id,),
    )

    with pytest.raises(ValueError, match="duplicate trace record ID"):
        writer.append(
            record_id="pose:1",
            stage=TraceStage.POSE_2D,
            status=TraceStatus.SUCCEEDED,
            event="duplicate",
        )
    with pytest.raises(ValueError, match="unknown parent"):
        writer.append(
            record_id="fusion:1",
            stage=TraceStage.RAW_FUSION,
            status=TraceStatus.SUCCEEDED,
            event="triangulated",
            parent_ids=("pose:missing",),
        )
    writer.close()

    assert second.ordinal == 1
    assert second.previous_hash == first.record_hash
    reader = RunArtifactReader(tmp_path / "run")
    assert reader.records(stage=TraceStage.POSE_2D) == (second,)
    assert reader.records(status=TraceStatus.SUCCEEDED) == (first,)
    assert reader.records(event="keypoints_estimated") == (second,)
    report = reader.validate()
    assert report.ok is True
    assert report.record_count == 2
    assert report.last_hash == second.record_hash


def test_blobs_are_content_addressed_deduplicated_and_verified(tmp_path: Path) -> None:
    writer = RunArtifactWriter.create(
        tmp_path / "run",
        run_id="run-blobs",
        pipeline_version="test-revision",
    )
    blob = writer.put_blob(
        b"frame pixels", role="source_left", media_type="image/png", suffix=".png"
    )
    duplicate = writer.put_blob(
        b"frame pixels", role="source_left", media_type="image/png", suffix=".png"
    )
    overlay = writer.put_blob(
        b"frame pixels", role="overlay", media_type="image/png", suffix=".png"
    )
    assert duplicate == blob
    assert overlay.role == "overlay"
    assert overlay.sha256 == blob.sha256
    assert overlay.relative_path == blob.relative_path
    assert blob.role == "source_left"
    assert blob.to_dict()["role"] == "source_left"
    assert blob.relative_path.startswith(f"blobs/sha256/{blob.sha256[:2]}/{blob.sha256}")
    assert (tmp_path / "run" / blob.relative_path).read_bytes() == b"frame pixels"
    writer.append(
        record_id="rectify:1",
        stage=TraceStage.RECTIFICATION,
        status=TraceStatus.SUCCEEDED,
        event="preview_rendered",
        blobs=(blob, overlay),
    )
    writer.close()

    report = RunArtifactReader(tmp_path / "run").validate()
    assert report.blob_count == 1

    (tmp_path / "run" / blob.relative_path).write_bytes(b"tampered")
    with pytest.raises(TraceValidationError, match="blob (size|hash) mismatch"):
        RunArtifactReader(tmp_path / "run").validate()


def test_blob_suffix_cannot_escape_the_run(tmp_path: Path) -> None:
    writer = RunArtifactWriter.create(
        tmp_path / "run",
        run_id="run-safe-paths",
        pipeline_version="test-revision",
    )
    with pytest.raises(ValueError, match="suffix"):
        writer.put_blob(
            b"payload",
            role="unsafe",
            media_type="application/octet-stream",
            suffix="/../../x",
        )
    writer.close()


def test_active_run_can_resume_then_finalize_once(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="run-resume",
        pipeline_version="test-revision",
    )
    first = writer.append(
        record_id="decode:1",
        stage=TraceStage.DECODE,
        status=TraceStatus.SUCCEEDED,
        event="decoded",
    )
    writer.close()

    resumed = RunArtifactWriter.open(run_dir)
    second = resumed.append(
        record_id="sync:1",
        stage=TraceStage.SYNCHRONIZATION,
        status=TraceStatus.SUCCEEDED,
        event="paired",
        parent_ids=(first.record_id,),
    )
    summary = resumed.finalize(summary={"accepted_pairs": 1})

    assert second.ordinal == 1
    assert summary["status"] == RunStatus.COMPLETED.value
    assert summary["record_count"] == 2
    assert summary["last_record_hash"] == second.record_hash
    report = RunArtifactReader(run_dir).validate()
    assert report.status is RunStatus.COMPLETED
    with pytest.raises(RuntimeError, match="finalized"):
        RunArtifactWriter.open(run_dir)


def test_active_run_allows_only_one_writer(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="run-exclusive",
        pipeline_version="test-revision",
    )

    with pytest.raises(RuntimeError, match="active writer"):
        RunArtifactWriter.open(run_dir)

    writer.close()
    reopened = RunArtifactWriter.open(run_dir)
    reopened.close()


def test_context_manager_records_failed_run_without_swallowing_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="model failed"):
        with RunArtifactWriter.create(
            run_dir,
            run_id="run-failed",
            pipeline_version="test-revision",
        ) as writer:
            writer.append(
                record_id="pose:started",
                stage=TraceStage.POSE_2D,
                status=TraceStatus.STARTED,
                event="inference_started",
            )
            raise RuntimeError("model failed")

    reader = RunArtifactReader(run_dir)
    assert reader.validate().status is RunStatus.FAILED
    summary = reader.summary
    assert summary is not None
    assert len(summary.pop("summary_hash")) == 64
    assert summary == {
        "schema_version": "fisheye-handpose/run-summary/v1",
        "run_id": "run-failed",
        "status": "FAILED",
        "finalized_at_utc": reader.summary["finalized_at_utc"],
        "record_count": 1,
        "last_record_hash": reader.records()[0].record_hash,
        "summary": {"error": {"type": "RuntimeError", "message": "model failed"}},
    }


def test_non_finite_json_is_rejected_before_any_record_is_appended(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="run-json",
        pipeline_version="test-revision",
    )
    with pytest.raises(ValueError, match="non-finite"):
        writer.append(
            record_id="fusion:bad",
            stage=TraceStage.RAW_FUSION,
            status=TraceStatus.SUCCEEDED,
            event="triangulated",
            payload={"xyz_m": [0.0, float("nan"), 1.0]},
        )
    writer.close()

    assert RunArtifactReader(run_dir).records() == ()


def test_validation_detects_modified_records_and_a_truncated_tail(tmp_path: Path) -> None:
    tampered_dir = tmp_path / "tampered"
    writer = RunArtifactWriter.create(
        tampered_dir,
        run_id="run-tampered",
        pipeline_version="test-revision",
    )
    writer.append(
        record_id="qa:1",
        stage=TraceStage.QA,
        status=TraceStatus.SUCCEEDED,
        event="checked",
        payload={"score": 1},
    )
    writer.close()
    trace_path = tampered_dir / "trace.jsonl"
    raw = json.loads(trace_path.read_text())
    raw["payload"]["score"] = 0
    trace_path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(TraceValidationError, match="record hash mismatch"):
        RunArtifactReader(tampered_dir).validate()

    truncated_dir = tmp_path / "truncated"
    writer = RunArtifactWriter.create(
        truncated_dir,
        run_id="run-truncated",
        pipeline_version="test-revision",
    )
    writer.append(
        record_id="qa:1",
        stage=TraceStage.QA,
        status=TraceStatus.SUCCEEDED,
        event="checked",
    )
    writer.close()
    truncated_path = truncated_dir / "trace.jsonl"
    truncated_path.write_bytes(truncated_path.read_bytes()[:-1])
    with pytest.raises(TraceValidationError, match="truncated tail"):
        RunArtifactReader(truncated_dir).validate()


def test_manifest_and_final_summary_are_hash_protected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="run-documents",
        pipeline_version="test-revision",
        config={"threshold": 0.25},
    )
    writer.finalize(summary={"accepted": 1})

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["threshold"] = 0.99
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(TraceValidationError, match="manifest hash mismatch"):
        RunArtifactReader(run_dir)

    manifest["config"]["threshold"] = 0.25
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["summary"]["accepted"] = 2
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(TraceValidationError, match="summary hash mismatch"):
        RunArtifactReader(run_dir).validate()
