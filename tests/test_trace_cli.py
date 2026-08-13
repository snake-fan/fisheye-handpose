from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye_handpose.cli import main
from tests.test_cli import CALIBRATION_YAML, _audit_args, _write_pts, _write_video


@pytest.fixture
def trace_tiny_session(tmp_path: Path) -> Path:
    prefix = "capture"
    (tmp_path / f"{prefix}_calibration_camera.yaml").write_text(CALIBRATION_YAML, encoding="utf-8")
    for side, delta in (("left", 0), ("right", 5)):
        _write_video(tmp_path / f"{prefix}_camera_{side}_part0001.mp4")
        _write_pts(
            tmp_path / f"{prefix}_camera_{side}_part0001_pts.csv",
            tuple(value + delta for value in (1_000_000, 1_033_333, 1_066_666, 1_099_999)),
        )
    return tmp_path


def test_trace_init_creates_a_resumable_active_run(tmp_path: Path, capsys) -> None:
    from fisheye_handpose.trace import RunArtifactWriter, TraceStage, TraceStatus

    run_dir = tmp_path / "trace-init"

    code = main(
        [
            "trace-init",
            str(run_dir),
            "--run-id",
            "trace-init-test",
            "--pipeline-version",
            "test-revision",
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "ok": True,
        "run_dir": str(run_dir.resolve()),
        "run_id": "trace-init-test",
        "status": "ACTIVE",
    }

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "trace-init-test"
    assert manifest["status"] == "ACTIVE"
    writer = RunArtifactWriter.open(run_dir)
    writer.append(
        record_id="system:resumed",
        stage=TraceStage.SYSTEM,
        status=TraceStatus.SUCCEEDED,
        event="run_resumed",
    )
    writer.close()


def test_trace_demo_persists_three_linked_frames_and_visual_blobs(tmp_path: Path, capsys) -> None:
    from fisheye_handpose.trace import RunArtifactReader, TraceStage

    run_dir = tmp_path / "trace-demo"

    assert main(["trace-demo", str(run_dir), "--run-id", "demo-test"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["frame_count"] == 3
    reader = RunArtifactReader(run_dir)
    assert reader.manifest["status"] == "ACTIVE"
    assert reader.validate().status.value == "COMPLETED"
    assert reader.summary["summary"]["frame_count"] == 3

    required_stages = {
        TraceStage.DECODE,
        TraceStage.DETECTION,
        TraceStage.POSE_2D,
        TraceStage.CROSS_VIEW_ASSOCIATION,
        TraceStage.RAW_FUSION,
        TraceStage.KINEMATIC_REFINEMENT,
        TraceStage.TEMPORAL_REFINEMENT,
        TraceStage.QA,
        TraceStage.EXPORT,
    }
    assert required_stages <= {record.stage for record in reader.records()}
    frame_ids = {
        record.payload["frame_id"]
        for record in reader.records()
        if isinstance(record.payload, dict) and "frame_id" in record.payload
    }
    assert frame_ids == {"frame/000000", "frame/000001", "frame/000002"}

    decoded = reader.records(stage=TraceStage.DECODE)
    assert len(decoded) == 6
    assert all(len(record.blobs) == 1 for record in decoded)
    for record in decoded:
        blob = record.blobs[0]
        content = (run_dir / blob.relative_path).read_text(encoding="utf-8")
        assert blob.media_type == "image/svg+xml"
        assert blob.role in {"source_left", "source_right"}
        assert content.startswith("<svg")

    detections = reader.records(stage=TraceStage.DETECTION)
    assert len(detections) == 6
    assert all(len(record.payload["detections"]) == 1 for record in detections)
    assert all(len(record.payload["detections"][0]["bbox_xyxy"]) == 4 for record in detections)
    poses = [
        record
        for record in reader.records(stage=TraceStage.POSE_2D)
        if record.event == "view_keypoints_inferred"
    ]
    assert len(poses) == 6
    assert all(len(record.payload["keypoints_uv"]) == 21 for record in poses)
    assert all(len(record.payload["keypoint_scores"]) == 21 for record in poses)


def test_trace_validate_returns_json_and_nonzero_for_a_corrupted_blob(
    tmp_path: Path, capsys
) -> None:
    from fisheye_handpose.trace import RunArtifactReader, TraceStage

    run_dir = tmp_path / "trace-corrupt"
    assert main(["trace-demo", str(run_dir)]) == 0
    capsys.readouterr()
    reader = RunArtifactReader(run_dir)
    blob = reader.records(stage=TraceStage.DECODE)[0].blobs[0]
    (run_dir / blob.relative_path).write_bytes(b"corrupted")

    assert main(["trace-validate", str(run_dir)]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["errors"]


def test_trace_validate_success_has_a_stable_machine_readable_shape(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "trace-valid"
    assert main(["trace-demo", str(run_dir)]) == 0
    capsys.readouterr()

    assert main(["trace-validate", str(run_dir)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["status"] == "COMPLETED"
    assert result["record_count"] > 0
    assert result["errors"] == []
    assert result["warnings"] == []


def test_trace_validate_missing_run_is_still_machine_readable(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "does-not-exist"

    assert main(["trace-validate", str(missing)]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert "does not exist" in result["errors"][0]


def test_trace_serve_opens_the_validated_run_in_the_viewer(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    run_dir = tmp_path / "trace-serve"
    assert main(["trace-demo", str(run_dir)]) == 0
    capsys.readouterr()
    received: dict[str, object] = {}

    def fake_serve(reader, *, host: str, port: int) -> None:
        received.update(reader=reader, host=host, port=port)

    monkeypatch.setattr("fisheye_handpose.viewer.serve_trace", fake_serve)

    assert main(["trace-serve", str(run_dir), "--host", "127.0.0.1", "--port", "8123"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "ok": True,
        "run_dir": str(run_dir.resolve()),
        "url": "http://127.0.0.1:8123/",
    }
    assert received["host"] == "127.0.0.1"
    assert received["port"] == 8123
    assert received["reader"].manifest["run_id"] == "trace-demo"


def test_trace_serve_rejects_a_non_loopback_host(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["trace-serve", "missing-run", "--host", "0.0.0.0"])

    assert "host must be 127.0.0.1 or localhost" in capsys.readouterr().err


def test_audit_session_can_emit_a_stage_trace_without_claiming_model_stages(
    trace_tiny_session: Path, tmp_path: Path
) -> None:
    from fisheye_handpose.trace import RunArtifactReader, RunStatus, TraceStage

    audit_path = tmp_path / "audit.json"
    trace_dir = tmp_path / "audit-trace"
    arguments = _audit_args(trace_tiny_session, audit_path)
    arguments.extend(("--trace-output", str(trace_dir)))

    assert main(arguments) == 0

    reader = RunArtifactReader(trace_dir)
    assert reader.manifest["status"] == RunStatus.ACTIVE.value
    assert reader.validate().status is RunStatus.COMPLETED
    stages = {record.stage for record in reader.records()}
    assert {
        TraceStage.DISCOVERY,
        TraceStage.CALIBRATION,
        TraceStage.RECTIFICATION,
        TraceStage.SYNCHRONIZATION,
        TraceStage.DECODE,
        TraceStage.QA,
    } <= stages
    assert not stages & {
        TraceStage.DETECTION,
        TraceStage.POSE_2D,
        TraceStage.CROSS_VIEW_ASSOCIATION,
        TraceStage.RAW_FUSION,
        TraceStage.KINEMATIC_REFINEMENT,
        TraceStage.TEMPORAL_REFINEMENT,
    }
    audit_record = next(
        record for record in reader.records() if record.event == "audit_report_persisted"
    )
    assert len(audit_record.blobs) == 1
    assert audit_record.blobs[0].role == "audit_report"
    traced_report = json.loads(
        (trace_dir / audit_record.blobs[0].relative_path).read_text(encoding="utf-8")
    )
    assert traced_report == json.loads(audit_path.read_text(encoding="utf-8"))


def test_failed_audit_still_publishes_a_valid_failed_trace(
    trace_tiny_session: Path, tmp_path: Path, capsys
) -> None:
    from fisheye_handpose.trace import RunArtifactReader, RunStatus, TraceStatus

    (trace_tiny_session / "capture_camera_left_part0001.mp4").write_bytes(b"not a video")
    audit_path = tmp_path / "failed-audit.json"
    trace_dir = tmp_path / "failed-audit-trace"
    arguments = _audit_args(trace_tiny_session, audit_path)
    arguments.extend(("--trace-output", str(trace_dir)))

    assert main(arguments) == 2

    reader = RunArtifactReader(trace_dir)
    assert reader.validate().status is RunStatus.FAILED
    assert any(record.status is TraceStatus.FAILED for record in reader.records())
    report_record = next(
        record for record in reader.records() if record.event == "audit_report_persisted"
    )
    assert report_record.status is TraceStatus.FAILED
    assert "session failed" in capsys.readouterr().err
