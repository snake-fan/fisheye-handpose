from __future__ import annotations

from pathlib import Path

import pytest

from fisheye_handpose.audit import AuditConfig
from fisheye_handpose.pipeline import (
    PipelineConfigurationError,
    PipelineExecutionSummary,
    PipelineRunRequest,
    run_data_item,
)
from fisheye_handpose.trace import RunArtifactReader, RunStatus, TraceStage, TraceStatus
from tests.test_cli import CALIBRATION_YAML, SIZE, _write_pts, _write_video


@pytest.fixture
def pipeline_session(tmp_path: Path) -> Path:
    session = tmp_path / "raw session 001"
    session.mkdir()
    prefix = "capture"
    (session / f"{prefix}_calibration_camera.yaml").write_text(
        CALIBRATION_YAML,
        encoding="utf-8",
    )
    for side, delta in (("left", 0), ("right", 5)):
        _write_video(session / f"{prefix}_camera_{side}_part0001.mp4")
        _write_pts(
            session / f"{prefix}_camera_{side}_part0001_pts.csv",
            tuple(value + delta for value in (1_000_000, 1_033_333, 1_066_666, 1_099_999)),
        )
    return session


def _audit_config() -> AuditConfig:
    return AuditConfig(
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="mm",
        extrinsics_convention="reference_to_camera",
        max_skew_ns=1_000_000,
        min_pair_count=1,
        min_overlap_duration_ns=1,
        min_timestamp_fps=29.0,
        max_timestamp_fps=31.0,
        output_size=SIZE,
        balance=0.0,
        min_common_valid_fraction=0.0,
        min_per_camera_valid_fraction=0.0,
        min_hfov_deg=1.0,
        min_vfov_deg=1.0,
        run_epipolar_qa=False,
    )


def test_run_data_item_publishes_one_valid_run_under_the_safe_item_directory(
    pipeline_session: Path,
    tmp_path: Path,
) -> None:
    request = PipelineRunRequest(
        session_path=pipeline_session,
        runs_root=tmp_path / "runs",
        audit_config=_audit_config(),
        item_id="capture-front-1",
        run_id="run-0001",
        pipeline_version="test-revision",
    )

    result = run_data_item(request)

    expected = (tmp_path / "runs" / "capture-front-1" / "run-0001").resolve()
    assert result.run_dir == expected
    assert result.status is RunStatus.COMPLETED
    assert result.output_status == "NOT_PRODUCED"
    reader = RunArtifactReader(expected)
    assert reader.validate().status is RunStatus.COMPLETED
    assert reader.manifest["run_id"] == "run-0001"


def test_failed_audit_retains_a_failed_run_and_marks_every_model_stage_skipped(
    pipeline_session: Path,
    tmp_path: Path,
) -> None:
    (pipeline_session / "capture_camera_left_part0001.mp4").write_bytes(b"not a video")
    request = PipelineRunRequest(
        session_path=pipeline_session,
        runs_root=tmp_path / "runs",
        audit_config=_audit_config(),
        item_id="broken-capture",
        run_id="run-failed",
    )

    result = run_data_item(request)

    assert result.status is RunStatus.FAILED
    assert result.output_status == "NOT_PRODUCED"
    reader = RunArtifactReader(result.run_dir)
    assert reader.validate().status is RunStatus.FAILED
    model_records = [
        record
        for record in reader.records()
        if record.stage
        in {
            TraceStage.DETECTION,
            TraceStage.POSE_2D,
            TraceStage.CROSS_VIEW_ASSOCIATION,
            TraceStage.RAW_FUSION,
            TraceStage.KINEMATIC_REFINEMENT,
            TraceStage.TEMPORAL_REFINEMENT,
            TraceStage.EXPORT,
        }
    ]
    assert len(model_records) == 7
    assert all(record.status is TraceStatus.SKIPPED for record in model_records)
    assert all(record.payload["output_status"] == "NOT_PRODUCED" for record in model_records)


def test_default_item_id_is_collision_resistant_when_the_source_name_needs_slugging(
    pipeline_session: Path,
    tmp_path: Path,
) -> None:
    result = run_data_item(
        PipelineRunRequest(
            session_path=pipeline_session,
            runs_root=tmp_path / "runs",
            audit_config=_audit_config(),
            run_id="run-default-item",
        )
    )

    assert result.item_id == "raw-session-001-3c4af71e"
    assert result.run_dir.parent.name == "raw-session-001-3c4af71e"


def test_existing_run_directory_is_never_overwritten(
    pipeline_session: Path,
    tmp_path: Path,
) -> None:
    request = PipelineRunRequest(
        session_path=pipeline_session,
        runs_root=tmp_path / "runs",
        audit_config=_audit_config(),
        item_id="capture-item",
        run_id="immutable-run",
    )
    first = run_data_item(request)
    summary_before = (first.run_dir / "run_summary.json").read_bytes()

    with pytest.raises(FileExistsError, match="File exists"):
        run_data_item(request)

    assert (first.run_dir / "run_summary.json").read_bytes() == summary_before


@pytest.mark.parametrize("unsafe", ("../escape", "nested/item", "with spaces", ".."))
def test_explicit_item_id_must_be_one_safe_directory_component(
    pipeline_session: Path,
    tmp_path: Path,
    unsafe: str,
) -> None:
    with pytest.raises(PipelineConfigurationError, match="item_id must match"):
        run_data_item(
            PipelineRunRequest(
                session_path=pipeline_session,
                runs_root=tmp_path / "runs",
                audit_config=_audit_config(),
                item_id=unsafe,
                run_id="never-created",
            )
        )

    assert not (tmp_path / "runs").exists()


def test_injected_executor_appends_export_to_the_same_writer_and_marks_output_produced(
    pipeline_session: Path,
    tmp_path: Path,
) -> None:
    class FakeExecutor:
        writer_root: Path | None = None

        def execute(self, context, writer) -> PipelineExecutionSummary:
            self.writer_root = writer.root
            assert context.audit_report["status"] == "WARN"
            export = writer.append(
                record_id="fake:export",
                stage=TraceStage.EXPORT,
                status=TraceStatus.SUCCEEDED,
                event="fhp21_pose_exported",
                payload={"output_status": "PRODUCED", "frame_count": 1},
                parent_ids=(context.audit_record_ids[-1],),
            )
            return PipelineExecutionSummary(
                output_status="PRODUCED",
                record_ids=(export.record_id,),
                details={"frame_count": 1},
            )

    executor = FakeExecutor()
    result = run_data_item(
        PipelineRunRequest(
            session_path=pipeline_session,
            runs_root=tmp_path / "runs",
            audit_config=_audit_config(),
            item_id="executor-item",
            run_id="executor-run",
        ),
        backends=executor,
    )

    assert result.output_status == "PRODUCED"
    assert executor.writer_root == result.run_dir
    reader = RunArtifactReader(result.run_dir)
    export = reader.records(stage=TraceStage.EXPORT)
    assert [record.record_id for record in export] == ["fake:export"]
    audit_report = next(
        record for record in reader.records() if record.event == "audit_report_persisted"
    )
    assert export[0].parent_ids == (audit_report.record_id,)
