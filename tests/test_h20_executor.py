from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fisheye_handpose.audit import AuditConfig
from fisheye_handpose.errors import FisheyeHandposeError
from fisheye_handpose.h20_executor import (
    H20ExecutorConfig,
    H20WorkerExecutionError,
    H20WorkerExecutor,
)
from fisheye_handpose.pipeline import PipelineExecutionContext, PipelineRunRequest
from fisheye_handpose.trace import (
    BlobRef,
    RunArtifactReader,
    RunArtifactWriter,
    RunStatus,
    TraceStage,
    TraceStatus,
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _valid_fhp21_record() -> dict[str, object]:
    points = [[index / 1000.0, 0.0, 0.5] for index in range(21)]
    raw = {
        "coordinate_frame": "rectified_left_camera",
        "length_unit": "m",
        "landmark_schema": "fhp21/v1",
        "landmarks_xyz_m": points,
        "validity": ["VALID"] * 21,
        "metrics": [
            {
                "joint_index": index,
                "epipolar_error_px": 0.1,
                "left_score": 0.9,
                "right_score": 0.9,
                "left_reprojection_error_px": 0.1,
                "right_reprojection_error_px": 0.1,
                "ray_angle_deg": 12.0,
            }
            for index in range(21)
        ],
        "valid_landmark_count": 21,
    }
    temporal = {
        "method": "causal_time_ema_v1",
        "timestamp_ns": 1_000_000_000,
        "landmarks_xyz_m": points,
        "validity": ["VALID"] * 21,
        "valid_landmark_count": 21,
        "reset_reason": "NEW_TRACK",
        "alpha": None,
        "refinement_applied": [False] * 21,
    }
    return {
        "schema_version": "fisheye-handpose/fhp21-output/v1",
        "record_type": "PoseEstimate",
        "sequence_id": "capture",
        "estimate_id": "part0001:pair000000:match-0:temporal-estimate",
        "frame_id": "part0001/pair000000",
        "frame_index": 0,
        "timestamp_ns": 1_000_000_000,
        "track_id": "track-0000",
        "source_observation_ids": ["part0001:pair000000:match-0"],
        "calibration_id": "sha256:test-calibration",
        "output_status": "PRODUCED",
        "output_frame": {
            "frame_id": "rectified_left_camera",
            "kind": "CAMERA",
            "axis_convention": "OPENCV_X_RIGHT_Y_DOWN_Z_FORWARD",
            "length_unit": "m",
        },
        "coordinate_frame": "rectified_left_camera",
        "length_unit": "m",
        "landmark_schema": "fhp21/v1",
        "handedness_probabilities": {"left": 0.0, "right": 0.0, "unknown": 1.0},
        "stage": "TEMPORAL_REFINEMENT",
        "selected_output_stage": "TEMPORAL_REFINEMENT",
        "kind": ["MEASURED"] * 21,
        "landmarks_xyz_m": points,
        "covariance_m2": [None] * 21,
        "covariance_status": ["NOT_ESTIMATED"] * 21,
        "validity": ["VALID"] * 21,
        "invalid_reason": [None] * 21,
        "evidence_source": ["MULTIVIEW"] * 21,
        "visibility_probability": [None] * 21,
        "visibility_status": ["NOT_ESTIMATED"] * 21,
        "confidence_probability": [None] * 21,
        "confidence_status": "NOT_CALIBRATED",
        "confidence_radius_m": None,
        "support_view_ids": [["left", "right"] for _ in range(21)],
        "reprojection_residuals_px": [{"left": 0.1, "right": 0.1} for _ in range(21)],
        "mapping_ids": ["rtmpose-hand5-native21-to-fhp21/v1"],
        "backend_provenance": {
            "producer": "fisheye_h20_worker",
            "producer_version": "test",
            "worker_request_sha256": "1" * 64,
            "model_manifest_sha256": "2" * 64,
            "mmpose_commit": "3" * 40,
            "detector": {"id": "det", "sha256": "4" * 64, "config": "det.py"},
            "pose": {"id": "pose", "sha256": "5" * 64, "config": "pose.py"},
            "fusion_method": "rectified_stereo_dlt_v1",
            "kinematic_method": "NONE",
            "temporal_method": "causal_time_ema_v1",
        },
        "raw": raw,
        "mano": None,
        "temporal": temporal,
    }


def _worker_result(result_dir: Path) -> None:
    result_dir.mkdir()
    blob_data = (
        json.dumps(_valid_fhp21_record(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    digest = hashlib.sha256(blob_data).hexdigest()
    relative = Path("blobs") / "sha256" / digest[:2] / f"{digest}.jsonl"
    blob_path = result_dir / relative
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(blob_data)
    output_path = result_dir / "fhp21.jsonl"
    output_path.write_bytes(blob_data)
    _json(
        result_dir / "manifest.json",
        {
            "schema_version": "fisheye-handpose/h20-worker-manifest/v1",
            "status": "ACTIVE",
        },
    )
    event = {
        "schema_version": "fisheye-handpose/h20-worker-event/v1",
        "ordinal": 0,
        "event_id": "part0001/pair000000:export:track-0000",
        "timestamp_utc": "2026-08-13T00:00:00.000000Z",
        "stage": "EXPORT",
        "status": "SUCCEEDED",
        "event": "fhp21_pose_exported",
        "parent_event_ids": [],
        "payload": {
            "frame_id": "part0001/pair000000",
            "frame_index": 0,
            "timestamp_ns": 1_000_000_000,
            "track_id": "track-0000",
            "output_status": "PRODUCED",
        },
        "blobs": [],
    }
    (result_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    _json(
        result_dir / "summary.json",
        {
            "schema_version": "fisheye-handpose/h20-worker-summary/v1",
            "status": "COMPLETED",
            "event_count": 1,
            "output_status": "PRODUCED",
            "export_count": 1,
            "output_file": "fhp21.jsonl",
            "output_artifact": {
                "role": "fhp21_output",
                "media_type": "application/x-ndjson",
                "bytes": len(blob_data),
                "sha256": digest,
                "relative_path": "fhp21.jsonl",
            },
        },
    )


@pytest.mark.parametrize("clock_offset_ns", [-7_123, 0, 7_123])
def test_h20_executor_builds_runtime_request_and_imports_verified_worker_bundle(
    tmp_path: Path,
    clock_offset_ns: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "capture"
    session.mkdir()
    calibration = session / "capture_calibration_camera.yaml"
    calibration.write_text("placeholder", encoding="utf-8")
    base_python = tmp_path / "uv-managed-python"
    base_python.write_text("", encoding="utf-8")
    worker_python = tmp_path / "deploy" / ".venv" / "bin" / "python"
    worker_python.parent.mkdir(parents=True)
    worker_python.symlink_to(base_python)
    worker_root = tmp_path / "worker-root"
    package = worker_root / "fisheye_h20_worker"
    package.mkdir(parents=True)
    # The real bridge is loaded from the configured worker source tree. The test copies
    # the public bridge package instead of mocking its validation behavior.
    source_package = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "mmpose-h20"
        / "worker"
        / "fisheye_h20_worker"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "_generated_project_contract.py",
        "artifacts.py",
        "bridge.py",
        "contracts.py",
        "output_contract.py",
    ):
        (package / name).write_bytes((source_package / name).read_bytes())

    config = H20ExecutorConfig.from_dict(
        {
            "schema_version": "fisheye-handpose/h20-executor/v1",
            "worker_python": str(worker_python),
            "worker_module_root": str(worker_root),
            "request": {
                "session": {"max_pairs": 3},
                "thresholds": {
                    "bbox_score": 0.3,
                    "keypoint_score": 0.2,
                    "association_epipolar_px": 5.0,
                    "max_reprojection_error_px": 3.0,
                    "min_ray_angle_deg": 0.1,
                },
                "models": {
                    "manifest": "/models/model-assets.json",
                    "model_dir": "/models/openmmlab",
                    "mmpose_source": "/vendor/mmpose",
                    "device": "cuda:0",
                    "detector_category_id": 0,
                    "license_risk_acknowledged": True,
                },
                "artifacts": {
                    "source_frames": "SAMPLED",
                    "sample_every": 10,
                    "image_format": "jpg",
                },
                "tracking": {"max_root_distance_m": 0.15, "max_gap_ms": 250},
                "mano": None,
                "temporal": {
                    "method": "causal_time_ema_v1",
                    "time_constant_ms": 80,
                    "gap_reset_ms": 250,
                },
            },
        }
    )
    captured: dict[str, object] = {}
    streamed_roles: list[str] = []
    original_put_blob_file = RunArtifactWriter.put_blob_file

    def recording_put_blob_file(
        self: RunArtifactWriter,
        source_path: str | Path,
        *,
        role: str,
        media_type: str,
        suffix: str = "",
    ) -> BlobRef:
        streamed_roles.append(role)
        return original_put_blob_file(
            self,
            source_path,
            role=role,
            media_type=media_type,
            suffix=suffix,
        )

    monkeypatch.setattr(RunArtifactWriter, "put_blob_file", recording_put_blob_file)

    def fake_process(command: list[str], *, env: dict[str, str]) -> tuple[int, bytes, bytes]:
        request_path = Path(command[-2])
        result_dir = Path(command[-1])
        captured["command"] = command
        captured["env"] = env
        captured["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        _worker_result(result_dir)
        return 0, b'{"ok":true}\n', b"worker log\n"

    run_dir = tmp_path / "run"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="run-1",
        pipeline_version="test",
    )
    audit_parent = writer.append(
        record_id="audit:complete",
        stage=TraceStage.QA,
        status=TraceStatus.SUCCEEDED,
        event="audit_completed",
        payload={},
    )
    audit_config = AuditConfig(
        left_id="cam_0",
        right_id="cam_1",
        translation_unit="mm",
        extrinsics_convention="reference_to_camera",
        clock_offset_ns=clock_offset_ns,
    )
    executor = H20WorkerExecutor(config, process_runner=fake_process)

    summary = executor.execute(
        PipelineExecutionContext(
            request=PipelineRunRequest(session, tmp_path, audit_config),
            item_id="capture",
            run_id="run-1",
            audit_report={
                "status": "PASS",
                "session": {"calibration_path": str(calibration)},
            },
            audit_record_ids=(audit_parent.record_id,),
        ),
        writer,
    )
    writer.finalize(summary={"output_status": summary.output_status})

    assert summary.output_status == "PRODUCED"
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["session"] == {
        "path": str(session.resolve()),
        "timestamp_column": "timestamp_us",
        "timestamp_unit": "us",
        "max_skew_us": 1000,
        "clock_offset_ns": clock_offset_ns,
        "max_pairs": 3,
    }
    assert request["calibration"]["path"] == str(calibration.resolve())
    assert request["calibration"]["left_camera_id"] == "cam_0"
    assert request["calibration"]["output_size"] == [1600, 1300]
    assert captured["env"] == {"PYTHONPATH": str(worker_root.resolve())}
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(worker_python.absolute())

    reader = RunArtifactReader(run_dir)
    assert reader.validate().ok
    export = reader.records(stage=TraceStage.EXPORT)
    assert len(export) == 1
    assert export[0].record_id == "h20:part0001/pair000000:export:track-0000"
    assert export[0].parent_ids == ("audit:complete",)
    roles = {blob.role for blob in export[0].blobs}
    assert {
        "worker_manifest",
        "worker_events",
        "worker_summary",
        "worker_fhp21_output",
    } <= roles
    assert {
        "worker_manifest",
        "worker_events",
        "worker_summary",
        "worker_fhp21_output",
    } <= set(streamed_roles)
    assert "result_dir" not in export[0].payload["worker_provenance"]
    assert {"worker_request", "worker_stdout", "worker_stderr"} <= roles


def test_h20_executor_rejects_missing_worker_paths_before_a_run_is_created(
    tmp_path: Path,
) -> None:
    from fisheye_handpose.h20_executor import H20ExecutorConfigurationError

    with pytest.raises(H20ExecutorConfigurationError, match="worker_python is not a file"):
        H20ExecutorConfig.from_dict(
            {
                "schema_version": "fisheye-handpose/h20-executor/v1",
                "worker_python": str(tmp_path / "missing-python"),
                "worker_module_root": str(tmp_path / "missing-worker"),
                "request": {
                    "session": {"max_pairs": 1},
                    "thresholds": {},
                    "models": {},
                    "artifacts": {},
                },
            }
        )

    assert issubclass(H20ExecutorConfigurationError, FisheyeHandposeError)


def test_invalid_worker_package_is_retained_as_a_failed_trace_record(tmp_path: Path) -> None:
    worker_python = tmp_path / "worker-python"
    worker_python.write_text("", encoding="utf-8")
    worker_root = tmp_path / "worker-root"
    package = worker_root / "fisheye_h20_worker"
    package.mkdir(parents=True)
    source_package = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "mmpose-h20"
        / "worker"
        / "fisheye_h20_worker"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "_generated_project_contract.py",
        "artifacts.py",
        "bridge.py",
        "contracts.py",
        "output_contract.py",
    ):
        (package / name).write_bytes((source_package / name).read_bytes())
    config = H20ExecutorConfig.from_dict(
        {
            "schema_version": "fisheye-handpose/h20-executor/v1",
            "worker_python": str(worker_python),
            "worker_module_root": str(worker_root),
            "request": {
                "session": {"max_pairs": 1},
                "thresholds": {},
                "models": {},
                "artifacts": {},
            },
        }
    )

    def invalid_process(command: list[str], *, env: dict[str, str]) -> tuple[int, bytes, bytes]:
        del env
        result_dir = Path(command[-1])
        result_dir.mkdir()
        (result_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
        return 1, b"worker stdout\n", b"worker stderr\n"

    run_dir = tmp_path / "failed-run"
    writer = RunArtifactWriter.create(run_dir, run_id="failed", pipeline_version="test")
    parent = writer.append(
        record_id="audit:complete",
        stage=TraceStage.QA,
        status=TraceStatus.SUCCEEDED,
        event="audit_completed",
        payload={},
    )
    context = PipelineExecutionContext(
        request=PipelineRunRequest(
            tmp_path / "capture",
            tmp_path,
            AuditConfig(
                left_id="cam_0",
                right_id="cam_1",
                translation_unit="mm",
                extrinsics_convention="reference_to_camera",
            ),
        ),
        item_id="capture",
        run_id="failed",
        audit_report={
            "status": "PASS",
            "session": {"calibration_path": str(tmp_path / "calibration.yaml")},
        },
        audit_record_ids=(parent.record_id,),
    )

    with pytest.raises(H20WorkerExecutionError, match="invalid result package"):
        H20WorkerExecutor(config, process_runner=invalid_process).execute(context, writer)
    writer.finalize(status=RunStatus.FAILED, summary={"output_status": "NOT_PRODUCED"})

    reader = RunArtifactReader(run_dir)
    failure = reader.get("h20:worker:failed")
    assert failure.status is TraceStatus.FAILED
    assert failure.payload["returncode"] == 1
    assert failure.payload["failure_phase"] == "package_validation"
    assert {blob.role for blob in failure.blobs} == {
        "worker_request",
        "worker_stdout",
        "worker_stderr",
        "invalid_worker_manifest",
    }
