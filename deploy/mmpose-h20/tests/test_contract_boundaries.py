from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.contracts import WorkerError, load_request  # noqa: E402
from fisheye_h20_worker.output_contract import (  # noqa: E402
    build_pose_estimate,
    validate_pose_estimate,
)


def _legacy_request(*, bbox_score: float) -> dict[str, object]:
    return {
        "schema_version": "fisheye-handpose/h20-worker-request/v1",
        "session": {
            "path": "/data/session",
            "timestamp_column": "timestamp_us",
            "timestamp_unit": "us",
            "max_skew_us": 1_000,
            "max_pairs": 1,
        },
        "calibration": {
            "path": "/data/calibration.yaml",
            "left_camera_id": "cam_0",
            "right_camera_id": "cam_1",
            "translation_unit": "mm",
            "extrinsics_convention": "reference_to_camera",
            "output_size": [640, 480],
            "balance": 0.0,
            "fov_scale": 1.0,
        },
        "thresholds": {
            "bbox_score": bbox_score,
            "keypoint_score": 0.2,
            "association_epipolar_px": 5.0,
            "max_reprojection_error_px": 3.0,
            "min_ray_angle_deg": 0.1,
        },
        "models": {
            "manifest": "/models/model-assets.json",
            "model_dir": "/models",
            "mmpose_source": "/src/mmpose",
            "device": "cuda:0",
            "detector_category_id": 0,
            "license_risk_acknowledged": True,
        },
        "artifacts": {"source_frames": "NONE", "sample_every": 1},
    }


def _write_request(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_legacy_request_below_recovery_default_remains_loadable(tmp_path: Path) -> None:
    request = load_request(_write_request(tmp_path, _legacy_request(bbox_score=0.1)))

    assert request.perception.recovery_bbox_score == 0.1


def test_explicit_recovery_threshold_above_bbox_threshold_is_rejected(tmp_path: Path) -> None:
    document = _legacy_request(bbox_score=0.1)
    document["perception"] = {"recovery_bbox_score": 0.2}

    with pytest.raises(WorkerError, match="recovery_bbox_score"):
        load_request(_write_request(tmp_path, document))


def _robust_pose_estimate() -> dict[str, object]:
    points = [[index / 1_000.0, 0.0, 0.5] for index in range(21)]
    raw = {
        "fusion_method": "robust_stereo_huber_irls_v1",
        "coordinate_frame": "rectified_left_camera",
        "length_unit": "m",
        "landmark_schema": "fhp21/v1",
        "landmarks_xyz_m": points,
        "validity": ["VALID"] * 21,
        "covariance_m2": [
            [[1.0e-6, 0.0, 0.0], [0.0, 2.0e-6, 0.0], [0.0, 0.0, 3.0e-6]] for _ in range(21)
        ],
        "covariance_status": ["HEURISTIC_UNCALIBRATED"] * 21,
        "metrics": [
            {
                "joint_index": index,
                "epipolar_error_px": 0.1,
                "left_score": 0.9,
                "right_score": 0.9,
                "left_reprojection_error_px": 0.1,
                "right_reprojection_error_px": 0.1,
                "ray_angle_deg": 12.0,
                "left_depth_m": 0.5,
                "right_depth_m": 0.5,
                "support_view_count": 2,
                "covariance_status": "HEURISTIC_UNCALIBRATED",
            }
            for index in range(21)
        ],
        "valid_landmark_count": 21,
        "hand_validity": "VALID",
        "hand_reason": None,
        "palm_support_count": 5,
        "minimum_palm_support": 3,
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
    record = build_pose_estimate(
        sequence_id="capture",
        estimate_id="estimate-0",
        frame_id="frame-0",
        frame_index=0,
        timestamp_ns=1_000_000_000,
        track_id="track-0000",
        source_observation_id="observation-0",
        calibration_id="sha256:test-calibration",
        raw=raw,
        mano=None,
        temporal=temporal,
        keypoint_score_threshold=0.2,
        backend_provenance={
            "producer": "fisheye_h20_worker",
            "producer_version": "test",
            "worker_request_sha256": "1" * 64,
            "model_manifest_sha256": "2" * 64,
            "mmpose_commit": "3" * 40,
            "detector": {"id": "det", "sha256": "4" * 64, "config": "det.py"},
            "pose": {"id": "pose", "sha256": "5" * 64, "config": "pose.py"},
            "fusion_method": "robust_stereo_huber_irls_v1",
            "kinematic_method": "NONE",
            "temporal_method": "causal_time_ema_v1",
        },
    )
    record["schema_version"] = "fisheye-handpose/fhp21-output/v1"
    return record


def test_robust_raw_covariance_rejects_an_asymmetric_matrix() -> None:
    record = _robust_pose_estimate()
    record["raw"]["covariance_m2"][0] = [
        [1.0e-18, 4.0e-19, 0.0],
        [0.0, 2.0e-18, 0.0],
        [0.0, 0.0, 3.0e-18],
    ]

    with pytest.raises(WorkerError, match="covariance.*symmetric"):
        validate_pose_estimate(record, line_number=1)


def test_robust_raw_covariance_rejects_a_negative_eigenvalue() -> None:
    record = _robust_pose_estimate()
    # The tiny upper-left block has one negative eigenvalue despite positive diagonals.
    # Its scale prevents a fixed absolute tolerance from hiding the invalid matrix.
    record["raw"]["covariance_m2"][0] = [
        [1.0e-18, 2.0e-18, 0.0],
        [2.0e-18, 1.0e-18, 0.0],
        [0.0, 0.0, 1.0e-18],
    ]

    with pytest.raises(WorkerError, match="covariance.*positive semidefinite"):
        validate_pose_estimate(record, line_number=1)


def test_selected_output_covariance_uses_the_same_matrix_validation() -> None:
    record = _robust_pose_estimate()
    record["covariance_m2"][0] = [
        [1.0e-18, 4.0e-19, 0.0],
        [0.0, 2.0e-18, 0.0],
        [0.0, 0.0, 3.0e-18],
    ]
    record["covariance_status"][0] = "ESTIMATED"

    with pytest.raises(WorkerError, match="covariance.*symmetric"):
        validate_pose_estimate(record, line_number=1)
