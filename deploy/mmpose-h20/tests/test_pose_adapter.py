from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.calibration import RectifiedStereo  # noqa: E402
from fisheye_h20_worker.contracts import WorkerError  # noqa: E402
from fisheye_h20_worker.crop import VirtualPerspectiveCropper  # noqa: E402
from fisheye_h20_worker.geometry import normalize_instances  # noqa: E402
from fisheye_h20_worker.pose_adapter import VirtualCropPoseAdapter  # noqa: E402


def _stereo() -> RectifiedStereo:
    k = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d = np.zeros((4, 1), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    p1 = np.column_stack((k, np.zeros(3, dtype=np.float64)))
    p2 = np.column_stack((k, np.array([-10.0, 0.0, 0.0])))
    dummy = np.zeros((200, 200), dtype=np.float32)
    return RectifiedStereo(
        calibration_id="fixture-calibration",
        image_size=(200, 200),
        output_size=(200, 200),
        left_k=k,
        left_d=d,
        right_k=k.copy(),
        right_d=d.copy(),
        right_from_left_rotation=identity.copy(),
        right_from_left_translation_m=np.array([-0.1, 0.0, 0.0]),
        r1=identity.copy(),
        r2=identity.copy(),
        p1=p1,
        p2=p2,
        q=np.eye(4, dtype=np.float64),
        left_undistort_maps=(dummy, dummy),
        right_undistort_maps=(dummy, dummy),
        left_rectify_maps=(dummy, dummy),
        right_rectify_maps=(dummy, dummy),
    )


class _PoseRuntime:
    def __init__(self, *, cardinality: int = 21, score: float = 0.9) -> None:
        self.pose_calls: list[dict[str, Any]] = []
        self.cardinality = cardinality
        self.score = score

    def infer_pose(
        self,
        models: object,
        frame: np.ndarray,
        *,
        bboxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        del models
        self.pose_calls.append({"shape": frame.shape, "bboxes": bboxes})
        height, width = frame.shape[:2]
        points = [[(width - 1.0) / 2.0, (height - 1.0) / 2.0]] * self.cardinality
        return [
            {
                "keypoints_uv": points,
                "keypoint_scores": [self.score] * self.cardinality,
            }
        ]


def _detections() -> list[dict[str, Any]]:
    return [
        {
            "bbox_xyxy": [60.0, 60.0, 140.0, 140.0],
            "bbox_score": 0.91,
            "label": 0,
        },
        {
            "bbox_xyxy": [82.0, 70.0, 154.0, 150.0],
            "bbox_score": 0.73,
            "label": 0,
        },
    ]


def test_virtual_adapter_runs_pose_on_each_physical_crop_and_maps_back_to_native() -> None:
    runtime = _PoseRuntime()
    adapter = VirtualCropPoseAdapter(
        cropper=VirtualPerspectiveCropper(output_size=(65, 65), bbox_scale=1.25),
        min_valid_fraction=0.8,
    )

    batch = adapter.infer(
        runtime=runtime,
        models=object(),
        frame=np.zeros((200, 200, 3), dtype=np.uint8),
        side="left",
        detections=_detections(),
        rectification=_stereo(),
    )

    assert len(runtime.pose_calls) == 2
    assert runtime.pose_calls == [
        {"shape": (65, 65, 3), "bboxes": [[0.0, 0.0, 64.0, 64.0]]},
        {"shape": (65, 65, 3), "bboxes": [[0.0, 0.0, 64.0, 64.0]]},
    ]
    assert len(batch.produced_instances) == 2
    first = batch.produced_instances[0]
    assert first["candidate_id"] == "left-0"
    assert first["bbox_xyxy"] == [60.0, 60.0, 140.0, 140.0]
    assert first["model_input_space"] == "virtual_pinhole"
    assert first["crop_policy_id"].startswith("virtual-perspective-kb4/v1")
    assert first["virtual_camera_id"].startswith("sha256:")
    assert len(first["keypoints_uv_crop"]) == 21
    assert len(first["keypoints_uv"]) == 21
    np.testing.assert_allclose(first["keypoints_uv"][0], [100.0, 100.0], atol=1e-8)
    assert first["keypoint_valid_mask"] == [True] * 21
    assert first["keypoint_scores"] == [0.9] * 21
    assert batch.results[0].crop is not None
    trace = batch.results[0].trace_payload()
    assert trace["output_status"] == "PRODUCED"
    assert trace["model_input_space"] == "virtual_pinhole"
    np.testing.assert_allclose(
        trace["virtual_camera"]["T_rig_from_virtual"],
        batch.results[0].crop.T_rig_from_virtual,
    )


def test_zero_detections_never_calls_pose_and_returns_an_empty_batch() -> None:
    runtime = _PoseRuntime()
    adapter = VirtualCropPoseAdapter(
        cropper=VirtualPerspectiveCropper(output_size=(65, 65)),
        min_valid_fraction=0.5,
    )

    batch = adapter.infer(
        runtime=runtime,
        models=object(),
        frame=np.zeros((200, 200, 3), dtype=np.uint8),
        side="left",
        detections=[],
        rectification=_stereo(),
    )

    assert runtime.pose_calls == []
    assert batch.results == ()
    assert batch.produced_instances == ()


def test_pose_cardinality_must_be_exactly_fhp21() -> None:
    runtime = _PoseRuntime(cardinality=20)
    adapter = VirtualCropPoseAdapter(
        cropper=VirtualPerspectiveCropper(output_size=(65, 65)),
        min_valid_fraction=0.5,
    )

    with pytest.raises(WorkerError, match="21"):
        adapter.infer(
            runtime=runtime,
            models=object(),
            frame=np.zeros((200, 200, 3), dtype=np.uint8),
            side="left",
            detections=_detections()[:1],
            rectification=_stereo(),
        )


def test_crop_below_valid_fraction_is_explicitly_not_produced() -> None:
    runtime = _PoseRuntime()
    adapter = VirtualCropPoseAdapter(
        cropper=VirtualPerspectiveCropper(output_size=(65, 65), bbox_scale=1.8),
        min_valid_fraction=0.99,
    )

    batch = adapter.infer(
        runtime=runtime,
        models=object(),
        frame=np.zeros((200, 200, 3), dtype=np.uint8),
        side="left",
        detections=[{"bbox_xyxy": [-30.0, -25.0, 55.0, 65.0], "bbox_score": 0.8, "label": 0}],
        rectification=_stereo(),
    )

    assert runtime.pose_calls == []
    assert batch.produced_instances == ()
    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.status == "NOT_PRODUCED"
    assert result.reason == "CROP_VALID_FRACTION_BELOW_THRESHOLD"
    assert result.trace_payload()["output_status"] == "NOT_PRODUCED"


def test_adapter_rejects_non_finite_pose_evidence_before_source_mapping() -> None:
    class NonFiniteRuntime(_PoseRuntime):
        def infer_pose(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            result = super().infer_pose(*args, **kwargs)
            result[0]["keypoints_uv"][3][0] = float("nan")
            return result

    adapter = VirtualCropPoseAdapter(
        cropper=VirtualPerspectiveCropper(output_size=(65, 65)),
        min_valid_fraction=0.5,
    )

    with pytest.raises(WorkerError, match="finite"):
        adapter.infer(
            runtime=NonFiniteRuntime(),
            models=object(),
            frame=np.zeros((200, 200, 3), dtype=np.uint8),
            side="left",
            detections=_detections()[:1],
            rectification=_stereo(),
        )


def test_unnormalized_simcc_response_is_preserved_but_not_used_as_overconfidence() -> None:
    adapter = VirtualCropPoseAdapter(
        cropper=VirtualPerspectiveCropper(output_size=(65, 65)),
        min_valid_fraction=0.5,
    )

    batch = adapter.infer(
        runtime=_PoseRuntime(score=1.0953675508499146),
        models=object(),
        frame=np.zeros((200, 200, 3), dtype=np.uint8),
        side="left",
        detections=_detections()[:1],
        rectification=_stereo(),
    )

    instance = batch.produced_instances[0]
    assert instance["model_keypoint_scores"] == pytest.approx([1.0953675508499146] * 21)
    assert instance["keypoint_scores"] == [1.0] * 21
    assert instance["keypoint_score_semantics"] == "RTMPOSE_SIMCC_MAX_RESPONSE_UNCALIBRATED"
    assert instance["keypoint_quality_weight_method"] == "CLIP_0_1_V1"


def test_native_pose_normalization_preserves_raw_simcc_response_and_bounds_quality() -> None:
    raw_score = 1.0953675508499146
    normalized = normalize_instances(
        [
            {
                "bbox_xyxy": [60.0, 60.0, 140.0, 140.0],
                "bbox_score": 0.91,
                "label": 0,
                "keypoints_uv": [[100.0, 100.0]] * 21,
                "keypoint_scores": [raw_score] * 21,
            }
        ],
        side="left",
        rectification=_stereo(),
    )[0]

    assert normalized["model_keypoint_scores"] == pytest.approx([raw_score] * 21)
    assert normalized["keypoint_scores"] == [1.0] * 21
    assert normalized["keypoint_score_semantics"] == ("RTMPOSE_SIMCC_MAX_RESPONSE_UNCALIBRATED")
    assert normalized["keypoint_quality_weight_method"] == "CLIP_0_1_V1"
