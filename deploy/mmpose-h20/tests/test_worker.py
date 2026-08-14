from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

import fisheye_h20_worker.runner as worker_runner  # noqa: E402
import fisheye_h20_worker.visualization as visualization  # noqa: E402
from fisheye_h20_worker.artifacts import ResultWriter  # noqa: E402
from fisheye_h20_worker.bridge import load_import_bundle  # noqa: E402
from fisheye_h20_worker.calibration import (  # noqa: E402
    load_rectified_stereo,
    project_rectified_keypoints,
)
from fisheye_h20_worker.candidates import CandidatePolicy  # noqa: E402
from fisheye_h20_worker.cli import main as worker_main  # noqa: E402
from fisheye_h20_worker.contracts import WorkerError, load_request  # noqa: E402
from fisheye_h20_worker.geometry import associate  # noqa: E402
from fisheye_h20_worker.mano import MANO_FHP21_MAPPING_ID, map_mano_to_fhp21  # noqa: E402
from fisheye_h20_worker.runner import run_worker  # noqa: E402
from fisheye_h20_worker.runtime import OpenMMLabRuntime  # noqa: E402
from fisheye_h20_worker.temporal import CausalTemporalRefiner  # noqa: E402
from fisheye_h20_worker.tracking import SequenceTracker  # noqa: E402

DET_CONFIG = "demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py"
POSE_CONFIG = "configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py"
MMPOSE_COMMIT = "5408bc76f5b848cf925a0d1857899011d8c5b497"

CALIBRATION_YAML = """
calibration_info:
  reference_camera: cam_0
cameras:
  - id: cam_0
    name: IR_L
    distortion_model: KB
    image_width: 640
    image_height: 480
    intrinsics: {fx: 200, fy: 200, cx: 320, cy: 240}
    distortion: {k1: 0, k2: 0, k3: 0, k4: 0, k5: 0, k6: 0, p1: 0, p2: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [0, 0, 0]
  - id: cam_1
    name: IR_R
    distortion_model: KB
    image_width: 640
    image_height: 480
    intrinsics: {fx: 200, fy: 200, cx: 320, cy: 240}
    distortion: {k1: 0, k2: 0, k3: 0, k4: 0, k5: 0, k6: 0, p1: 0, p2: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [-100, 0, 0]
"""


def _write_fixture(root: Path, *, pair_count: int = 2) -> dict[str, Path]:
    session = root / "session"
    session.mkdir()
    calibration = session / "capture_calibration_camera.yaml"
    calibration.write_text(CALIBRATION_YAML, encoding="utf-8")
    timestamps = [1_000_000 + index * 33_333 for index in range(pair_count)]
    for side, delta in (("left", 0), ("right", 5)):
        (session / f"capture_camera_{side}_part0001.mp4").write_bytes(b"fake video")
        (session / f"capture_camera_{side}_part0001_pts.csv").write_text(
            "timestamp_us\n" + "".join(f"{value + delta}\n" for value in timestamps),
            encoding="utf-8",
        )

    model_dir = root / "models"
    model_dir.mkdir()
    detector = model_dir / "detector.pth"
    pose = model_dir / "pose.pth"
    detector.write_bytes(b"verified detector")
    pose.write_bytes(b"verified pose")
    source = root / "mmpose"
    for relative in (DET_CONFIG, POSE_CONFIG):
        config = source / relative
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("# pinned config\n", encoding="utf-8")
    manifest = root / "model-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fisheye-handpose/model-assets/v1",
                "artifacts": [
                    {
                        "id": "rtmdet-nano-hand",
                        "filename": detector.name,
                        "sha256": hashlib.sha256(detector.read_bytes()).hexdigest(),
                        "bytes": detector.stat().st_size,
                        "config": DET_CONFIG,
                        "license_status": "REVIEW_REQUIRED",
                    },
                    {
                        "id": "rtmpose-m-hand5",
                        "filename": pose.name,
                        "sha256": hashlib.sha256(pose.read_bytes()).hexdigest(),
                        "bytes": pose.stat().st_size,
                        "config": POSE_CONFIG,
                        "license_status": "REVIEW_REQUIRED",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    request = root / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "fisheye-handpose/h20-worker-request/v1",
                "session": {
                    "path": str(session),
                    "timestamp_column": "timestamp_us",
                    "timestamp_unit": "us",
                    "max_skew_us": 1000,
                    "max_pairs": pair_count,
                },
                "calibration": {
                    "path": str(calibration),
                    "left_camera_id": "cam_0",
                    "right_camera_id": "cam_1",
                    "translation_unit": "mm",
                    "extrinsics_convention": "reference_to_camera",
                    "output_size": [640, 480],
                    "balance": 0.0,
                    "fov_scale": 1.0,
                },
                "thresholds": {
                    "bbox_score": 0.3,
                    "keypoint_score": 0.2,
                    "association_epipolar_px": 5.0,
                    "max_reprojection_error_px": 3.0,
                    "min_ray_angle_deg": 0.1,
                },
                "models": {
                    "manifest": str(manifest),
                    "model_dir": str(model_dir),
                    "mmpose_source": str(source),
                    "device": "cuda:0",
                    "detector_category_id": 0,
                    "license_risk_acknowledged": True,
                },
                "artifacts": {"source_frames": "NONE", "sample_every": 1},
            }
        ),
        encoding="utf-8",
    )
    return {
        "request": request,
        "session": session,
        "manifest": manifest,
        "model_dir": model_dir,
        "source": source,
    }


def _configure_mano(fixture: dict[str, Path], root: Path) -> dict[str, Path]:
    model_root = root / "mano-models"
    mano_dir = model_root / "mano"
    mano_dir.mkdir(parents=True)
    left = mano_dir / "MANO_LEFT.pkl"
    right = mano_dir / "MANO_RIGHT.pkl"
    left.write_bytes(b"verified private MANO left")
    right.write_bytes(b"verified private MANO right")
    manifest = root / "private-mano-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fisheye-handpose/mano-assets/v1",
                "license": {
                    "acknowledged": True,
                    "reference": "user-reviewed MANO license record",
                },
                "provenance": {
                    "acknowledged": True,
                    "source": "official user-supplied MANO v1.2 archive",
                },
                "artifacts": [
                    {
                        "side": side,
                        "filename": f"mano/MANO_{side.upper()}.pkl",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                    for side, path in (("left", left), ("right", right))
                ],
            }
        ),
        encoding="utf-8",
    )
    request = json.loads(fixture["request"].read_text(encoding="utf-8"))
    request["mano"] = {
        "model_root": str(model_root),
        "manifest": str(manifest),
        "min_valid_landmarks": 15,
        "max_fit_rmse_m": 0.02,
        "iterations": 4,
        "learning_rate": 0.03,
    }
    fixture["request"].write_text(json.dumps(request), encoding="utf-8")
    return {"model_root": model_root, "manifest": manifest, "left": left, "right": right}


def _rewrite_first_output_record(
    result_dir: Path,
    mutate: Any,
) -> None:
    output = result_dir / "fhp21.jsonl"
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    mutate(records[0])
    data = b"".join(
        (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for record in records
    )
    _rewrite_output_bytes(result_dir, data)


def _rewrite_output_bytes(result_dir: Path, data: bytes) -> None:
    output = result_dir / "fhp21.jsonl"
    output.write_bytes(data)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_artifact"]["bytes"] = len(data)
    summary["output_artifact"]["sha256"] = hashlib.sha256(data).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


class FakeRuntime:
    def __init__(self) -> None:
        self.load_calls = 0
        self.infer_calls = 0
        self.source_calls = 0
        self.seen_frames: list[tuple[str, int]] = []
        self.encoded_frames: list[tuple[str, int, str, str]] = []

    def verify_source(self, **kwargs: Any) -> dict[str, Any]:
        self.source_calls += 1
        return {
            "commit": MMPOSE_COMMIT,
            "configs": {relative: "git-blob" for relative in (DET_CONFIG, POSE_CONFIG)},
        }

    def load_models(self, **kwargs: Any) -> object:
        self.load_calls += 1
        return object()

    def iter_video_frames(self, path: Path):
        side = "left" if "_left_" in path.name else "right"
        for index in range(4):
            yield {"side": side, "index": index}

    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        self.infer_calls += 1
        self.seen_frames.append((frame["side"], frame["index"]))
        shift = 0.0 if frame["side"] == "left" else -20.0
        points = [[320.0 + shift + joint * 0.01, 240.0 + joint * 0.01] for joint in range(21)]
        return [
            {
                "bbox_xyxy": [260.0 + shift, 180.0, 380.0 + shift, 340.0],
                "bbox_score": 0.95,
                "label": 0,
                "keypoints_uv": points,
                "keypoint_scores": [0.9] * 21,
            }
        ]

    def encode_frame(self, frame: dict[str, Any], image_format: str) -> bytes:
        self.encoded_frames.append(
            (frame["side"], frame["index"], frame.get("rendering", "source"), image_format)
        )
        return f"{frame}:{image_format}".encode()

    def render_rectification(
        self,
        rectification: object,
        side: str,
        frame: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        del rectification
        return {
            "undistorted": {**frame, "side": side, "rendering": "undistorted"},
            "rectified": {**frame, "side": side, "rendering": "rectified"},
        }


class MovingHandRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        instances = super().infer(models, frame, **kwargs)
        offset = 2.0 * frame["index"]
        for instance in instances:
            instance["bbox_xyxy"][0] += offset
            instance["bbox_xyxy"][2] += offset
            for point in instance["keypoints_uv"]:
                point[0] += offset
        return instances


class MissingMiddleFrameRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        if frame["index"] == 1:
            return []
        return super().infer(models, frame, **kwargs)


class VirtualPoseRuntime(FakeRuntime):
    def __init__(self, *, model_score: float = 0.9) -> None:
        super().__init__()
        self.detect_calls = 0
        self.pose_calls = 0
        self.model_score = model_score

    def iter_video_frames(self, path: Path):
        import numpy as np

        side_value = 40 if "_left_" in path.name else 80
        for _ in range(4):
            yield np.full((480, 640, 3), side_value, dtype=np.uint8)

    def detect(self, models: object, frame: Any, **kwargs: Any) -> list[dict[str, Any]]:
        del models, kwargs
        self.detect_calls += 1
        shift = 0.0 if float(frame.mean()) < 60.0 else -20.0
        return [
            {
                "bbox_xyxy": [260.0 + shift, 180.0, 380.0 + shift, 340.0],
                "bbox_score": 0.95,
                "label": 0,
            }
        ]

    def infer_pose(
        self,
        models: object,
        frame: Any,
        *,
        bboxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        del models
        self.pose_calls += 1
        self.assert_virtual_crop(frame, bboxes)
        center = [(frame.shape[1] - 1.0) / 2.0, (frame.shape[0] - 1.0) / 2.0]
        return [
            {
                "keypoints_uv": [center] * 21,
                "keypoint_scores": [self.model_score] * 21,
            }
        ]

    @staticmethod
    def assert_virtual_crop(frame: Any, bboxes: list[list[float]]) -> None:
        if frame.shape != (160, 192, 3):
            raise AssertionError(f"pose did not receive configured virtual crop: {frame.shape}")
        if bboxes != [[0.0, 0.0, 191.0, 159.0]]:
            raise AssertionError(f"pose bbox is not the physical crop extent: {bboxes}")

    def encode_frame(self, frame: Any, image_format: str) -> bytes:
        return f"{frame.shape}:{image_format}".encode()

    def render_rectification(
        self,
        rectification: object,
        side: str,
        frame: Any,
    ) -> dict[str, Any]:
        del rectification, side
        return {"undistorted": frame.copy(), "rectified": frame.copy()}


class CandidateAwareRuntime(VirtualPoseRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.detect_candidate_calls = 0

    def detect_candidates(
        self,
        models: object,
        frame: Any,
        *,
        policy: CandidatePolicy,
        category_id: int,
        view_id: str,
    ) -> Any:
        del models, frame
        self.detect_candidate_calls += 1
        shift = 0.0 if view_id == "left" else -20.0
        return policy.classify(
            bboxes=[
                [260.0 + shift, 180.0, 380.0 + shift, 260.0],
                [260.0 + shift, 260.0, 380.0 + shift, 340.0],
                [100.0 + shift, 100.0, 180.0 + shift, 180.0],
                [420.0 + shift, 100.0, 500.0 + shift, 180.0],
            ],
            scores=[0.90, 0.25, 0.19, 0.99],
            labels=[category_id, category_id, category_id, category_id + 1],
            category_id=category_id,
            view_id=view_id,
        )

    def infer_pose(
        self,
        models: object,
        frame: Any,
        *,
        bboxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        del models
        self.pose_calls += 1
        results = []
        for bbox in bboxes:
            center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
            results.append(
                {
                    "keypoints_uv": [center] * 21,
                    "keypoint_scores": [0.9] * 21,
                }
            )
        return results


def _instance(*, side: str, y: float) -> dict[str, Any]:
    shift = 0.0 if side == "left" else -20.0
    return {
        "bbox_xyxy": [260.0 + shift, y - 40.0, 380.0 + shift, y + 80.0],
        "bbox_score": 0.94,
        "label": 0,
        "keypoints_uv": [[320.0 + shift + joint * 0.01, y + joint * 0.01] for joint in range(21)],
        "keypoint_scores": [0.91] * 21,
    }


class AssociationScenarioRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        self.infer_calls += 1
        self.seen_frames.append((frame["side"], frame["index"]))
        side = frame["side"]
        frame_index = frame["index"]
        scenarios = {
            0: {
                "left": [_instance(side="left", y=200.0), _instance(side="left", y=300.0)],
                "right": [_instance(side="right", y=300.0), _instance(side="right", y=200.0)],
            },
            1: {"left": [_instance(side="left", y=220.0)], "right": []},
            2: {"left": [], "right": [_instance(side="right", y=250.0)]},
            3: {
                "left": [_instance(side="left", y=180.0), _instance(side="left", y=280.0)],
                "right": [_instance(side="right", y=180.0)],
            },
        }
        return scenarios[frame_index][side]


class QualityFailureRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        self.infer_calls += 1
        self.seen_frames.append((frame["side"], frame["index"]))
        instance = _instance(side=frame["side"], y=240.0)
        if frame["side"] == "right":
            instance["keypoint_scores"][0] = 0.1
            instance["keypoints_uv"][1][1] += 20.0
        return [instance]


class DegenerateTriangulationRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        instances = super().infer(models, frame, **kwargs)
        if frame["side"] == "right":
            for point in instances[0]["keypoints_uv"]:
                point[0] += 20.0
        return instances


class InsufficientPalmRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        instances = super().infer(models, frame, **kwargs)
        for index in (0, 5, 9):
            instances[0]["keypoint_scores"][index] = 0.01
        return instances


class MixedPalmRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        del models, kwargs
        instances = [
            _instance(side=frame["side"], y=210.0),
            _instance(side=frame["side"], y=310.0),
        ]
        for index in (0, 5, 9):
            instances[1]["keypoint_scores"][index] = 0.01
        return instances


class CardinalityRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        self.infer_calls += 1
        self.seen_frames.append((frame["side"], frame["index"]))
        values = {"left": (220.0, 230.0), "right": (221.0, 218.0)}
        return [_instance(side=frame["side"], y=y) for y in values[frame["side"]]]


class ModelLoadFailureRuntime(FakeRuntime):
    def load_models(self, **kwargs: Any) -> object:
        self.load_calls += 1
        raise RuntimeError("simulated model initialization failure")


class LateInferenceFailureRuntime(FakeRuntime):
    def infer(self, models: object, frame: dict[str, Any], **kwargs: Any):
        if frame["index"] == 1 and frame["side"] == "left":
            raise RuntimeError("simulated late inference failure")
        return super().infer(models, frame, **kwargs)


class FakeManoRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.mano_load_calls = 0
        self.mano_fit_calls: list[dict[str, Any]] = []

    def load_mano_models(self, **kwargs: Any) -> dict[str, object]:
        self.mano_load_calls += 1
        return {"left": object(), "right": object()}

    def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
        self.mano_fit_calls.append(kwargs)
        side = kwargs["side"]
        fixed_beta = kwargs["fixed_beta"]
        beta = (
            list(fixed_beta) if fixed_beta is not None else [0.2 if side == "right" else 0.1] * 10
        )
        landmarks = [
            list(point) if point is not None else [0.0, 0.0, 0.0]
            for point in kwargs["target_xyz_m"]
        ]
        return {
            "side": side,
            "mapping_id": MANO_FHP21_MAPPING_ID,
            "landmarks_xyz_m": landmarks,
            "validity": ["VALID"] * 21,
            "rmse_m": 0.005 if side == "right" else 0.01,
            "global_orient": [0.0, 0.0, 0.0],
            "hand_pose": [0.0] * 45,
            "transl": [0.0, 0.0, 0.0],
            "beta": beta,
            "iterations_run": 17,
            "best_loss": 0.0001,
            "final_loss": 0.0002,
            "joint_residuals_m": [0.005 if side == "right" else 0.01] * 21,
            "converged": True,
        }


class IntermittentManoRuntime(FakeManoRuntime):
    def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
        result = super().fit_mano(models, **kwargs)
        # Frame two rejects its warm full/weighted attempt and cold-recovery
        # full/weighted attempt. Frame three can then verify accepted-state ancestry.
        if len(self.mano_fit_calls) in {3, 4, 5, 6}:
            result["rmse_m"] = 0.03
            result["joint_residuals_m"] = [0.03] * 21
        return result


class RobustOutlierManoRuntime(FakeManoRuntime):
    def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
        result = super().fit_mano(models, **kwargs)
        if kwargs.get("joint_weights") is None:
            result["rmse_m"] = 0.03
            result["joint_residuals_m"] = [0.01] * 19 + [0.09, 0.07]
        else:
            result["rmse_m"] = 0.028
            result["joint_residuals_m"] = [0.009] * 19 + [0.088, 0.068]
        return result


class WorkerContractTests(unittest.TestCase):
    def test_overlay_track_colors_match_stage_comparison_rgb_contract(self) -> None:
        self.assertEqual(visualization.track_color_rgb("track-0000"), (117, 246, 196))
        self.assertEqual(visualization.track_color_rgb("track-0001"), (255, 180, 84))

    def test_overlay_renderer_uses_seekable_h264_contract_and_exact_cfr_timeline(self) -> None:
        import numpy as np

        commands: list[list[str]] = []

        class FakeInput:
            def __init__(self) -> None:
                self.payload = bytearray()

            def write(self, value: bytes) -> int:
                self.payload.extend(value)
                return len(value)

            def close(self) -> None:
                return None

        class FakeError:
            def read(self) -> bytes:
                return b""

        class FakeProcess:
            def __init__(self, command: list[str]) -> None:
                commands.append(command)
                self.stdin = FakeInput()
                self.stderr = FakeError()
                self.returncode: int | None = None

            def wait(self, timeout: int | None = None) -> int:
                del timeout
                Path(commands[-1][-1]).write_bytes(b"synthetic seekable mp4")
                self.returncode = 0
                return 0

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.returncode = -9

        timestamps = [1_000_000_000, 1_033_333_333, 1_066_666_666]
        projected = {
            "left": [[32.0, 24.0], *([None] * 20)],
            "right": [[30.0, 24.0], *([None] * 20)],
        }
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "overlay.mp4"
            with (
                patch.object(
                    visualization,
                    "_ffmpeg_details",
                    return_value={
                        "executable": "/usr/bin/ffmpeg",
                        "version": "ffmpeg version test",
                        "encoder": "libx264",
                    },
                ),
                patch.object(
                    visualization.subprocess,
                    "Popen",
                    side_effect=lambda command, **kwargs: FakeProcess(command),
                ),
            ):
                renderer = visualization.RawVsStableOverlayVideo(
                    output_path=output,
                    image_size=(64, 48),
                    timestamps_ns=timestamps,
                    temporal_method="causal_time_ema_v1",
                )
                for index, timestamp_ns in enumerate(timestamps):
                    renderer.append_frame(
                        left_frame=np.zeros((48, 64, 3), dtype=np.uint8),
                        right_frame=np.zeros((48, 64, 3), dtype=np.uint8),
                        frame_id=f"frame/{index:06d}",
                        frame_index=index,
                        timestamp_ns=timestamp_ns,
                        tracks=[
                            {
                                "track_id": "track-0000",
                                "raw": projected,
                                "stable": projected,
                                "stable_input_stage": "RAW_FUSION",
                            }
                        ],
                    )
                result = renderer.close()

        command = commands[0]
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertEqual(command[command.index("-framerate") + 1], "30/1")
        self.assertEqual(command[command.index("-video_track_timescale") + 1], "30")
        self.assertEqual(result["timeline"]["time_base"], {"numerator": 1, "denominator": 30})
        self.assertEqual(
            [frame["video_pts"] for frame in result["timeline"]["frames"]],
            [0, 1, 2],
        )
        self.assertEqual(result["metadata"]["frame_count"], 3)
        self.assertEqual(result["metadata"]["stable_input_stages"], ["RAW_FUSION"])
        self.assertEqual(result["metadata"]["temporal_method"], "causal_time_ema_v1")
        self.assertEqual(
            result["metadata"]["comparison_stages"],
            ["RAW_FUSION", "TEMPORAL_REFINEMENT"],
        )

    def test_result_writer_and_bridge_stream_a_large_blob_without_path_read_bytes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "large.mp4"
            with source.open("wb") as handle:
                handle.write(b"0123456789abcdef" * 327_680)
            expected_size = source.stat().st_size
            result_dir = root / "result"
            writer = ResultWriter(result_dir, {"test": True})
            reference = writer.put_blob_file(
                source,
                role="overlay_video_raw_vs_stable_stereo_rectified",
                media_type="video/mp4",
                suffix=".mp4",
            )
            writer.append(
                event_id="overlay:export",
                stage="EXPORT",
                status="SUCCEEDED",
                event="raw_vs_stable_overlay_video_exported",
                payload={"output_status": "PRODUCED"},
                blobs=[reference],
            )
            writer.finalize(
                status="COMPLETED",
                summary={"output_status": "NOT_PRODUCED", "output_file": None},
            )
            blob_path = result_dir / reference["relative_path"]
            original_read_bytes = Path.read_bytes

            def guarded_read_bytes(path: Path) -> bytes:
                if path == blob_path:
                    raise AssertionError("large blob must be verified through streaming reads")
                return original_read_bytes(path)

            with patch.object(Path, "read_bytes", guarded_read_bytes):
                bundle = load_import_bundle(result_dir)

        self.assertEqual(bundle.blobs_by_event[0][0].bytes, expected_size)
        self.assertEqual(bundle.blobs_by_event[0][0].source_path, blob_path.resolve())

    def test_overlay_video_request_is_backward_compatible_and_strictly_boolean(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)

            self.assertFalse(load_request(fixture["request"]).artifacts.overlay_video)

            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["artifacts"]["overlay_video"] = True
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            self.assertTrue(load_request(fixture["request"]).artifacts.overlay_video)

            request["artifacts"]["overlay_video"] = 1
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaisesRegex(WorkerError, "artifacts.overlay_video must be boolean"):
                load_request(fixture["request"])

    def test_virtual_pose_profile_is_optional_versioned_and_validated(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)

            baseline = load_request(fixture["request"])
            self.assertEqual(baseline.perception.pose_input, "baseline_native_v1")
            self.assertEqual(baseline.perception.crop_output_size, (256, 256))
            self.assertEqual(baseline.perception.recovery_bbox_score, 0.20)
            self.assertEqual(baseline.perception.max_candidates_per_view, 4)
            self.assertEqual(baseline.thresholds.min_depth_m, 0.1)
            self.assertEqual(baseline.thresholds.max_depth_m, 2.0)

            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["perception"] = {
                "pose_input": "virtual_perspective_kb4_v1",
                "crop_output_size": [192, 160],
                "crop_bbox_scale": 1.4,
                "crop_min_valid_fraction": 0.75,
                "recovery_bbox_score": 0.25,
                "max_candidates_per_view": 3,
            }
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            virtual = load_request(fixture["request"])
            self.assertEqual(virtual.perception.pose_input, "virtual_perspective_kb4_v1")
            self.assertEqual(virtual.perception.crop_output_size, (192, 160))
            self.assertEqual(virtual.perception.crop_bbox_scale, 1.4)
            self.assertEqual(virtual.perception.crop_min_valid_fraction, 0.75)
            self.assertEqual(virtual.perception.recovery_bbox_score, 0.25)
            self.assertEqual(virtual.perception.max_candidates_per_view, 3)

            request["perception"]["pose_input"] = "whole_rectified_frame"
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaisesRegex(WorkerError, "perception.pose_input"):
                load_request(fixture["request"])

            request["perception"]["pose_input"] = "baseline_native_v1"
            request["thresholds"]["min_depth_m"] = 1.0
            request["thresholds"]["max_depth_m"] = 0.5
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaisesRegex(WorkerError, "depth"):
                load_request(fixture["request"])

    def test_candidate_recovery_threshold_cannot_exceed_seed_threshold(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["perception"] = {
                "recovery_bbox_score": 0.31,
                "max_candidates_per_view": 4,
            }
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")

            with self.assertRaisesRegex(WorkerError, "recovery_bbox_score"):
                load_request(fixture["request"])

    def test_worker_virtual_pose_profile_records_crop_and_maps_pose_to_native(self) -> None:
        raw_model_score = 1.0953675508499146
        runtime = VirtualPoseRuntime(model_score=raw_model_score)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["perception"] = {
                "pose_input": "virtual_perspective_kb4_v1",
                "crop_output_size": [192, 160],
                "crop_bbox_scale": 1.3,
                "crop_min_valid_fraction": 0.8,
            }
            request["artifacts"]["source_frames"] = "ALL"
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
            exported = [
                json.loads(line)
                for line in (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(runtime.infer_calls, 0)
        self.assertEqual(runtime.detect_calls, 2)
        self.assertEqual(runtime.pose_calls, 2)
        self.assertEqual(
            manifest["configuration"]["perception"],
            {
                "pose_input": "virtual_perspective_kb4_v1",
                "crop_output_size": [192, 160],
                "crop_bbox_scale": 1.3,
                "crop_min_valid_fraction": 0.8,
                "crop_policy_id": "virtual-perspective-kb4/v1",
                "recovery_bbox_score": 0.2,
                "max_candidates_per_view": 4,
            },
        )
        crop_events = [event for event in events if event["event"] == "virtual_crop_pose_inferred"]
        self.assertEqual(len(crop_events), 2)
        self.assertTrue(
            all(event["payload"]["output_status"] == "PRODUCED" for event in crop_events)
        )
        self.assertTrue(
            all(
                event["payload"]["model_keypoint_scores"] == [raw_model_score] * 21
                for event in crop_events
            )
        )
        self.assertTrue(
            all(event["payload"]["keypoint_scores"] == [1.0] * 21 for event in crop_events)
        )
        self.assertTrue(
            all(
                {blob["role"] for blob in event["blobs"]}
                == {"virtual_crop", "virtual_crop_valid_mask"}
                for event in crop_events
            )
        )
        poses = [event for event in events if event["event"] == "view_keypoints_inferred"]
        self.assertEqual(len(poses), 2)
        for pose in poses:
            instance = pose["payload"]["instances"][0]
            self.assertEqual(instance["model_input_space"], "virtual_pinhole")
            self.assertEqual(len(instance["keypoints_uv"]), 21)
            self.assertEqual(len(instance["keypoints_uv_crop"]), 21)
            self.assertEqual(len(instance["keypoints_uv_rectified"]), 21)
            self.assertEqual(instance["model_keypoint_scores"], [raw_model_score] * 21)
            self.assertEqual(instance["keypoint_scores"], [1.0] * 21)
        tracked = [event for event in events if event["event"] == "tracked_view_keypoints_recorded"]
        self.assertTrue(
            all(
                event["payload"]["model_keypoint_scores"] == [raw_model_score] * 21
                for event in tracked
            )
        )
        self.assertTrue(all(event["payload"]["keypoint_scores"] == [1.0] * 21 for event in tracked))
        self.assertTrue(
            all(
                metric["left_score"] == 1.0 and metric["right_score"] == 1.0
                for metric in exported[0]["raw"]["metrics"]
            )
        )
        self.assertEqual(
            exported[0]["backend_provenance"]["model_keypoint_score_semantics"],
            "RTMPOSE_SIMCC_MAX_RESPONSE_UNCALIBRATED",
        )

    def test_native_pose_profile_traces_all_detector_decisions_and_poses_the_bounded_pool(
        self,
    ) -> None:
        runtime = CandidateAwareRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(runtime.infer_calls, 0)
        self.assertEqual(runtime.detect_candidate_calls, 2)
        self.assertEqual(runtime.pose_calls, 2)
        detections = [event for event in events if event["event"] == "hand_candidates_detected"]
        self.assertEqual(len(detections), 2)
        for detection in detections:
            decisions = detection["payload"]["candidate_decisions"]
            self.assertEqual(len(decisions), 4)
            self.assertEqual(
                [decision["source_index"] for decision in decisions],
                [0, 1, 2, 3],
            )
            self.assertEqual(
                [decision["classification"] for decision in decisions],
                ["SEED", "RECOVERY", "REJECTED", "REJECTED"],
            )
            self.assertEqual(
                [decision["reason"] for decision in decisions],
                [
                    "SCORE_MEETS_SEED_THRESHOLD",
                    "SCORE_MEETS_RECOVERY_THRESHOLD",
                    "SCORE_BELOW_RECOVERY_THRESHOLD",
                    "CATEGORY_MISMATCH",
                ],
            )
            self.assertEqual(len(detection["payload"]["candidate_pool"]), 2)
        poses = [event for event in events if event["event"] == "view_keypoints_inferred"]
        self.assertEqual(
            [
                [instance["classification"] for instance in event["payload"]["instances"]]
                for event in poses
            ],
            [["SEED", "RECOVERY"], ["SEED", "RECOVERY"]],
        )
        association = next(
            event for event in events if event["event"] == "cross_view_hands_associated"
        )
        self.assertEqual(len(association["payload"]["matches"]), 2)

    def test_virtual_pose_profile_preserves_pool_candidate_ids_through_each_crop(self) -> None:
        runtime = CandidateAwareRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["perception"] = {
                "pose_input": "virtual_perspective_kb4_v1",
                "crop_output_size": [192, 160],
                "crop_bbox_scale": 1.3,
                "crop_min_valid_fraction": 0.8,
                "recovery_bbox_score": 0.2,
                "max_candidates_per_view": 4,
            }
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(runtime.detect_candidate_calls, 2)
        self.assertEqual(runtime.pose_calls, 4)
        crop_events = [event for event in events if event["event"] == "virtual_crop_pose_inferred"]
        self.assertEqual(
            [event["payload"]["candidate_id"] for event in crop_events],
            [
                "left-det-0000",
                "left-det-0001",
                "right-det-0000",
                "right-det-0001",
            ],
        )
        self.assertEqual(
            [event["payload"]["detection"]["classification"] for event in crop_events],
            ["SEED", "RECOVERY", "SEED", "RECOVERY"],
        )

    def test_rectified_projection_preserves_landmark_cardinality_and_nulls_invalid_points(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = load_request(fixture["request"])
            stereo = load_rectified_stereo(request.calibration)

            landmarks: list[list[float] | None] = [[0.0, 0.0, 1.0]] + [None] * 20
            validity = ["VALID", "LOW_KEYPOINT_SCORE"] + ["VALID"] * 19
            projected = project_rectified_keypoints(stereo, landmarks, validity)

        self.assertEqual(set(projected), {"left", "right"})
        self.assertEqual(len(projected["left"]), 21)
        self.assertEqual(len(projected["right"]), 21)
        self.assertIsNotNone(projected["left"][0])
        self.assertIsNotNone(projected["right"][0])
        self.assertEqual(projected["left"][1:], [None] * 20)
        self.assertEqual(projected["right"][1:], [None] * 20)

        with self.assertRaisesRegex(WorkerError, "21 landmarks"):
            project_rectified_keypoints(stereo, landmarks[:-1], validity[:-1])

    def test_calibration_id_binds_rectified_output_geometry_parameters(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request_document = json.loads(fixture["request"].read_text(encoding="utf-8"))

            baseline = load_rectified_stereo(load_request(fixture["request"]).calibration)
            identifiers = {baseline.calibration_id}
            for update in (
                {"output_size": [320, 240]},
                {"balance": 0.5},
                {"fov_scale": 1.25},
            ):
                candidate = json.loads(json.dumps(request_document))
                candidate["calibration"].update(update)
                fixture["request"].write_text(json.dumps(candidate), encoding="utf-8")
                identifiers.add(
                    load_rectified_stereo(
                        load_request(fixture["request"]).calibration
                    ).calibration_id
                )

        self.assertEqual(len(identifiers), 4)

    def test_sequence_tracker_is_one_to_one_stable_and_resets_after_a_real_time_gap(
        self,
    ) -> None:
        tracker = SequenceTracker(max_root_distance_m=0.2, max_gap_ms=250.0)

        def observation(identifier: str, x: float) -> dict[str, Any]:
            return {
                "observation_id": identifier,
                "landmarks_xyz_m": [[x, 0.0, 1.0] for _ in range(21)],
                "validity": ["VALID"] * 21,
            }

        first = tracker.assign(
            [observation("left", 0.0), observation("right", 1.0)],
            timestamp_ns=1_000_000_000,
        )
        second = tracker.assign(
            [observation("right-again", 1.01), observation("left-again", 0.01)],
            timestamp_ns=1_100_000_000,
        )
        after_gap = tracker.assign(
            [observation("left-after-gap", 0.02)],
            timestamp_ns=1_500_000_000,
        )

        self.assertEqual([value["decision"] for value in first], ["NEW", "NEW"])
        self.assertEqual(second[0]["track_id"], first[1]["track_id"])
        self.assertEqual(second[1]["track_id"], first[0]["track_id"])
        self.assertEqual([value["decision"] for value in second], ["MATCHED", "MATCHED"])
        self.assertAlmostEqual(second[0]["distance_m"], 0.01)
        self.assertEqual(after_gap[0]["decision"], "NEW")
        self.assertNotIn(after_gap[0]["track_id"], {value["track_id"] for value in first})

    def test_causal_temporal_baseline_uses_timestamp_delta_and_explicit_gap_reset(
        self,
    ) -> None:
        refiner = CausalTemporalRefiner(time_constant_ms=100.0, gap_reset_ms=200.0)

        def points(value: float) -> list[list[float]]:
            return [[value, value, value] for _ in range(21)]

        first = refiner.refine(
            track_id="track-0000",
            timestamp_ns=0,
            landmarks_xyz_m=points(0.0),
            validity=["VALID"] * 21,
        )
        second = refiner.refine(
            track_id="track-0000",
            timestamp_ns=100_000_000,
            landmarks_xyz_m=points(1.0),
            validity=["VALID"] * 21,
        )
        after_gap = refiner.refine(
            track_id="track-0000",
            timestamp_ns=400_000_000,
            landmarks_xyz_m=points(2.0),
            validity=["VALID"] * 21,
        )
        after_source_change = refiner.refine(
            track_id="track-0000",
            timestamp_ns=450_000_000,
            landmarks_xyz_m=points(3.0),
            validity=["VALID"] * 21,
            input_stage="KINEMATIC_REFINEMENT",
        )

        self.assertEqual(first["reset_reason"], "FIRST_OBSERVATION")
        self.assertEqual(first["refinement_applied"], [False] * 21)
        self.assertAlmostEqual(second["alpha"], 1.0 - __import__("math").exp(-1.0))
        self.assertAlmostEqual(second["landmarks_xyz_m"][0][0], 1.0 - __import__("math").exp(-1.0))
        self.assertEqual(second["refinement_applied"], [True] * 21)
        self.assertEqual(after_gap["reset_reason"], "GAP_EXCEEDED")
        self.assertEqual(after_gap["landmarks_xyz_m"][0], [2.0, 2.0, 2.0])
        self.assertEqual(after_gap["refinement_applied"], [False] * 21)
        self.assertEqual(after_source_change["reset_reason"], "INPUT_STAGE_CHANGED")
        self.assertEqual(after_source_change["landmarks_xyz_m"][0], [3.0, 3.0, 3.0])
        self.assertEqual(after_source_change["refinement_applied"], [False] * 21)

    def test_mano_native_16_plus_five_tip_vertices_have_an_explicit_fhp21_mapping(
        self,
    ) -> None:
        joints = [[float(index), 0.0, 0.0] for index in range(16)]
        vertices = [[float(index), 1.0, 0.0] for index in range(745)]

        mapped = map_mano_to_fhp21(joints, vertices)

        self.assertEqual(MANO_FHP21_MAPPING_ID, "mano-v1.2-j16-tips-to-fhp21/v1")
        self.assertEqual(
            [point[0] for point in mapped],
            [0, 13, 14, 15, 744, 1, 2, 3, 320, 4, 5, 6, 443, 10, 11, 12, 554, 7, 8, 9, 671],
        )

    def test_no_mano_is_explicitly_skipped_but_tracking_temporal_and_fhp21_are_real(
        self,
    ) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            exported = [
                json.loads(line)
                for line in (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        skipped = next(
            event
            for event in events
            if event["stage"] == "KINEMATIC_REFINEMENT" and event["status"] == "SKIPPED"
        )
        frame_kinematic = next(
            event
            for event in events
            if event["event"] == "mano_frame_not_configured"
            and event["payload"]["frame_id"] == "part0001/pair000000"
        )
        raw = next(event for event in events if event["event"] == "raw_landmarks_triangulated")
        temporal = next(event for event in events if event["stage"] == "TEMPORAL_REFINEMENT")
        export_event = next(event for event in events if event["event"] == "fhp21_record_exported")
        detection = next(event for event in events if event["stage"] == "DETECTION")
        tracked_pose = next(
            event for event in events if event["event"] == "tracked_view_keypoints_recorded"
        )
        self.assertEqual(skipped["payload"]["output_status"], "NOT_PRODUCED")
        self.assertEqual(frame_kinematic["payload"]["track_id"], "track-0000")
        self.assertEqual(
            frame_kinematic["parent_event_ids"],
            ["part0001:pair000000:tracking"],
        )
        self.assertEqual(raw["payload"]["track_assignment"]["decision"], "NEW")
        self.assertEqual(raw["payload"]["track_id"], "track-0000")
        self.assertEqual(raw["payload"]["fusion_method"], "robust_stereo_huber_irls_v1")
        self.assertEqual(raw["payload"]["hand_validity"], "VALID")
        self.assertGreaterEqual(raw["payload"]["palm_support_count"], 3)
        self.assertEqual(len(raw["payload"]["covariance_m2"]), 21)
        self.assertEqual(
            raw["payload"]["covariance_status"],
            ["HEURISTIC_UNCALIBRATED"] * 21,
        )
        self.assertTrue(all(value is not None for value in raw["payload"]["covariance_m2"]))
        for event in (raw, temporal, export_event):
            projected = event["payload"]["projected_keypoints_uv"]
            self.assertEqual(set(projected), {"left", "right"})
            self.assertEqual(len(projected["left"]), 21)
            self.assertEqual(len(projected["right"]), 21)
            self.assertTrue(all(point is not None for point in projected["left"]))
            self.assertTrue(all(point is not None for point in projected["right"]))
            self.assertEqual(event["payload"]["projected_keypoints_space"], "rectified")
        self.assertEqual(
            frame_kinematic["payload"]["projected_keypoints_uv"],
            {"left": [None] * 21, "right": [None] * 21},
        )
        self.assertIn("detections", detection["payload"])
        self.assertEqual(tracked_pose["payload"]["track_id"], "track-0000")
        self.assertEqual(len(tracked_pose["payload"]["keypoints_uv"]), 21)
        self.assertEqual(len(tracked_pose["payload"]["keypoint_scores"]), 21)
        self.assertEqual(len(tracked_pose["payload"]["detections"]), 1)
        self.assertEqual(
            exported[0]["raw"]["covariance_status"],
            ["HEURISTIC_UNCALIBRATED"] * 21,
        )
        self.assertEqual(temporal["payload"]["input_stage"], "RAW_FUSION")
        self.assertEqual(temporal["parent_event_ids"], [frame_kinematic["event_id"]])
        self.assertEqual(temporal["payload"]["output_status"], "PRODUCED")
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["schema_version"], "fisheye-handpose/fhp21-output/v1")
        self.assertEqual(exported[0]["output_status"], "PRODUCED")
        self.assertEqual(exported[0]["track_id"], "track-0000")
        self.assertIsNone(exported[0]["mano"])
        self.assertEqual(exported[0]["selected_output_stage"], "TEMPORAL_REFINEMENT")
        self.assertEqual(exported[0]["record_type"], "PoseEstimate")
        self.assertEqual(exported[0]["stage"], "TEMPORAL_REFINEMENT")
        self.assertEqual(exported[0]["source_observation_ids"], ["part0001:pair000000:match-0"])
        self.assertEqual(
            exported[0]["output_frame"],
            {
                "frame_id": "rectified_left_camera",
                "kind": "CAMERA",
                "axis_convention": "OPENCV_X_RIGHT_Y_DOWN_Z_FORWARD",
                "length_unit": "m",
            },
        )
        self.assertEqual(
            exported[0]["handedness_probabilities"],
            {"left": 0.0, "right": 0.0, "unknown": 1.0},
        )
        self.assertEqual(exported[0]["validity"], ["VALID"] * 21)
        self.assertEqual(exported[0]["evidence_source"], ["MULTIVIEW"] * 21)
        self.assertEqual(exported[0]["kind"], ["MEASURED"] * 21)
        self.assertEqual(exported[0]["covariance_m2"], [None] * 21)
        self.assertEqual(exported[0]["covariance_status"], ["NOT_ESTIMATED"] * 21)
        self.assertEqual(exported[0]["visibility_probability"], [None] * 21)
        self.assertEqual(exported[0]["visibility_status"], ["NOT_ESTIMATED"] * 21)
        self.assertEqual(exported[0]["confidence_probability"], [None] * 21)
        self.assertEqual(exported[0]["confidence_status"], "NOT_CALIBRATED")
        self.assertIsNone(exported[0]["confidence_radius_m"])
        self.assertEqual(exported[0]["support_view_ids"], [["left", "right"]] * 21)
        self.assertEqual(len(exported[0]["reprojection_residuals_px"]), 21)
        self.assertEqual(set(exported[0]["reprojection_residuals_px"][0]), {"left", "right"})
        self.assertEqual(
            exported[0]["mapping_ids"],
            ["rtmpose-hand5-native21-to-fhp21/v1"],
        )
        self.assertEqual(exported[0]["backend_provenance"]["mmpose_commit"], MMPOSE_COMMIT)
        self.assertEqual(
            exported[0]["backend_provenance"]["producer_version"],
            "h20-worker/v1",
        )
        self.assertEqual(
            exported[0]["backend_provenance"]["temporal_method"],
            "causal_time_ema_v1",
        )
        self.assertEqual(export_event["payload"]["estimate_id"], exported[0]["estimate_id"])
        self.assertEqual(export_event["payload"]["covariance_m2"], [None] * 21)
        self.assertEqual(
            export_event["payload"]["source_observation_ids"],
            exported[0]["source_observation_ids"],
        )
        self.assertEqual(result["output_status"], "PRODUCED")
        self.assertEqual(result["export_count"], 1)

    def test_rejected_mano_and_first_temporal_observation_preserve_measured_kind(self) -> None:
        runtime = FakeManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            _configure_mano(fixture, root)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["mano"]["max_fit_rmse_m"] = 0.001
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            exported = json.loads(
                (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            bundle = load_import_bundle(result_dir)

        mano_event = next(event for event in events if event["event"] == "mano_frame_not_produced")
        self.assertEqual(mano_event["payload"]["selection"]["decision"], "NO_HIGH_QUALITY_FIT")
        self.assertEqual(
            [attempt["status"] for attempt in mano_event["payload"]["selection"]["attempts"]],
            ["REJECTED", "REJECTED"],
        )
        self.assertIsNone(exported["mano"])
        self.assertEqual(exported["temporal"]["reset_reason"], "FIRST_OBSERVATION")
        self.assertEqual(exported["temporal"]["refinement_applied"], [False] * 21)
        self.assertEqual(exported["kind"], ["MEASURED"] * 21)
        self.assertEqual(result["mano_output_count"], 0)
        self.assertEqual(bundle.summary["export_count"], 1)

    def test_second_frame_with_real_ema_update_is_refined(self) -> None:
        runtime = MovingHandRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            exported = [
                json.loads(line)
                for line in (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            bundle = load_import_bundle(result_dir)

        self.assertEqual(exported[0]["kind"], ["MEASURED"] * 21)
        self.assertIsNone(exported[0]["temporal"]["alpha"])
        self.assertNotEqual(exported[1]["raw"]["landmarks_xyz_m"], exported[0]["landmarks_xyz_m"])
        self.assertIsNotNone(exported[1]["temporal"]["alpha"])
        self.assertIsNone(exported[1]["temporal"]["reset_reason"])
        self.assertEqual(exported[1]["temporal"]["refinement_applied"], [True] * 21)
        self.assertEqual(exported[1]["kind"], ["REFINED"] * 21)
        self.assertEqual(bundle.summary["export_count"], 2)

    def test_configured_mano_selects_handedness_once_and_freezes_track_beta(self) -> None:
        runtime = FakeManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            _configure_mano(fixture, root)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            exported = [
                json.loads(line)
                for line in (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(runtime.mano_load_calls, 1)
        self.assertEqual(
            [call["side"] for call in runtime.mano_fit_calls], ["left", "right", "right"]
        )
        self.assertEqual(
            [call["seed_id"] for call in runtime.mano_fit_calls],
            ["mano_mean", "mano_mean", "accepted_state"],
        )
        self.assertIsNone(runtime.mano_fit_calls[0]["fixed_beta"])
        self.assertIsNone(runtime.mano_fit_calls[1]["fixed_beta"])
        self.assertEqual(runtime.mano_fit_calls[2]["fixed_beta"], [0.2] * 10)
        self.assertIsNone(runtime.mano_fit_calls[0]["initial_parameters"])
        self.assertEqual(
            runtime.mano_fit_calls[2]["initial_parameters"],
            {
                "global_orient": [0.0, 0.0, 0.0],
                "hand_pose": [0.0] * 45,
                "transl": [0.0, 0.0, 0.0],
                "beta": [0.2] * 10,
            },
        )
        mano_events = [event for event in events if event["event"] == "mano_frame_fitted"]
        self.assertEqual(len(mano_events), 2)
        self.assertEqual(mano_events[0]["payload"]["handedness"], "right")
        self.assertEqual(mano_events[0]["payload"]["selection"]["decision"], "SELECTED")
        self.assertEqual(mano_events[0]["payload"]["loss"]["metric"], "RMSE_M")
        self.assertEqual(mano_events[0]["payload"]["loss"]["value"], 0.005)
        robust_gate = mano_events[0]["payload"]["selection"]["gate"]
        self.assertEqual(robust_gate["method"], "RESIDUAL_TRIM_10PCT_V1")
        self.assertEqual(robust_gate["status"], "HEURISTIC_UNCALIBRATED")
        self.assertFalse(robust_gate["triggered"])
        self.assertEqual(robust_gate["first_pass_rmse_m"], 0.005)
        self.assertEqual(robust_gate["raw_rmse_m"], 0.005)
        self.assertEqual(robust_gate["full_rmse_m"], 0.005)
        self.assertEqual(robust_gate["weighted_rmse_m"], 0.005)
        self.assertEqual(robust_gate["inlier_rmse_m"], 0.005)
        self.assertEqual(robust_gate["joint_weights"], [1.0] * 21)
        self.assertEqual(robust_gate["inlier_mask"], [True] * 21)
        self.assertEqual(robust_gate["effective_joint_count"], 21)
        self.assertEqual(mano_events[0]["payload"]["fit_quality"], robust_gate)
        self.assertEqual(mano_events[0]["payload"]["raw_rmse_m"], 0.005)
        self.assertEqual(mano_events[0]["payload"]["full_rmse_m"], 0.005)
        self.assertEqual(mano_events[0]["payload"]["weighted_rmse_m"], 0.005)
        self.assertEqual(mano_events[0]["payload"]["inlier_rmse_m"], 0.005)
        self.assertEqual(mano_events[0]["payload"]["joint_weights"], [1.0] * 21)
        self.assertEqual(mano_events[0]["payload"]["inlier_mask"], [True] * 21)
        self.assertEqual(mano_events[0]["payload"]["effective_joint_count"], 21)
        self.assertEqual(
            mano_events[0]["payload"]["robust_gate_method"],
            "RESIDUAL_TRIM_10PCT_V1",
        )
        self.assertEqual(
            mano_events[0]["payload"]["robust_gate_status"],
            "HEURISTIC_UNCALIBRATED",
        )
        self.assertEqual(mano_events[0]["payload"]["optimizer"]["iterations_run"], 17)
        self.assertEqual(mano_events[0]["payload"]["selection"]["init_source"], "COLD_START")
        self.assertEqual(mano_events[1]["payload"]["selection"]["init_source"], "ACCEPTED_STATE")
        self.assertEqual(mano_events[0]["payload"]["projected_keypoints_space"], "rectified")
        self.assertTrue(
            all(
                point is not None
                for side in ("left", "right")
                for point in mano_events[0]["payload"]["projected_keypoints_uv"][side]
            )
        )
        self.assertTrue(mano_events[1]["payload"]["beta_frozen"])
        self.assertEqual(exported[0]["mano"]["mapping_id"], MANO_FHP21_MAPPING_ID)
        self.assertEqual(
            exported[0]["backend_provenance"]["kinematic_method"],
            "mano_v1.2_full45_robust_weighted_v3",
        )
        self.assertEqual(exported[0]["kind"], ["REFINED"] * 21)
        self.assertEqual(result["mano_output_count"], 2)

    def test_stateful_tracking_mano_and_temporal_events_reference_prior_state_events(
        self,
    ) -> None:
        runtime = FakeManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            _configure_mano(fixture, root)
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            events = {
                event["event_id"]: event
                for event in (
                    json.loads(line)
                    for line in (result_dir / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }

        prior_prefix = "part0001:pair000000"
        current_prefix = "part0001:pair000001"
        tracking = events[f"{current_prefix}:tracking"]
        mano = events[f"{current_prefix}:mano:match-0"]
        temporal = events[f"{current_prefix}:temporal:match-0"]
        self.assertIn(f"{prior_prefix}:tracking", tracking["parent_event_ids"])
        self.assertEqual(
            tracking["payload"]["state_predecessor_event_ids"],
            [f"{prior_prefix}:tracking"],
        )
        self.assertIn(f"{prior_prefix}:mano:match-0", mano["parent_event_ids"])
        self.assertIn(f"{current_prefix}:tracking", mano["parent_event_ids"])
        self.assertEqual(
            mano["payload"]["state_predecessor_event_id"],
            f"{prior_prefix}:mano:match-0",
        )
        self.assertIn(f"{prior_prefix}:temporal:match-0", temporal["parent_event_ids"])
        self.assertEqual(
            temporal["payload"]["state_predecessor_event_id"],
            f"{prior_prefix}:temporal:match-0",
        )

    def test_robust_mano_refit_trace_keeps_full_and_inlier_metrics_distinct(self) -> None:
        runtime = RobustOutlierManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            _configure_mano(fixture, root)
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            exported = json.loads(
                (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        mano_event = next(event for event in events if event["event"] == "mano_frame_fitted")
        gate = mano_event["payload"]["selection"]["gate"]
        self.assertEqual(len(runtime.mano_fit_calls), 4)
        self.assertTrue(gate["triggered"])
        self.assertEqual(gate["reason"], "ROBUST_INLIER_GATE_PASSED")
        self.assertEqual(gate["first_pass_rmse_m"], 0.03)
        self.assertEqual(gate["raw_rmse_m"], 0.028)
        self.assertEqual(gate["full_rmse_m"], 0.028)
        self.assertAlmostEqual(gate["weighted_rmse_m"], 0.009)
        self.assertAlmostEqual(gate["inlier_rmse_m"], 0.009)
        self.assertEqual(gate["joint_weights"], [1.0] * 19 + [0.0, 0.0])
        self.assertEqual(gate["inlier_mask"], [True] * 19 + [False, False])
        self.assertEqual(gate["effective_joint_count"], 19)
        self.assertEqual(gate["trimmed_joint_indices"], [19, 20])
        self.assertEqual(
            gate["stage_iterations"],
            [
                {"stage": "FULL_HUBER", "iterations_run": 17},
                {"stage": "WEIGHTED_REFIT", "iterations_run": 17},
            ],
        )
        self.assertEqual(mano_event["payload"]["fit_quality"], gate)
        self.assertEqual(mano_event["payload"]["rmse_m"], 0.028)
        self.assertEqual(exported["mano"]["rmse_m"], 0.028)
        self.assertEqual(exported["schema_version"], "fisheye-handpose/fhp21-output/v1")

    def test_rejected_mano_attempt_is_not_used_as_the_next_accepted_state_predecessor(
        self,
    ) -> None:
        runtime = IntermittentManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=3)
            _configure_mano(fixture, root)
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            events = {
                event["event_id"]: event
                for event in (
                    json.loads(line)
                    for line in (result_dir / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }

        first = events["part0001:pair000000:mano:match-0"]
        rejected = events["part0001:pair000001:mano:match-0"]
        recovered = events["part0001:pair000002:mano:match-0"]
        self.assertEqual(first["event"], "mano_frame_fitted")
        self.assertEqual(rejected["event"], "mano_frame_not_produced")
        self.assertEqual(recovered["event"], "mano_frame_fitted")
        self.assertEqual(
            rejected["payload"]["state_predecessor_event_id"],
            first["event_id"],
        )
        self.assertEqual(
            recovered["payload"]["state_predecessor_event_id"],
            first["event_id"],
        )
        self.assertIn(first["event_id"], recovered["parent_event_ids"])
        self.assertNotIn(rejected["event_id"], recovered["parent_event_ids"])

    def test_recovered_track_state_references_the_intervening_missing_frame(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=3)
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=MissingMiddleFrameRuntime())
            events = {
                event["event_id"]: event
                for event in (
                    json.loads(line)
                    for line in (result_dir / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }

        missing = events["part0001:pair000001:tracking"]
        recovered = events["part0001:pair000002:tracking"]
        self.assertIn(missing["event_id"], recovered["parent_event_ids"])
        self.assertEqual(
            recovered["payload"]["state_predecessor_event_ids"],
            [missing["event_id"]],
        )
        self.assertTrue(recovered["payload"]["assignments"][0]["recovered"])

    def test_tampered_mano_fails_before_any_mano_deserialization(self) -> None:
        runtime = FakeManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            mano = _configure_mano(fixture, root)
            mano["right"].write_bytes(b"tampered private MANO right")

            with self.assertRaisesRegex(Exception, "mismatch"):
                run_worker(fixture["request"], root / "result", runtime=runtime)

        self.assertEqual(runtime.mano_load_calls, 0)

    def test_real_runtime_decodes_video_with_opencv_presentation_order_and_releases_it(
        self,
    ) -> None:
        captures: list[Any] = []

        class FakeCapture:
            def __init__(self, path: str) -> None:
                self.path = path
                self.frames = ["presentation-0", "presentation-1", "presentation-2"]
                self.released = False
                captures.append(self)

            def isOpened(self) -> bool:
                return True

            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)

            def release(self) -> None:
                self.released = True

        class FakeCv2:
            VideoCapture = FakeCapture

        with patch.dict(sys.modules, {"cv2": FakeCv2}):
            frames = list(OpenMMLabRuntime().iter_video_frames(Path("capture.mp4")))

        self.assertEqual(frames, ["presentation-0", "presentation-1", "presentation-2"])
        self.assertEqual(captures[0].path, "capture.mp4")
        self.assertTrue(captures[0].released)

    def test_real_runtime_infers_each_model_in_its_openmmlab_scope_and_restores_prior_scope(
        self,
    ) -> None:
        from contextlib import contextmanager
        from types import ModuleType, SimpleNamespace

        import numpy as np

        active_scope = ["mmpose"]
        calls: list[tuple[str, str]] = []

        class FakeDefaultScope:
            @classmethod
            @contextmanager
            def overwrite_default_scope(cls, scope_name: str):
                previous_scope = active_scope[0]
                active_scope[0] = scope_name
                try:
                    yield
                finally:
                    active_scope[0] = previous_scope

        def inference_detector(model: object, frame: object) -> object:
            calls.append(("detector", active_scope[0]))
            return SimpleNamespace(
                pred_instances=SimpleNamespace(
                    bboxes=np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
                    scores=np.asarray([0.9], dtype=np.float32),
                    labels=np.asarray([0], dtype=np.int64),
                )
            )

        def inference_topdown(model: object, frame: object, *, bboxes: object) -> list[object]:
            calls.append(("pose", active_scope[0]))
            return [
                SimpleNamespace(
                    pred_instances=SimpleNamespace(
                        keypoints=np.zeros((1, 21, 2), dtype=np.float32),
                        keypoint_scores=np.full((1, 21), 0.8, dtype=np.float32),
                    )
                )
            ]

        mmengine = ModuleType("mmengine")
        mmengine_registry = ModuleType("mmengine.registry")
        mmengine_registry.DefaultScope = FakeDefaultScope
        mmdet = ModuleType("mmdet")
        mmdet_apis = ModuleType("mmdet.apis")
        mmdet_apis.inference_detector = inference_detector
        mmpose = ModuleType("mmpose")
        mmpose_apis = ModuleType("mmpose.apis")
        mmpose_apis.inference_topdown = inference_topdown

        models = SimpleNamespace(detector=object(), pose=object())
        with patch.dict(
            sys.modules,
            {
                "mmengine": mmengine,
                "mmengine.registry": mmengine_registry,
                "mmdet": mmdet,
                "mmdet.apis": mmdet_apis,
                "mmpose": mmpose,
                "mmpose.apis": mmpose_apis,
            },
        ):
            instances = OpenMMLabRuntime().infer(
                models,
                object(),
                bbox_threshold=0.3,
                category_id=0,
            )

        self.assertEqual(calls, [("detector", "mmdet"), ("pose", "mmpose")])
        self.assertEqual(active_scope, ["mmpose"])
        self.assertEqual(len(instances), 1)

    def test_real_runtime_classifies_every_raw_detector_instance_before_pooling(self) -> None:
        from contextlib import contextmanager
        from types import ModuleType, SimpleNamespace

        import numpy as np

        class FakeDefaultScope:
            @classmethod
            @contextmanager
            def overwrite_default_scope(cls, scope_name: str):
                self.assertEqual(scope_name, "mmdet")
                yield

        def inference_detector(model: object, frame: object) -> object:
            del model, frame
            return SimpleNamespace(
                pred_instances=SimpleNamespace(
                    bboxes=np.asarray(
                        [
                            [10.0, 20.0, 30.0, 40.0],
                            [50.0, 20.0, 70.0, 40.0],
                            [90.0, 20.0, 110.0, 40.0],
                            [130.0, 20.0, 150.0, 40.0],
                        ],
                        dtype=np.float32,
                    ),
                    scores=np.asarray([0.9, 0.25, 0.19, 0.99], dtype=np.float32),
                    labels=np.asarray([0, 0, 0, 1], dtype=np.int64),
                )
            )

        mmengine = ModuleType("mmengine")
        mmengine_registry = ModuleType("mmengine.registry")
        mmengine_registry.DefaultScope = FakeDefaultScope
        mmdet = ModuleType("mmdet")
        mmdet_apis = ModuleType("mmdet.apis")
        mmdet_apis.inference_detector = inference_detector

        models = SimpleNamespace(detector=object())
        with patch.dict(
            sys.modules,
            {
                "mmengine": mmengine,
                "mmengine.registry": mmengine_registry,
                "mmdet": mmdet,
                "mmdet.apis": mmdet_apis,
            },
        ):
            batch = OpenMMLabRuntime().detect_candidates(
                models,
                object(),
                policy=CandidatePolicy(
                    seed_threshold=0.3,
                    recovery_threshold=0.2,
                    max_candidates=4,
                ),
                category_id=0,
                view_id="left",
            )

        self.assertEqual(len(batch.decisions), 4)
        self.assertEqual(
            [decision.classification for decision in batch.decisions],
            ["SEED", "RECOVERY", "REJECTED", "REJECTED"],
        )
        self.assertEqual(
            [decision.reason for decision in batch.decisions],
            [
                "SCORE_MEETS_SEED_THRESHOLD",
                "SCORE_MEETS_RECOVERY_THRESHOLD",
                "SCORE_BELOW_RECOVERY_THRESHOLD",
                "CATEGORY_MISMATCH",
            ],
        )
        self.assertEqual(
            [decision.candidate_id for decision in batch.candidate_pool],
            ["left-det-0000", "left-det-0001"],
        )

    def test_models_load_once_and_real_detection_pose_events_are_persisted(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)

            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(runtime.source_calls, 1)
        self.assertEqual(runtime.load_calls, 1)
        self.assertEqual(runtime.infer_calls, 4)
        poses = [event for event in events if event["event"] == "view_keypoints_inferred"]
        self.assertEqual(len(poses), 4)
        self.assertTrue(
            all(len(event["payload"]["instances"][0]["keypoints_uv"]) == 21 for event in poses)
        )
        self.assertTrue(
            all(len(event["payload"]["instances"][0]["keypoint_scores"]) == 21 for event in poses)
        )
        self.assertEqual(manifest["models"]["mmpose_source"]["commit"], MMPOSE_COMMIT)
        self.assertEqual(manifest["configuration"]["session"]["max_pairs"], 2)
        self.assertEqual(manifest["configuration"]["thresholds"]["bbox_score"], 0.3)
        self.assertEqual(
            set(manifest["models"]["artifacts"]),
            {"rtmdet-nano-hand", "rtmpose-m-hand5"},
        )

    def test_association_handles_zero_one_two_and_unmatched_then_exports_raw_metrics(
        self,
    ) -> None:
        runtime = AssociationScenarioRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=4)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        associations = [
            event["payload"] for event in events if event["event"] == "cross_view_hands_associated"
        ]
        self.assertEqual([len(value["matches"]) for value in associations], [2, 0, 0, 1])
        self.assertEqual(associations[0]["matches"][0]["left_index"], 0)
        self.assertEqual(associations[0]["matches"][0]["right_index"], 1)
        self.assertEqual(associations[1]["unmatched_left_indices"], [0])
        self.assertEqual(associations[2]["unmatched_right_indices"], [0])
        self.assertEqual(associations[3]["unmatched_left_indices"], [1])

        raw_events = [event for event in events if event["event"] == "raw_landmarks_triangulated"]
        self.assertEqual(len(raw_events), 3)
        for event in raw_events:
            payload = event["payload"]
            self.assertEqual(payload["coordinate_frame"], "rectified_left_camera")
            self.assertEqual(payload["length_unit"], "m")
            self.assertEqual(len(payload["landmarks_xyz_m"]), 21)
            self.assertEqual(len(payload["validity"]), 21)
            self.assertEqual(len(payload["metrics"]), 21)
            self.assertTrue(all(value == "VALID" for value in payload["validity"]))
            self.assertTrue(all(point[2] > 0 for point in payload["landmarks_xyz_m"]))
            self.assertTrue(
                all(
                    {
                        "epipolar_error_px",
                        "left_reprojection_error_px",
                        "right_reprojection_error_px",
                        "ray_angle_deg",
                    }
                    <= metric.keys()
                    for metric in payload["metrics"]
                )
            )
        self.assertEqual(result["matched_hand_count"], 3)
        self.assertEqual(result["valid_landmark_count"], 63)

    def test_association_accepts_four_candidates_per_view_but_selects_at_most_two_hands(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            fixture = _write_fixture(Path(temporary_directory), pair_count=1)
            thresholds = load_request(fixture["request"]).thresholds

        def instance(candidate_id: str, y: float) -> dict[str, Any]:
            return {
                "candidate_id": candidate_id,
                "keypoints_uv_rectified": [[100.0, y]] * 21,
                "keypoint_scores": [0.9] * 21,
            }

        left = [instance(f"left-det-{index:04d}", 100.0 + 20.0 * index) for index in range(4)]
        right = [instance(f"right-det-{index:04d}", 100.0 + 20.0 * index) for index in range(4)]

        result = associate(left, right, thresholds)

        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(
            [
                (match["left_candidate_id"], match["right_candidate_id"])
                for match in result["matches"]
            ],
            [
                ("left-det-0000", "right-det-0000"),
                ("left-det-0001", "right-det-0001"),
            ],
        )
        self.assertEqual(result["unmatched_left_indices"], [2, 3])
        self.assertEqual(result["unmatched_right_indices"], [2, 3])

    def test_timestamp_matches_select_video_presentation_indices_after_a_drop(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=3)
            (fixture["session"] / "capture_camera_right_part0001_pts.csv").write_text(
                "timestamp_us\n1000005\n1066671\n",
                encoding="utf-8",
            )

            result = run_worker(fixture["request"], root / "result", runtime=runtime)

        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(
            runtime.seen_frames,
            [("left", 0), ("right", 0), ("left", 2), ("right", 1)],
        )

    def test_clock_offset_is_applied_to_right_timestamps_without_losing_nanoseconds(self) -> None:
        for offset_ns, expected_corrected_ns, expected_pair_ns, expected_skew_ns in (
            (-5_123, 999_999_877, 999_999_938, -123),
            (7_123, 1_000_012_123, 1_000_006_061, 12_123),
        ):
            with (
                self.subTest(clock_offset_ns=offset_ns),
                TemporaryDirectory() as temporary_directory,
            ):
                root = Path(temporary_directory)
                fixture = _write_fixture(root, pair_count=1)
                request = json.loads(fixture["request"].read_text(encoding="utf-8"))
                request["session"]["clock_offset_ns"] = offset_ns
                fixture["request"].write_text(json.dumps(request), encoding="utf-8")
                result_dir = root / "result"

                run_worker(fixture["request"], result_dir, runtime=FakeRuntime())
                events = [
                    json.loads(line)
                    for line in (result_dir / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                bundle = load_import_bundle(result_dir)
                exported = json.loads(
                    (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()[0]
                )
                manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

            sync = next(event["payload"] for event in events if event["stage"] == "SYNCHRONIZATION")
            self.assertEqual(sync["left_timestamp_ns"], 1_000_000_000)
            self.assertEqual(sync["left_timestamp_raw_ns"], 1_000_000_000)
            self.assertEqual(sync["left_timestamp_corrected_ns"], 1_000_000_000)
            self.assertEqual(sync["right_timestamp_ns"], 1_000_005_000)
            self.assertEqual(sync["right_timestamp_raw_ns"], 1_000_005_000)
            self.assertEqual(sync["right_timestamp_corrected_ns"], expected_corrected_ns)
            self.assertEqual(sync["clock_offset_ns"], offset_ns)
            self.assertEqual(sync["pair_timestamp_ns"], expected_pair_ns)
            self.assertEqual(sync["skew_ns"], expected_skew_ns)
            self.assertEqual(sync["corrected_skew_ns"], expected_skew_ns)
            self.assertEqual(exported["timestamp_ns"], expected_pair_ns)
            self.assertEqual(exported["temporal"]["timestamp_ns"], expected_pair_ns)
            self.assertEqual(manifest["configuration"]["session"]["clock_offset_ns"], offset_ns)
            self.assertEqual(bundle.output_status, "PRODUCED")

    def test_missing_clock_offset_defaults_to_zero_through_bridge_and_output(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=FakeRuntime())
            bundle = load_import_bundle(result_dir)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            exported = json.loads(
                (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

        sync = next(event["payload"] for event in events if event["stage"] == "SYNCHRONIZATION")
        self.assertEqual(sync["clock_offset_ns"], 0)
        self.assertEqual(sync["right_timestamp_raw_ns"], 1_000_005_000)
        self.assertEqual(sync["right_timestamp_corrected_ns"], 1_000_005_000)
        self.assertEqual(sync["pair_timestamp_ns"], 1_000_002_500)
        self.assertEqual(exported["timestamp_ns"], 1_000_002_500)
        self.assertEqual(exported["temporal"]["timestamp_ns"], 1_000_002_500)
        self.assertEqual(manifest["configuration"]["session"]["clock_offset_ns"], 0)
        self.assertEqual(bundle.output_status, "PRODUCED")

    def test_frame_index_is_global_when_pair_index_restarts_in_the_next_part(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            for side, delta in (("left", 0), ("right", 5)):
                (fixture["session"] / f"capture_camera_{side}_part0002.mp4").write_bytes(
                    b"fake video part two"
                )
                (fixture["session"] / f"capture_camera_{side}_part0002_pts.csv").write_text(
                    f"timestamp_us\n{2_000_000 + delta}\n",
                    encoding="utf-8",
                )
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["session"]["max_pairs"] = 2
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        sync = [event["payload"] for event in events if event["stage"] == "SYNCHRONIZATION"]
        self.assertEqual([event["pair_index"] for event in sync], [0, 0])
        self.assertEqual([event["frame_index"] for event in sync], [0, 1])
        self.assertEqual(
            [event["frame_id"] for event in sync],
            ["part0001/pair000000", "part0002/pair000000"],
        )
        self.assertTrue(all("/" not in event["event_id"] for event in events))

    def test_raw_landmarks_keep_explicit_per_joint_quality_failures(self) -> None:
        runtime = QualityFailureRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            exported = json.loads(
                (result_dir / "fhp21.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        raw = next(event["payload"] for event in events if event["stage"] == "RAW_FUSION")
        self.assertEqual(raw["validity"][0], "LOW_KEYPOINT_SCORE")
        self.assertEqual(raw["validity"][1], "EPIPOLAR_ERROR")
        self.assertIsNone(raw["landmarks_xyz_m"][0])
        self.assertIsNone(raw["landmarks_xyz_m"][1])
        self.assertEqual(raw["valid_landmark_count"], 19)
        self.assertEqual(result["valid_landmark_count"], 19)
        self.assertEqual(exported["validity"][:2], ["INVALID", "INVALID"])
        self.assertEqual(
            exported["invalid_reason"][:2],
            ["LOW_KEYPOINT_SCORE", "EPIPOLAR_ERROR"],
        )
        self.assertEqual(exported["landmarks_xyz_m"][:2], [None, None])
        self.assertEqual(exported["evidence_source"][:2], ["MONOCULAR", "MULTIVIEW"])
        self.assertEqual(exported["covariance_m2"][:2], [None, None])

    def test_insufficient_current_palm_support_keeps_raw_evidence_but_does_not_export(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=InsufficientPalmRuntime())
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        rejected = next(event for event in events if event["event"] == "raw_hand_gate_not_produced")
        self.assertEqual(rejected["payload"]["hand_validity"], "INVALID")
        self.assertEqual(rejected["payload"]["hand_reason"], "INSUFFICIENT_PALM_SUPPORT")
        self.assertEqual(rejected["payload"]["palm_support_count"], 2)
        self.assertEqual(rejected["payload"]["valid_landmark_count"], 18)
        self.assertTrue(any(point is not None for point in rejected["payload"]["landmarks_xyz_m"]))
        self.assertEqual(result["output_status"], "NOT_PRODUCED")
        self.assertEqual(result["export_count"], 0)
        self.assertFalse((result_dir / "fhp21.jsonl").exists())

    def test_assignment_maximizes_valid_match_count_before_epipolar_cost(self) -> None:
        runtime = CardinalityRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["thresholds"]["association_epipolar_px"] = 5.0
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        association = next(
            event["payload"] for event in events if event["stage"] == "CROSS_VIEW_ASSOCIATION"
        )
        self.assertEqual(len(association["matches"]), 2)
        self.assertEqual(
            {(match["left_index"], match["right_index"]) for match in association["matches"]},
            {(0, 1), (1, 0)},
        )

    def test_zero_match_frame_retains_explicit_not_produced_events_through_export(self) -> None:
        runtime = AssociationScenarioRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        empty_frame = [
            event
            for event in events
            if event["payload"].get("frame_id") == "part0001/pair000001"
            and event["stage"]
            in {"RAW_FUSION", "KINEMATIC_REFINEMENT", "TEMPORAL_REFINEMENT", "EXPORT"}
        ]
        self.assertEqual(
            [event["stage"] for event in empty_frame],
            ["RAW_FUSION", "KINEMATIC_REFINEMENT", "TEMPORAL_REFINEMENT", "EXPORT"],
        )
        self.assertTrue(
            all(event["payload"]["output_status"] == "NOT_PRODUCED" for event in empty_frame)
        )
        for event in empty_frame:
            self.assertEqual(
                event["payload"]["projected_keypoints_uv"],
                {"left": [None] * 21, "right": [None] * 21},
            )
            self.assertEqual(event["payload"]["projected_keypoints_space"], "rectified")
        empty_tracking = next(
            event for event in events if event["event_id"] == "part0001:pair000001:tracking"
        )
        self.assertEqual(
            empty_tracking["payload"]["state_predecessor_event_ids"],
            ["part0001:pair000000:tracking"],
        )
        self.assertIn(
            "part0001:pair000000:tracking",
            empty_tracking["parent_event_ids"],
        )
        self.assertEqual(
            empty_frame[0]["parent_event_ids"],
            ["part0001:pair000001:association"],
        )
        self.assertIn(empty_frame[0]["event_id"], empty_tracking["parent_event_ids"])
        self.assertEqual(
            empty_frame[1]["parent_event_ids"],
            ["part0001:pair000001:tracking"],
        )
        for predecessor, event in zip(empty_frame[1:-1], empty_frame[2:], strict=True):
            self.assertEqual(event["parent_event_ids"], [predecessor["event_id"]])

    def test_zero_valid_triangulation_retains_an_explicit_not_produced_export_event(
        self,
    ) -> None:
        runtime = DegenerateTriangulationRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        rejected_raw = next(
            event for event in events if event["event"] == "raw_hand_gate_not_produced"
        )
        temporal = next(
            event for event in events if event["event"] == "temporal_refinement_not_produced"
        )
        export = next(event for event in events if event["event"] == "fhp21_record_not_produced")
        self.assertEqual(result["output_status"], "NOT_PRODUCED")
        self.assertEqual(rejected_raw["payload"]["output_status"], "NOT_PRODUCED")
        self.assertEqual(rejected_raw["payload"]["valid_landmark_count"], 0)
        self.assertEqual(rejected_raw["payload"]["hand_validity"], "INVALID")
        self.assertEqual(export["payload"]["output_status"], "NOT_PRODUCED")
        self.assertIsNone(temporal["payload"]["track_id"])
        self.assertIsNone(export["payload"]["track_id"])
        self.assertEqual(export["parent_event_ids"], [temporal["event_id"]])

    def test_mixed_valid_and_rejected_hands_keep_complete_causal_trace_chains(self) -> None:
        runtime = MixedPalmRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"

            result = run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        association = next(
            event for event in events if event["event"] == "cross_view_hands_associated"
        )
        accepted_raw = next(
            event for event in events if event["event"] == "raw_landmarks_triangulated"
        )
        rejected_raw = next(
            event for event in events if event["event"] == "raw_hand_gate_not_produced"
        )
        tracking = next(event for event in events if event["event"] == "sequence_tracks_assigned")
        rejected_mano = next(
            event for event in events if event["event"] == "kinematic_refinement_not_produced"
        )
        rejected_temporal = next(
            event for event in events if event["event"] == "temporal_refinement_not_produced"
        )
        rejected_export = next(
            event
            for event in events
            if event["event"] == "fhp21_record_not_produced"
            and event["payload"]["reason"] == "RAW_HAND_GATE_REJECTED"
        )

        self.assertEqual(result["export_count"], 1)
        self.assertEqual(accepted_raw["parent_event_ids"], [association["event_id"]])
        self.assertIn(accepted_raw["event_id"], tracking["parent_event_ids"])
        self.assertIn(rejected_raw["event_id"], tracking["parent_event_ids"])
        self.assertEqual(rejected_mano["parent_event_ids"], [tracking["event_id"]])
        self.assertEqual(rejected_temporal["parent_event_ids"], [rejected_mano["event_id"]])
        self.assertEqual(rejected_export["parent_event_ids"], [rejected_temporal["event_id"]])
        self.assertIsNone(rejected_export["payload"]["track_id"])

    def test_tampered_checkpoint_fails_before_source_verification_or_model_loading(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            (fixture["model_dir"] / "pose.pth").write_bytes(b"tampered")

            with self.assertRaisesRegex(Exception, "mismatch"):
                run_worker(fixture["request"], root / "result", runtime=runtime)

        self.assertEqual(runtime.source_calls, 0)
        self.assertEqual(runtime.load_calls, 0)

    def test_source_frame_policy_writes_verified_content_addressed_blobs(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["artifacts"] = {
                "source_frames": "SAMPLED",
                "sample_every": 2,
                "image_format": "png",
            }
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"

            run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            sync_events = [event for event in events if event["stage"] == "SYNCHRONIZATION"]
            refs = sync_events[0]["blobs"]
            self.assertEqual(sync_events[1]["blobs"], [])
            rectification_events = [
                event for event in events if event["event"] == "stereo_pair_rectification_rendered"
            ]
            self.assertEqual(len(rectification_events), 1)
            self.assertEqual(
                rectification_events[0]["parent_event_ids"],
                [sync_events[0]["event_id"]],
            )
            self.assertEqual(
                {reference["role"] for reference in rectification_events[0]["blobs"]},
                {
                    "undistorted_left",
                    "undistorted_right",
                    "rectified_left",
                    "rectified_right",
                },
            )
            self.assertEqual(
                rectification_events[0]["payload"]["calibration_id"],
                next(
                    event["payload"]["calibration_id"]
                    for event in events
                    if event["event"] == "worker_rectification_loaded"
                ),
            )
            for reference in refs:
                data = (result_dir / reference["relative_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), reference["sha256"])
                self.assertEqual(len(data), reference["bytes"])

        self.assertEqual(
            runtime.encoded_frames,
            [
                ("left", 0, "source", "png"),
                ("right", 0, "source", "png"),
                ("left", 0, "undistorted", "png"),
                ("left", 0, "rectified", "png"),
                ("right", 0, "undistorted", "png"),
                ("right", 0, "rectified", "png"),
            ],
        )
        self.assertEqual({reference["role"] for reference in refs}, {"source_left", "source_right"})

    def test_optional_overlay_video_writes_one_all_track_frame_per_pair_and_global_artifacts(
        self,
    ) -> None:
        instances: list[Any] = []

        class FakeOverlayVideo:
            def __init__(
                self,
                *,
                output_path: Path,
                image_size: tuple[int, int],
                timestamps_ns: list[int],
                temporal_method: str,
            ) -> None:
                self.output_path = output_path
                self.image_size = image_size
                self.timestamps_ns = timestamps_ns
                self.temporal_method = temporal_method
                self.frames: list[dict[str, Any]] = []
                self.aborted = False
                instances.append(self)

            def append_frame(self, **value: Any) -> None:
                self.frames.append(value)

            def close(self) -> dict[str, Any]:
                self.output_path.write_bytes(b"test h264 mp4 bytes")
                timeline = {
                    "schema_version": "fisheye-handpose/overlay-video-timeline/v1",
                    "frame_rate": {"numerator": 30, "denominator": 1},
                    "time_base": {"numerator": 1, "denominator": 30},
                    "frames": [
                        {
                            "video_frame_index": index,
                            "video_pts": index,
                            "duration_pts": 1,
                            "frame_id": frame["frame_id"],
                            "frame_index": frame["frame_index"],
                            "timestamp_ns": frame["timestamp_ns"],
                            "track_ids": sorted(track["track_id"] for track in frame["tracks"]),
                        }
                        for index, frame in enumerate(self.frames)
                    ],
                }
                return {
                    "path": self.output_path,
                    "timeline": timeline,
                    "metadata": {
                        "schema_version": "fisheye-handpose/overlay-video/v1",
                        "layout": "RAW_LEFT_RAW_RIGHT_STABLE_LEFT_STABLE_RIGHT",
                        "image_space": "rectified",
                        "comparison_stages": ["RAW_FUSION", "TEMPORAL_REFINEMENT"],
                        "frame_count": len(self.frames),
                        "width": self.image_size[0],
                        "height": self.image_size[1],
                        "codec": "h264",
                        "pixel_format": "yuv420p",
                        "tracks": ["track-0000", "track-0001"],
                        "ffmpeg": {"executable": "/usr/bin/ffmpeg", "encoder": "libx264"},
                    },
                }

            def abort(self) -> None:
                self.aborted = True

        runtime = AssociationScenarioRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["artifacts"]["overlay_video"] = True
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"

            with patch.object(
                worker_runner,
                "RawVsStableOverlayVideo",
                FakeOverlayVideo,
                create=True,
            ):
                run_worker(fixture["request"], result_dir, runtime=runtime)
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            bundle = load_import_bundle(result_dir)

        self.assertEqual(len(instances), 1)
        overlay = instances[0]
        self.assertEqual(overlay.temporal_method, "causal_time_ema_v1")
        self.assertEqual(len(overlay.frames), 2)
        self.assertEqual([len(frame["tracks"]) for frame in overlay.frames], [2, 0])
        self.assertEqual(
            sorted(track["track_id"] for track in overlay.frames[0]["tracks"]),
            ["track-0000", "track-0001"],
        )
        global_event = next(
            event for event in events if event["event"] == "raw_vs_stable_overlay_video_exported"
        )
        self.assertNotIn("frame_id", global_event["payload"])
        self.assertEqual(global_event["payload"]["frame_count"], 2)
        self.assertEqual(global_event["payload"]["temporal_method"], "causal_time_ema_v1")
        self.assertEqual(
            global_event["payload"]["comparison_stages"],
            ["RAW_FUSION", "TEMPORAL_REFINEMENT"],
        )
        self.assertEqual(
            {blob["role"] for blob in global_event["blobs"]},
            {
                "overlay_video_raw_vs_stable_stereo_rectified",
                "overlay_video_timeline",
            },
        )
        imported = next(
            record
            for record in bundle.core_records(external_parent_id="audit:report")
            if record.event == "raw_vs_stable_overlay_video_exported"
        )
        self.assertEqual(
            {blob.role for blob in imported.blobs},
            {
                "overlay_video_raw_vs_stable_stereo_rectified",
                "overlay_video_timeline",
            },
        )

    def test_overlay_ffmpeg_is_aborted_when_a_later_pipeline_stage_fails(self) -> None:
        instances: list[Any] = []

        class FailingRunOverlay:
            def __init__(self, **kwargs: Any) -> None:
                self.output_path = kwargs["output_path"]
                self.aborted = False
                self.frame_count = 0
                instances.append(self)

            def append_frame(self, **kwargs: Any) -> None:
                del kwargs
                self.frame_count += 1

            def close(self) -> dict[str, Any]:
                raise AssertionError("failed worker must not close/export the overlay")

            def abort(self) -> None:
                self.aborted = True
                self.output_path.unlink(missing_ok=True)

        runtime = LateInferenceFailureRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["artifacts"]["overlay_video"] = True
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "result"
            with (
                patch.object(
                    worker_runner,
                    "RawVsStableOverlayVideo",
                    FailingRunOverlay,
                    create=True,
                ),
                self.assertRaisesRegex(RuntimeError, "simulated late inference failure"),
            ):
                run_worker(fixture["request"], result_dir, runtime=runtime)

            summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].frame_count, 1)
        self.assertTrue(instances[0].aborted)
        self.assertEqual(summary["status"], "FAILED")
        self.assertEqual(summary["overlay_video_output_count"], 0)

    def test_existing_result_is_never_overwritten(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            result_dir = root / "result"
            run_worker(fixture["request"], result_dir, runtime=runtime)
            summary_before = (result_dir / "summary.json").read_bytes()

            with self.assertRaises(FileExistsError):
                run_worker(fixture["request"], result_dir, runtime=runtime)

            self.assertEqual((result_dir / "summary.json").read_bytes(), summary_before)

    def test_failure_after_manifest_creation_retains_a_failed_process_package(self) -> None:
        runtime = ModelLoadFailureRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            result_dir = root / "failed-result"

            with self.assertRaisesRegex(RuntimeError, "initialization"):
                run_worker(fixture["request"], result_dir, runtime=runtime)

            summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
            events = [
                json.loads(line)
                for line in (result_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["status"], "FAILED")
        self.assertEqual(summary["output_status"], "NOT_PRODUCED")
        self.assertEqual(summary["error"]["type"], "RuntimeError")
        self.assertGreaterEqual(len(events), 2)
        failed = [event for event in events if event["status"] == "FAILED"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["event"], "worker_execution_failed")
        self.assertEqual(failed[0]["payload"]["error"]["type"], "RuntimeError")

    def test_late_failure_marks_an_integrity_checked_output_as_partial_not_produced(self) -> None:
        runtime = LateInferenceFailureRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            result_dir = root / "failed-result"

            with self.assertRaisesRegex(RuntimeError, "late inference"):
                run_worker(fixture["request"], result_dir, runtime=runtime)

            summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
            bundle = load_import_bundle(result_dir)
            failed_event = next(event for event in bundle.events if event["status"] == "FAILED")

        self.assertEqual(summary["status"], "FAILED")
        self.assertEqual(summary["output_status"], "NOT_PRODUCED")
        self.assertIsNone(summary["output_file"])
        self.assertEqual(summary["partial_output_artifact"]["role"], "partial_fhp21_output")
        self.assertEqual(bundle.output_status, "NOT_PRODUCED")
        self.assertEqual(failed_event["stage"], "DETECTION")
        self.assertEqual(failed_event["payload"]["frame_id"], "part0001/pair000001")
        self.assertIn(
            "worker_partial_fhp21_output",
            {blob.role for blob in bundle.package_blobs},
        )

    def test_cli_emits_one_json_report_using_the_same_injected_runtime(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "result"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = worker_main(
                    [str(fixture["request"]), str(result_dir)],
                    runtime=runtime,
                )

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "COMPLETED")
        self.assertEqual(report["pair_count"], 1)

    def test_import_bridge_validates_blobs_and_maps_the_event_dag_to_core_records(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["artifacts"]["source_frames"] = "ALL"
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)

            bundle = load_import_bundle(result_dir)
            records = list(bundle.core_records(external_parent_id="audit:report"))
            imported_blob_sources_exist = all(
                blob.source_path.is_file() for record in records for blob in record.blobs
            )

        self.assertEqual(bundle.output_status, "PRODUCED")
        self.assertEqual(records[0].parent_ids, ("audit:report",))
        self.assertEqual(records[0].record_id, "h20:system:verified")
        self.assertEqual(
            {blob.role for blob in records[0].blobs},
            {
                "worker_manifest",
                "worker_events",
                "worker_summary",
                "worker_fhp21_output",
            },
        )
        sync = next(record for record in records if record.stage == "SYNCHRONIZATION")
        left_detection = next(
            record for record in records if record.record_id.endswith(":detection:left")
        )
        left_pose = next(record for record in records if record.record_id.endswith(":pose2d:left"))
        association = next(
            record for record in records if record.event == "cross_view_hands_associated"
        )
        tracking = next(record for record in records if record.event == "sequence_tracks_assigned")
        raw = next(record for record in records if record.stage == "RAW_FUSION")
        self.assertEqual(sync.payload["frame_index"], 0)
        self.assertEqual(left_detection.parent_ids, (sync.record_id,))
        self.assertEqual(left_pose.parent_ids, (left_detection.record_id,))
        self.assertEqual(len(association.parent_ids), 2)
        self.assertEqual(raw.parent_ids, (association.record_id,))
        self.assertEqual(tracking.parent_ids, (raw.record_id,))
        self.assertEqual({blob.role for blob in sync.blobs}, {"source_left", "source_right"})
        self.assertTrue(imported_blob_sources_exist)

    def test_import_bridge_rejects_a_tampered_worker_blob(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["artifacts"]["source_frames"] = "ALL"
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)
            blob = next((result_dir / "blobs").rglob("*.jpg"))
            blob.write_bytes(b"tampered")

            with self.assertRaisesRegex(Exception, "identity mismatch"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_a_tampered_final_fhp21_file(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)
            with (result_dir / "fhp21.jsonl").open("ab") as handle:
                handle.write(b'\n{"tampered":true}\n')

            with self.assertRaisesRegex(Exception, "output identity mismatch"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_an_integrity_valid_but_incomplete_fhp21_record(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)
            _rewrite_first_output_record(
                result_dir,
                lambda record: record.pop("covariance_m2"),
            )

            with self.assertRaisesRegex(Exception, "missing required fields.*covariance_m2"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_an_unknown_top_level_v1_output_field(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)
            _rewrite_first_output_record(
                result_dir,
                lambda record: record.__setitem__("unversioned_extension", True),
            )

            with self.assertRaisesRegex(Exception, "unknown fields.*unversioned_extension"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_a_record_without_source_observation_payload(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)
            _rewrite_first_output_record(
                result_dir,
                lambda record: record.__setitem__("raw", None),
            )

            with self.assertRaisesRegex(Exception, "raw source observation is invalid"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_a_malformed_raw_source_observation(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)

            def truncate_raw_landmarks(record: dict[str, Any]) -> None:
                record["raw"]["landmarks_xyz_m"].pop()

            _rewrite_first_output_record(result_dir, truncate_raw_landmarks)

            with self.assertRaisesRegex(Exception, "raw.landmarks_xyz_m.*21"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_inconsistent_evidence_and_support_views(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)

            def remove_right_support(record: dict[str, Any]) -> None:
                record["support_view_ids"][0] = ["left"]

            _rewrite_first_output_record(result_dir, remove_right_support)

            with self.assertRaisesRegex(Exception, "evidence_source.*support_view_ids"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_selected_output_that_disagrees_with_temporal_source(
        self,
    ) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)

            def change_temporal_source(record: dict[str, Any]) -> None:
                record["temporal"]["landmarks_xyz_m"][0][0] += 0.25

            _rewrite_first_output_record(result_dir, change_temporal_source)

            with self.assertRaisesRegex(Exception, "selected landmarks.*temporal"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_refined_kind_without_mano_or_ema_provenance(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)

            def invent_refinement(record: dict[str, Any]) -> None:
                record["kind"][0] = "REFINED"

            _rewrite_first_output_record(result_dir, invent_refinement)

            with self.assertRaisesRegex(Exception, "kind.*MANO/temporal provenance"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_empty_mano_as_refinement_provenance(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)

            def invent_empty_mano(record: dict[str, Any]) -> None:
                record["mano"] = {}
                record["kind"] = ["REFINED"] * 21

            _rewrite_first_output_record(result_dir, invent_empty_mano)

            with self.assertRaisesRegex(Exception, "MANO refinement payload fields"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_nonfinite_and_duplicate_json_values_in_fhp21(self) -> None:
        for mutation, expected in (
            (
                lambda data: data.replace(b'"frame_index": 0', b'"frame_index": 1e400', 1),
                "non-finite",
            ),
            (
                lambda data: data.replace(
                    b'"track_id": "track-0000"',
                    b'"track_id": "track-0000", "track_id": "track-0000"',
                    1,
                ),
                "duplicate JSON key: track_id",
            ),
        ):
            with self.subTest(expected=expected), TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                fixture = _write_fixture(root, pair_count=1)
                result_dir = root / "worker-result"
                run_worker(fixture["request"], result_dir, runtime=FakeRuntime())
                original = (result_dir / "fhp21.jsonl").read_bytes()
                changed = mutation(original)
                self.assertNotEqual(changed, original)
                _rewrite_output_bytes(result_dir, changed)

                with self.assertRaisesRegex(Exception, expected):
                    load_import_bundle(result_dir)

    def test_import_bridge_rejects_null_for_a_valid_landmark(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)

            def null_valid_point(record: dict[str, Any]) -> None:
                record["landmarks_xyz_m"][0] = None
                record["temporal"]["landmarks_xyz_m"][0] = None

            _rewrite_first_output_record(result_dir, null_valid_point)

            with self.assertRaisesRegex(Exception, "valid coordinate must contain xyz"):
                load_import_bundle(result_dir)

    def test_import_bridge_rejects_non_increasing_timestamp_for_one_track(self) -> None:
        runtime = FakeRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=2)
            result_dir = root / "worker-result"
            run_worker(fixture["request"], result_dir, runtime=runtime)
            output = result_dir / "fhp21.jsonl"
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            records[1]["timestamp_ns"] = records[0]["timestamp_ns"]
            records[1]["temporal"]["timestamp_ns"] = records[0]["timestamp_ns"]
            data = b"".join(
                (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                for record in records
            )
            _rewrite_output_bytes(result_dir, data)

            with self.assertRaisesRegex(Exception, "non-increasing track timestamp"):
                load_import_bundle(result_dir)

    def test_completed_package_cannot_claim_produced_without_a_final_output_file(self) -> None:
        runtime = AssociationScenarioRuntime()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, pair_count=1)
            request = json.loads(fixture["request"].read_text(encoding="utf-8"))
            request["session"]["max_pairs"] = 1
            fixture["request"].write_text(json.dumps(request), encoding="utf-8")
            result_dir = root / "worker-result"
            # Frame zero does produce outputs; remove the final output then rewrite only
            # the summary claim to demonstrate that bridge verification is fail-closed.
            run_worker(fixture["request"], result_dir, runtime=runtime)
            (result_dir / "fhp21.jsonl").unlink()

            with self.assertRaisesRegex(Exception, "output is missing"):
                load_import_bundle(result_dir)


if __name__ == "__main__":
    unittest.main()
