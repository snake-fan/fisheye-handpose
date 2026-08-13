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

from fisheye_h20_worker.bridge import load_import_bundle  # noqa: E402
from fisheye_h20_worker.cli import main as worker_main  # noqa: E402
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
        self.encoded_frames: list[tuple[str, int, str]] = []

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
        self.encoded_frames.append((frame["side"], frame["index"], image_format))
        return f"{frame}:{image_format}".encode()


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
        }


class WorkerContractTests(unittest.TestCase):
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
            ["part0001:pair000000:raw:match-0"],
        )
        self.assertEqual(raw["payload"]["track_assignment"]["decision"], "NEW")
        self.assertEqual(raw["payload"]["track_id"], "track-0000")
        self.assertIn("detections", detection["payload"])
        self.assertEqual(tracked_pose["payload"]["track_id"], "track-0000")
        self.assertEqual(len(tracked_pose["payload"]["keypoints_uv"]), 21)
        self.assertEqual(len(tracked_pose["payload"]["keypoint_scores"]), 21)
        self.assertEqual(len(tracked_pose["payload"]["detections"]), 1)
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
        self.assertIsNone(runtime.mano_fit_calls[0]["fixed_beta"])
        self.assertIsNone(runtime.mano_fit_calls[1]["fixed_beta"])
        self.assertEqual(runtime.mano_fit_calls[2]["fixed_beta"], [0.2] * 10)
        mano_events = [event for event in events if event["event"] == "mano_frame_fitted"]
        self.assertEqual(len(mano_events), 2)
        self.assertEqual(mano_events[0]["payload"]["handedness"], "right")
        self.assertEqual(mano_events[0]["payload"]["selection"]["decision"], "SELECTED")
        self.assertEqual(mano_events[0]["payload"]["loss"]["metric"], "RMSE_M")
        self.assertEqual(mano_events[0]["payload"]["loss"]["value"], 0.005)
        self.assertTrue(mano_events[1]["payload"]["beta_frozen"])
        self.assertEqual(exported[0]["mano"]["mapping_id"], MANO_FHP21_MAPPING_ID)
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
        self.assertEqual(
            mano["payload"]["state_predecessor_event_id"],
            f"{prior_prefix}:mano:match-0",
        )
        self.assertIn(f"{prior_prefix}:temporal:match-0", temporal["parent_event_ids"])
        self.assertEqual(
            temporal["payload"]["state_predecessor_event_id"],
            f"{prior_prefix}:temporal:match-0",
        )

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
            ["part0001:pair000001:tracking"],
        )
        for predecessor, event in zip(empty_frame[:-1], empty_frame[1:], strict=True):
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

        temporal = next(
            event for event in events if event["event"] == "temporal_landmarks_not_produced"
        )
        export = next(event for event in events if event["event"] == "fhp21_record_not_produced")
        self.assertEqual(result["output_status"], "NOT_PRODUCED")
        self.assertEqual(export["payload"]["output_status"], "NOT_PRODUCED")
        self.assertEqual(export["payload"]["track_id"], "track-0000")
        self.assertEqual(export["parent_event_ids"], [temporal["event_id"]])

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
            for reference in refs:
                data = (result_dir / reference["relative_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), reference["sha256"])
                self.assertEqual(len(data), reference["bytes"])

        self.assertEqual(
            runtime.encoded_frames,
            [("left", 0, "png"), ("right", 0, "png")],
        )
        self.assertEqual({reference["role"] for reference in refs}, {"source_left", "source_right"})

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
        self.assertEqual(tracking.parent_ids, (association.record_id,))
        self.assertEqual(len(raw.parent_ids), 2)
        self.assertTrue(all(parent.endswith((":left", ":right")) for parent in raw.parent_ids))
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
