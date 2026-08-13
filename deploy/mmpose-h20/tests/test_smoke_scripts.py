from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = DEPLOY_ROOT / "scripts"
MMPOSE_COMMIT = "5408bc76f5b848cf925a0d1857899011d8c5b497"
DET_CONFIG = Path("demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py")
POSE_CONFIG = Path("configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py")


def load_script(name: str) -> ModuleType:
    path = SCRIPTS_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def touch_inputs(root: Path, *names: str) -> dict[str, Path]:
    paths = {name: root / name for name in names}
    for path in paths.values():
        path.write_bytes(b"local test fixture")
    return paths


def write_rtmpose_assets(root: Path) -> dict[str, Path]:
    model_dir = root / "models"
    model_dir.mkdir()
    detector = model_dir / "detector.pth"
    pose = model_dir / "pose.pth"
    detector.write_bytes(b"verified detector checkpoint")
    pose.write_bytes(b"verified pose checkpoint")

    source = root / "mmpose"
    det_config = source / DET_CONFIG
    pose_config = source / POSE_CONFIG
    det_config.parent.mkdir(parents=True)
    pose_config.parent.mkdir(parents=True)
    det_config.write_text("# pinned detector config\n", encoding="utf-8")
    pose_config.write_text("# pinned pose config\n", encoding="utf-8")

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
                        "config": DET_CONFIG.as_posix(),
                    },
                    {
                        "id": "rtmpose-m-hand5",
                        "filename": pose.name,
                        "sha256": hashlib.sha256(pose.read_bytes()).hexdigest(),
                        "bytes": pose.stat().st_size,
                        "config": POSE_CONFIG.as_posix(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "model_dir": model_dir,
        "detector": detector,
        "pose": pose,
        "source": source,
        "det_config": det_config,
        "pose_config": pose_config,
        "manifest": manifest,
    }


def write_mano_assets(root: Path) -> dict[str, Path]:
    model_dir = root / "models"
    mano_dir = model_dir / "mano"
    mano_dir.mkdir(parents=True)
    left = mano_dir / "MANO_LEFT.pkl"
    right = mano_dir / "MANO_RIGHT.pkl"
    left.write_bytes(b"licensed left MANO fixture")
    right.write_bytes(b"licensed right MANO fixture")
    manifest = root / "mano-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fisheye-handpose/mano-assets/v1",
                "license": {
                    "acknowledged": True,
                    "reference": "MANO model license accepted by the supplying user",
                },
                "provenance": {
                    "acknowledged": True,
                    "source": "User-supplied MANO v1.2 files from the official MANO portal",
                },
                "artifacts": [
                    {
                        "side": "left",
                        "filename": "mano/MANO_LEFT.pkl",
                        "sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
                        "bytes": left.stat().st_size,
                    },
                    {
                        "side": "right",
                        "filename": "mano/MANO_RIGHT.pkl",
                        "sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
                        "bytes": right.stat().st_size,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {"model_dir": model_dir, "left": left, "right": right, "manifest": manifest}


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class FakeRTMPoseRuntime:
    def __init__(
        self,
        detections: list[dict[str, Any]],
        pose_report: dict[str, Any] | None = None,
        emit_dependency_log: bool = False,
        source_commit: str = MMPOSE_COMMIT,
    ) -> None:
        self.detections = detections
        self.pose_report = pose_report
        self.emit_dependency_log = emit_dependency_log
        self.source_commit = source_commit
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def cuda_smoke(self, device: str) -> dict[str, Any]:
        if self.emit_dependency_log:
            print("simulated dependency log")
        self.calls.append(("cuda_smoke", {"device": device}))
        return {
            "available": True,
            "device": device,
            "device_name": "NVIDIA H20",
            "compute_capability": [9, 0],
            "torch_cuda": "12.1",
            "tensor_finite": True,
        }

    def verify_mmpose_source(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("verify_mmpose_source", kwargs))
        return {
            "commit": self.source_commit,
            "configs": {
                relative_path.as_posix(): "verified-git-blob"
                for relative_path in kwargs["config_relative_paths"]
            },
        }

    def detect(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("detect", kwargs))
        return self.detections

    def infer_pose(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("infer_pose", kwargs))
        if self.pose_report is None:
            raise AssertionError("pose inference must not run when the detector returns no boxes")
        return self.pose_report


class RTMPoseSmokeContractTests(unittest.TestCase):
    def test_source_verification_rejects_dirty_base_config(self) -> None:
        module = load_script("rtmpose_smoke")
        with TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "mmpose"
            source.mkdir()
            run_git(source, "init", "-q")
            run_git(source, "config", "user.name", "Smoke Test")
            run_git(source, "config", "user.email", "smoke@example.invalid")
            det_config = source / DET_CONFIG
            pose_config = source / POSE_CONFIG
            base_config = source / "configs/_base_/runtime.py"
            det_config.parent.mkdir(parents=True)
            pose_config.parent.mkdir(parents=True)
            base_config.parent.mkdir(parents=True)
            det_config.write_text("# detector config\n", encoding="utf-8")
            pose_config.write_text("_base_ = '../../../_base_/runtime.py'\n", encoding="utf-8")
            base_config.write_text("default_scope = 'mmpose'\n", encoding="utf-8")
            run_git(source, "add", ".")
            run_git(source, "commit", "-q", "-m", "pinned source fixture")
            expected_commit = run_git(source, "rev-parse", "HEAD")
            base_config.write_text("default_scope = 'attacker'\n", encoding="utf-8")

            with self.assertRaises(module.SmokeError):
                module.OpenMMLabRuntime().verify_mmpose_source(
                    source=source.resolve(),
                    expected_commit=expected_commit,
                    config_relative_paths=[DET_CONFIG, POSE_CONFIG],
                )

    def test_zero_detections_short_circuit_pose_and_emit_json(self) -> None:
        module = load_script("rtmpose_smoke")
        runtime = FakeRTMPoseRuntime(detections=[])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = write_rtmpose_assets(root)
            inputs.update(touch_inputs(root, "frame.jpg"))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--mmpose-source",
                        str(inputs["source"]),
                        "--image",
                        str(inputs["frame.jpg"]),
                    ],
                    runtime=runtime,
                    asset_manifest=inputs["manifest"],
                )

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema_version"], "fisheye-handpose/rtmpose-smoke/v1")
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "no_detections")
        self.assertEqual(report["detector"]["count"], 0)
        self.assertFalse(report["pose"]["called"])
        self.assertEqual(
            [name for name, _ in runtime.calls],
            ["verify_mmpose_source", "cuda_smoke", "detect"],
        )
        source_call = runtime.calls[0][1]
        self.assertEqual(source_call["source"], inputs["source"].resolve())
        self.assertEqual(source_call["expected_commit"], MMPOSE_COMMIT)
        self.assertEqual(source_call["config_relative_paths"], [DET_CONFIG, POSE_CONFIG])
        detector_call = runtime.calls[2][1]
        self.assertEqual(detector_call["config"], inputs["det_config"].resolve())
        self.assertEqual(detector_call["checkpoint"], inputs["detector"].resolve())

    def test_detection_flows_to_explicit_pose_model_as_detector_boxes(self) -> None:
        module = load_script("rtmpose_smoke")
        detection = {"bbox": [10.0, 20.0, 110.0, 220.0], "score": 0.91, "label": 0}
        runtime = FakeRTMPoseRuntime(
            detections=[detection],
            pose_report={
                "called": True,
                "instances": 1,
                "landmarks_per_instance": [21],
                "finite": True,
            },
            emit_dependency_log=True,
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = write_rtmpose_assets(root)
            inputs.update(touch_inputs(root, "frame.jpg"))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--mmpose-source",
                        str(inputs["source"]),
                        "--image",
                        str(inputs["frame.jpg"]),
                    ],
                    runtime=runtime,
                    asset_manifest=inputs["manifest"],
                )

        self.assertEqual(return_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["pose"]["landmarks_per_instance"], [21])
        self.assertEqual(
            [name for name, _ in runtime.calls],
            ["verify_mmpose_source", "cuda_smoke", "detect", "infer_pose"],
        )
        pose_call = runtime.calls[3][1]
        self.assertEqual(pose_call["config"], inputs["pose_config"].resolve())
        self.assertEqual(pose_call["checkpoint"], inputs["pose"].resolve())
        self.assertEqual(pose_call["bboxes"], [detection["bbox"]])

    def test_missing_arguments_are_reported_as_json(self) -> None:
        module = load_script("rtmpose_smoke")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            return_code = module.main([])

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "failed")

    def test_tampered_checkpoint_fails_before_source_cuda_or_model_loading(self) -> None:
        module = load_script("rtmpose_smoke")
        runtime = FakeRTMPoseRuntime(detections=[])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = write_rtmpose_assets(root)
            inputs.update(touch_inputs(root, "frame.jpg"))
            inputs["pose"].write_bytes(b"tampered without updating the trusted manifest")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--mmpose-source",
                        str(inputs["source"]),
                        "--image",
                        str(inputs["frame.jpg"]),
                    ],
                    runtime=runtime,
                    asset_manifest=inputs["manifest"],
                )

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["ok"])
        self.assertIn("mismatch", report["error"]["message"].lower())
        self.assertEqual(runtime.calls, [])

    def test_wrong_mmpose_commit_fails_before_cuda_or_model_loading(self) -> None:
        module = load_script("rtmpose_smoke")
        runtime = FakeRTMPoseRuntime(detections=[], source_commit="0" * 40)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = write_rtmpose_assets(root)
            inputs.update(touch_inputs(root, "frame.jpg"))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--mmpose-source",
                        str(inputs["source"]),
                        "--image",
                        str(inputs["frame.jpg"]),
                    ],
                    runtime=runtime,
                    asset_manifest=inputs["manifest"],
                )

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["ok"])
        self.assertIn("commit", report["error"]["message"].lower())
        self.assertEqual([name for name, _ in runtime.calls], ["verify_mmpose_source"])


class FakeManoRuntime:
    def __init__(self, *, emit_dependency_log: bool = False) -> None:
        self.emit_dependency_log = emit_dependency_log
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def cuda_smoke(self, device: str) -> dict[str, Any]:
        if self.emit_dependency_log:
            print("simulated dependency log")
        self.calls.append(("cuda_smoke", {"device": device}))
        return {
            "available": True,
            "device": device,
            "device_name": "NVIDIA H20",
            "compute_capability": [9, 0],
            "torch_cuda": "12.1",
            "tensor_finite": True,
        }

    def load_model(self, **kwargs: Any) -> str:
        self.calls.append(("load_model", kwargs))
        return f"{kwargs['side']}-model"

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("forward", kwargs))
        return {
            "finite": True,
            "vertices_shape": [1, 778, 3],
            "joints_shape": [1, 16, 3],
        }

    def adam_backward_smoke(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("adam_backward_smoke", kwargs))
        return {"backward": True, "gradients_finite": True, "parameters_finite": True}

    def lbfgs_closure_smoke(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("lbfgs_closure_smoke", kwargs))
        return {"closure_calls": 2, "gradients_finite": True, "parameters_finite": True}


class ManoSmokeContractTests(unittest.TestCase):
    def test_local_left_and_right_models_run_forward_adam_and_lbfgs(self) -> None:
        module = load_script("mano_smoke")
        runtime = FakeManoRuntime(emit_dependency_log=True)
        with TemporaryDirectory() as temporary_directory:
            inputs = write_mano_assets(Path(temporary_directory))
            expected_left_sha256 = hashlib.sha256(inputs["left"].read_bytes()).hexdigest()
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--manifest",
                        str(inputs["manifest"]),
                    ],
                    runtime=runtime,
                )

        self.assertEqual(return_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema_version"], "fisheye-handpose/mano-smoke/v1")
        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["models"]["left"]["finite"])
        self.assertTrue(report["models"]["right"]["finite"])
        self.assertEqual(report["mano_assets"]["license"]["acknowledged"], True)
        self.assertEqual(report["mano_assets"]["provenance"]["acknowledged"], True)
        self.assertEqual(
            report["mano_assets"]["artifacts"]["left"]["sha256"],
            expected_left_sha256,
        )
        self.assertTrue(report["optimizers"]["adam"]["backward"])
        self.assertGreaterEqual(report["optimizers"]["lbfgs"]["closure_calls"], 1)
        self.assertEqual(
            [name for name, _ in runtime.calls],
            [
                "cuda_smoke",
                "load_model",
                "forward",
                "load_model",
                "forward",
                "adam_backward_smoke",
                "lbfgs_closure_smoke",
            ],
        )
        self.assertEqual(runtime.calls[1][1]["side"], "left")
        self.assertEqual(runtime.calls[3][1]["side"], "right")
        self.assertEqual(runtime.calls[1][1]["model_dir"], inputs["model_dir"].resolve())
        self.assertEqual(runtime.calls[3][1]["model_dir"], inputs["model_dir"].resolve())

    def test_missing_arguments_are_reported_as_json(self) -> None:
        module = load_script("mano_smoke")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            return_code = module.main([])

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "failed")

    def test_missing_mano_file_fails_before_heavy_runtime_is_used(self) -> None:
        module = load_script("mano_smoke")
        runtime = FakeManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            inputs = write_mano_assets(Path(temporary_directory))
            inputs["left"].unlink()
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--manifest",
                        str(inputs["manifest"]),
                    ],
                    runtime=runtime,
                )

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["ok"])
        self.assertIn("MANO_LEFT.pkl", report["error"]["message"])
        self.assertEqual(runtime.calls, [])

    def test_unacknowledged_license_fails_before_heavy_runtime_is_used(self) -> None:
        module = load_script("mano_smoke")
        runtime = FakeManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            inputs = write_mano_assets(Path(temporary_directory))
            manifest = json.loads(inputs["manifest"].read_text(encoding="utf-8"))
            manifest["license"]["acknowledged"] = False
            inputs["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--manifest",
                        str(inputs["manifest"]),
                    ],
                    runtime=runtime,
                )

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["ok"])
        self.assertIn("license", report["error"]["message"].lower())
        self.assertEqual(runtime.calls, [])

    def test_tampered_mano_file_fails_before_heavy_runtime_is_used(self) -> None:
        module = load_script("mano_smoke")
        runtime = FakeManoRuntime()
        with TemporaryDirectory() as temporary_directory:
            inputs = write_mano_assets(Path(temporary_directory))
            inputs["left"].write_bytes(b"tampered without updating the user manifest")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--model-dir",
                        str(inputs["model_dir"]),
                        "--manifest",
                        str(inputs["manifest"]),
                    ],
                    runtime=runtime,
                )

        self.assertEqual(return_code, 1)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["ok"])
        self.assertIn("mismatch", report["error"]["message"].lower())
        self.assertEqual(runtime.calls, [])


if __name__ == "__main__":
    unittest.main()
