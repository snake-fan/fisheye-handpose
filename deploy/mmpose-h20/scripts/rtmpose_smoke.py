#!/usr/bin/env python3
"""Run an explicit detector + RTMPose H20 deployment smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "fisheye-handpose/rtmpose-smoke/v1"
H20_COMPUTE_CAPABILITY = (9, 0)
MMPOSE_COMMIT = "5408bc76f5b848cf925a0d1857899011d8c5b497"
DEPLOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_MANIFEST = DEPLOY_ROOT / "model-assets.json"
EXPECTED_ARTIFACTS = {
    "rtmdet-nano-hand": Path("demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py"),
    "rtmpose-m-hand5": Path(
        "configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py"
    ),
}


class SmokeError(RuntimeError):
    """A deployment contract failed."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SmokeError(f"invalid arguments: {message}")


@dataclass(frozen=True)
class SmokeConfig:
    asset_manifest: Path
    asset_report: dict[str, Any]
    mmpose_source: Path
    det_config: Path
    det_checkpoint: Path
    pose_config: Path
    pose_checkpoint: Path
    image: Path
    device: str
    bbox_threshold: float
    det_category_id: int


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        description=(
            "Smoke-test an explicit MMDetection hand detector followed by low-level "
            "MMPose inference_topdown on an NVIDIA H20."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing the two checkpoints declared by model-assets.json",
    )
    parser.add_argument(
        "--mmpose-source",
        required=True,
        type=Path,
        help=f"Clean MMPose source checkout at commit {MMPOSE_COMMIT}",
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-threshold", type=float, default=0.30)
    parser.add_argument("--det-category-id", type=int, default=0)
    return parser


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SmokeError(f"{label} is not a local file: {resolved}")
    return resolved


def _resolve_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SmokeError(f"{label} is not a local directory: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_assets(manifest_path: Path, model_dir: Path) -> dict[str, Any]:
    manifest_path = _resolve_file(manifest_path, "model asset manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"cannot parse model asset manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise SmokeError("model asset manifest must be a JSON object")
    if manifest.get("schema_version") != "fisheye-handpose/model-assets/v1":
        raise SmokeError("unexpected model asset manifest schema_version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SmokeError("model asset manifest artifacts must be a list")

    by_id: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise SmokeError("each model asset entry must be an object")
        artifact_id = artifact.get("id")
        if artifact_id in by_id:
            raise SmokeError(f"duplicate model asset id: {artifact_id}")
        if isinstance(artifact_id, str):
            by_id[artifact_id] = artifact

    verified: dict[str, dict[str, Any]] = {}
    for artifact_id, expected_config in EXPECTED_ARTIFACTS.items():
        artifact = by_id.get(artifact_id)
        if artifact is None:
            raise SmokeError(f"model asset manifest is missing required id: {artifact_id}")
        filename = artifact.get("filename")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise SmokeError(f"model asset {artifact_id} filename must be a plain basename")
        if artifact.get("config") != expected_config.as_posix():
            raise SmokeError(
                f"model asset {artifact_id} must bind config {expected_config.as_posix()}"
            )
        expected_sha256 = artifact.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
        ):
            raise SmokeError(f"model asset {artifact_id} has an invalid SHA-256")
        expected_bytes = artifact.get("bytes")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
        ):
            raise SmokeError(f"model asset {artifact_id} has an invalid byte count")

        checkpoint = _resolve_file(model_dir / filename, f"{artifact_id} checkpoint")
        actual_bytes = checkpoint.stat().st_size
        if actual_bytes != expected_bytes:
            raise SmokeError(
                f"model asset {artifact_id} size mismatch: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256.lower() != expected_sha256.lower():
            raise SmokeError(f"model asset {artifact_id} SHA-256 mismatch")
        verified[artifact_id] = {
            "id": artifact_id,
            "path": checkpoint,
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "config": expected_config,
        }

    return {
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "artifacts": verified,
    }


def _config_from_namespace(
    namespace: argparse.Namespace,
    asset_manifest: Path,
) -> SmokeConfig:
    if not 0.0 <= namespace.bbox_threshold <= 1.0:
        raise SmokeError("--bbox-threshold must be between 0 and 1")
    if re.fullmatch(r"cuda(?::\d+)?", namespace.device) is None:
        raise SmokeError("--device must be a CUDA device such as cuda:0")
    model_dir = _resolve_directory(namespace.model_dir, "model directory")
    mmpose_source = _resolve_directory(namespace.mmpose_source, "MMPose source")
    asset_report = _load_verified_assets(asset_manifest, model_dir)
    detector = asset_report["artifacts"]["rtmdet-nano-hand"]
    pose = asset_report["artifacts"]["rtmpose-m-hand5"]
    return SmokeConfig(
        asset_manifest=asset_report["manifest"],
        asset_report=asset_report,
        mmpose_source=mmpose_source,
        det_config=mmpose_source / detector["config"],
        det_checkpoint=detector["path"],
        pose_config=mmpose_source / pose["config"],
        pose_checkpoint=pose["path"],
        image=_resolve_file(namespace.image, "input image"),
        device=namespace.device,
        bbox_threshold=namespace.bbox_threshold,
        det_category_id=namespace.det_category_id,
    )


def _require_h20(report: dict[str, Any]) -> None:
    if not report.get("available"):
        raise SmokeError("CUDA is not available")
    device_name = str(report.get("device_name", ""))
    if "H20" not in device_name.upper():
        raise SmokeError(f"expected an NVIDIA H20, got {device_name or 'an unnamed GPU'}")
    capability = tuple(report.get("compute_capability", ()))
    if capability != H20_COMPUTE_CAPABILITY:
        raise SmokeError(
            "expected H20 compute capability 9.0, got "
            + (".".join(str(value) for value in capability) or "unknown")
        )
    if not report.get("tensor_finite"):
        raise SmokeError("the CUDA tensor smoke calculation was not finite")


def _to_numpy(value: Any, numpy_module: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return numpy_module.asarray(value)


class OpenMMLabRuntime:
    """Lazy heavy-dependency adapter used by the CLI."""

    @staticmethod
    def _git(source: Path, *arguments: str, allow_dirty_status: bool = False) -> str:
        completed = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise SmokeError(f"git {' '.join(arguments)} failed: {detail}")
        output = completed.stdout.strip()
        if output and not allow_dirty_status:
            return output
        return output

    def verify_mmpose_source(
        self,
        *,
        source: Path,
        expected_commit: str,
        config_relative_paths: list[Path],
    ) -> dict[str, Any]:
        repository_root = Path(self._git(source, "rev-parse", "--show-toplevel")).resolve()
        if repository_root != source:
            raise SmokeError(
                f"--mmpose-source must be the repository root: expected {repository_root}"
            )
        actual_commit = self._git(source, "rev-parse", "--verify", "HEAD")
        if actual_commit != expected_commit:
            raise SmokeError(
                f"MMPose source commit mismatch: expected {expected_commit}, got {actual_commit}"
            )

        relative_names = [path.as_posix() for path in config_relative_paths]
        for relative_path in config_relative_paths:
            _resolve_file(source / relative_path, f"MMPose config {relative_path.as_posix()}")
        status = self._git(
            source,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            allow_dirty_status=True,
        )
        if status:
            raise SmokeError("MMPose source checkout must be completely clean")
        config_blobs = {
            relative_path: self._git(source, "rev-parse", f"HEAD:{relative_path}")
            for relative_path in relative_names
        }
        return {"commit": actual_commit, "configs": config_blobs}

    def cuda_smoke(self, device: str) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {
                "available": False,
                "device": device,
                "device_name": None,
                "compute_capability": None,
                "torch_cuda": torch.version.cuda,
                "tensor_finite": False,
            }

        torch_device = torch.device(device)
        device_index = (
            torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        )
        values = torch.arange(16, dtype=torch.float32, device=torch_device).reshape(4, 4)
        product = values @ values.transpose(0, 1)
        torch.cuda.synchronize(torch_device)
        return {
            "available": True,
            "device": str(torch_device),
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensor_finite": bool(torch.isfinite(product).all().item()),
        }

    def detect(
        self,
        *,
        config: Path,
        checkpoint: Path,
        image: Path,
        device: str,
        bbox_threshold: float,
        category_id: int,
    ) -> list[dict[str, Any]]:
        import numpy as np
        from mmdet.apis import inference_detector, init_detector

        detector = init_detector(str(config), str(checkpoint), device=device)
        result = inference_detector(detector, str(image))
        predicted = getattr(result, "pred_instances", None)
        if predicted is None:
            raise SmokeError("the detector result has no pred_instances")

        bboxes = _to_numpy(getattr(predicted, "bboxes", []), np)
        scores = _to_numpy(getattr(predicted, "scores", []), np)
        labels = _to_numpy(getattr(predicted, "labels", []), np)
        if bboxes.ndim != 2 or bboxes.shape[1:] != (4,):
            raise SmokeError(f"detector bboxes must have shape (N, 4), got {bboxes.shape}")
        if scores.shape != (bboxes.shape[0],) or labels.shape != (bboxes.shape[0],):
            raise SmokeError(
                "detector bboxes, scores, and labels have inconsistent instance counts"
            )
        if not (
            np.isfinite(bboxes).all() and np.isfinite(scores).all() and np.isfinite(labels).all()
        ):
            raise SmokeError("detector output contains non-finite values")

        selected = (scores >= bbox_threshold) & (labels.astype(np.int64) == category_id)
        detections: list[dict[str, Any]] = []
        for bbox, score, label in zip(
            bboxes[selected], scores[selected], labels[selected], strict=True
        ):
            detections.append(
                {
                    "bbox": [float(value) for value in bbox],
                    "score": float(score),
                    "label": int(label),
                }
            )
        return detections

    def infer_pose(
        self,
        *,
        config: Path,
        checkpoint: Path,
        image: Path,
        bboxes: list[list[float]],
        device: str,
    ) -> dict[str, Any]:
        import numpy as np
        from mmpose.apis import inference_topdown, init_model

        if not bboxes:
            raise SmokeError("pose inference was called without detector boxes")
        pose_model = init_model(str(config), str(checkpoint), device=device)
        bbox_array = np.asarray(bboxes, dtype=np.float32)
        pose_results = inference_topdown(pose_model, str(image), bboxes=bbox_array)
        if len(pose_results) != len(bboxes):
            raise SmokeError(f"pose returned {len(pose_results)} instances for {len(bboxes)} boxes")

        landmark_counts: list[int] = []
        for instance_index, result in enumerate(pose_results):
            predicted = getattr(result, "pred_instances", None)
            if predicted is None or not hasattr(predicted, "keypoints"):
                raise SmokeError(f"pose instance {instance_index} has no keypoints")
            keypoints = _to_numpy(predicted.keypoints, np)
            if keypoints.ndim == 3 and keypoints.shape[0] == 1:
                keypoints = keypoints[0]
            if keypoints.ndim != 2 or keypoints.shape[1] < 2:
                raise SmokeError(f"pose keypoints must have shape (K, C>=2), got {keypoints.shape}")
            landmark_count = int(keypoints.shape[0])
            if landmark_count != 21:
                raise SmokeError(
                    f"pose instance {instance_index} has {landmark_count} landmarks, expected 21"
                )
            if not np.isfinite(keypoints).all():
                raise SmokeError(f"pose instance {instance_index} contains non-finite keypoints")
            landmark_counts.append(landmark_count)

        return {
            "called": True,
            "instances": len(pose_results),
            "landmarks_per_instance": landmark_counts,
            "finite": True,
        }


def run_smoke(config: SmokeConfig, runtime: Any | None = None) -> dict[str, Any]:
    runtime = runtime if runtime is not None else OpenMMLabRuntime()
    source_report = runtime.verify_mmpose_source(
        source=config.mmpose_source,
        expected_commit=MMPOSE_COMMIT,
        config_relative_paths=list(EXPECTED_ARTIFACTS.values()),
    )
    if source_report.get("commit") != MMPOSE_COMMIT:
        raise SmokeError("MMPose source verification returned the wrong commit")
    expected_configs = {path.as_posix() for path in EXPECTED_ARTIFACTS.values()}
    if set(source_report.get("configs", {})) != expected_configs:
        raise SmokeError("MMPose source verification did not bind both fixed config files")
    cuda_report = runtime.cuda_smoke(config.device)
    _require_h20(cuda_report)

    detections = runtime.detect(
        config=config.det_config,
        checkpoint=config.det_checkpoint,
        image=config.image,
        device=config.device,
        bbox_threshold=config.bbox_threshold,
        category_id=config.det_category_id,
    )
    detector_report = {
        "called": True,
        "count": len(detections),
        "bbox_threshold": config.bbox_threshold,
        "category_id": config.det_category_id,
        "detections": detections,
    }
    common = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "cuda": cuda_report,
        "mmpose_source": {
            "path": str(config.mmpose_source),
            **source_report,
        },
        "model_assets": {
            "manifest": str(config.asset_manifest),
            "manifest_sha256": config.asset_report["manifest_sha256"],
            "artifacts": {
                artifact_id: {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in artifact.items()
                }
                for artifact_id, artifact in config.asset_report["artifacts"].items()
            },
        },
        "inputs": {
            "det_config": str(config.det_config),
            "det_checkpoint": str(config.det_checkpoint),
            "pose_config": str(config.pose_config),
            "pose_checkpoint": str(config.pose_checkpoint),
            "image": str(config.image),
        },
        "detector": detector_report,
    }
    if not detections:
        return {
            **common,
            "ok": False,
            "status": "no_detections",
            "pose": {"called": False, "instances": 0},
        }

    pose_report = runtime.infer_pose(
        config=config.pose_config,
        checkpoint=config.pose_checkpoint,
        image=config.image,
        bboxes=[detection["bbox"] for detection in detections],
        device=config.device,
    )
    return {**common, "status": "passed", "pose": pose_report}


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime: Any | None = None,
    asset_manifest: Path = DEFAULT_ASSET_MANIFEST,
) -> int:
    try:
        namespace = build_parser().parse_args(argv)
        config = _config_from_namespace(namespace, asset_manifest)
        with redirect_stdout(sys.stderr):
            report = run_smoke(config, runtime=runtime)
    except Exception as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        return_code = 1
    else:
        return_code = 0 if report.get("ok") else 1
    print(json.dumps(report, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
