"""Lazy OpenMMLab/OpenCV adapter; imported only by the real H20 CLI."""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidates import CandidateBatch, CandidatePolicy
from .contracts import WorkerError
from .mano import MANO_FHP21_MAPPING_ID, MANO_FHP21_SOURCES


@dataclass(frozen=True)
class LoadedModels:
    detector: Any
    pose: Any


def _numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class OpenMMLabRuntime:
    """Own both initialized models for the lifetime of one worker invocation."""

    @staticmethod
    def _git(source: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise WorkerError(f"git {' '.join(arguments)} failed: {detail}")
        return completed.stdout.strip()

    def verify_source(
        self,
        *,
        source: Path,
        expected_commit: str,
        config_relative_paths: list[str],
    ) -> dict[str, Any]:
        if not source.is_dir():
            raise WorkerError(f"MMPose source directory does not exist: {source}")
        repository_root = Path(self._git(source, "rev-parse", "--show-toplevel")).resolve()
        if repository_root != source:
            raise WorkerError("mmpose_source must be the repository root")
        commit = self._git(source, "rev-parse", "--verify", "HEAD")
        if commit != expected_commit:
            raise WorkerError(f"MMPose source commit mismatch: {commit}")
        if self._git(source, "status", "--porcelain=v1", "--untracked-files=all"):
            raise WorkerError("MMPose source checkout must be completely clean")
        configs: dict[str, str] = {}
        for relative in config_relative_paths:
            path = (source / relative).resolve()
            try:
                path.relative_to(source)
            except ValueError as exc:
                raise WorkerError("MMPose config escapes source root") from exc
            if not path.is_file():
                raise WorkerError(f"MMPose config is missing: {relative}")
            configs[relative] = self._git(source, "rev-parse", f"HEAD:{relative}")
        return {"commit": commit, "configs": configs}

    def load_models(
        self,
        *,
        det_config: Path,
        det_checkpoint: Path,
        pose_config: Path,
        pose_checkpoint: Path,
        device: str,
    ) -> LoadedModels:
        import torch
        from mmdet.apis import init_detector
        from mmpose.apis import init_model

        if not torch.cuda.is_available():
            raise WorkerError("CUDA is unavailable")
        torch_device = torch.device(device)
        device_index = (
            torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        )
        if tuple(torch.cuda.get_device_capability(device_index)) != (9, 0):
            raise WorkerError("worker requires H20/SM90 compute capability 9.0")
        if "H20" not in torch.cuda.get_device_name(device_index).upper():
            raise WorkerError("worker CUDA device is not an NVIDIA H20")
        return LoadedModels(
            detector=init_detector(str(det_config), str(det_checkpoint), device=device),
            pose=init_model(str(pose_config), str(pose_checkpoint), device=device),
        )

    def iter_video_frames(self, path: Path):
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise WorkerError(f"OpenCV cannot open video: {path}")
            decoded = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None:
                    raise WorkerError(f"OpenCV returned an empty decoded frame: {path}")
                decoded += 1
                yield frame
            if decoded == 0:
                raise WorkerError(f"OpenCV decoded no presentation-order frames: {path}")
        except WorkerError:
            raise
        except Exception as exc:
            raise WorkerError(f"video decode failed for {path}: {exc}") from exc
        finally:
            capture.release()

    def detect(
        self,
        models: LoadedModels,
        frame: Any,
        *,
        bbox_threshold: float,
        category_id: int,
        max_instances: int = 2,
    ) -> list[dict[str, Any]]:
        """Return deterministic native-fisheye detector proposals without running pose."""

        import numpy as np
        from mmdet.apis import inference_detector
        from mmengine.registry import DefaultScope

        if (
            isinstance(max_instances, bool)
            or not isinstance(max_instances, int)
            or max_instances < 1
        ):
            raise WorkerError("max_instances must be a positive integer")

        with DefaultScope.overwrite_default_scope("mmdet"):
            detection_result = inference_detector(models.detector, frame)
        predicted = getattr(detection_result, "pred_instances", None)
        if predicted is None:
            raise WorkerError("detector result has no pred_instances")
        bboxes = _numpy(getattr(predicted, "bboxes", []), np)
        scores = _numpy(getattr(predicted, "scores", []), np)
        labels = _numpy(getattr(predicted, "labels", []), np)
        if bboxes.ndim != 2 or bboxes.shape[1:] != (4,):
            raise WorkerError(f"detector bbox shape is invalid: {bboxes.shape}")
        if scores.shape != (len(bboxes),) or labels.shape != (len(bboxes),):
            raise WorkerError("detector outputs have inconsistent instance counts")
        selected = [
            index
            for index in range(len(bboxes))
            if float(scores[index]) >= bbox_threshold and int(labels[index]) == category_id
        ]
        selected.sort(key=lambda index: (-float(scores[index]), index))
        selected = selected[:max_instances]
        return [
            {
                "bbox_xyxy": bboxes[index].astype(float).tolist(),
                "bbox_score": float(scores[index]),
                "label": int(labels[index]),
            }
            for index in selected
        ]

    def detect_candidates(
        self,
        models: LoadedModels,
        frame: Any,
        *,
        policy: CandidatePolicy,
        category_id: int,
        view_id: str,
    ) -> CandidateBatch:
        """Classify every raw detector proposal before bounding the association pool."""

        import numpy as np
        from mmdet.apis import inference_detector
        from mmengine.registry import DefaultScope

        if not isinstance(policy, CandidatePolicy):
            raise WorkerError("candidate policy must be a CandidatePolicy")
        with DefaultScope.overwrite_default_scope("mmdet"):
            detection_result = inference_detector(models.detector, frame)
        predicted = getattr(detection_result, "pred_instances", None)
        if predicted is None:
            raise WorkerError("detector result has no pred_instances")
        return policy.classify(
            bboxes=_numpy(getattr(predicted, "bboxes", []), np),
            scores=_numpy(getattr(predicted, "scores", []), np),
            labels=_numpy(getattr(predicted, "labels", []), np),
            category_id=category_id,
            view_id=view_id,
        )

    def infer_pose(
        self,
        models: LoadedModels,
        frame: Any,
        *,
        bboxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Run top-down pose for explicit boxes in the supplied physical image space."""

        import numpy as np
        from mmengine.registry import DefaultScope
        from mmpose.apis import inference_topdown

        selected_boxes = np.asarray(bboxes, dtype=np.float32)
        if selected_boxes.size == 0:
            return []
        if selected_boxes.ndim != 2 or selected_boxes.shape[1:] != (4,):
            raise WorkerError(f"pose bbox shape is invalid: {selected_boxes.shape}")
        if not np.all(np.isfinite(selected_boxes)):
            raise WorkerError("pose bboxes contain non-finite values")
        with DefaultScope.overwrite_default_scope("mmpose"):
            pose_results = inference_topdown(models.pose, frame, bboxes=selected_boxes)
        if len(pose_results) != len(selected_boxes):
            raise WorkerError("pose result count differs from detector box count")
        evidence: list[dict[str, Any]] = []
        for output_index, pose_result in enumerate(pose_results):
            pose = getattr(pose_result, "pred_instances", None)
            if pose is None or not hasattr(pose, "keypoints"):
                raise WorkerError(f"pose instance {output_index} has no keypoints")
            keypoints = _numpy(pose.keypoints, np)
            keypoint_scores = _numpy(getattr(pose, "keypoint_scores", []), np)
            if keypoints.ndim == 3 and keypoints.shape[0] == 1:
                keypoints = keypoints[0]
            if keypoint_scores.ndim == 2 and keypoint_scores.shape[0] == 1:
                keypoint_scores = keypoint_scores[0]
            if keypoints.shape != (21, 2) or keypoint_scores.shape != (21,):
                raise WorkerError(
                    f"pose instance must have (21,2)/(21,) output, got "
                    f"{keypoints.shape}/{keypoint_scores.shape}"
                )
            if not (np.all(np.isfinite(keypoints)) and np.all(np.isfinite(keypoint_scores))):
                raise WorkerError("model output contains non-finite values")
            evidence.append(
                {
                    "keypoints_uv": keypoints.astype(float).tolist(),
                    "keypoint_scores": keypoint_scores.astype(float).tolist(),
                }
            )
        return evidence

    def infer(
        self,
        models: LoadedModels,
        frame: Any,
        *,
        bbox_threshold: float,
        category_id: int,
        max_instances: int = 2,
    ) -> list[dict[str, Any]]:
        """Compatibility composition for the recorded native-fisheye baseline profile."""

        detections = self.detect(
            models,
            frame,
            bbox_threshold=bbox_threshold,
            category_id=category_id,
            max_instances=max_instances,
        )
        if not detections:
            return []
        poses = self.infer_pose(
            models,
            frame,
            bboxes=[detection["bbox_xyxy"] for detection in detections],
        )
        return [{**detection, **pose} for detection, pose in zip(detections, poses, strict=True)]

    def encode_frame(self, frame: Any, image_format: str) -> bytes:
        import cv2

        extension = ".jpg" if image_format == "jpg" else ".png"
        ok, encoded = cv2.imencode(extension, frame)
        if not ok:
            raise WorkerError(f"cannot encode source frame as {image_format}")
        return bytes(encoded)

    def render_rectification(
        self,
        rectification: Any,
        side: str,
        frame: Any,
    ) -> dict[str, Any]:
        """Render diagnostic still/video spaces without changing model inference input."""

        return {
            "undistorted": rectification.render_frame(
                side,
                frame,
                image_space="undistorted",
            ),
            "rectified": rectification.render_frame(
                side,
                frame,
                image_space="rectified",
            ),
        }

    def load_mano_models(self, *, model_root: Path, device: str) -> dict[str, Any]:
        """Load each already-hash-verified MANO pickle exactly once."""
        import smplx
        import torch

        models: dict[str, Any] = {}
        for side in ("left", "right"):
            model = smplx.create(
                str(model_root),
                model_type="mano",
                ext="pkl",
                is_rhand=side == "right",
                use_pca=False,
                flat_hand_mean=False,
                batch_size=1,
                dtype=torch.float32,
            ).to(torch.device(device))
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            models[side] = model
        return models

    @staticmethod
    def _mapped_mano_tensor(output: Any, torch: Any) -> Any:
        joints = getattr(output, "joints", None)
        vertices = getattr(output, "vertices", None)
        if joints is None or vertices is None:
            raise WorkerError("MANO output must contain joints and vertices")
        if joints.ndim != 3 or joints.shape[0] != 1 or joints.shape[1] < 16:
            raise WorkerError(f"MANO joint output shape is invalid: {tuple(joints.shape)}")
        if vertices.ndim != 3 or vertices.shape[0] != 1 or vertices.shape[1] <= 744:
            raise WorkerError(f"MANO vertex output shape is invalid: {tuple(vertices.shape)}")
        return torch.stack(
            [
                (joints[0, index] if source == "joint" else vertices[0, index])
                for source, index in MANO_FHP21_SOURCES
            ],
            dim=0,
        )

    def fit_mano(
        self,
        models: dict[str, Any],
        *,
        side: str,
        target_xyz_m: list[list[float] | None],
        validity: list[str],
        fixed_beta: list[float] | None,
        device: str,
        iterations: int,
        learning_rate: float,
        joint_weights: list[float] | None = None,
        initial_parameters: dict[str, Any] | None = None,
        seed_id: str = "mano_mean",
    ) -> dict[str, Any]:
        """Robust full-pose fit with accepted-state warm start and best-state recovery."""
        import torch

        if side not in {"left", "right"} or side not in models:
            raise WorkerError(f"unsupported or unloaded MANO side: {side}")
        if len(target_xyz_m) != 21 or len(validity) != 21:
            raise WorkerError("MANO fit target must contain 21 landmarks and validity values")
        torch_device = torch.device(device)
        valid_indices = [
            index
            for index, (point, flag) in enumerate(zip(target_xyz_m, validity, strict=True))
            if flag == "VALID" and isinstance(point, list) and len(point) == 3
        ]
        if not valid_indices:
            raise WorkerError("MANO fit requires at least one valid target landmark")
        if joint_weights is None:
            normalized_joint_weights = [
                1.0 if index in valid_indices else 0.0 for index in range(21)
            ]
        else:
            if not isinstance(joint_weights, list) or len(joint_weights) != 21:
                raise WorkerError("MANO joint_weights must contain 21 values")
            normalized_joint_weights = []
            for index, value in enumerate(joint_weights):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise WorkerError("MANO joint_weights must be finite values in [0, 1]")
                weight = float(value)
                if index not in valid_indices and weight != 0.0:
                    raise WorkerError("MANO joint_weights must be zero for invalid landmarks")
                normalized_joint_weights.append(weight)
        weighted_indices = [
            index for index in valid_indices if normalized_joint_weights[index] > 0.0
        ]
        if not weighted_indices:
            raise WorkerError("MANO fit requires at least one positive valid joint weight")
        if not isinstance(seed_id, str) or not seed_id:
            raise WorkerError("MANO fit seed_id must be non-empty")
        if initial_parameters is not None and not isinstance(initial_parameters, dict):
            raise WorkerError("MANO initial_parameters must be an object or null")
        dense_target = [([0.0, 0.0, 0.0] if point is None else point) for point in target_xyz_m]
        target = torch.tensor(dense_target, dtype=torch.float32, device=torch_device)
        mask = torch.tensor(valid_indices, dtype=torch.long, device=torch_device)
        selected_weights = torch.tensor(
            [normalized_joint_weights[index] for index in valid_indices],
            dtype=torch.float32,
            device=torch_device,
        )
        if not bool(torch.isfinite(target.index_select(0, mask)).all().item()):
            raise WorkerError("MANO fit target contains non-finite values")

        def initial_tensor(field: str, length: int, default: Any) -> Any:
            value = None if initial_parameters is None else initial_parameters.get(field)
            if value is None:
                return default.detach().clone()
            if not isinstance(value, list) or len(value) != length:
                raise WorkerError(f"MANO initial {field} must contain {length} values")
            tensor = torch.tensor([value], dtype=torch.float32, device=torch_device)
            if not bool(torch.isfinite(tensor).all().item()):
                raise WorkerError(f"MANO initial {field} must be finite")
            return tensor

        hand_pose = torch.nn.Parameter(
            initial_tensor(
                "hand_pose",
                45,
                torch.zeros((1, 45), dtype=torch.float32, device=torch_device),
            )
        )
        global_orient = torch.nn.Parameter(
            initial_tensor(
                "global_orient",
                3,
                torch.zeros((1, 3), dtype=torch.float32, device=torch_device),
            )
        )
        selected_target = target.index_select(0, mask)
        target_center = (selected_target * selected_weights[:, None]).sum(
            dim=0, keepdim=True
        ) / selected_weights.sum()
        transl = torch.nn.Parameter(initial_tensor("transl", 3, target_center))
        if fixed_beta is None:
            beta: Any = torch.nn.Parameter(
                initial_tensor(
                    "beta",
                    10,
                    torch.zeros((1, 10), dtype=torch.float32, device=torch_device),
                )
            )
            parameters = [hand_pose, global_orient, transl, beta]
        else:
            if len(fixed_beta) != 10:
                raise WorkerError("fixed MANO beta must contain 10 values")
            beta = torch.tensor([fixed_beta], dtype=torch.float32, device=torch_device).detach()
            if not bool(torch.isfinite(beta).all().item()):
                raise WorkerError("fixed MANO beta must be finite")
            parameters = [hand_pose, global_orient, transl]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        model = models[side]
        best_loss = math.inf
        final_loss = math.inf
        best_values: tuple[Any, Any, Any, Any] | None = None
        stale_steps = 0
        patience = min(40, max(10, iterations // 5))
        iterations_run = 0

        def objective() -> Any:
            output = model(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=beta,
                transl=transl,
                return_verts=True,
            )
            mapped = self._mapped_mano_tensor(output, torch)
            residual = mapped.index_select(0, mask) - target.index_select(0, mask)
            residual_norm = torch.linalg.vector_norm(residual, dim=1)
            huber_delta = torch.tensor(0.02, dtype=torch.float32, device=torch_device)
            quadratic = torch.minimum(residual_norm, huber_delta)
            linear = residual_norm - quadratic
            per_joint_loss = 0.5 * quadratic.square() + huber_delta * linear
            data_loss = (per_joint_loss * selected_weights).sum() / selected_weights.sum()
            regularization = 1e-4 * hand_pose.square().mean()
            regularization = regularization + 1e-5 * global_orient.square().mean()
            if fixed_beta is None:
                regularization = regularization + 1e-5 * beta.square().mean()
            return data_loss + regularization

        def parameter_snapshot() -> tuple[Any, Any, Any, Any]:
            return (
                hand_pose.detach().clone(),
                global_orient.detach().clone(),
                transl.detach().clone(),
                beta.detach().clone(),
            )

        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not bool(torch.isfinite(loss).item()):
                raise WorkerError("MANO optimization loss is non-finite")
            if best_values is None:
                best_loss = float(loss.detach().item())
                best_values = parameter_snapshot()
            loss.backward()
            if any(
                parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in parameters
            ):
                raise WorkerError("MANO optimization gradient is missing or non-finite")
            optimizer.step()
            iterations_run = step + 1
            with torch.no_grad():
                post_step_loss = objective()
            if not bool(torch.isfinite(post_step_loss).item()):
                raise WorkerError("MANO post-step optimization loss is non-finite")
            final_loss = float(post_step_loss.detach().item())
            if final_loss < best_loss - 1e-10:
                best_loss = final_loss
                best_values = parameter_snapshot()
                stale_steps = 0
            else:
                stale_steps += 1
            if iterations_run >= 20 and stale_steps >= patience:
                break
        if best_values is None:
            raise WorkerError("MANO optimizer did not produce a finite state")
        with torch.no_grad():
            hand_pose.copy_(best_values[0])
            global_orient.copy_(best_values[1])
            transl.copy_(best_values[2])
            if fixed_beta is None:
                beta.copy_(best_values[3])
            else:
                beta = best_values[3]
        with torch.no_grad():
            output = model(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=beta,
                transl=transl,
                return_verts=True,
            )
            mapped = self._mapped_mano_tensor(output, torch)
            residual = mapped.index_select(0, mask) - target.index_select(0, mask)
            rmse = torch.sqrt(residual.square().sum(dim=1).mean())
            residual_square = residual.square().sum(dim=1)
            weighted_rmse = torch.sqrt(
                (residual_square * selected_weights).sum() / selected_weights.sum()
            )
            all_residuals = torch.linalg.vector_norm(mapped - target, dim=1)
        tensors = (mapped, rmse, weighted_rmse, hand_pose, global_orient, transl, beta)
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise WorkerError("MANO fit output contains non-finite values")
        return {
            "side": side,
            "mapping_id": MANO_FHP21_MAPPING_ID,
            "landmarks_xyz_m": mapped.detach().cpu().tolist(),
            "validity": ["VALID"] * 21,
            "rmse_m": float(rmse.detach().item()),
            "weighted_rmse_m": float(weighted_rmse.detach().item()),
            "effective_joint_count": len(weighted_indices),
            "joint_weights": normalized_joint_weights,
            "global_orient": global_orient.detach().cpu()[0].tolist(),
            "hand_pose": hand_pose.detach().cpu()[0].tolist(),
            "transl": transl.detach().cpu()[0].tolist(),
            "beta": beta.detach().cpu()[0].tolist(),
            "iterations_run": iterations_run,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "joint_residuals_m": [
                (float(all_residuals[index].detach().item()) if index in valid_indices else None)
                for index in range(21)
            ],
            "converged": iterations_run < iterations,
            "seed_id": seed_id,
        }


__all__ = ["LoadedModels", "OpenMMLabRuntime"]
