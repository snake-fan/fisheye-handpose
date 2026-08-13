#!/usr/bin/env python3
"""Run local MANO left/right and optimizer smoke tests on an NVIDIA H20."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "fisheye-handpose/mano-smoke/v1"
H20_COMPUTE_CAPABILITY = (9, 0)
MANO_FILES = {
    "left": Path("mano") / "MANO_LEFT.pkl",
    "right": Path("mano") / "MANO_RIGHT.pkl",
}


class SmokeError(RuntimeError):
    """A deployment contract failed."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SmokeError(f"invalid arguments: {message}")


@dataclass(frozen=True)
class SmokeConfig:
    model_dir: Path
    manifest: Path
    asset_report: dict[str, Any]
    device: str
    adam_lr: float
    lbfgs_lr: float
    lbfgs_max_iter: int


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        description=(
            "Smoke-test user-supplied MANO_LEFT.pkl and MANO_RIGHT.pkl files with smplx, "
            "including forward, Adam backward, and an LBFGS closure on NVIDIA H20."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Local smplx model root containing mano/MANO_LEFT.pkl and MANO_RIGHT.pkl",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="User-maintained MANO manifest with hashes and license/provenance acknowledgements",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs-lr", type=float, default=0.1)
    parser.add_argument("--lbfgs-max-iter", type=int, default=2)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_mano_assets(model_dir: Path, manifest_path: Path) -> dict[str, Any]:
    resolved_manifest = manifest_path.expanduser().resolve()
    if not resolved_manifest.is_file():
        raise SmokeError(f"MANO manifest is not a local file: {resolved_manifest}")
    try:
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"cannot parse MANO manifest {resolved_manifest}: {error}") from error
    if not isinstance(manifest, dict):
        raise SmokeError("MANO manifest must be a JSON object")
    if manifest.get("schema_version") != "fisheye-handpose/mano-assets/v1":
        raise SmokeError("unexpected MANO manifest schema_version")

    license_record = manifest.get("license")
    if not isinstance(license_record, dict) or license_record.get("acknowledged") is not True:
        raise SmokeError("MANO manifest must explicitly acknowledge the model license")
    if (
        not isinstance(license_record.get("reference"), str)
        or not license_record["reference"].strip()
    ):
        raise SmokeError("MANO manifest license.reference must be a non-empty string")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("acknowledged") is not True:
        raise SmokeError("MANO manifest must explicitly acknowledge artifact provenance")
    if not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
        raise SmokeError("MANO manifest provenance.source must be a non-empty string")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SmokeError("MANO manifest artifacts must be a list")
    by_side: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise SmokeError("each MANO artifact entry must be an object")
        side = artifact.get("side")
        if side in by_side:
            raise SmokeError(f"duplicate MANO artifact side: {side}")
        if isinstance(side, str):
            by_side[side] = artifact

    verified: dict[str, dict[str, Any]] = {}
    for side, relative_path in MANO_FILES.items():
        artifact = by_side.get(side)
        if artifact is None:
            raise SmokeError(f"MANO manifest is missing the {side} artifact")
        if artifact.get("filename") != relative_path.as_posix():
            raise SmokeError(f"MANO {side} filename must be exactly {relative_path.as_posix()}")
        expected_sha256 = artifact.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
        ):
            raise SmokeError(f"MANO {side} artifact has an invalid SHA-256")
        expected_bytes = artifact.get("bytes")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
        ):
            raise SmokeError(f"MANO {side} artifact has an invalid byte count")

        model_file = model_dir / relative_path
        if not model_file.is_file():
            raise SmokeError(f"required local MANO model file is missing: {model_file}")
        actual_bytes = model_file.stat().st_size
        if actual_bytes != expected_bytes:
            raise SmokeError(
                f"MANO {side} size mismatch: expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha256 = _sha256(model_file)
        if actual_sha256.lower() != expected_sha256.lower():
            raise SmokeError(f"MANO {side} SHA-256 mismatch")
        verified[side] = {
            "path": str(model_file),
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }

    return {
        "manifest": str(resolved_manifest),
        "manifest_sha256": _sha256(resolved_manifest),
        "license": license_record,
        "provenance": provenance,
        "artifacts": verified,
    }


def _resolve_model_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SmokeError(f"MANO model directory does not exist: {resolved}")
    return resolved


def _config_from_namespace(namespace: argparse.Namespace) -> SmokeConfig:
    if re.fullmatch(r"cuda(?::\d+)?", namespace.device) is None:
        raise SmokeError("--device must be a CUDA device such as cuda:0")
    if namespace.adam_lr <= 0:
        raise SmokeError("--adam-lr must be positive")
    if namespace.lbfgs_lr <= 0:
        raise SmokeError("--lbfgs-lr must be positive")
    if namespace.lbfgs_max_iter < 1:
        raise SmokeError("--lbfgs-max-iter must be at least 1")
    model_dir = _resolve_model_dir(namespace.model_dir)
    asset_report = _load_verified_mano_assets(model_dir, namespace.manifest)
    return SmokeConfig(
        model_dir=model_dir,
        manifest=Path(asset_report["manifest"]),
        asset_report=asset_report,
        device=namespace.device,
        adam_lr=namespace.adam_lr,
        lbfgs_lr=namespace.lbfgs_lr,
        lbfgs_max_iter=namespace.lbfgs_max_iter,
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


class TorchManoRuntime:
    """Lazy torch/smplx adapter used by the CLI."""

    def __init__(self) -> None:
        import smplx
        import torch

        self.smplx = smplx
        self.torch = torch

    def cuda_smoke(self, device: str) -> dict[str, Any]:
        torch = self.torch
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

    def load_model(self, *, model_dir: Path, side: str, device: str) -> Any:
        if side not in MANO_FILES:
            raise SmokeError(f"unsupported MANO side: {side}")
        model = self.smplx.create(
            str(model_dir),
            model_type="mano",
            ext="pkl",
            is_rhand=side == "right",
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
            dtype=self.torch.float32,
        )
        model = model.to(self.torch.device(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _forward_with_pose(self, model: Any, hand_pose: Any, device: str) -> Any:
        torch = self.torch
        torch_device = torch.device(device)
        return model(
            global_orient=torch.zeros((1, 3), dtype=torch.float32, device=torch_device),
            hand_pose=hand_pose,
            betas=torch.zeros((1, 10), dtype=torch.float32, device=torch_device),
            transl=torch.zeros((1, 3), dtype=torch.float32, device=torch_device),
            return_verts=True,
        )

    def forward(self, *, model: Any, side: str, device: str) -> dict[str, Any]:
        torch = self.torch
        hand_pose = torch.zeros((1, 45), dtype=torch.float32, device=torch.device(device))
        with torch.no_grad():
            output = self._forward_with_pose(model, hand_pose, device)
        vertices = getattr(output, "vertices", None)
        joints = getattr(output, "joints", None)
        if vertices is None or joints is None:
            raise SmokeError(f"{side} MANO output must contain vertices and joints")
        finite = bool(torch.isfinite(vertices).all().item() and torch.isfinite(joints).all().item())
        if not finite:
            raise SmokeError(f"{side} MANO forward contains non-finite values")
        return {
            "finite": True,
            "vertices_shape": list(vertices.shape),
            "joints_shape": list(joints.shape),
        }

    def _new_pose_parameters(self, device: str, initial_value: float) -> dict[str, Any]:
        torch = self.torch
        torch_device = torch.device(device)
        return {
            side: torch.nn.Parameter(
                torch.full(
                    (1, 45),
                    initial_value,
                    dtype=torch.float32,
                    device=torch_device,
                )
            )
            for side in ("left", "right")
        }

    def _optimization_loss(
        self,
        models: dict[str, Any],
        poses: dict[str, Any],
        device: str,
    ) -> Any:
        torch = self.torch
        total = torch.zeros((), dtype=torch.float32, device=torch.device(device))
        for side in ("left", "right"):
            output = self._forward_with_pose(models[side], poses[side], device)
            vertices = getattr(output, "vertices", None)
            joints = getattr(output, "joints", None)
            if vertices is None or joints is None:
                raise SmokeError(f"{side} MANO optimization output is incomplete")
            total = total + vertices.square().mean() + joints.square().mean()
            total = total + 1e-4 * poses[side].square().mean()
        return total

    def _validate_gradients(self, poses: dict[str, Any], optimizer_name: str) -> None:
        torch = self.torch
        for side, pose in poses.items():
            if pose.grad is None:
                raise SmokeError(f"{optimizer_name} did not produce a {side} hand-pose gradient")
            if not bool(torch.isfinite(pose.grad).all().item()):
                raise SmokeError(f"{optimizer_name} produced a non-finite {side} gradient")

    def adam_backward_smoke(
        self,
        *,
        models: dict[str, Any],
        device: str,
        learning_rate: float,
    ) -> dict[str, Any]:
        torch = self.torch
        poses = self._new_pose_parameters(device, initial_value=0.01)
        optimizer = torch.optim.Adam(list(poses.values()), lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)
        loss = self._optimization_loss(models, poses, device)
        if not bool(torch.isfinite(loss).item()):
            raise SmokeError("Adam loss is not finite")
        loss.backward()
        self._validate_gradients(poses, "Adam")
        optimizer.step()
        parameters_finite = all(
            bool(torch.isfinite(parameter).all().item()) for parameter in poses.values()
        )
        if not parameters_finite:
            raise SmokeError("Adam produced non-finite hand-pose parameters")
        torch.cuda.synchronize(torch.device(device))
        return {
            "backward": True,
            "loss": float(loss.detach().item()),
            "gradients_finite": True,
            "parameters_finite": True,
        }

    def lbfgs_closure_smoke(
        self,
        *,
        models: dict[str, Any],
        device: str,
        learning_rate: float,
        max_iter: int,
    ) -> dict[str, Any]:
        torch = self.torch
        poses = self._new_pose_parameters(device, initial_value=0.02)
        optimizer = torch.optim.LBFGS(
            list(poses.values()),
            lr=learning_rate,
            max_iter=max_iter,
            tolerance_grad=0.0,
            tolerance_change=0.0,
            line_search_fn=None,
        )
        closure_calls = 0

        def closure() -> Any:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            loss = self._optimization_loss(models, poses, device)
            if not bool(torch.isfinite(loss).item()):
                raise SmokeError("LBFGS closure loss is not finite")
            loss.backward()
            self._validate_gradients(poses, "LBFGS")
            closure_calls += 1
            return loss

        initial_loss = float(self._optimization_loss(models, poses, device).detach().item())
        optimizer.step(closure)
        final_loss_tensor = self._optimization_loss(models, poses, device).detach()
        if not bool(torch.isfinite(final_loss_tensor).item()):
            raise SmokeError("LBFGS final loss is not finite")
        parameters_finite = all(
            bool(torch.isfinite(parameter).all().item()) for parameter in poses.values()
        )
        if closure_calls < 1:
            raise SmokeError("LBFGS did not invoke its closure")
        if not parameters_finite:
            raise SmokeError("LBFGS produced non-finite hand-pose parameters")
        torch.cuda.synchronize(torch.device(device))
        return {
            "closure_calls": closure_calls,
            "initial_loss": initial_loss,
            "final_loss": float(final_loss_tensor.item()),
            "gradients_finite": True,
            "parameters_finite": True,
        }


def _validate_optimizer_report(name: str, report: dict[str, Any]) -> None:
    if name == "Adam" and not report.get("backward"):
        raise SmokeError("Adam backward did not run")
    if name == "LBFGS" and int(report.get("closure_calls", 0)) < 1:
        raise SmokeError("LBFGS closure did not run")
    if not report.get("gradients_finite"):
        raise SmokeError(f"{name} gradients were not finite")
    if not report.get("parameters_finite"):
        raise SmokeError(f"{name} parameters were not finite")


def run_smoke(config: SmokeConfig, runtime: Any | None = None) -> dict[str, Any]:
    runtime = runtime if runtime is not None else TorchManoRuntime()
    cuda_report = runtime.cuda_smoke(config.device)
    _require_h20(cuda_report)

    models: dict[str, Any] = {}
    model_reports: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        model = runtime.load_model(
            model_dir=config.model_dir,
            side=side,
            device=config.device,
        )
        models[side] = model
        report = runtime.forward(model=model, side=side, device=config.device)
        if not report.get("finite"):
            raise SmokeError(f"{side} MANO forward was not finite")
        model_reports[side] = report

    adam_report = runtime.adam_backward_smoke(
        models=models,
        device=config.device,
        learning_rate=config.adam_lr,
    )
    _validate_optimizer_report("Adam", adam_report)
    lbfgs_report = runtime.lbfgs_closure_smoke(
        models=models,
        device=config.device,
        learning_rate=config.lbfgs_lr,
        max_iter=config.lbfgs_max_iter,
    )
    _validate_optimizer_report("LBFGS", lbfgs_report)

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "status": "passed",
        "cuda": cuda_report,
        "model_dir": str(config.model_dir),
        "mano_assets": config.asset_report,
        "models": model_reports,
        "optimizers": {"adam": adam_report, "lbfgs": lbfgs_report},
    }


def main(argv: Sequence[str] | None = None, *, runtime: Any | None = None) -> int:
    try:
        namespace = build_parser().parse_args(argv)
        config = _config_from_namespace(namespace)
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
        return_code = 0
    print(json.dumps(report, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
