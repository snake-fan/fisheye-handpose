#!/usr/bin/env python3
"""Fail-closed deployment checks for the pinned H20 MMPose environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "fisheye-handpose/doctor/v1"
MANIFEST_SCHEMA = "fisheye-handpose/h20-environment/v1"
DEPLOY_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = DEPLOY_ROOT / "environment.json"
EXPECTED_TARGET = {
    "platform": "linux-x86_64",
    "python": "3.10",
    "cuda": "12.1",
    "gpu": "NVIDIA H20",
    "compute_capability": "9.0",
}
EXPECTED_RESOLUTION = {
    "uv": ">=0.12.3,<0.13",
    "exclude_newer": "2024-08-01T00:00:00Z",
}
EXPECTED_PACKAGES = {
    "torch": "2.1.0",
    "torchvision": "0.16.0",
    "mmcv": "2.1.0",
    "mmengine": "0.10.3",
    "mmdet": "3.2.0",
    "mmpose": "1.3.2",
    "numpy": "1.26.4",
    "chumpy": "0.71",
    "smplx": "0.1.28",
}
EXPECTED_BINARY_SOURCES = {
    "torch": "https://download.pytorch.org/whl/cu121",
    "torchvision": "https://download.pytorch.org/whl/cu121",
    "mmcv": (
        "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/"
        "mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl"
    ),
    "chumpy": ("git+https://github.com/nim65s/chumpy.git@2816a138d2f60bc8a77eddb9962c4c825179cb56"),
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    expected: Any = None
    actual: Any = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "ok": self.ok}
        if self.expected is not None:
            result["expected"] = self.expected
        if self.actual is not None:
            result["actual"] = self.actual
        if self.detail is not None:
            result["detail"] = self.detail
        return result


def equality_check(name: str, expected: Any, actual: Any) -> Check:
    return Check(name=name, ok=actual == expected, expected=expected, actual=actual)


def load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[Check]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [Check("manifest.load", False, actual=str(path), detail=str(exc))]
    if not isinstance(value, dict):
        return None, [
            Check("manifest.load", False, expected="JSON object", actual=type(value).__name__)
        ]
    return value, [Check("manifest.load", True, actual=str(path))]


def manifest_checks(path: Path) -> tuple[dict[str, Any] | None, list[Check]]:
    manifest, checks = load_manifest(path)
    if manifest is None:
        return None, checks

    checks.extend(
        [
            equality_check("manifest.schema", MANIFEST_SCHEMA, manifest.get("schema_version")),
            equality_check(
                "manifest.intent",
                "legacy-compatibility-reproduction",
                manifest.get("intent"),
            ),
            equality_check(
                "manifest.security_status",
                "not-approved-for-production",
                manifest.get("security_status"),
            ),
            equality_check("manifest.target", EXPECTED_TARGET, manifest.get("target")),
            equality_check(
                "manifest.resolution",
                EXPECTED_RESOLUTION,
                manifest.get("resolution"),
            ),
            equality_check("manifest.packages", EXPECTED_PACKAGES, manifest.get("packages")),
            equality_check(
                "manifest.binary_sources",
                EXPECTED_BINARY_SOURCES,
                manifest.get("binary_sources"),
            ),
            equality_check("python-version", "3.10", read_python_version()),
        ]
    )
    return manifest, checks


def read_python_version() -> str | None:
    try:
        return (DEPLOY_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def base_version(version: str | None) -> str | None:
    return version.split("+", 1)[0] if version is not None else None


def guarded_check(name: str, function: Callable[[], Check]) -> Check:
    try:
        return function()
    except Exception as exc:  # Runtime imports and native extensions can fail arbitrarily.
        return Check(name, False, detail=f"{type(exc).__name__}: {exc}")


def package_checks() -> list[Check]:
    return [
        equality_check(f"package.{name}", expected, base_version(installed_version(name)))
        for name, expected in EXPECTED_PACKAGES.items()
    ]


def import_checks() -> list[Check]:
    modules = ("torch", "torchvision", "mmcv", "mmengine", "mmdet", "mmpose", "chumpy", "smplx")
    checks: list[Check] = []
    for module_name in modules:
        try:
            module = __import__(module_name)
        except Exception as exc:
            checks.append(
                Check(
                    f"runtime.import.{module_name}",
                    False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            checks.append(
                Check(
                    f"runtime.import.{module_name}",
                    True,
                    actual=getattr(module, "__version__", None),
                )
            )
    return checks


def check_host() -> Check:
    system = platform.system().lower()
    machine = platform.machine().lower()
    actual = f"{system}-{'x86_64' if machine in {'x86_64', 'amd64'} else machine}"
    return equality_check("runtime.platform", "linux-x86_64", actual)


def check_python() -> Check:
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    return equality_check("runtime.python", "3.10", actual)


def check_torch_cuda(torch: Any) -> Check:
    return equality_check("runtime.torch_cuda", "12.1", torch.version.cuda)


def check_cuda_available(torch: Any) -> Check:
    available = bool(torch.cuda.is_available())
    return equality_check("runtime.cuda_available", True, available)


def check_sm90(torch: Any) -> Check:
    if not torch.cuda.is_available():
        return Check("runtime.compute_capability", False, expected="9.0", actual=None)
    devices = []
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": f"{major}.{minor}",
            }
        )
    ok = bool(devices) and all(item["compute_capability"] == "9.0" for item in devices)
    return Check("runtime.compute_capability", ok, expected="9.0", actual=devices)


def check_mmcv_nms(torch: Any) -> Check:
    from mmcv.ops import nms

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]],
        device="cuda",
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8], device="cuda", dtype=torch.float32)
    detections, keep = nms(boxes, scores, 0.5)
    ok = detections.is_cuda and keep.is_cuda and keep.detach().cpu().tolist() == [0]
    return Check(
        "runtime.mmcv.ops.nms",
        bool(ok),
        expected={"device": "cuda", "keep": [0]},
        actual={"device": detections.device.type, "keep": keep.detach().cpu().tolist()},
    )


def check_fp16(torch: Any) -> Check:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float16)
    right = torch.eye(2, device="cuda", dtype=torch.float16)
    result = left @ right
    torch.cuda.synchronize()
    ok = result.dtype == torch.float16 and torch.isfinite(result).all().item()
    return Check(
        "runtime.float16",
        bool(ok),
        expected={"device": "cuda", "dtype": "torch.float16", "finite": True},
        actual={
            "device": result.device.type,
            "dtype": str(result.dtype),
            "finite": bool(torch.isfinite(result).all().item()),
        },
    )


def model_directory_check(path: Path) -> Check:
    try:
        has_file = path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))
    except OSError as exc:
        return Check("runtime.model_directory", False, actual=str(path), detail=str(exc))
    return Check(
        "runtime.model_directory",
        has_file,
        expected="existing non-empty directory",
        actual=str(path),
    )


def runtime_checks(model_dirs: list[Path]) -> list[Check]:
    checks = [check_host(), check_python(), *package_checks(), *import_checks()]
    try:
        import torch
    except Exception as exc:
        checks.append(Check("runtime.torch_import", False, detail=f"{type(exc).__name__}: {exc}"))
        return checks

    checks.extend(
        [
            Check("runtime.torch_import", True, actual=getattr(torch, "__version__", None)),
            guarded_check("runtime.torch_cuda", lambda: check_torch_cuda(torch)),
            guarded_check("runtime.cuda_available", lambda: check_cuda_available(torch)),
            guarded_check("runtime.compute_capability", lambda: check_sm90(torch)),
            guarded_check("runtime.mmcv.ops.nms", lambda: check_mmcv_nms(torch)),
            guarded_check("runtime.float16", lambda: check_fp16(torch)),
        ]
    )
    checks.extend(model_directory_check(path) for path in model_dirs)
    return checks


def build_report(mode: str, target: dict[str, Any], checks: list[Check]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "ok": bool(checks) and all(check.ok for check in checks),
        "target": target,
        "checks": [check.as_dict() for check in checks],
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="runtime")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", action="append", type=Path, default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        checks = [Check("arguments.parse", False, detail=f"argparse exited with {exc.code}")]
        print(json.dumps(build_report("invalid", EXPECTED_TARGET, checks), sort_keys=True))
        return 1

    if args.mode not in {"manifest", "runtime"}:
        checks = [
            Check(
                "arguments.mode",
                False,
                expected=["manifest", "runtime"],
                actual=args.mode,
            )
        ]
        print(json.dumps(build_report(args.mode, EXPECTED_TARGET, checks), sort_keys=True))
        return 1

    manifest, checks = manifest_checks(args.manifest)
    target = manifest.get("target", EXPECTED_TARGET) if manifest is not None else EXPECTED_TARGET
    if args.mode == "runtime" and all(check.ok for check in checks):
        checks.extend(runtime_checks(args.model_dir))
    report = build_report(args.mode, target, checks)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # Keep all unexpected failures machine-readable and fail closed.
        failure = Check("doctor.internal", False, detail=f"{type(exc).__name__}: {exc}")
        report = build_report("internal-error", EXPECTED_TARGET, [failure])
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(1) from None
