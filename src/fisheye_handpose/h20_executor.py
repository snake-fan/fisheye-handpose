"""Process-isolated H20 worker adapter for the canonical run trace.

The core stays on its Python 3.11 geometry/runtime environment.  CUDA perception and
MANO run in the separately locked Python 3.10 environment and return a self-contained,
hash-verified package.  Only this adapter owns the canonical trace writer.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import FisheyeHandposeError
from .pipeline import PipelineExecutionContext, PipelineExecutionSummary
from .trace import BlobRef, RunArtifactWriter, TraceStage, TraceStatus

EXECUTOR_SCHEMA = "fisheye-handpose/h20-executor/v1"
WORKER_REQUEST_SCHEMA = "fisheye-handpose/h20-worker-request/v1"


class H20ExecutorConfigurationError(FisheyeHandposeError):
    """The process-isolated worker configuration is incomplete or unsafe."""


class H20WorkerExecutionError(FisheyeHandposeError):
    """The isolated worker failed or returned an invalid evidence package."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise H20ExecutorConfigurationError(f"{label} must be a JSON object")
    return value


def _path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise H20ExecutorConfigurationError(f"{label} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _worker_python_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise H20ExecutorConfigurationError("worker_python must be a non-empty path")
    # A venv's Python is commonly a symlink to its base interpreter. Keep the venv
    # entry point so Python discovers pyvenv.cfg and activates that environment.
    return Path(value).expanduser().absolute()


@dataclass(frozen=True, slots=True)
class H20ExecutorConfig:
    worker_python: Path
    worker_module_root: Path
    request_template: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> H20ExecutorConfig:
        root = _mapping(value, "executor config")
        if root.get("schema_version") != EXECUTOR_SCHEMA:
            raise H20ExecutorConfigurationError(f"schema_version must be {EXECUTOR_SCHEMA!r}")
        request = _mapping(root.get("request"), "request")
        required = {"session", "thresholds", "models", "artifacts"}
        missing = sorted(required - request.keys())
        if missing:
            raise H20ExecutorConfigurationError(f"request template is missing sections: {missing}")
        worker_python = _worker_python_path(root.get("worker_python"))
        worker_module_root = _path(root.get("worker_module_root"), "worker_module_root")
        if not worker_python.is_file():
            raise H20ExecutorConfigurationError(f"worker_python is not a file: {worker_python}")
        if not worker_module_root.is_dir():
            raise H20ExecutorConfigurationError(
                f"worker_module_root is not a directory: {worker_module_root}"
            )
        return cls(
            worker_python=worker_python,
            worker_module_root=worker_module_root,
            request_template=json.loads(json.dumps(request, allow_nan=False, ensure_ascii=False)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> H20ExecutorConfig:
        source = Path(path).expanduser().resolve()
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise H20ExecutorConfigurationError(
                f"cannot load H20 executor config {source}: {exc}"
            ) from exc
        return cls.from_dict(value)


def _default_process_runner(command: list[str], *, env: dict[str, str]) -> tuple[int, bytes, bytes]:
    completed = subprocess.run(
        command,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _load_bridge(worker_root: Path) -> ModuleType:
    bridge_path = worker_root / "fisheye_h20_worker" / "bridge.py"
    if not bridge_path.is_file():
        raise H20ExecutorConfigurationError(f"worker bridge is missing: {bridge_path}")
    package_name = "fisheye_h20_worker"
    package_init = worker_root / package_name / "__init__.py"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        package_init,
        submodule_search_locations=[str(package_init.parent)],
    )
    if package_spec is None or package_spec.loader is None:
        raise H20ExecutorConfigurationError("cannot load H20 worker package")
    package = importlib.util.module_from_spec(package_spec)
    import sys

    previous_package = sys.modules.get(package_name)
    previous_bridge = sys.modules.get(f"{package_name}.bridge")
    try:
        sys.modules[package_name] = package
        package_spec.loader.exec_module(package)
        bridge_spec = importlib.util.spec_from_file_location(f"{package_name}.bridge", bridge_path)
        if bridge_spec is None or bridge_spec.loader is None:
            raise H20ExecutorConfigurationError("cannot load H20 worker bridge")
        bridge = importlib.util.module_from_spec(bridge_spec)
        sys.modules[f"{package_name}.bridge"] = bridge
        bridge_spec.loader.exec_module(bridge)
        return bridge
    finally:
        if previous_package is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous_package
        if previous_bridge is None:
            sys.modules.pop(f"{package_name}.bridge", None)
        else:
            sys.modules[f"{package_name}.bridge"] = previous_bridge


def _worker_request(config: H20ExecutorConfig, context: PipelineExecutionContext) -> dict[str, Any]:
    template = json.loads(json.dumps(config.request_template, allow_nan=False))
    audit = context.request.audit_config
    session = _mapping(template["session"], "request.session")
    max_pairs = session.get("max_pairs")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs <= 0:
        raise H20ExecutorConfigurationError("request.session.max_pairs must be positive")
    report_session = _mapping(context.audit_report.get("session"), "audit session")
    calibration_path = report_session.get("calibration_path")
    if not isinstance(calibration_path, str) or not calibration_path:
        raise H20ExecutorConfigurationError(
            "audit report does not contain session.calibration_path"
        )
    template["schema_version"] = WORKER_REQUEST_SCHEMA
    template["session"] = {
        "path": str(Path(context.request.session_path).expanduser().resolve()),
        "timestamp_column": audit.timestamp_column,
        "timestamp_unit": audit.timestamp_unit,
        "max_skew_us": audit.max_skew_ns // 1_000,
        "clock_offset_ns": audit.clock_offset_ns,
        "max_pairs": max_pairs,
    }
    template["calibration"] = {
        "path": str(Path(calibration_path).expanduser().resolve()),
        "left_camera_id": audit.left_id,
        "right_camera_id": audit.right_id,
        "translation_unit": audit.translation_unit,
        "extrinsics_convention": audit.extrinsics_convention,
        "output_size": list(audit.output_size),
        "balance": audit.balance,
        "fov_scale": audit.fov_scale,
    }
    return template


def _log_blob(writer: RunArtifactWriter, data: bytes, *, role: str) -> BlobRef | None:
    if not data:
        return None
    return writer.put_blob(
        data,
        role=role,
        media_type="text/plain; charset=utf-8",
        suffix=".log",
    )


def _append_worker_failure(
    writer: RunArtifactWriter,
    context: PipelineExecutionContext,
    *,
    returncode: int | None,
    phase: str,
    message: str,
    blobs: tuple[BlobRef, ...],
) -> str:
    record = writer.append(
        record_id="h20:worker:failed",
        stage=TraceStage.SYSTEM,
        status=TraceStatus.FAILED,
        event="h20_worker_failed",
        payload={
            "returncode": returncode,
            "failure_phase": phase,
            "message": message,
            "output_status": "NOT_PRODUCED",
        },
        parent_ids=(context.audit_record_ids[-1],),
        blobs=blobs,
    )
    return record.record_id


def _diagnostic_package_blobs(
    writer: RunArtifactWriter,
    result_dir: Path,
) -> tuple[BlobRef, ...]:
    blobs: list[BlobRef] = []
    media_types = {
        "manifest.json": "application/json",
        "events.jsonl": "application/x-ndjson",
        "summary.json": "application/json",
        "fhp21.jsonl": "application/x-ndjson",
    }
    for name, media_type in media_types.items():
        path = result_dir / name
        if path.is_file():
            blobs.append(
                writer.put_blob(
                    path.read_bytes(),
                    role=f"invalid_worker_{path.stem}",
                    media_type=media_type,
                    suffix=path.suffix,
                )
            )
    return tuple(blobs)


class H20WorkerExecutor:
    """Run a configured CUDA worker and import its verified result package."""

    def __init__(
        self,
        config: H20ExecutorConfig,
        *,
        process_runner: Callable[..., tuple[int, bytes, bytes]] = _default_process_runner,
    ) -> None:
        if not isinstance(config, H20ExecutorConfig):
            raise TypeError("config must be H20ExecutorConfig")
        self.config = config
        self._process_runner = process_runner

    def execute(
        self,
        context: PipelineExecutionContext,
        writer: RunArtifactWriter,
    ) -> PipelineExecutionSummary:
        if not self.config.worker_python.is_file():
            raise H20ExecutorConfigurationError(
                f"worker Python is not a file: {self.config.worker_python}"
            )
        if not self.config.worker_module_root.is_dir():
            raise H20ExecutorConfigurationError(
                f"worker module root is not a directory: {self.config.worker_module_root}"
            )
        request = _worker_request(self.config, context)
        bridge = _load_bridge(self.config.worker_module_root)
        with tempfile.TemporaryDirectory(prefix="fhp-h20-worker-") as temporary:
            temporary_root = Path(temporary)
            request_path = temporary_root / "request.json"
            result_dir = temporary_root / "result"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            request_blob = writer.put_blob(
                request_path.read_bytes(),
                role="worker_request",
                media_type="application/json",
                suffix=".json",
            )
            command = [
                str(self.config.worker_python),
                "-m",
                "fisheye_h20_worker",
                str(request_path),
                str(result_dir),
            ]
            try:
                returncode, stdout, stderr = self._process_runner(
                    command, env={"PYTHONPATH": str(self.config.worker_module_root)}
                )
            except (OSError, subprocess.SubprocessError) as exc:
                record_id = _append_worker_failure(
                    writer,
                    context,
                    returncode=None,
                    phase="process_start",
                    message=str(exc),
                    blobs=(request_blob,),
                )
                raise H20WorkerExecutionError(
                    f"cannot start H20 worker; trace record={record_id}: {exc}"
                ) from exc
            stdout_blob = _log_blob(writer, stdout, role="worker_stdout")
            stderr_blob = _log_blob(writer, stderr, role="worker_stderr")
            failure_blobs = (request_blob,) + tuple(
                blob for blob in (stdout_blob, stderr_blob) if blob is not None
            )
            if not result_dir.is_dir():
                record_id = _append_worker_failure(
                    writer,
                    context,
                    returncode=returncode,
                    phase="worker_process",
                    message="worker did not create a result package",
                    blobs=failure_blobs,
                )
                raise H20WorkerExecutionError(
                    f"H20 worker exited {returncode} without a result package; "
                    f"trace record={record_id}"
                )
            try:
                bundle = bridge.load_import_bundle(result_dir)
            except Exception as exc:
                failure_blobs += _diagnostic_package_blobs(writer, result_dir)
                record_id = _append_worker_failure(
                    writer,
                    context,
                    returncode=returncode,
                    phase="package_validation",
                    message=str(exc),
                    blobs=failure_blobs,
                )
                raise H20WorkerExecutionError(
                    f"H20 worker returned an invalid result package; "
                    f"trace record={record_id}: {exc}"
                ) from exc
            imported_ids: list[str] = []
            for ordinal, record in enumerate(
                bundle.core_records(external_parent_id=context.audit_record_ids[-1])
            ):
                blobs: list[BlobRef] = []
                for value in record.blobs:
                    blobs.append(
                        writer.put_blob(
                            value.source_path.read_bytes(),
                            role=value.role,
                            media_type=value.media_type,
                            suffix=value.suffix,
                        )
                    )
                if ordinal == 0:
                    blobs.extend(failure_blobs)
                payload = dict(record.payload)
                provenance = payload.get("worker_provenance")
                if isinstance(provenance, dict):
                    provenance = dict(provenance)
                    provenance.pop("result_dir", None)
                    payload["worker_provenance"] = provenance
                imported = writer.append(
                    record_id=record.record_id,
                    stage=TraceStage(record.stage),
                    status=TraceStatus(record.status),
                    event=record.event,
                    payload=payload,
                    parent_ids=record.parent_ids,
                    blobs=tuple(blobs),
                )
                imported_ids.append(imported.record_id)
            if not imported_ids:
                record_id = _append_worker_failure(
                    writer,
                    context,
                    returncode=returncode,
                    phase="package_import",
                    message="worker produced no importable stage records",
                    blobs=failure_blobs,
                )
                raise H20WorkerExecutionError(
                    f"H20 worker produced no importable stage records; trace record={record_id}"
                )
            if returncode != 0 or bundle.terminal_status != "COMPLETED":
                raise H20WorkerExecutionError(
                    f"H20 worker failed: returncode={returncode}, "
                    f"terminal_status={bundle.terminal_status}"
                )
            return PipelineExecutionSummary(
                output_status=bundle.output_status,
                record_ids=tuple(imported_ids),
                details={
                    "worker_returncode": returncode,
                    "worker_terminal_status": bundle.terminal_status,
                    "worker_record_count": len(imported_ids),
                    "worker_summary": bundle.summary,
                },
            )


__all__ = [
    "EXECUTOR_SCHEMA",
    "H20ExecutorConfig",
    "H20ExecutorConfigurationError",
    "H20WorkerExecutionError",
    "H20WorkerExecutor",
]
