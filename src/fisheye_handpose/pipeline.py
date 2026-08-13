"""One immutable, inspectable run directory per stereo data item."""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from . import __version__
from .audit import AuditConfig, audit_session
from .audit_trace import append_audit_trace
from .errors import FisheyeHandposeError
from .trace import RunArtifactWriter, RunStatus, TraceStage, TraceStatus

OUTPUT_NOT_PRODUCED = "NOT_PRODUCED"
OUTPUT_PRODUCED = "PRODUCED"
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MODEL_STAGES = (
    TraceStage.DETECTION,
    TraceStage.POSE_2D,
    TraceStage.CROSS_VIEW_ASSOCIATION,
    TraceStage.RAW_FUSION,
    TraceStage.KINEMATIC_REFINEMENT,
    TraceStage.TEMPORAL_REFINEMENT,
    TraceStage.EXPORT,
)


class PipelineConfigurationError(FisheyeHandposeError):
    """A run request cannot be represented by the immutable directory protocol."""


@dataclass(frozen=True, slots=True)
class PipelineRunRequest:
    session_path: str | Path
    runs_root: str | Path
    audit_config: AuditConfig
    item_id: str | None = None
    run_id: str | None = None
    pipeline_version: str = __version__


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    item_id: str
    run_id: str
    run_dir: Path
    status: RunStatus
    output_status: str
    audit_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": self.status.value,
            "output_status": self.output_status,
            "audit_status": self.audit_status,
        }


@dataclass(frozen=True, slots=True)
class PipelineExecutionContext:
    request: PipelineRunRequest
    item_id: str
    run_id: str
    audit_report: dict[str, Any]
    audit_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineExecutionSummary:
    output_status: str
    record_ids: tuple[str, ...]
    details: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_status, str) or not self.output_status.strip():
            raise ValueError("output_status must be a non-empty string")
        if any(not isinstance(record_id, str) or not record_id for record_id in self.record_ids):
            raise ValueError("record_ids must contain non-empty strings")


class PipelineStageExecutor(Protocol):
    """Minimal injection boundary for an external model-stage worker."""

    def execute(
        self,
        context: PipelineExecutionContext,
        writer: RunArtifactWriter,
    ) -> PipelineExecutionSummary: ...


def _validate_explicit_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise PipelineConfigurationError(f"{label} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}")
    return value


def _default_item_id(source_id: str) -> str:
    if _SAFE_COMPONENT.fullmatch(source_id) and source_id not in {".", ".."}:
        return source_id
    ascii_value = unicodedata.normalize("NFKD", source_id).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower() or "item"
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:118]}-{digest}"


def _default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _append_unproduced_stages(
    writer: RunArtifactWriter,
    *,
    parent_id: str,
    reason: str,
) -> tuple[str, ...]:
    record_ids: list[str] = []
    current_parent = parent_id
    for stage in _MODEL_STAGES:
        record = writer.append(
            record_id=f"pipeline:skipped:{stage.value.lower()}",
            stage=stage,
            status=TraceStatus.SKIPPED,
            event="stage_output_not_produced",
            payload={
                "output_status": OUTPUT_NOT_PRODUCED,
                "reason": reason,
            },
            parent_ids=(current_parent,),
        )
        record_ids.append(record.record_id)
        current_parent = record.record_id
    return tuple(record_ids)


def run_data_item(
    request: PipelineRunRequest,
    backends: PipelineStageExecutor | None = None,
) -> PipelineRunResult:
    """Run the available stages and retain their evidence in one non-overwriting folder.

    ``backends`` is the injection seam for the perception composition. Until that
    composition is provided, every model-dependent stage is persisted as ``SKIPPED`` and
    the independent output state is ``NOT_PRODUCED``.
    """

    if not isinstance(request, PipelineRunRequest):
        raise TypeError("request must be a PipelineRunRequest")
    if not isinstance(request.audit_config, AuditConfig):
        raise TypeError("request.audit_config must be an AuditConfig")
    session_path = Path(request.session_path).expanduser().resolve()
    runs_root = Path(request.runs_root).expanduser().resolve()
    source_item_id = session_path.name or "session"
    item_id = (
        _default_item_id(source_item_id)
        if request.item_id is None
        else _validate_explicit_component(request.item_id, "item_id")
    )
    run_id = (
        _default_run_id()
        if request.run_id is None
        else _validate_explicit_component(request.run_id, "run_id")
    )
    item_dir = runs_root / item_id
    run_dir = item_dir / run_id
    item_dir.mkdir(parents=True, exist_ok=True)

    writer = RunArtifactWriter.create(
        run_dir,
        run_id=run_id,
        pipeline_version=request.pipeline_version,
        config={"audit": request.audit_config.to_dict()},
        inputs=[{"kind": "stereo_session", "path": str(session_path)}],
        metadata={
            "producer": "run-item",
            "item_id": item_id,
            "source_item_id": source_item_id,
            "preview_images": {
                "status": "OMITTED",
                "reason": "audit does not decode video twice for preview generation",
            },
        },
    )
    report: dict[str, Any] | None = None
    try:
        report = audit_session(session_path, request.audit_config)
        audit_record_ids = append_audit_trace(report, writer)
        audit_status = str(report.get("status") or "FAIL")
        execution_details: Any = None
        if audit_status == "FAIL" or backends is None:
            reason = (
                "audit failed; downstream model stages were not eligible to run"
                if audit_status == "FAIL"
                else "no perception, MANO, or temporal backend bundle was configured"
            )
            execution_record_ids = _append_unproduced_stages(
                writer,
                parent_id=audit_record_ids[-1],
                reason=reason,
            )
            output_status = OUTPUT_NOT_PRODUCED
        else:
            execution = backends.execute(
                PipelineExecutionContext(
                    request=request,
                    item_id=item_id,
                    run_id=run_id,
                    audit_report=report,
                    audit_record_ids=audit_record_ids,
                ),
                writer,
            )
            if not isinstance(execution, PipelineExecutionSummary):
                raise TypeError("executor must return PipelineExecutionSummary")
            execution_record_ids = execution.record_ids
            execution_details = execution.details
            output_status = execution.output_status
        status = RunStatus.FAILED if audit_status == "FAIL" else RunStatus.COMPLETED
        writer.finalize(
            status=status,
            summary={
                "item_id": item_id,
                "audit_status": audit_status,
                "output_status": output_status,
                "audit_record_count": len(audit_record_ids),
                "execution_record_count": len(execution_record_ids),
                "execution_details": execution_details,
            },
        )
    except BaseException as exc:
        try:
            writer.finalize(
                status=RunStatus.FAILED,
                summary={
                    "item_id": item_id,
                    "audit_status": None if report is None else report.get("status"),
                    "output_status": OUTPUT_NOT_PRODUCED,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        except BaseException:
            writer.close()
        raise

    return PipelineRunResult(
        item_id=item_id,
        run_id=run_id,
        run_dir=run_dir.resolve(),
        status=status,
        output_status=output_status,
        audit_status=audit_status,
    )


__all__ = [
    "OUTPUT_NOT_PRODUCED",
    "OUTPUT_PRODUCED",
    "PipelineConfigurationError",
    "PipelineExecutionContext",
    "PipelineExecutionSummary",
    "PipelineRunRequest",
    "PipelineRunResult",
    "PipelineStageExecutor",
    "run_data_item",
]
