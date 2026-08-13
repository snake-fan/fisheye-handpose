"""Fail-closed checkpoint identity verification before Torch deserialization."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .contracts import ModelRequest, WorkerError

MMPOSE_COMMIT = "5408bc76f5b848cf925a0d1857899011d8c5b497"
EXPECTED_CONFIGS = {
    "rtmdet-nano-hand": "demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py",
    "rtmpose-m-hand5": (
        "configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(config: ModelRequest) -> dict[str, Any]:
    if not config.manifest.is_file():
        raise WorkerError(f"model manifest is not a file: {config.manifest}")
    if not config.model_dir.is_dir():
        raise WorkerError(f"model directory does not exist: {config.model_dir}")
    try:
        document = json.loads(config.manifest.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"cannot parse model manifest: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != (
        "fisheye-handpose/model-assets/v1"
    ):
        raise WorkerError("unexpected model asset manifest schema")
    entries = document.get("artifacts")
    if not isinstance(entries, list):
        raise WorkerError("model asset manifest artifacts must be a list")
    by_id = {
        entry.get("id"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    if len(by_id) != len(entries):
        raise WorkerError("model asset IDs must be unique strings")
    verified: dict[str, Any] = {}
    for artifact_id, expected_config in EXPECTED_CONFIGS.items():
        entry = by_id.get(artifact_id)
        if entry is None:
            raise WorkerError(f"model manifest is missing {artifact_id}")
        filename = entry.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise WorkerError(f"{artifact_id} filename must be a plain basename")
        if entry.get("config") != expected_config:
            raise WorkerError(f"{artifact_id} is bound to an unexpected config")
        license_status = entry.get("license_status")
        if license_status not in {"ALLOWED", "REVIEW_REQUIRED"}:
            raise WorkerError(f"{artifact_id} has unsupported license_status")
        if license_status == "REVIEW_REQUIRED" and not config.license_risk_acknowledged:
            raise WorkerError(f"{artifact_id} requires explicit license-risk acknowledgement")
        expected_hash = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise WorkerError(f"{artifact_id} has invalid SHA-256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise WorkerError(f"{artifact_id} has invalid byte count")
        checkpoint = (config.model_dir / filename).resolve()
        if checkpoint.parent != config.model_dir or not checkpoint.is_file():
            raise WorkerError(f"{artifact_id} checkpoint is missing")
        actual_bytes = checkpoint.stat().st_size
        if actual_bytes != expected_bytes:
            raise WorkerError(f"{artifact_id} checkpoint size mismatch")
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            raise WorkerError(f"{artifact_id} checkpoint SHA-256 mismatch")
        config_path = (config.mmpose_source / expected_config).resolve()
        try:
            config_path.relative_to(config.mmpose_source)
        except ValueError as exc:
            raise WorkerError(f"{artifact_id} config escapes MMPose source") from exc
        if not config_path.is_file():
            raise WorkerError(f"{artifact_id} config is missing: {config_path}")
        verified[artifact_id] = {
            "checkpoint": checkpoint,
            "sha256": actual_hash,
            "bytes": actual_bytes,
            "config": expected_config,
            "config_path": config_path,
            "license_status": license_status,
        }
    return {
        "manifest": config.manifest,
        "manifest_sha256": sha256_file(config.manifest),
        "artifacts": verified,
    }


def verify_source_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("commit") != MMPOSE_COMMIT:
        raise WorkerError("MMPose source verification returned the wrong commit")
    configs = report.get("configs")
    if not isinstance(configs, dict) or set(configs) != set(EXPECTED_CONFIGS.values()):
        raise WorkerError("MMPose source verification did not bind both configs")
    return report


__all__ = [
    "EXPECTED_CONFIGS",
    "MMPOSE_COMMIT",
    "sha256_file",
    "verify_assets",
    "verify_source_report",
]
