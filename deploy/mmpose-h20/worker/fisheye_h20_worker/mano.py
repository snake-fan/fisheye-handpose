"""Private MANO asset verification and the explicit MANO-to-FHP21 joint contract."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .contracts import WorkerError

MANO_ASSET_SCHEMA = "fisheye-handpose/mano-assets/v1"
MANO_FHP21_MAPPING_ID = "mano-v1.2-j16-tips-to-fhp21/v1"
MANO_FILES = {
    "left": Path("mano") / "MANO_LEFT.pkl",
    "right": Path("mano") / "MANO_RIGHT.pkl",
}
# smplx MANO native 16-joint order plus public MANO topology fingertip vertices.
MANO_FHP21_SOURCES: tuple[tuple[str, int], ...] = (
    ("joint", 0),
    ("joint", 13),
    ("joint", 14),
    ("joint", 15),
    ("vertex", 744),
    ("joint", 1),
    ("joint", 2),
    ("joint", 3),
    ("vertex", 320),
    ("joint", 4),
    ("joint", 5),
    ("joint", 6),
    ("vertex", 443),
    ("joint", 10),
    ("joint", 11),
    ("joint", 12),
    ("vertex", 554),
    ("joint", 7),
    ("joint", 8),
    ("joint", 9),
    ("vertex", 671),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_mano_to_fhp21(joints: Any, vertices: Any) -> list[list[float]]:
    if len(joints) < 16 or len(vertices) <= 744:
        raise WorkerError("MANO output is too small for the declared FHP21 mapping")
    mapped: list[list[float]] = []
    for source, index in MANO_FHP21_SOURCES:
        point = joints[index] if source == "joint" else vertices[index]
        values = [float(value) for value in point]
        if len(values) != 3:
            raise WorkerError("MANO mapped point must contain xyz")
        mapped.append(values)
    return mapped


def verify_mano_assets(model_root: Path, manifest_path: Path) -> dict[str, Any]:
    if not model_root.is_dir():
        raise WorkerError(f"MANO model root does not exist: {model_root}")
    if not manifest_path.is_file():
        raise WorkerError(f"MANO manifest is not a local file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"cannot parse MANO manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANO_ASSET_SCHEMA:
        raise WorkerError("unexpected MANO manifest schema_version")
    license_record = manifest.get("license")
    if not isinstance(license_record, dict) or license_record.get("acknowledged") is not True:
        raise WorkerError("MANO manifest must explicitly acknowledge the model license")
    if (
        not isinstance(license_record.get("reference"), str)
        or not license_record["reference"].strip()
    ):
        raise WorkerError("MANO manifest license.reference must be non-empty")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("acknowledged") is not True:
        raise WorkerError("MANO manifest must explicitly acknowledge artifact provenance")
    if not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
        raise WorkerError("MANO manifest provenance.source must be non-empty")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise WorkerError("MANO manifest artifacts must be a list")
    by_side: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise WorkerError("each MANO artifact entry must be an object")
        side = artifact.get("side")
        if side in by_side:
            raise WorkerError(f"duplicate MANO artifact side: {side}")
        if isinstance(side, str):
            by_side[side] = artifact
    verified: dict[str, dict[str, Any]] = {}
    for side, relative in MANO_FILES.items():
        artifact = by_side.get(side)
        if artifact is None:
            raise WorkerError(f"MANO manifest is missing the {side} artifact")
        if artifact.get("filename") != relative.as_posix():
            raise WorkerError(f"MANO {side} filename must be exactly {relative.as_posix()}")
        expected_hash = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None
        ):
            raise WorkerError(f"MANO {side} artifact has an invalid SHA-256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise WorkerError(f"MANO {side} artifact has an invalid byte count")
        path = (model_root / relative).resolve()
        try:
            path.relative_to(model_root)
        except ValueError as exc:
            raise WorkerError("MANO artifact escapes model root") from exc
        if not path.is_file():
            raise WorkerError(f"required local MANO model file is missing: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise WorkerError(
                f"MANO {side} size mismatch: expected {expected_bytes}, got {actual_bytes}"
            )
        actual_hash = _sha256(path)
        if actual_hash.lower() != expected_hash.lower():
            raise WorkerError(f"MANO {side} SHA-256 mismatch")
        verified[side] = {"path": str(path), "bytes": actual_bytes, "sha256": actual_hash}
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "license": license_record,
        "provenance": provenance,
        "artifacts": verified,
        "mapping_id": MANO_FHP21_MAPPING_ID,
    }


__all__ = [
    "MANO_FHP21_MAPPING_ID",
    "MANO_FHP21_SOURCES",
    "map_mano_to_fhp21",
    "verify_mano_assets",
]
