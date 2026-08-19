#!/usr/bin/env python3
"""Fetch and verify allowlisted OpenMMLab model artifacts.

This script never deserializes checkpoints.  It exists to keep network acquisition,
license acknowledgement, and cryptographic verification ahead of ``torch.load``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from runpy import run_path
from typing import Any
from urllib.parse import urlsplit

DEPLOY_ROOT = Path(__file__).resolve().parent
_PROJECT_CONTRACT = run_path(str(DEPLOY_ROOT / "_generated_project_contract.py"))
SCHEMA_VERSION = str(_PROJECT_CONTRACT["MODEL_ASSETS_SCHEMA"])
DEFAULT_MANIFEST = DEPLOY_ROOT / "model-assets.json"
ALLOWED_DOWNLOAD_HOST = "download.openmmlab.com"
LICENSE_STATUSES = frozenset({"ALLOWED", "REVIEW_REQUIRED"})


class AssetError(RuntimeError):
    """Raised when the model-asset contract is not satisfied."""

    def __init__(self, message: str, *, code: str = "ASSET_ERROR") -> None:
        super().__init__(message)
        self.code = code


class JsonArgumentParser(argparse.ArgumentParser):
    """Turn argparse failures into the same machine-readable error path."""

    def error(self, message: str) -> None:
        raise AssetError(message, code="INVALID_ARGUMENTS")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AssetError(
            f"cannot parse model manifest {path}: {exc}",
            code="INVALID_MANIFEST",
        ) from exc
    if not isinstance(payload, dict):
        raise AssetError("manifest root must be a JSON object", code="INVALID_MANIFEST")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssetError(
            f"manifest schema_version must be {SCHEMA_VERSION!r}",
            code="INVALID_MANIFEST",
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AssetError(
            "manifest must contain a non-empty artifacts list",
            code="INVALID_MANIFEST",
        )
    seen_filenames: set[str] = set()
    seen_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        label = f"artifact[{index}]"
        if not isinstance(artifact, dict):
            raise AssetError(f"{label} must be an object", code="INVALID_MANIFEST")

        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise AssetError(
                f"{label} id must be a non-empty string",
                code="INVALID_MANIFEST",
            )
        if artifact_id in seen_ids:
            raise AssetError(
                f"duplicate artifact id: {artifact_id}",
                code="INVALID_MANIFEST",
            )
        seen_ids.add(artifact_id)

        filename = artifact.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            raise AssetError(
                f"{label} filename must be a plain basename",
                code="INVALID_MANIFEST",
            )
        if filename in seen_filenames:
            raise AssetError(
                f"duplicate artifact filename: {filename}",
                code="INVALID_MANIFEST",
            )
        seen_filenames.add(filename)

        config = artifact.get("config")
        if not isinstance(config, str) or not config.strip():
            raise AssetError(
                f"{label} config must be a non-empty relative path",
                code="INVALID_MANIFEST",
            )
        config_path = PurePosixPath(config)
        if (
            config_path.is_absolute()
            or any(part in {"", ".", ".."} for part in config_path.parts)
            or "\\" in config
            or config_path.suffix != ".py"
        ):
            raise AssetError(
                f"{label} config must be a safe relative .py path",
                code="INVALID_MANIFEST",
            )

        license_status = artifact.get("license_status")
        if license_status not in LICENSE_STATUSES:
            allowed = ", ".join(sorted(LICENSE_STATUSES))
            raise AssetError(
                f"{label} license_status must be one of: {allowed}",
                code="INVALID_MANIFEST",
            )

        sha256 = artifact.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise AssetError(
                f"artifact {filename} has no full SHA-256",
                code="INVALID_MANIFEST",
            )
        try:
            int(sha256, 16)
        except ValueError as exc:
            raise AssetError(
                f"artifact {filename} has invalid SHA-256",
                code="INVALID_MANIFEST",
            ) from exc
        expected_bytes = artifact.get("bytes")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
            raise AssetError(
                f"artifact {filename} has invalid byte count",
                code="INVALID_MANIFEST",
            )
        if expected_bytes <= 0:
            raise AssetError(
                f"artifact {filename} has invalid byte count",
                code="INVALID_MANIFEST",
            )

        url = artifact.get("url")
        if not isinstance(url, str):
            raise AssetError(
                f"artifact {filename} has no URL",
                code="INVALID_MANIFEST",
            )
        try:
            parsed_url = urlsplit(url)
            hostname = parsed_url.hostname
            port = parsed_url.port
        except ValueError as exc:
            raise AssetError(
                f"artifact {filename} has an invalid URL: {exc}",
                code="INVALID_MANIFEST",
            ) from exc
        if parsed_url.scheme != "https":
            raise AssetError(
                f"artifact {filename} URL must use https",
                code="INVALID_MANIFEST",
            )
        if hostname != ALLOWED_DOWNLOAD_HOST:
            raise AssetError(
                f"artifact {filename} URL host must be {ALLOWED_DOWNLOAD_HOST}",
                code="INVALID_MANIFEST",
            )
        if parsed_url.username is not None or parsed_url.password is not None:
            raise AssetError(
                f"artifact {filename} URL must not include credentials",
                code="INVALID_MANIFEST",
            )
        if port not in {None, 443}:
            raise AssetError(
                f"artifact {filename} URL must use the default HTTPS port",
                code="INVALID_MANIFEST",
            )
        if not parsed_url.path or parsed_url.path.endswith("/"):
            raise AssetError(
                f"artifact {filename} URL must identify a file",
                code="INVALID_MANIFEST",
            )
    return payload


def _inspect(artifact: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    path = output_dir / artifact["filename"]
    report = {
        "id": artifact.get("id"),
        "filename": artifact["filename"],
        "path": str(path.resolve()),
        "expected_sha256": artifact["sha256"],
        "expected_bytes": artifact["bytes"],
        "license_status": artifact.get("license_status", "UNKNOWN"),
    }
    if not path.is_file():
        return {**report, "status": "MISSING"}
    actual_bytes = path.stat().st_size
    actual_sha256 = _sha256(path)
    report.update(actual_bytes=actual_bytes, actual_sha256=actual_sha256)
    if actual_bytes != artifact["bytes"]:
        return {**report, "status": "SIZE_MISMATCH"}
    if actual_sha256 != artifact["sha256"]:
        return {**report, "status": "HASH_MISMATCH"}
    return {**report, "status": "PASS"}


def verify(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    reports = [_inspect(artifact, output_dir) for artifact in manifest["artifacts"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "artifacts": reports,
    }


def _fetch_one(artifact: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / artifact["filename"]
    current = _inspect(artifact, output_dir)
    if current["status"] == "PASS":
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{destination.name}.",
            suffix=".download",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(artifact["url"]) as response:  # noqa: S310
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        downloaded = {
            "bytes": temporary.stat().st_size,
            "sha256": _sha256(temporary),
        }
        if downloaded["bytes"] != artifact["bytes"]:
            raise AssetError(
                f"downloaded size mismatch for {artifact['filename']}: {downloaded['bytes']}"
            )
        if downloaded["sha256"] != artifact["sha256"]:
            raise AssetError(f"downloaded SHA-256 mismatch for {artifact['filename']}")
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "fetch"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "fetch":
            command.add_argument("--acknowledge-license-risk", action="store_true")
    return parser


def _error_report(code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ERROR",
        "error": {"code": code, "message": message},
    }


def _write_report(report: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        manifest = _load_manifest(args.manifest.resolve())
        if args.command == "fetch":
            review_required = [
                artifact["id"]
                for artifact in manifest["artifacts"]
                if artifact.get("license_status") != "ALLOWED"
            ]
            if review_required and not args.acknowledge_license_risk:
                raise AssetError(
                    "license review is required; pass --acknowledge-license-risk only after "
                    f"approving intended use for: {', '.join(review_required)}",
                    code="LICENSE_ACKNOWLEDGEMENT_REQUIRED",
                )
            for artifact in manifest["artifacts"]:
                _fetch_one(artifact, args.output_dir.resolve())
        report = verify(manifest, args.output_dir.resolve())
        _write_report(report)
        return 0 if report["status"] == "PASS" else 2
    except AssetError as exc:
        _write_report(_error_report(exc.code, str(exc)))
        return 2
    except OSError as exc:
        _write_report(_error_report("IO_ERROR", str(exc)))
        return 2
    except Exception as exc:  # Keep every CLI failure machine-readable and fail closed.
        message = f"{type(exc).__name__}: {exc}"
        _write_report(_error_report("INTERNAL_ERROR", message))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
