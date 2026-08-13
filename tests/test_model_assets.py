from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "deploy" / "mmpose-h20" / "model_assets.py"
SCHEMA_VERSION = "fisheye-handpose/model-assets/v1"
OFFICIAL_URL = "https://download.openmmlab.com/test/model.pth"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_model_assets_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(payload: bytes = b"controlled test checkpoint", **updates: Any) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "id": "test-model",
        "filename": "test-model.pth",
        "url": OFFICIAL_URL,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "config": "configs/test-model.py",
        "license_status": "REVIEW_REQUIRED",
    }
    artifact.update(updates)
    return artifact


def _write_manifest(path: Path, artifacts: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "artifacts": artifacts}),
        encoding="utf-8",
    )


def _assert_single_json_error(stdout: str, stderr: str) -> dict[str, Any]:
    assert stderr == ""
    report = json.loads(stdout)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "ERROR"
    assert isinstance(report["error"]["code"], str)
    assert isinstance(report["error"]["message"], str)
    return report


def test_argparse_failure_is_one_json_document_on_stdout() -> None:
    result = _run("fetch")

    assert result.returncode != 0
    report = _assert_single_json_error(result.stdout, result.stderr)
    assert report["error"]["code"] == "INVALID_ARGUMENTS"
    assert "output-dir" in report["error"]["message"]


def test_asset_fetch_is_license_gated_and_hash_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"controlled test checkpoint"
    source = tmp_path / "source.pth"
    source.write_bytes(payload)
    manifest = tmp_path / "assets.json"
    _write_manifest(manifest, [_artifact(payload)])
    output = tmp_path / "models"

    refused = _run("fetch", "--manifest", str(manifest), "--output-dir", str(output))
    assert refused.returncode != 0
    report = _assert_single_json_error(refused.stdout, refused.stderr)
    assert report["error"]["code"] == "LICENSE_ACKNOWLEDGEMENT_REQUIRED"
    assert "acknowledge" in report["error"]["message"].lower()
    assert not (output / "test-model.pth").exists()

    module = _load_script()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _url: source.open("rb"))
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return_code = module.main(
            [
                "fetch",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--acknowledge-license-risk",
            ]
        )

    assert return_code == 0
    assert stderr.getvalue() == ""
    fetch_report = json.loads(stdout.getvalue())
    assert fetch_report["status"] == "PASS"
    assert (output / "test-model.pth").read_bytes() == payload


@pytest.mark.parametrize(
    ("mutate", "message_fragment"),
    [
        (lambda item: item.pop("id"), "id"),
        (lambda item: item.pop("config"), "config"),
        (lambda item: item.pop("license_status"), "license_status"),
        (lambda item: item.update(url="http://download.openmmlab.com/model.pth"), "https"),
        (lambda item: item.update(url="https://evil.example/model.pth"), "host"),
        (lambda item: item.update(url="https://[broken/model.pth"), "url"),
        (lambda item: item.update(url="https://download.openmmlab.com:bad/model.pth"), "url"),
        (lambda item: item.update(id=""), "id"),
        (lambda item: item.update(config="../escape.py"), "config"),
        (lambda item: item.update(license_status="MAYBE"), "license_status"),
    ],
)
def test_manifest_rejects_missing_or_untrusted_fields_without_traceback(
    tmp_path: Path, mutate: Any, message_fragment: str
) -> None:
    artifact = _artifact()
    mutate(artifact)
    manifest = tmp_path / "bad.json"
    _write_manifest(manifest, [artifact])

    result = _run("verify", "--manifest", str(manifest), "--output-dir", str(tmp_path))

    assert result.returncode != 0
    report = _assert_single_json_error(result.stdout, result.stderr)
    assert report["error"]["code"] == "INVALID_MANIFEST"
    assert message_fragment.lower() in report["error"]["message"].lower()
    assert "traceback" not in result.stdout.lower()


def test_manifest_root_and_artifact_types_fail_as_json(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.json"
    manifest.write_text("[]", encoding="utf-8")

    result = _run("verify", "--manifest", str(manifest), "--output-dir", str(tmp_path))

    assert result.returncode != 0
    report = _assert_single_json_error(result.stdout, result.stderr)
    assert report["error"]["code"] == "INVALID_MANIFEST"


def test_network_oserror_is_one_json_document_and_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    manifest = tmp_path / "assets.json"
    _write_manifest(manifest, [_artifact(license_status="ALLOWED")])
    output = tmp_path / "models"

    def fail_download(_url: str) -> None:
        raise OSError("simulated network outage")

    monkeypatch.setattr(module.urllib.request, "urlopen", fail_download)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return_code = module.main(
            ["fetch", "--manifest", str(manifest), "--output-dir", str(output)]
        )

    assert return_code != 0
    report = _assert_single_json_error(stdout.getvalue(), stderr.getvalue())
    assert report["error"]["code"] == "IO_ERROR"
    assert "network outage" in report["error"]["message"]
    assert not (output / "test-model.pth").exists()
    assert not list(output.glob("*.download"))


def test_unexpected_downloader_failure_is_one_json_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    manifest = tmp_path / "assets.json"
    _write_manifest(manifest, [_artifact(license_status="ALLOWED")])

    def fail_download(_url: str) -> None:
        raise RuntimeError("simulated downloader bug")

    monkeypatch.setattr(module.urllib.request, "urlopen", fail_download)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return_code = module.main(
            ["fetch", "--manifest", str(manifest), "--output-dir", str(tmp_path / "models")]
        )

    assert return_code != 0
    report = _assert_single_json_error(stdout.getvalue(), stderr.getvalue())
    assert report["error"]["code"] == "INTERNAL_ERROR"
    assert "downloader bug" in report["error"]["message"]


def test_asset_verification_fails_closed_after_tampering(tmp_path: Path) -> None:
    payload = b"expected"
    model = tmp_path / "test-model.pth"
    model.write_bytes(b"tampered")
    manifest = tmp_path / "assets.json"
    _write_manifest(manifest, [_artifact(payload)])

    result = _run("verify", "--manifest", str(manifest), "--output-dir", str(tmp_path))

    assert result.returncode != 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["status"] == "FAIL"
    assert report["artifacts"][0]["status"] == "HASH_MISMATCH"
