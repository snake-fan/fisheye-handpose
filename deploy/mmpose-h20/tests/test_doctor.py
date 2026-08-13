from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
DOCTOR = DEPLOY_ROOT / "doctor.py"


def run_doctor(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DOCTOR), *args],
        cwd=DEPLOY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


class DoctorContractTests(unittest.TestCase):
    def test_manifest_mode_validates_the_pinned_environment_on_any_host(self) -> None:
        completed = run_doctor("--mode", "manifest")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema_version"], "fisheye-handpose/doctor/v1")
        self.assertEqual(report["mode"], "manifest")
        self.assertTrue(report["ok"])
        self.assertEqual(report["target"]["platform"], "linux-x86_64")
        self.assertEqual(report["target"]["python"], "3.10")
        self.assertEqual(report["target"]["cuda"], "12.1")
        self.assertEqual(report["target"]["compute_capability"], "9.0")
        resolution_check = next(
            check for check in report["checks"] if check["name"] == "manifest.resolution"
        )
        self.assertEqual(
            resolution_check["actual"]["exclude_newer"],
            "2024-08-01T00:00:00Z",
        )
        package_check = next(
            check for check in report["checks"] if check["name"] == "manifest.packages"
        )
        self.assertEqual(package_check["actual"]["chumpy"], "0.71")
        source_check = next(
            check for check in report["checks"] if check["name"] == "manifest.binary_sources"
        )
        self.assertIn("2816a138d2f60bc8a77eddb9962c4c825179cb56", source_check["actual"]["chumpy"])
        self.assertTrue(report["checks"])
        self.assertTrue(all(check["ok"] for check in report["checks"]))

    def test_manifest_mode_fails_closed_for_a_missing_manifest(self) -> None:
        completed = run_doctor(
            "--mode",
            "manifest",
            "--manifest",
            str(DEPLOY_ROOT / "missing-environment.json"),
        )

        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertFalse(report["ok"])
        self.assertEqual(report["mode"], "manifest")
        self.assertEqual(report["checks"][0]["name"], "manifest.load")
        self.assertFalse(report["checks"][0]["ok"])

    def test_unknown_mode_is_reported_as_json_and_fails_closed(self) -> None:
        completed = run_doctor("--mode", "unknown")

        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertFalse(report["ok"])
        self.assertEqual(report["checks"][0]["name"], "arguments.mode")


if __name__ == "__main__":
    unittest.main()
