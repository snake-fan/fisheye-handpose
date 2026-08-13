from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_installed_console_script_imports_package_and_emits_schema():
    """Catch packaging failures hidden by pytest's source-tree pythonpath."""

    script = Path(sys.executable).with_name("fisheye-handpose")
    if not script.is_file():
        pytest.skip("console script is only available after installing the project")

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(script), "schema"],
        cwd=Path.home(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["version"] == "fhp21/v1"
    assert report["landmark_count"] == 21
    assert completed.stderr == ""
