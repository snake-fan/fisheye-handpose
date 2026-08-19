from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_project_contract_outputs_are_current() -> None:
    result = subprocess.run(
        [sys.executable, PROJECT_ROOT / "scripts" / "generate_contracts.py", "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
