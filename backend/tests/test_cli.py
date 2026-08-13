from __future__ import annotations

from pathlib import Path

import pytest

from fisheye_trace_api.cli import parse_args


def test_cli_defaults_to_loopback_and_accepts_explicit_catalog_root(tmp_path: Path) -> None:
    args = parse_args(["--catalog-root", str(tmp_path)])

    assert args.catalog_root == tmp_path.resolve()
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_cli_rejects_missing_catalog_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--catalog-root", str(tmp_path / "missing")])
