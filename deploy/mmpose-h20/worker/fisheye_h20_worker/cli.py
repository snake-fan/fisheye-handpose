"""Machine-readable CLI for one H20 perception worker result package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .runner import run_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fisheye-h20-worker")
    parser.add_argument("request", type=Path)
    parser.add_argument("result_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, runtime: Any | None = None) -> int:
    try:
        namespace = build_parser().parse_args(argv)
        report = run_worker(namespace.request, namespace.result_dir, runtime=runtime)
    except Exception as exc:
        report = {
            "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        code = 1
    else:
        code = 0
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return code


__all__ = ["main"]
