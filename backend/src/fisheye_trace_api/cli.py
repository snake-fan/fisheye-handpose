"""Command-line launcher for the standalone trace API."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .app import DEFAULT_CORS_ORIGINS, create_app


def _directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"catalog root is not a directory: {path}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a read-only pipeline trace catalog")
    parser.add_argument("--catalog-root", required=True, type=_directory)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        help="allowed browser origin; may be repeated",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    origins = tuple(args.cors_origins) if args.cors_origins else DEFAULT_CORS_ORIGINS
    uvicorn.run(
        create_app(args.catalog_root, cors_origins=origins),
        host=args.host,
        port=args.port,
        access_log=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
