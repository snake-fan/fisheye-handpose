"""Standalone Python 3.10 H20 perception worker.

The import bridge is intentionally usable from a copied standard-library-only subset of
the package. Keep the heavy runner import lazy so core-side validation does not import
OpenMMLab composition modules.
"""

from typing import Any


def run_worker(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .runner import run_worker as implementation

    return implementation(*args, **kwargs)


__all__ = ["run_worker"]
