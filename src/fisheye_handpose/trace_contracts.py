"""JSON-safe serialization for versioned hand-pose trace contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .contracts import PoseEstimate, SpatialObservation, Validity


class TraceSerializationError(ValueError):
    """Raised when a typed contract cannot be represented without losing semantics."""


def _array_value(value: Any) -> Any:
    """Move backend arrays to ordinary nested values without importing their libraries."""

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        return tolist()
    return candidate


def _json_value(value: Any, *, allow_nonfinite: bool = False) -> Any:
    value = _array_value(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number):
            return number
        if allow_nonfinite:
            return None
        raise TraceSerializationError("non-finite numeric value is not JSON-safe")
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name), allow_nonfinite=allow_nonfinite)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, allow_nonfinite=allow_nonfinite)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, allow_nonfinite=allow_nonfinite) for item in value]
    raise TraceSerializationError(f"unsupported trace payload value: {type(value).__name__}")


def _landmark_values(value: Any, validity: Sequence[Validity], field_name: str) -> list[Any]:
    value = _array_value(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TraceSerializationError(f"{field_name} must be a landmark sequence")
    if len(value) != len(validity):
        raise TraceSerializationError(f"{field_name} and validity counts disagree")
    result: list[Any] = []
    for index, (item, state) in enumerate(zip(value, validity, strict=True)):
        try:
            result.append(_json_value(item, allow_nonfinite=state is Validity.INVALID))
        except TraceSerializationError as exc:
            raise TraceSerializationError(
                f"valid landmark {index} has invalid {field_name}: {exc}"
            ) from exc
    return result


def contract_to_trace_payload(value: SpatialObservation | PoseEstimate) -> dict[str, Any]:
    """Serialize one raw or refined ``fhp21/v1`` record as standards-compliant JSON.

    Invalid landmarks may use non-finite internal sentinels; those values become explicit
    JSON ``null``. A valid landmark containing a non-finite coordinate, covariance, or
    probability is rejected instead of silently degrading the measurement.
    """

    if not isinstance(value, (SpatialObservation, PoseEstimate)):
        raise TypeError("trace contract must be SpatialObservation or PoseEstimate")

    payload: dict[str, Any] = {"record_type": type(value).__name__}
    special = {
        "landmarks_xyz_m",
        "covariance_m2",
        "visibility_probability",
        "confidence_probability",
    }
    for field in fields(value):
        item = getattr(value, field.name)
        if field.name in special and item is not None:
            payload[field.name] = _landmark_values(item, value.validity, field.name)
        else:
            payload[field.name] = _json_value(item)
    if isinstance(value, SpatialObservation):
        payload["stage"] = value.stage.value
    return payload


__all__ = ["TraceSerializationError", "contract_to_trace_payload"]
