"""Timestamp-aware causal baseline; no fixed-frame-rate alpha is used."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .contracts import WorkerError

TEMPORAL_METHOD = "causal_time_ema_v1"


@dataclass
class _TemporalState:
    timestamp_ns: int
    input_stage: str
    points: list[list[float] | None]
    point_timestamps_ns: list[int | None]


class CausalTemporalRefiner:
    def __init__(self, *, time_constant_ms: float, gap_reset_ms: float) -> None:
        if not math.isfinite(time_constant_ms) or time_constant_ms <= 0:
            raise WorkerError("temporal time_constant_ms must be positive")
        if not math.isfinite(gap_reset_ms) or gap_reset_ms <= 0:
            raise WorkerError("temporal gap_reset_ms must be positive")
        self.time_constant_ns = float(time_constant_ms) * 1_000_000.0
        self.gap_reset_ns = int(gap_reset_ms * 1_000_000)
        self._states: dict[str, _TemporalState] = {}

    def refine(
        self,
        *,
        track_id: str,
        timestamp_ns: int,
        landmarks_xyz_m: list[list[float] | None],
        validity: list[str],
        input_stage: str = "RAW_FUSION",
    ) -> dict[str, Any]:
        if not isinstance(track_id, str) or not track_id:
            raise WorkerError("temporal track_id must be non-empty")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise WorkerError("temporal timestamp_ns must be an integer")
        if input_stage not in {"RAW_FUSION", "KINEMATIC_REFINEMENT"}:
            raise WorkerError("temporal input_stage is unsupported")
        if len(landmarks_xyz_m) != 21 or len(validity) != 21:
            raise WorkerError("temporal input must contain 21 landmarks and validity values")
        current: list[list[float] | None] = []
        for point, flag in zip(landmarks_xyz_m, validity, strict=True):
            if flag != "VALID" or not isinstance(point, list) or len(point) != 3:
                current.append(None)
                continue
            values = [float(value) for value in point]
            if not all(math.isfinite(value) for value in values):
                raise WorkerError("temporal input contains a non-finite valid point")
            current.append(values)

        previous = self._states.get(track_id)
        reset_reason: str | None = None
        if previous is None:
            reset_reason = "FIRST_OBSERVATION"
        elif input_stage != previous.input_stage:
            reset_reason = "INPUT_STAGE_CHANGED"
        elif timestamp_ns <= previous.timestamp_ns:
            reset_reason = "NON_MONOTONIC_TIMESTAMP"
        elif timestamp_ns - previous.timestamp_ns > self.gap_reset_ns:
            reset_reason = "GAP_EXCEEDED"

        if reset_reason is not None:
            output = [list(point) if point is not None else None for point in current]
            point_timestamps = [timestamp_ns if point is not None else None for point in current]
            refinement_applied = [False] * 21
            alpha: float | None = None
        else:
            assert previous is not None
            frame_dt_ns = timestamp_ns - previous.timestamp_ns
            alpha = 1.0 - math.exp(-frame_dt_ns / self.time_constant_ns)
            output = []
            refinement_applied = []
            point_timestamps = list(previous.point_timestamps_ns)
            for index, point in enumerate(current):
                if point is None:
                    output.append(None)
                    refinement_applied.append(False)
                    continue
                prior = previous.points[index]
                prior_timestamp = previous.point_timestamps_ns[index]
                if prior is None or prior_timestamp is None:
                    output.append(list(point))
                    refinement_applied.append(False)
                else:
                    point_alpha = 1.0 - math.exp(
                        -(timestamp_ns - prior_timestamp) / self.time_constant_ns
                    )
                    output.append(
                        [
                            old + point_alpha * (new - old)
                            for old, new in zip(prior, point, strict=True)
                        ]
                    )
                    refinement_applied.append(True)
                point_timestamps[index] = timestamp_ns
        stored_points = [
            (
                list(output[index])
                if output[index] is not None
                else (
                    list(previous.points[index])
                    if previous is not None
                    and reset_reason is None
                    and previous.points[index] is not None
                    else None
                )
            )
            for index in range(21)
        ]
        self._states[track_id] = _TemporalState(
            timestamp_ns, input_stage, stored_points, point_timestamps
        )
        output_validity = [
            "VALID" if point is not None else flag
            for point, flag in zip(output, validity, strict=True)
        ]
        return {
            "method": TEMPORAL_METHOD,
            "timestamp_ns": timestamp_ns,
            "alpha": alpha,
            "reset_reason": reset_reason,
            "refinement_applied": refinement_applied,
            "landmarks_xyz_m": output,
            "validity": output_validity,
            "valid_landmark_count": sum(point is not None for point in output),
        }


__all__ = ["CausalTemporalRefiner", "TEMPORAL_METHOD"]
