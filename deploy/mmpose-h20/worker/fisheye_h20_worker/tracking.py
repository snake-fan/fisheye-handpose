"""Small deterministic sequence-local tracker for stereo 3D observations."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

from .contracts import WorkerError

PALM_ANCHOR_INDICES = (0, 5, 9, 13, 17)
PALM_ANCHOR_METHOD_ID = "fhp21_palm_coordinate_median_v1"
MOTION_METHOD_ID = "constant_velocity_metric_v1"


def _anchor(
    observation: dict[str, Any],
) -> tuple[tuple[float, float, float] | None, int]:
    points = observation.get("landmarks_xyz_m")
    validity = observation.get("validity")
    if not isinstance(points, list) or len(points) != 21:
        raise WorkerError("tracking observation must contain 21 landmarks_xyz_m")
    if not isinstance(validity, list) or len(validity) != 21:
        raise WorkerError("tracking observation must contain 21 validity values")
    valid: list[tuple[float, float, float]] = []
    for index in PALM_ANCHOR_INDICES:
        point = points[index]
        flag = validity[index]
        if flag != "VALID" or not isinstance(point, list) or len(point) != 3:
            continue
        values = tuple(float(value) for value in point)
        if all(math.isfinite(value) for value in values):
            valid.append(values)
    if not valid:
        return None, 0
    center = tuple(statistics.median(point[axis] for point in valid) for axis in range(3))
    return center, len(valid)


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


@dataclass
class _Track:
    track_id: str
    anchor: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    timestamp_ns: int
    missed_updates: int = 0

    def predict(self, timestamp_ns: int) -> tuple[float, float, float]:
        delta_seconds = (timestamp_ns - self.timestamp_ns) / 1_000_000_000.0
        return tuple(
            position + velocity * delta_seconds
            for position, velocity in zip(self.anchor, self.velocity_mps, strict=True)
        )


class SequenceTracker:
    """Assign at most two hands per frame with max-cardinality/min-distance matching."""

    def __init__(self, *, max_root_distance_m: float, max_gap_ms: float) -> None:
        if not math.isfinite(max_root_distance_m) or max_root_distance_m <= 0:
            raise WorkerError("tracking max_root_distance_m must be positive")
        if not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
            raise WorkerError("tracking max_gap_ms must be positive")
        self.max_root_distance_m = float(max_root_distance_m)
        self.max_gap_ns = int(max_gap_ms * 1_000_000)
        self._next_id = 0
        self._tracks: dict[str, _Track] = {}
        self._last_timestamp_ns: int | None = None

    def _new(self, anchor: tuple[float, float, float] | None, timestamp_ns: int) -> str:
        track_id = f"track-{self._next_id:04d}"
        self._next_id += 1
        if anchor is not None:
            self._tracks[track_id] = _Track(track_id, anchor, (0.0, 0.0, 0.0), timestamp_ns)
        return track_id

    def assign(
        self,
        observations: list[dict[str, Any]],
        *,
        timestamp_ns: int,
    ) -> list[dict[str, Any]]:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise WorkerError("tracking timestamp_ns must be an integer")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise WorkerError("tracking timestamps must be monotonically non-decreasing")
        if len(observations) > 2:
            raise WorkerError("tracking supports at most two observations per frame")
        observation_ids: list[str] = []
        for observation in observations:
            observation_id = observation.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise WorkerError("tracking observation_id must be non-empty")
            observation_ids.append(observation_id)
        anchor_results = [_anchor(observation) for observation in observations]
        anchors = [value[0] for value in anchor_results]
        active = [
            track
            for track in sorted(self._tracks.values(), key=lambda value: value.track_id)
            if 0 <= timestamp_ns - track.timestamp_ns <= self.max_gap_ns
        ]
        self._tracks = {track.track_id: track for track in active}
        predictions = [track.predict(timestamp_ns) for track in active]
        candidates: list[tuple[int, int, float]] = []
        for observation_index, anchor in enumerate(anchors):
            if anchor is None:
                continue
            for track_index, prediction in enumerate(predictions):
                distance = _distance(anchor, prediction)
                if distance <= self.max_root_distance_m:
                    candidates.append((observation_index, track_index, distance))

        assignments: list[tuple[tuple[int, int, float], ...]] = [()]
        for size in range(1, min(len(observations), len(active)) + 1):
            for observation_indices in combinations(range(len(observations)), size):
                for track_indices in permutations(range(len(active)), size):
                    selected: list[tuple[int, int, float]] = []
                    for observation_index, track_index in zip(
                        observation_indices, track_indices, strict=True
                    ):
                        candidate = next(
                            (
                                value
                                for value in candidates
                                if value[0] == observation_index and value[1] == track_index
                            ),
                            None,
                        )
                        if candidate is None:
                            break
                        selected.append(candidate)
                    if len(selected) == size:
                        assignments.append(tuple(selected))
        selected = min(
            assignments,
            key=lambda values: (
                -len(values),
                sum(value[2] for value in values),
                tuple((value[0], active[value[1]].track_id) for value in values),
            ),
        )
        matched = {
            observation_index: (track_index, distance)
            for observation_index, track_index, distance in selected
        }
        matched_track_ids: set[str] = set()
        results: list[dict[str, Any]] = []
        for index, (observation_id, anchor) in enumerate(
            zip(observation_ids, anchors, strict=True)
        ):
            if index in matched:
                track_index, distance = matched[index]
                previous = active[track_index]
                track_id = previous.track_id
                matched_track_ids.add(track_id)
                decision = "MATCHED"
                recovered = previous.missed_updates > 0
                decision_reason = "RECOVERED_WITHIN_TTL" if recovered else "PREDICTION_GATE_MATCH"
                delta_ms: float | None = (timestamp_ns - previous.timestamp_ns) / 1_000_000.0
                predicted_anchor = predictions[track_index]
                velocity = previous.velocity_mps
                if anchor is not None:
                    delta_seconds = (timestamp_ns - previous.timestamp_ns) / 1_000_000_000.0
                    if delta_seconds > 0.0:
                        velocity = tuple(
                            (current - prior) / delta_seconds
                            for current, prior in zip(anchor, previous.anchor, strict=True)
                        )
                    self._tracks[track_id] = _Track(track_id, anchor, velocity, timestamp_ns)
            else:
                track_id = self._new(anchor, timestamp_ns)
                distance = None
                delta_ms = None
                decision = "NEW"
                recovered = False
                decision_reason = "NEW_TRACK" if anchor is not None else "INSUFFICIENT_PALM_SUPPORT"
                predicted_anchor = None
                velocity = (0.0, 0.0, 0.0) if anchor is not None else None
            results.append(
                {
                    "observation_id": observation_id,
                    "track_id": track_id,
                    "decision": decision,
                    "decision_reason": decision_reason,
                    "recovered": recovered,
                    "distance_m": distance,
                    "delta_ms": delta_ms,
                    "anchor_method": PALM_ANCHOR_METHOD_ID,
                    "anchor_indices": list(PALM_ANCHOR_INDICES),
                    "anchor_support": anchor_results[index][1],
                    "anchor_xyz_m": list(anchor) if anchor is not None else None,
                    "trackable": anchor is not None,
                    "motion_method": MOTION_METHOD_ID,
                    "predicted_anchor_xyz_m": (
                        list(predicted_anchor) if predicted_anchor is not None else None
                    ),
                    "velocity_xyz_mps": list(velocity) if velocity is not None else None,
                }
            )
        for track in active:
            if track.track_id not in matched_track_ids:
                self._tracks[track.track_id] = _Track(
                    track.track_id,
                    track.anchor,
                    track.velocity_mps,
                    track.timestamp_ns,
                    track.missed_updates + 1,
                )
        self._last_timestamp_ns = timestamp_ns
        return results


__all__ = [
    "MOTION_METHOD_ID",
    "PALM_ANCHOR_INDICES",
    "PALM_ANCHOR_METHOD_ID",
    "SequenceTracker",
]
