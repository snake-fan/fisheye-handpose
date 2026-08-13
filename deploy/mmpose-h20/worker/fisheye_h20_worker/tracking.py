"""Small deterministic sequence-local tracker for stereo 3D observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

from .contracts import WorkerError


def _anchor(observation: dict[str, Any]) -> tuple[float, float, float] | None:
    points = observation.get("landmarks_xyz_m")
    validity = observation.get("validity")
    if not isinstance(points, list) or len(points) != 21:
        raise WorkerError("tracking observation must contain 21 landmarks_xyz_m")
    if not isinstance(validity, list) or len(validity) != 21:
        raise WorkerError("tracking observation must contain 21 validity values")
    valid: list[tuple[float, float, float]] = []
    for point, flag in zip(points, validity, strict=True):
        if flag != "VALID" or not isinstance(point, list) or len(point) != 3:
            continue
        values = tuple(float(value) for value in point)
        if all(math.isfinite(value) for value in values):
            valid.append(values)
    if validity[0] == "VALID" and isinstance(points[0], list) and len(points[0]) == 3:
        wrist = tuple(float(value) for value in points[0])
        if all(math.isfinite(value) for value in wrist):
            return wrist
    if not valid:
        return None
    count = float(len(valid))
    return tuple(sum(point[axis] for point in valid) / count for axis in range(3))


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


@dataclass
class _Track:
    track_id: str
    anchor: tuple[float, float, float]
    timestamp_ns: int


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

    def _new(self, anchor: tuple[float, float, float] | None, timestamp_ns: int) -> str:
        track_id = f"track-{self._next_id:04d}"
        self._next_id += 1
        if anchor is not None:
            self._tracks[track_id] = _Track(track_id, anchor, timestamp_ns)
        return track_id

    def assign(
        self,
        observations: list[dict[str, Any]],
        *,
        timestamp_ns: int,
    ) -> list[dict[str, Any]]:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise WorkerError("tracking timestamp_ns must be an integer")
        if len(observations) > 2:
            raise WorkerError("tracking supports at most two observations per frame")
        anchors = [_anchor(observation) for observation in observations]
        active = [
            track
            for track in sorted(self._tracks.values(), key=lambda value: value.track_id)
            if 0 <= timestamp_ns - track.timestamp_ns <= self.max_gap_ns
        ]
        self._tracks = {track.track_id: track for track in active}
        candidates: list[tuple[int, int, float]] = []
        for observation_index, anchor in enumerate(anchors):
            if anchor is None:
                continue
            for track_index, track in enumerate(active):
                distance = _distance(anchor, track.anchor)
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
        results: list[dict[str, Any]] = []
        for index, (observation, anchor) in enumerate(zip(observations, anchors, strict=True)):
            observation_id = observation.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise WorkerError("tracking observation_id must be non-empty")
            if index in matched:
                track_index, distance = matched[index]
                previous = active[track_index]
                track_id = previous.track_id
                decision = "MATCHED"
                delta_ms: float | None = (timestamp_ns - previous.timestamp_ns) / 1_000_000.0
                if anchor is not None:
                    self._tracks[track_id] = _Track(track_id, anchor, timestamp_ns)
            else:
                track_id = self._new(anchor, timestamp_ns)
                distance = None
                delta_ms = None
                decision = "NEW"
            results.append(
                {
                    "observation_id": observation_id,
                    "track_id": track_id,
                    "decision": decision,
                    "distance_m": distance,
                    "delta_ms": delta_ms,
                    "anchor_method": "WRIST_ELSE_VALID_CENTROID",
                    "anchor_xyz_m": list(anchor) if anchor is not None else None,
                }
            )
        return results


__all__ = ["SequenceTracker"]
