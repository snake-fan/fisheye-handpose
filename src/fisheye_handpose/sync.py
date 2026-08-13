"""Strict hardware-timestamp parsing and monotonic stereo pairing."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

from .errors import SyncError, TimestampError

TimestampUnit = Literal["ns", "us", "ms"]
_UNIT_TO_NS: dict[TimestampUnit, int] = {"ns": 1, "us": 1_000, "ms": 1_000_000}


@dataclass(frozen=True, slots=True)
class TimestampSeries:
    source_path: Path
    values_ns: tuple[int, ...]
    source_unit: TimestampUnit
    column: str

    def __post_init__(self) -> None:
        if self.source_unit not in _UNIT_TO_NS:
            raise TimestampError(f"unsupported timestamp unit {self.source_unit!r}")
        if not isinstance(self.column, str) or not self.column:
            raise TimestampError("timestamp column must be a non-empty string")
        if not self.values_ns:
            raise TimestampError(f"timestamp stream is empty: {self.source_path}")
        for index, value in enumerate(self.values_ns):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise TimestampError(
                    f"timestamps must be non-negative integers at index {index}, got {value!r}"
                )
        for index in range(1, len(self.values_ns)):
            if self.values_ns[index] <= self.values_ns[index - 1]:
                raise TimestampError(
                    f"timestamps must be strictly increasing at row {index + 2}: "
                    f"{self.values_ns[index - 1]} -> {self.values_ns[index]}"
                )

    @property
    def nominal_period_ns(self) -> int | None:
        if len(self.values_ns) < 2:
            return None
        return int(
            statistics.median(
                b - a for a, b in zip(self.values_ns, self.values_ns[1:], strict=False)
            )
        )

    @property
    def intervals_ns(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.values_ns, self.values_ns[1:], strict=False))

    @property
    def minimum_interval_ns(self) -> int | None:
        intervals = self.intervals_ns
        return min(intervals) if intervals else None

    @property
    def maximum_interval_ns(self) -> int | None:
        intervals = self.intervals_ns
        return max(intervals) if intervals else None

    @property
    def gap_threshold_ns(self) -> int | None:
        period = self.nominal_period_ns
        return None if period is None else period * 3 // 2

    @property
    def gap_after_indices(self) -> tuple[int, ...]:
        threshold = self.gap_threshold_ns
        if threshold is None:
            return ()
        return tuple(
            index for index, interval in enumerate(self.intervals_ns) if interval > threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "count": len(self.values_ns),
            "source_unit": self.source_unit,
            "column": self.column,
            "first_timestamp_ns": self.values_ns[0],
            "last_timestamp_ns": self.values_ns[-1],
            "nominal_period_ns": self.nominal_period_ns,
            "minimum_interval_ns": self.minimum_interval_ns,
            "maximum_interval_ns": self.maximum_interval_ns,
            "gap_threshold_ns": self.gap_threshold_ns,
            "gap_count": len(self.gap_after_indices),
            "gap_after_indices": list(self.gap_after_indices),
        }


@dataclass(frozen=True, slots=True)
class FrameMatch:
    pair_index: int
    left_index: int
    right_index: int
    left_timestamp_ns: int
    right_timestamp_ns: int
    right_timestamp_corrected_ns: int
    pair_timestamp_ns: int
    skew_ns: int

    @property
    def right_timestamp_raw_ns(self) -> int:
        return self.right_timestamp_ns

    def to_dict(self) -> dict[str, int]:
        return {
            "pair_index": self.pair_index,
            "left_index": self.left_index,
            "right_index": self.right_index,
            "left_timestamp_ns": self.left_timestamp_ns,
            "right_timestamp_ns": self.right_timestamp_ns,
            "right_timestamp_raw_ns": self.right_timestamp_ns,
            "right_timestamp_corrected_ns": self.right_timestamp_corrected_ns,
            "pair_timestamp_ns": self.pair_timestamp_ns,
            "skew_ns": self.skew_ns,
            "corrected_skew_ns": self.skew_ns,
        }


@dataclass(frozen=True, slots=True)
class SyncResult:
    matches: tuple[FrameMatch, ...]
    unmatched_left_indices: tuple[int, ...]
    unmatched_right_indices: tuple[int, ...]
    overlap_unmatched_left_indices: tuple[int, ...]
    overlap_unmatched_right_indices: tuple[int, ...]
    left_tail_indices: tuple[int, ...]
    right_tail_indices: tuple[int, ...]
    max_skew_ns: int
    clock_offset_ns: int
    overlap_start_ns: int
    overlap_end_ns: int
    overlap_match_count: int
    left_overlap_frame_count: int
    right_overlap_frame_count: int
    left_gap_after_indices: tuple[int, ...]
    right_gap_after_indices: tuple[int, ...]

    @property
    def absolute_skews_ns(self) -> tuple[int, ...]:
        return tuple(abs(match.skew_ns) for match in self.matches)

    def to_dict(self, *, include_matches: bool = False) -> dict[str, Any]:
        skews = sorted(self.absolute_skews_ns)

        def percentile(fraction: float) -> int | None:
            if not skews:
                return None
            return skews[round((len(skews) - 1) * fraction)]

        result: dict[str, Any] = {
            "pair_count": len(self.matches),
            "overlap_pair_count": self.overlap_match_count,
            "max_skew_ns_configured": self.max_skew_ns,
            "clock_offset_ns": self.clock_offset_ns,
            "median_abs_skew_ns": percentile(0.5),
            "p95_abs_skew_ns": percentile(0.95),
            "p99_abs_skew_ns": percentile(0.99),
            "max_abs_skew_ns": max(skews) if skews else None,
            "overlap_start_ns": self.overlap_start_ns,
            "overlap_end_ns": self.overlap_end_ns,
            "overlap_duration_ns": max(0, self.overlap_end_ns - self.overlap_start_ns),
            "unmatched_left_count": len(self.unmatched_left_indices),
            "unmatched_right_count": len(self.unmatched_right_indices),
            "overlap_unmatched_left_count": len(self.overlap_unmatched_left_indices),
            "overlap_unmatched_right_count": len(self.overlap_unmatched_right_indices),
            "left_tail_count": len(self.left_tail_indices),
            "right_tail_count": len(self.right_tail_indices),
            "left_overlap_frame_count": self.left_overlap_frame_count,
            "right_overlap_frame_count": self.right_overlap_frame_count,
            "left_gap_count": len(self.left_gap_after_indices),
            "right_gap_count": len(self.right_gap_after_indices),
            "left_gap_after_indices": list(self.left_gap_after_indices),
            "right_gap_after_indices": list(self.right_gap_after_indices),
            "left_overlap_match_rate": (
                self.overlap_match_count / self.left_overlap_frame_count
                if self.left_overlap_frame_count
                else 0.0
            ),
            "right_overlap_match_rate": (
                self.overlap_match_count / self.right_overlap_frame_count
                if self.right_overlap_frame_count
                else 0.0
            ),
        }
        if include_matches:
            result["matches"] = [match.to_dict() for match in self.matches]
            result["unmatched_left_indices"] = list(self.unmatched_left_indices)
            result["unmatched_right_indices"] = list(self.unmatched_right_indices)
        return result


def read_timestamp_csv(
    path: str | Path,
    *,
    column: str = "timestamp_us",
    unit: TimestampUnit = "us",
) -> TimestampSeries:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise TimestampError(f"timestamp file does not exist: {source}")
    if unit not in _UNIT_TO_NS:
        raise TimestampError(f"unsupported timestamp unit {unit!r}")
    values: list[int] = []
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise TimestampError(
                    f"{source}: required column {column!r} is absent; fields={reader.fieldnames}"
                )
            for row_number, row in enumerate(reader, start=2):
                raw = row.get(column)
                if raw is None or not raw.strip():
                    raise TimestampError(f"{source}:{row_number}: blank timestamp")
                token = raw.strip()
                if token.startswith("+"):
                    token = token[1:]
                if not token.isdigit():
                    raise TimestampError(
                        f"{source}:{row_number}: timestamp must be an integer, got {raw!r}"
                    )
                values.append(int(token) * _UNIT_TO_NS[unit])
    except UnicodeError as exc:
        raise TimestampError(f"{source}: timestamp CSV must be UTF-8/UTF-8-SIG") from exc
    return TimestampSeries(source, tuple(values), unit, column)


def match_timestamps(
    left: TimestampSeries,
    right: TimestampSeries,
    *,
    max_skew_ns: int,
    clock_offset_ns: int = 0,
    enforce_unambiguous: bool = True,
) -> SyncResult:
    """Pair timestamps in order without reusing or shifting frames after a drop.

    ``clock_offset_ns`` is explicitly added to right timestamps for comparison. It is
    never estimated. With a tolerance below half the minimum observed local interval,
    the simple monotonic matcher has at most one candidate from either stream.
    """

    if isinstance(max_skew_ns, bool) or not isinstance(max_skew_ns, Integral) or max_skew_ns <= 0:
        raise SyncError("max_skew_ns must be positive")
    if isinstance(clock_offset_ns, bool) or not isinstance(clock_offset_ns, Integral):
        raise SyncError("clock_offset_ns must be an integer")
    local_intervals = [
        interval
        for interval in (left.minimum_interval_ns, right.minimum_interval_ns)
        if interval is not None
    ]
    if enforce_unambiguous and local_intervals and max_skew_ns * 2 >= min(local_intervals):
        raise SyncError(
            f"max_skew_ns={max_skew_ns} must be less than half the minimum local interval "
            f"({min(local_intervals)} ns)"
        )
    adjusted_right = tuple(value + clock_offset_ns for value in right.values_ns)
    overlap_start = max(left.values_ns[0], adjusted_right[0])
    overlap_end = min(left.values_ns[-1], adjusted_right[-1])
    if overlap_start > overlap_end:
        raise SyncError("left and right timestamp streams have no common time interval")

    matches: list[FrameMatch] = []
    unmatched_left: list[int] = []
    unmatched_right: list[int] = []
    i = j = 0
    while i < len(left.values_ns) and j < len(adjusted_right):
        left_ts = left.values_ns[i]
        right_ts = adjusted_right[j]
        skew = right_ts - left_ts
        if skew < -max_skew_ns:
            unmatched_right.append(j)
            j += 1
        elif skew > max_skew_ns:
            unmatched_left.append(i)
            i += 1
        else:
            matches.append(
                FrameMatch(
                    pair_index=len(matches),
                    left_index=i,
                    right_index=j,
                    left_timestamp_ns=left_ts,
                    right_timestamp_ns=right.values_ns[j],
                    right_timestamp_corrected_ns=right_ts,
                    pair_timestamp_ns=(left_ts + right_ts) // 2,
                    skew_ns=skew,
                )
            )
            i += 1
            j += 1
    unmatched_left.extend(range(i, len(left.values_ns)))
    unmatched_right.extend(range(j, len(right.values_ns)))

    def in_overlap_left(index: int) -> bool:
        return overlap_start <= left.values_ns[index] <= overlap_end

    def in_overlap_right(index: int) -> bool:
        return overlap_start <= adjusted_right[index] <= overlap_end

    if not matches:
        raise SyncError("no frame pairs satisfy the configured skew tolerance")

    left_overlap_indices = tuple(
        index for index in range(len(left.values_ns)) if in_overlap_left(index)
    )
    right_overlap_indices = tuple(
        index for index in range(len(right.values_ns)) if in_overlap_right(index)
    )
    strict_overlap_matches = tuple(
        match
        for match in matches
        if in_overlap_left(match.left_index) and in_overlap_right(match.right_index)
    )
    matched_left_in_overlap = {match.left_index for match in strict_overlap_matches}
    matched_right_in_overlap = {match.right_index for match in strict_overlap_matches}
    overlap_unmatched_left = tuple(
        index for index in left_overlap_indices if index not in matched_left_in_overlap
    )
    overlap_unmatched_right = tuple(
        index for index in right_overlap_indices if index not in matched_right_in_overlap
    )
    # Preserve the public meaning of tail indices as *unmatched* recording-window
    # tails. Strict overlap rates above deliberately exclude tolerance-edge pairs
    # whose two timestamps do not both lie inside the common interval.
    left_tails = tuple(index for index in unmatched_left if not in_overlap_left(index))
    right_tails = tuple(index for index in unmatched_right if not in_overlap_right(index))
    return SyncResult(
        matches=tuple(matches),
        unmatched_left_indices=tuple(unmatched_left),
        unmatched_right_indices=tuple(unmatched_right),
        overlap_unmatched_left_indices=overlap_unmatched_left,
        overlap_unmatched_right_indices=overlap_unmatched_right,
        left_tail_indices=left_tails,
        right_tail_indices=right_tails,
        max_skew_ns=max_skew_ns,
        clock_offset_ns=clock_offset_ns,
        overlap_start_ns=overlap_start,
        overlap_end_ns=overlap_end,
        overlap_match_count=len(strict_overlap_matches),
        left_overlap_frame_count=len(left_overlap_indices),
        right_overlap_frame_count=len(right_overlap_indices),
        left_gap_after_indices=left.gap_after_indices,
        right_gap_after_indices=right.gap_after_indices,
    )
