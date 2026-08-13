"""Standalone copy of the v1 session/timestamp wire semantics.

This module intentionally does not import the Python 3.11 core package. File names,
timestamp units, and monotonic pairing fields mirror its JSON protocol so the H20 Python
3.10 process can exchange artifacts without sharing an interpreter.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import SessionRequest, WorkerError

_PART = re.compile(
    r"^(?P<prefix>.+)_camera_(?P<side>left|right)_part(?P<part>\d+)"
    r"(?P<pts>_pts)?\.(?P<extension>mp4|csv)$"
)
_UNIT_TO_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000}


@dataclass(frozen=True)
class StereoPart:
    part_number: int
    left_video: Path
    right_video: Path
    left_timestamps: Path
    right_timestamps: Path


@dataclass(frozen=True)
class FramePair:
    part_number: int
    pair_index: int
    left_index: int
    right_index: int
    left_timestamp_ns: int
    right_timestamp_ns: int
    right_timestamp_corrected_ns: int
    clock_offset_ns: int
    pair_timestamp_ns: int
    skew_ns: int

    def to_dict(self) -> dict[str, int]:
        return {
            "part_number": self.part_number,
            "pair_index": self.pair_index,
            "left_index": self.left_index,
            "right_index": self.right_index,
            "left_timestamp_ns": self.left_timestamp_ns,
            "left_timestamp_raw_ns": self.left_timestamp_ns,
            "left_timestamp_corrected_ns": self.left_timestamp_ns,
            "right_timestamp_ns": self.right_timestamp_ns,
            "right_timestamp_raw_ns": self.right_timestamp_ns,
            "right_timestamp_corrected_ns": self.right_timestamp_corrected_ns,
            "clock_offset_ns": self.clock_offset_ns,
            "pair_timestamp_ns": self.pair_timestamp_ns,
            "skew_ns": self.skew_ns,
            "corrected_skew_ns": self.skew_ns,
        }


def discover_parts(session_dir: Path) -> tuple[StereoPart, ...]:
    if not session_dir.is_dir():
        raise WorkerError(f"session directory does not exist: {session_dir}")
    records: dict[int, dict[tuple[str, str], Path]] = {}
    prefixes: set[str] = set()
    for child in sorted(session_dir.iterdir(), key=lambda path: path.name):
        match = _PART.fullmatch(child.name)
        if match is None:
            continue
        if not child.is_file():
            raise WorkerError(f"session artifact is not a file: {child}")
        part_number = int(match.group("part"))
        if part_number < 1:
            raise WorkerError("part numbering must begin at one")
        kind = "timestamps" if match.group("pts") else "video"
        expected_extension = "csv" if kind == "timestamps" else "mp4"
        if match.group("extension") != expected_extension:
            raise WorkerError(f"session artifact extension mismatch: {child}")
        key = (match.group("side"), kind)
        if key in records.setdefault(part_number, {}):
            raise WorkerError(f"duplicate session artifact: {child}")
        records[part_number][key] = child.resolve()
        prefixes.add(match.group("prefix"))
    if not records or len(prefixes) != 1:
        raise WorkerError("session must contain one complete stereo capture prefix")
    numbers = sorted(records)
    if numbers != list(range(1, numbers[-1] + 1)):
        raise WorkerError("session part numbers must be contiguous from one")
    required = {
        ("left", "video"),
        ("right", "video"),
        ("left", "timestamps"),
        ("right", "timestamps"),
    }
    parts: list[StereoPart] = []
    for number in numbers:
        if set(records[number]) != required:
            raise WorkerError(f"session part {number} is incomplete")
        values = records[number]
        parts.append(
            StereoPart(
                number,
                values[("left", "video")],
                values[("right", "video")],
                values[("left", "timestamps")],
                values[("right", "timestamps")],
            )
        )
    return tuple(parts)


def read_timestamps(path: Path, *, column: str, unit: str) -> tuple[int, ...]:
    factor = _UNIT_TO_NS[unit]
    values: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise WorkerError(f"{path}: timestamp column {column!r} is absent")
        for row_number, row in enumerate(reader, start=2):
            token = str(row.get(column, "")).strip()
            if token.startswith("+"):
                token = token[1:]
            if not token.isdigit():
                raise WorkerError(f"{path}:{row_number}: timestamp must be an integer")
            values.append(int(token) * factor)
    if not values:
        raise WorkerError(f"timestamp stream is empty: {path}")
    if any(current <= previous for previous, current in zip(values, values[1:], strict=False)):
        raise WorkerError(f"timestamp stream is not strictly increasing: {path}")
    return tuple(values)


def match_part(part: StereoPart, request: SessionRequest) -> tuple[FramePair, ...]:
    left = read_timestamps(
        part.left_timestamps,
        column=request.timestamp_column,
        unit=request.timestamp_unit,
    )
    right = read_timestamps(
        part.right_timestamps,
        column=request.timestamp_column,
        unit=request.timestamp_unit,
    )
    local_intervals = [
        current - previous
        for values in (left, right)
        for previous, current in zip(values, values[1:], strict=False)
    ]
    if local_intervals and request.max_skew_ns * 2 >= min(local_intervals):
        raise WorkerError(
            "max timestamp skew must be less than half the minimum local frame interval"
        )
    matches: list[FramePair] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        right_corrected_ns = right[right_index] + request.clock_offset_ns
        skew = right_corrected_ns - left[left_index]
        if skew < -request.max_skew_ns:
            right_index += 1
        elif skew > request.max_skew_ns:
            left_index += 1
        else:
            matches.append(
                FramePair(
                    part_number=part.part_number,
                    pair_index=len(matches),
                    left_index=left_index,
                    right_index=right_index,
                    left_timestamp_ns=left[left_index],
                    right_timestamp_ns=right[right_index],
                    right_timestamp_corrected_ns=right_corrected_ns,
                    clock_offset_ns=request.clock_offset_ns,
                    pair_timestamp_ns=(left[left_index] + right_corrected_ns) // 2,
                    skew_ns=skew,
                )
            )
            left_index += 1
            right_index += 1
    if not matches:
        raise WorkerError(f"part {part.part_number}: no timestamps satisfy max skew")
    return tuple(matches)


def selected_frames(
    runtime: Any,
    video: Path,
    indices: Iterable[int],
) -> dict[int, Any]:
    requested = set(indices)
    if not requested:
        return {}
    found: dict[int, Any] = {}
    maximum = max(requested)
    for index, frame in enumerate(runtime.iter_video_frames(video)):
        if index in requested:
            found[index] = frame
        if index >= maximum:
            break
    missing = sorted(requested - found.keys())
    if missing:
        raise WorkerError(f"video ended before presentation-order frame indices {missing}: {video}")
    return found


__all__ = [
    "FramePair",
    "StereoPart",
    "discover_parts",
    "match_part",
    "read_timestamps",
    "selected_frames",
]
