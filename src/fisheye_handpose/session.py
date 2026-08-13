"""Deterministic discovery of complete stereo capture sessions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DiscoveryError

_PART_PATTERN = re.compile(
    r"^(?P<prefix>.+)_camera_(?P<side>left|right)_part(?P<part>\d+)"
    r"(?P<pts>_pts)?\.(?P<extension>mp4|csv)$"
)


@dataclass(frozen=True, slots=True)
class StereoPartSpec:
    part_number: int
    left_video: Path
    right_video: Path
    left_timestamps: Path
    right_timestamps: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_number": self.part_number,
            "left_video": str(self.left_video),
            "right_video": str(self.right_video),
            "left_timestamps": str(self.left_timestamps),
            "right_timestamps": str(self.right_timestamps),
        }


@dataclass(frozen=True, slots=True)
class StereoSessionSpec:
    session_id: str
    session_dir: Path
    calibration_path: Path
    parts: tuple[StereoPartSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "calibration_path": str(self.calibration_path),
            "parts": [part.to_dict() for part in self.parts],
        }


def discover_session(path: str | Path) -> StereoSessionSpec:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise DiscoveryError(f"session directory does not exist: {directory}")
    calibration_files = sorted(
        candidate
        for candidate in directory.glob("*_calibration_camera.yaml")
        if candidate.is_file()
    )
    if len(calibration_files) != 1:
        raise DiscoveryError(
            f"{directory}: expected exactly one *_calibration_camera.yaml, "
            f"found {len(calibration_files)}"
        )

    records: dict[int, dict[tuple[str, str], Path]] = {}
    prefixes: set[str] = set()
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        match = _PART_PATTERN.match(child.name)
        if not match:
            continue
        if not child.is_file():
            raise DiscoveryError(f"{child}: matched a session artifact name but is not a file")
        side = match.group("side")
        part = int(match.group("part"))
        if part < 1:
            raise DiscoveryError(f"{child}: part number must start at 1")
        kind = "timestamps" if match.group("pts") else "video"
        expected_extension = "csv" if kind == "timestamps" else "mp4"
        if match.group("extension") != expected_extension:
            raise DiscoveryError(f"{child}: extension does not match its role")
        key = (side, kind)
        if key in records.setdefault(part, {}):
            raise DiscoveryError(f"duplicate {side} {kind} for part {part}: {child}")
        records[part][key] = child.resolve()
        prefixes.add(match.group("prefix"))
    if not records:
        raise DiscoveryError(f"{directory}: no stereo video parts found")
    if len(prefixes) != 1:
        raise DiscoveryError(
            f"{directory}: files contain multiple session prefixes: {sorted(prefixes)}"
        )
    prefix = next(iter(prefixes))
    expected_calibration_name = f"{prefix}_calibration_camera.yaml"
    if calibration_files[0].name != expected_calibration_name:
        raise DiscoveryError(
            f"{directory}: calibration prefix does not match the video prefix; expected "
            f"{expected_calibration_name!r}, found {calibration_files[0].name!r}"
        )

    part_numbers = sorted(records)
    expected_part_numbers = list(range(1, part_numbers[-1] + 1))
    if part_numbers != expected_part_numbers:
        missing_parts = sorted(set(expected_part_numbers) - set(part_numbers))
        raise DiscoveryError(
            f"{directory}: part numbers must be contiguous from 1; "
            f"found={part_numbers}, missing={missing_parts}"
        )

    required = {
        ("left", "video"),
        ("right", "video"),
        ("left", "timestamps"),
        ("right", "timestamps"),
    }
    parts: list[StereoPartSpec] = []
    for number in part_numbers:
        missing = required - records[number].keys()
        if missing:
            raise DiscoveryError(
                f"{directory}: part {number} is incomplete; missing={sorted(missing)}"
            )
        values = records[number]
        parts.append(
            StereoPartSpec(
                part_number=number,
                left_video=values[("left", "video")],
                right_video=values[("right", "video")],
                left_timestamps=values[("left", "timestamps")],
                right_timestamps=values[("right", "timestamps")],
            )
        )
    return StereoSessionSpec(
        directory.name,
        directory,
        calibration_files[0].resolve(),
        tuple(parts),
    )


def discover_sessions(root: str | Path) -> tuple[StereoSessionSpec, ...]:
    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise DiscoveryError(f"root directory does not exist: {directory}")
    candidates = (
        [directory]
        if any(path.is_file() for path in directory.glob("*_calibration_camera.yaml"))
        else []
    )
    candidates.extend(
        child
        for child in sorted(directory.iterdir(), key=lambda item: item.name)
        if child.is_dir()
        and any(path.is_file() for path in child.glob("*_calibration_camera.yaml"))
    )
    if not candidates:
        raise DiscoveryError(f"{directory}: no candidate session directories found")
    sessions = [discover_session(candidate) for candidate in candidates]
    if len({session.session_id for session in sessions}) != len(sessions):
        raise DiscoveryError("discovered session IDs are not unique")
    return tuple(sessions)
