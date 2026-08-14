"""Streaming rectified raw-versus-stable diagnostic video rendering."""

from __future__ import annotations

import math
import re
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from .contracts import WorkerError

TIMELINE_SCHEMA = "fisheye-handpose/overlay-video-timeline/v1"
VIDEO_SCHEMA = "fisheye-handpose/overlay-video/v1"

_FHP21_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

# RGB metadata values. OpenCV receives the reversed BGR tuple while rendering.
_TRACK_PALETTE = (
    (117, 246, 196),
    (255, 180, 84),
    (123, 166, 255),
    (239, 134, 184),
    (196, 240, 106),
)
_NUMBERED_TRACK_ID = re.compile(r"^track-([0-9]+)$")


def track_color_rgb(track_id: str) -> tuple[int, int, int]:
    """Return the RGB color shared with React's StageComparison."""

    match = _NUMBERED_TRACK_ID.fullmatch(track_id)
    if match is not None:
        palette_index = int(match.group(1)) % len(_TRACK_PALETTE)
    else:
        hash_value = 0
        for character in track_id:
            hash_value = (hash_value * 31 + ord(character)) & 0xFFFFFFFF
        palette_index = hash_value % len(_TRACK_PALETTE)
    return _TRACK_PALETTE[palette_index]


def _frame_rate(timestamps_ns: list[int]) -> Fraction:
    deltas = sorted(
        current - previous for previous, current in pairwise(timestamps_ns) if current > previous
    )
    if not deltas:
        return Fraction(30, 1)
    median_delta_ns = deltas[len(deltas) // 2]
    rate = Fraction(1_000_000_000, median_delta_ns).limit_denominator(1001)
    if not 1 <= float(rate) <= 120:
        raise WorkerError(f"overlay video frame rate is implausible: {float(rate):.6f} fps")
    return rate


def _ffmpeg_details(executable: Path) -> dict[str, str]:
    if not executable.is_file():
        raise WorkerError(f"overlay video ffmpeg is missing: {executable}")
    try:
        version = subprocess.run(
            [str(executable), "-hide_banner", "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
        encoders = subprocess.run(
            [str(executable), "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise WorkerError(f"cannot inspect overlay video ffmpeg: {exc}") from exc
    version_line = (version.stdout or version.stderr).splitlines()
    if version.returncode != 0 or not version_line:
        raise WorkerError("overlay video ffmpeg version probe failed")
    if encoders.returncode != 0 or "libx264" not in encoders.stdout:
        raise WorkerError("overlay video ffmpeg does not provide libx264")
    return {
        "executable": str(executable),
        "version": version_line[0],
        "encoder": "libx264",
    }


def _valid_projected_points(value: Any) -> list[list[float] | None]:
    if not isinstance(value, list) or len(value) != 21:
        raise WorkerError("overlay projected keypoints must contain 21 values")
    points: list[list[float] | None] = []
    for point in value:
        if point is None:
            points.append(None)
            continue
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                for number in point
            )
        ):
            raise WorkerError("overlay projected keypoint is invalid")
        points.append([float(point[0]), float(point[1])])
    return points


class RawVsStableOverlayVideo:
    """Feed one 2x2 BGR diagnostic frame per synchronized pair to libx264."""

    def __init__(
        self,
        *,
        output_path: Path,
        image_size: tuple[int, int],
        timestamps_ns: list[int],
        temporal_method: str,
    ) -> None:
        if not timestamps_ns or any(
            isinstance(value, bool) or not isinstance(value, int) for value in timestamps_ns
        ):
            raise WorkerError("overlay video requires integer synchronized timestamps")
        if any(current <= previous for previous, current in pairwise(timestamps_ns)):
            raise WorkerError("overlay video timestamps must be strictly increasing")
        width, height = image_size
        if width <= 1 or height <= 1:
            raise WorkerError("overlay video image size is invalid")
        if not isinstance(temporal_method, str) or not temporal_method.strip():
            raise WorkerError("overlay video temporal method is invalid")
        self.output_path = Path(output_path).resolve()
        if self.output_path.exists():
            raise WorkerError(f"overlay video output already exists: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._timestamps_ns = list(timestamps_ns)
        self._temporal_method = temporal_method
        self._source_size = (width, height)
        self._panel_size = (max(1, width // 2), max(1, height // 2))
        self._output_size = (self._panel_size[0] * 2, self._panel_size[1] * 2)
        self._rate = _frame_rate(timestamps_ns)
        self._time_base = Fraction(1, self._rate.numerator)
        self._duration_pts = self._rate.denominator
        self._ffmpeg = _ffmpeg_details(Path("/usr/bin/ffmpeg"))
        gop = max(1, round(float(self._rate) / 2.0))
        command = [
            self._ffmpeg["executable"],
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self._output_size[0]}x{self._output_size[1]}",
            "-framerate",
            f"{self._rate.numerator}/{self._rate.denominator}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-fps_mode",
            "cfr",
            "-video_track_timescale",
            str(self._rate.numerator),
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise WorkerError(f"cannot start overlay video ffmpeg: {exc}") from exc
        if self._process.stdin is None or self._process.stderr is None:
            self.abort()
            raise WorkerError("overlay video ffmpeg pipes are unavailable")
        self._frames: list[dict[str, Any]] = []
        self._track_colors: dict[str, tuple[int, int, int]] = {}
        self._stable_input_stages: set[str] = set()
        self._closed = False

    def _color(self, track_id: str) -> tuple[int, int, int]:
        color = self._track_colors.get(track_id)
        if color is None:
            color = track_color_rgb(track_id)
            self._track_colors[track_id] = color
        return color

    def _panel(
        self,
        frame: Any,
        *,
        tracks: list[dict[str, Any]],
        stage_key: str,
        stage_label: str,
        side: str,
        frame_id: str,
        timestamp_ns: int,
    ) -> Any:
        import cv2

        try:
            if (
                frame is None
                or len(frame.shape) != 3
                or frame.shape[2] != 3
                or frame.shape[1::-1] != self._source_size
            ):
                raise WorkerError("overlay rectified frame has an invalid size")
        except (AttributeError, IndexError) as exc:
            raise WorkerError("overlay rectified frame is not an image") from exc
        panel = cv2.resize(frame, self._panel_size, interpolation=cv2.INTER_AREA)
        scale_x = self._panel_size[0] / self._source_size[0]
        scale_y = self._panel_size[1] / self._source_size[1]
        visible_tracks = 0
        for track in tracks:
            track_id = track.get("track_id")
            projection = track.get(stage_key)
            if not isinstance(track_id, str) or not track_id or not isinstance(projection, dict):
                raise WorkerError("overlay track payload is invalid")
            points = _valid_projected_points(projection.get(side))
            points = [
                point
                if point is not None
                and 0.0 <= point[0] < self._source_size[0]
                and 0.0 <= point[1] < self._source_size[1]
                else None
                for point in points
            ]
            if not any(point is not None for point in points):
                continue
            visible_tracks += 1
            rgb = self._color(track_id)
            bgr = (rgb[2], rgb[1], rgb[0])
            for start_index, end_index in _FHP21_EDGES:
                start = points[start_index]
                end = points[end_index]
                if start is None or end is None:
                    continue
                cv2.line(
                    panel,
                    (round(start[0] * scale_x), round(start[1] * scale_y)),
                    (round(end[0] * scale_x), round(end[1] * scale_y)),
                    bgr,
                    2,
                    cv2.LINE_AA,
                )
            for point in points:
                if point is None:
                    continue
                cv2.circle(
                    panel,
                    (round(point[0] * scale_x), round(point[1] * scale_y)),
                    3,
                    bgr,
                    -1,
                    cv2.LINE_AA,
                )
            wrist = points[0]
            if wrist is not None:
                cv2.putText(
                    panel,
                    track_id,
                    (round(wrist[0] * scale_x) + 5, round(wrist[1] * scale_y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    bgr,
                    1,
                    cv2.LINE_AA,
                )
        cv2.rectangle(panel, (0, 0), (self._panel_size[0], 43), (5, 9, 12), -1)
        cv2.putText(
            panel,
            f"{stage_label} / {side.upper()}",
            (9, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (240, 246, 244),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{frame_id}  {timestamp_ns} ns",
            (9, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (150, 165, 170),
            1,
            cv2.LINE_AA,
        )
        if visible_tracks == 0:
            cv2.putText(
                panel,
                "NO HAND OUTPUT",
                (max(8, self._panel_size[0] // 2 - 62), self._panel_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (106, 189, 244),
                1,
                cv2.LINE_AA,
            )
        return panel

    def append_frame(
        self,
        *,
        left_frame: Any,
        right_frame: Any,
        frame_id: str,
        frame_index: int,
        timestamp_ns: int,
        tracks: list[dict[str, Any]],
    ) -> None:
        if self._closed:
            raise WorkerError("cannot append to a closed overlay video")
        expected_index = len(self._frames)
        if expected_index >= len(self._timestamps_ns):
            raise WorkerError("overlay video received too many frames")
        if timestamp_ns != self._timestamps_ns[expected_index]:
            raise WorkerError("overlay video frame timestamp/order mismatch")
        if not isinstance(frame_id, str) or not frame_id:
            raise WorkerError("overlay video frame_id is invalid")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise WorkerError("overlay video frame_index is invalid")
        if not isinstance(tracks, list):
            raise WorkerError("overlay video tracks must be a list")
        for track in tracks:
            input_stage = track.get("stable_input_stage")
            if isinstance(input_stage, str) and input_stage:
                self._stable_input_stages.add(input_stage)
        import cv2
        import numpy as np

        raw_left = self._panel(
            left_frame,
            tracks=tracks,
            stage_key="raw",
            stage_label="RAW_FUSION",
            side="left",
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
        )
        raw_right = self._panel(
            right_frame,
            tracks=tracks,
            stage_key="raw",
            stage_label="RAW_FUSION",
            side="right",
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
        )
        stable_left = self._panel(
            left_frame,
            tracks=tracks,
            stage_key="stable",
            stage_label="TEMPORAL_REFINEMENT",
            side="left",
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
        )
        stable_right = self._panel(
            right_frame,
            tracks=tracks,
            stage_key="stable",
            stage_label="TEMPORAL_REFINEMENT",
            side="right",
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
        )
        composed = np.ascontiguousarray(
            cv2.vconcat(
                (cv2.hconcat((raw_left, raw_right)), cv2.hconcat((stable_left, stable_right)))
            )
        )
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(composed.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError(f"overlay video ffmpeg pipe failed: {exc}") from exc
        self._frames.append(
            {
                "video_frame_index": expected_index,
                "video_pts": expected_index * self._duration_pts,
                "duration_pts": self._duration_pts,
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "track_ids": sorted(
                    track["track_id"] for track in tracks if isinstance(track.get("track_id"), str)
                ),
            }
        )

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise WorkerError("overlay video is already closed")
        if len(self._frames) != len(self._timestamps_ns):
            self.abort()
            raise WorkerError(
                f"overlay video frame count mismatch: {len(self._frames)} != "
                f"{len(self._timestamps_ns)}"
            )
        assert self._process.stdin is not None
        assert self._process.stderr is not None
        self._process.stdin.close()
        try:
            returncode = self._process.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            self._process.wait()
            self._closed = True
            raise WorkerError("overlay video ffmpeg did not finish") from exc
        stderr = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        self._closed = True
        if returncode != 0:
            self.output_path.unlink(missing_ok=True)
            raise WorkerError(f"overlay video ffmpeg exited {returncode}: {stderr}")
        if not self.output_path.is_file() or self.output_path.stat().st_size <= 0:
            raise WorkerError("overlay video ffmpeg produced no MP4")
        timeline = {
            "schema_version": TIMELINE_SCHEMA,
            "frame_rate": {
                "numerator": self._rate.numerator,
                "denominator": self._rate.denominator,
            },
            "time_base": {
                "numerator": self._time_base.numerator,
                "denominator": self._time_base.denominator,
            },
            "frames": self._frames,
        }
        metadata = {
            "schema_version": VIDEO_SCHEMA,
            "output_status": "PRODUCED",
            "layout": "RAW_LEFT_RAW_RIGHT_STABLE_LEFT_STABLE_RIGHT",
            "image_space": "rectified",
            "comparison_stages": ["RAW_FUSION", "TEMPORAL_REFINEMENT"],
            "temporal_method": self._temporal_method,
            "stable_input_stages": sorted(self._stable_input_stages),
            "frame_count": len(self._frames),
            "width": self._output_size[0],
            "height": self._output_size[1],
            "codec": "h264",
            "pixel_format": "yuv420p",
            "container": "mp4",
            "frame_rate": timeline["frame_rate"],
            "time_base": timeline["time_base"],
            "tracks": [
                {"track_id": track_id, "color_rgb": list(color)}
                for track_id, color in self._track_colors.items()
            ],
            "ffmpeg": self._ffmpeg,
        }
        return {"path": self.output_path, "timeline": timeline, "metadata": metadata}

    def abort(self) -> None:
        if getattr(self, "_closed", False):
            return
        process = getattr(self, "_process", None)
        if process is not None:
            stdin = getattr(process, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.output_path.unlink(missing_ok=True)
        self._closed = True


__all__ = ["RawVsStableOverlayVideo", "TIMELINE_SCHEMA", "VIDEO_SCHEMA"]
