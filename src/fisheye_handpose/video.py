"""Strict video audit and streaming access to synchronized stereo frame pairs.

Decoded frame indices in this module always mean PyAV presentation-order indices.  A
video must pass a complete decode audit before :class:`StereoPairReader` will use it;
container frame-count metadata is deliberately not trusted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import FisheyeHandposeError
from .sync import FrameMatch, SyncResult


class VideoError(FisheyeHandposeError):
    """A video cannot be decoded or does not satisfy the strict frame contract."""


def _require_av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise VideoError("PyAV is required for video audit and stereo frame reading") from exc
    return av


def _positive_size(value: tuple[int, int]) -> tuple[int, int]:
    try:
        width, height = value
    except (TypeError, ValueError) as exc:
        raise VideoError("expected_size must be a (width, height) pair") from exc
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise VideoError("expected_size must contain two positive integers")
    return width, height


def _time_ns(pts: int, time_base: Any) -> int:
    """Convert a presentation timestamp to integer nanoseconds without float drift."""

    return int(pts * time_base * 1_000_000_000)


@dataclass(frozen=True, slots=True)
class VideoReport:
    """Result of decoding every frame of one video in presentation order."""

    path: Path
    passed: bool
    expected_size: tuple[int, int]
    expected_frame_count: int
    video_stream_count: int
    stream_index: int | None
    codec_name: str | None
    stream_size: tuple[int, int] | None
    decoded_size: tuple[int, int] | None
    decoded_frame_count: int
    presentation_timestamps_complete: bool
    presentation_order_strict: bool
    first_presentation_time_ns: int | None
    last_presentation_time_ns: int | None
    file_size_bytes: int | None
    file_mtime_ns: int | None
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "path": str(self.path),
            "expected_size": list(self.expected_size),
            "expected_frame_count": self.expected_frame_count,
            "video_stream_count": self.video_stream_count,
            "stream_index": self.stream_index,
            "codec_name": self.codec_name,
            "stream_size": list(self.stream_size) if self.stream_size is not None else None,
            "decoded_size": list(self.decoded_size) if self.decoded_size is not None else None,
            "decoded_frame_count": self.decoded_frame_count,
            "presentation_timestamps_complete": self.presentation_timestamps_complete,
            "presentation_order_strict": self.presentation_order_strict,
            "first_presentation_time_ns": self.first_presentation_time_ns,
            "last_presentation_time_ns": self.last_presentation_time_ns,
            "file_size_bytes": self.file_size_bytes,
            "file_mtime_ns": self.file_mtime_ns,
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
        }


def audit_video(
    path: str | Path,
    expected_size: tuple[int, int],
    expected_frame_count: int,
) -> VideoReport:
    """Fully decode a video and verify its mapping to a hardware timestamp stream.

    Expected input defects are represented by a failed :class:`VideoReport`, allowing
    callers to persist an audit artifact.  A missing PyAV runtime or an invalid audit
    configuration raises :class:`VideoError` because no meaningful audit can run.
    """

    av = _require_av()
    expected_size = _positive_size(expected_size)
    if not isinstance(expected_frame_count, int) or expected_frame_count <= 0:
        raise VideoError("expected_frame_count must be a positive integer")

    source = Path(path).expanduser().resolve()
    hard_failures: list[str] = []
    warnings: list[str] = []
    video_stream_count = 0
    stream_index: int | None = None
    codec_name: str | None = None
    stream_size: tuple[int, int] | None = None
    decoded_size: tuple[int, int] | None = None
    decoded_frame_count = 0
    timestamps_complete = True
    order_strict = True
    first_time_ns: int | None = None
    last_time_ns: int | None = None
    file_size_bytes: int | None = None
    file_mtime_ns: int | None = None

    try:
        initial_stat = source.stat()
        file_size_bytes = initial_stat.st_size
        file_mtime_ns = initial_stat.st_mtime_ns
    except OSError as exc:
        hard_failures.append(f"video file is not readable: {exc}")
        return VideoReport(
            source,
            False,
            expected_size,
            expected_frame_count,
            video_stream_count,
            stream_index,
            codec_name,
            stream_size,
            decoded_size,
            decoded_frame_count,
            timestamps_complete,
            order_strict,
            first_time_ns,
            last_time_ns,
            file_size_bytes,
            file_mtime_ns,
            tuple(hard_failures),
            tuple(warnings),
        )

    container = None
    try:
        container = av.open(str(source), mode="r")
        streams = list(container.streams.video)
        video_stream_count = len(streams)
        if video_stream_count != 1:
            hard_failures.append(f"expected exactly one video stream, found {video_stream_count}")
        if not streams:
            return VideoReport(
                source,
                False,
                expected_size,
                expected_frame_count,
                video_stream_count,
                stream_index,
                codec_name,
                stream_size,
                decoded_size,
                decoded_frame_count,
                timestamps_complete,
                order_strict,
                first_time_ns,
                last_time_ns,
                file_size_bytes,
                file_mtime_ns,
                tuple(hard_failures),
                tuple(warnings),
            )

        stream = streams[0]
        stream_index = int(stream.index)
        codec_name = str(getattr(stream.codec_context, "name", "") or "") or None
        if codec_name is None:
            hard_failures.append("video stream does not declare a decodable codec")
        stream_size = (int(stream.codec_context.width), int(stream.codec_context.height))
        if stream_size != expected_size:
            hard_failures.append(
                f"stream size {stream_size} does not match expected size {expected_size}"
            )

        previous_time: Any | None = None
        observed_sizes: set[tuple[int, int]] = set()
        for frame in container.decode(stream):
            decoded_frame_count += 1
            observed_sizes.add((int(frame.width), int(frame.height)))
            if frame.pts is None or frame.time_base is None:
                timestamps_complete = False
                order_strict = False
                continue
            current_time = frame.pts * frame.time_base
            current_time_ns = _time_ns(frame.pts, frame.time_base)
            if first_time_ns is None:
                first_time_ns = current_time_ns
            last_time_ns = current_time_ns
            if previous_time is not None and current_time <= previous_time:
                order_strict = False
            previous_time = current_time

        if len(observed_sizes) == 1:
            decoded_size = next(iter(observed_sizes))
        elif len(observed_sizes) > 1:
            hard_failures.append(
                "decoded frame resolution is not fixed: "
                + ", ".join(str(size) for size in sorted(observed_sizes))
            )
        if decoded_size is not None and decoded_size != expected_size:
            hard_failures.append(
                f"decoded frame size {decoded_size} does not match expected size {expected_size}"
            )
        if decoded_frame_count == 0:
            hard_failures.append("video decoded zero frames")
        if decoded_frame_count != expected_frame_count:
            hard_failures.append(
                f"decoded frame count {decoded_frame_count} does not match timestamp count "
                f"{expected_frame_count}"
            )
        # Decoder iteration defines presentation order. Container PTS is useful
        # diagnostics, but is not the hardware-clock truth and is not a hard gate.
        if not timestamps_complete:
            warnings.append("one or more decoded frames have no container presentation timestamp")
        if timestamps_complete and not order_strict:
            warnings.append("container presentation timestamps are not strictly increasing")
    except Exception as exc:
        hard_failures.append(f"complete video decode failed: {type(exc).__name__}: {exc}")
    finally:
        if container is not None:
            container.close()

    try:
        final_stat = source.stat()
    except OSError as exc:
        hard_failures.append(f"video disappeared during audit: {exc}")
        file_size_bytes = None
        file_mtime_ns = None
    else:
        if final_stat.st_size != file_size_bytes or final_stat.st_mtime_ns != file_mtime_ns:
            hard_failures.append("video file changed while it was being audited")
        file_size_bytes = final_stat.st_size
        file_mtime_ns = final_stat.st_mtime_ns

    return VideoReport(
        path=source,
        passed=not hard_failures,
        expected_size=expected_size,
        expected_frame_count=expected_frame_count,
        video_stream_count=video_stream_count,
        stream_index=stream_index,
        codec_name=codec_name,
        stream_size=stream_size,
        decoded_size=decoded_size,
        decoded_frame_count=decoded_frame_count,
        presentation_timestamps_complete=timestamps_complete,
        presentation_order_strict=order_strict,
        first_presentation_time_ns=first_time_ns,
        last_presentation_time_ns=last_time_ns,
        file_size_bytes=file_size_bytes,
        file_mtime_ns=file_mtime_ns,
        hard_failures=tuple(hard_failures),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class StereoFramePair:
    """One synchronized pair of presentation-order BGR frames."""

    left_bgr: Any
    right_bgr: Any
    match: FrameMatch

    @property
    def pair_index(self) -> int:
        return self.match.pair_index

    @property
    def left_index(self) -> int:
        return self.match.left_index

    @property
    def right_index(self) -> int:
        return self.match.right_index

    @property
    def left_timestamp_ns_raw(self) -> int:
        return self.match.left_timestamp_ns

    @property
    def right_timestamp_ns_raw(self) -> int:
        return self.match.right_timestamp_raw_ns

    @property
    def left_timestamp_ns_corrected(self) -> int:
        # The left hardware clock is the pairing reference in match_timestamps().
        return self.match.left_timestamp_ns

    @property
    def right_timestamp_ns_corrected(self) -> int:
        return self.match.right_timestamp_corrected_ns

    @property
    def clock_offset_ns(self) -> int:
        return self.right_timestamp_ns_corrected - self.right_timestamp_ns_raw

    @property
    def corrected_skew_ns(self) -> int:
        return self.right_timestamp_ns_corrected - self.left_timestamp_ns_corrected

    @property
    def pair_timestamp_ns(self) -> int:
        return self.match.pair_timestamp_ns


class StereoPairReader:
    """Stream selected stereo frames without seeking or materializing either video."""

    def __init__(
        self,
        left_path: str | Path,
        right_path: str | Path,
        left_report: VideoReport,
        right_report: VideoReport,
        sync_result: SyncResult,
    ) -> None:
        self.left_path = Path(left_path).expanduser().resolve()
        self.right_path = Path(right_path).expanduser().resolve()
        self.left_report = left_report
        self.right_report = right_report
        self.sync_result = sync_result
        self._validate_contract()
        self._left_container: Any | None = None
        self._right_container: Any | None = None
        self._left_stream: Any | None = None
        self._right_stream: Any | None = None
        self._iteration_started = False

    def _validate_contract(self) -> None:
        if not isinstance(self.left_report, VideoReport) or not isinstance(
            self.right_report, VideoReport
        ):
            raise VideoError("StereoPairReader requires two VideoReport instances")
        if not isinstance(self.sync_result, SyncResult):
            raise VideoError("StereoPairReader requires a SyncResult")
        for side, path, report in (
            ("left", self.left_path, self.left_report),
            ("right", self.right_path, self.right_report),
        ):
            if not report.passed:
                raise VideoError(f"{side} video did not pass its complete decode audit")
            if report.path != path:
                raise VideoError(f"{side} report belongs to {report.path}, not {path}")
            try:
                stat = path.stat()
            except OSError as exc:
                raise VideoError(f"{side} video is no longer readable: {exc}") from exc
            if stat.st_size != report.file_size_bytes or stat.st_mtime_ns != report.file_mtime_ns:
                raise VideoError(f"{side} video changed after its decode audit")

        matches = self.sync_result.matches
        if not matches:
            raise VideoError("SyncResult contains no stereo frame matches")
        previous_pair = previous_left = previous_right = -1
        for match in matches:
            if match.pair_index != previous_pair + 1:
                raise VideoError("SyncResult pair indices must be contiguous from zero")
            if match.left_index <= previous_left or match.right_index <= previous_right:
                raise VideoError("SyncResult frame indices must be strictly increasing")
            if match.left_index < 0 or match.left_index >= self.left_report.decoded_frame_count:
                raise VideoError(
                    f"left match index {match.left_index} is outside the audited video"
                )
            if match.right_index < 0 or match.right_index >= self.right_report.decoded_frame_count:
                raise VideoError(
                    f"right match index {match.right_index} is outside the audited video"
                )
            if (
                match.right_timestamp_corrected_ns - match.right_timestamp_raw_ns
                != self.sync_result.clock_offset_ns
            ):
                raise VideoError("FrameMatch is inconsistent with SyncResult.clock_offset_ns")
            if match.right_timestamp_corrected_ns - match.left_timestamp_ns != match.skew_ns:
                raise VideoError("FrameMatch corrected timestamp and skew are inconsistent")
            previous_pair = match.pair_index
            previous_left = match.left_index
            previous_right = match.right_index

    @staticmethod
    def _open_verified(av: Any, path: Path, report: VideoReport) -> tuple[Any, Any]:
        try:
            container = av.open(str(path), mode="r")
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise VideoError(
                    f"{path}: expected one video stream after audit, found {len(streams)}"
                )
            stream = streams[0]
            codec_name = str(getattr(stream.codec_context, "name", "") or "") or None
            stream_size = (int(stream.codec_context.width), int(stream.codec_context.height))
            if codec_name != report.codec_name or stream_size != report.stream_size:
                raise VideoError(f"{path}: codec or dimensions changed after audit")
            return container, stream
        except Exception:
            if "container" in locals():
                container.close()
            raise

    def __enter__(self) -> StereoPairReader:
        if self._left_container is not None or self._right_container is not None:
            raise VideoError("StereoPairReader is already open")
        av = _require_av()
        self._validate_contract()
        self._left_container, self._left_stream = self._open_verified(
            av, self.left_path, self.left_report
        )
        try:
            self._right_container, self._right_stream = self._open_verified(
                av, self.right_path, self.right_report
            )
        except Exception:
            self._left_container.close()
            self._left_container = None
            self._left_stream = None
            raise
        self._iteration_started = False
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._right_container is not None:
            self._right_container.close()
        if self._left_container is not None:
            self._left_container.close()
        self._left_container = None
        self._right_container = None
        self._left_stream = None
        self._right_stream = None
        self._iteration_started = False

    @staticmethod
    def _next_frame(iterator: Iterator[Any], path: Path, target_index: int) -> Any:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise VideoError(
                f"{path}: decoder ended before audited frame index {target_index}; "
                "file may have changed"
            ) from exc
        except Exception as exc:
            raise VideoError(
                f"{path}: decode failed while advancing to frame {target_index}: {exc}"
            ) from exc

    def _iterate_open(self) -> Iterator[StereoFramePair]:
        if (
            self._left_container is None
            or self._right_container is None
            or self._left_stream is None
            or self._right_stream is None
        ):
            raise VideoError("StereoPairReader is not open")
        if self._iteration_started:
            raise VideoError("an open StereoPairReader can only be iterated once")
        self._iteration_started = True

        left_frames = iter(self._left_container.decode(self._left_stream))
        right_frames = iter(self._right_container.decode(self._right_stream))
        left_index = right_index = -1
        for match in self.sync_result.matches:
            left_frame = right_frame = None
            while left_index < match.left_index:
                left_frame = self._next_frame(left_frames, self.left_path, match.left_index)
                left_index += 1
            while right_index < match.right_index:
                right_frame = self._next_frame(right_frames, self.right_path, match.right_index)
                right_index += 1
            if left_frame is None or right_frame is None:
                # Strictly increasing match indices make this unreachable unless the
                # SyncResult was mutated after validation.
                raise VideoError("failed to advance both decoders to the requested pair")
            if (left_frame.width, left_frame.height) != self.left_report.expected_size:
                raise VideoError("left frame dimensions changed after audit")
            if (right_frame.width, right_frame.height) != self.right_report.expected_size:
                raise VideoError("right frame dimensions changed after audit")
            try:
                left_bgr = left_frame.to_ndarray(format="bgr24")
                right_bgr = right_frame.to_ndarray(format="bgr24")
            except Exception as exc:
                raise VideoError(f"failed to convert a stereo pair to BGR: {exc}") from exc
            yield StereoFramePair(left_bgr=left_bgr, right_bgr=right_bgr, match=match)

    def _iterate_managed(self) -> Iterator[StereoFramePair]:
        with self:
            yield from self._iterate_open()

    def __iter__(self) -> Iterator[StereoFramePair]:
        if self._left_container is None and self._right_container is None:
            return self._iterate_managed()
        return self._iterate_open()


__all__ = [
    "StereoFramePair",
    "StereoPairReader",
    "VideoError",
    "VideoReport",
    "audit_video",
]
