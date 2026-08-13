from __future__ import annotations

import builtins
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from fisheye_handpose.sync import TimestampSeries, match_timestamps
from fisheye_handpose.video import StereoPairReader, VideoError, audit_video

SIZE = (32, 24)
PERIOD_US = 33_333


def _write_video(path: Path, levels: list[int], size: tuple[int, int] = SIZE) -> Path:
    width, height = size
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"bf": "2"}
        for index, level in enumerate(levels):
            image = np.full((height, width, 3), level, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _series(path: Path, values_us: list[int]) -> TimestampSeries:
    return TimestampSeries(
        source_path=path,
        values_ns=tuple(value * 1_000 for value in values_us),
        source_unit="us",
        column="timestamp_us",
    )


def test_audit_fully_decodes_and_reports_codec_size_count_and_presentation_order(tmp_path):
    video = _write_video(tmp_path / "tiny.mp4", [20, 50, 80, 110])

    report = audit_video(video, SIZE, 4)

    assert report.passed
    assert report.status == "PASS"
    assert report.codec_name
    assert report.stream_size == SIZE
    assert report.decoded_size == SIZE
    assert report.decoded_frame_count == 4
    assert report.presentation_timestamps_complete
    assert report.presentation_order_strict
    assert report.first_presentation_time_ns is not None
    assert report.last_presentation_time_ns > report.first_presentation_time_ns
    assert report.to_dict()["hard_failures"] == []


@pytest.mark.parametrize(
    ("expected_size", "expected_count", "failure_text"),
    [
        ((64, 48), 4, "size"),
        (SIZE, 5, "timestamp count"),
    ],
)
def test_audit_returns_failed_report_for_contract_mismatch(
    tmp_path, expected_size, expected_count, failure_text
):
    video = _write_video(tmp_path / "bad_contract.mp4", [10, 20, 30, 40])

    report = audit_video(video, expected_size, expected_count)

    assert not report.passed
    assert report.status == "FAIL"
    assert any(failure_text in failure for failure in report.hard_failures)


def test_stereo_reader_advances_monotonically_over_dropped_frames_and_exposes_times(tmp_path):
    left_video = _write_video(tmp_path / "left.mp4", [20, 60, 100, 140])
    right_video = _write_video(tmp_path / "right.mp4", [30, 110, 150])
    left_times = _series(
        tmp_path / "left_pts.csv",
        [1_000_000, 1_033_333, 1_066_666, 1_099_999],
    )
    right_times = _series(
        tmp_path / "right_pts.csv",
        [1_000_200, 1_066_866, 1_100_199],
    )
    sync = match_timestamps(
        left_times,
        right_times,
        max_skew_ns=1_000_000,
        clock_offset_ns=-200_000,
    )
    left_report = audit_video(left_video, SIZE, len(left_times.values_ns))
    right_report = audit_video(right_video, SIZE, len(right_times.values_ns))

    reader = StereoPairReader(left_video, right_video, left_report, right_report, sync)
    with reader as opened:
        pairs = list(opened)

    assert [(pair.left_index, pair.right_index) for pair in pairs] == [(0, 0), (2, 1), (3, 2)]
    assert all(pair.left_bgr.shape == (SIZE[1], SIZE[0], 3) for pair in pairs)
    assert all(pair.right_bgr.shape == (SIZE[1], SIZE[0], 3) for pair in pairs)
    assert [float(pair.left_bgr.mean()) for pair in pairs] == pytest.approx([20, 100, 140], abs=8)
    assert pairs[0].match is sync.matches[0]
    assert pairs[0].left_timestamp_ns_raw == 1_000_000_000
    assert pairs[0].right_timestamp_ns_raw == 1_000_200_000
    assert pairs[0].right_timestamp_ns_corrected == 1_000_000_000
    assert pairs[0].clock_offset_ns == -200_000
    assert pairs[0].pair_timestamp_ns == 1_000_000_000


def test_reader_can_manage_its_own_context_and_rejects_failed_or_stale_reports(tmp_path):
    left_video = _write_video(tmp_path / "left.mp4", [10, 20])
    right_video = _write_video(tmp_path / "right.mp4", [30, 40])
    left_times = _series(tmp_path / "left_pts.csv", [0, PERIOD_US])
    right_times = _series(tmp_path / "right_pts.csv", [0, PERIOD_US])
    sync = match_timestamps(left_times, right_times, max_skew_ns=1_000_000)
    left_report = audit_video(left_video, SIZE, 2)
    right_report = audit_video(right_video, SIZE, 2)

    reader = StereoPairReader(left_video, right_video, left_report, right_report, sync)
    assert len(list(reader)) == 2

    failed_report = audit_video(left_video, SIZE, 3)
    with pytest.raises(VideoError, match="did not pass"):
        StereoPairReader(left_video, right_video, failed_report, right_report, sync)

    left_video.write_bytes(left_video.read_bytes() + b"changed")
    with pytest.raises(VideoError, match="changed after"):
        StereoPairReader(left_video, right_video, left_report, right_report, sync)


def test_missing_pyav_is_a_domain_error(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def import_without_av(name, *args, **kwargs):
        if name == "av":
            raise ImportError("simulated missing PyAV")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_av)
    with pytest.raises(VideoError, match="PyAV is required"):
        audit_video(tmp_path / "unused.mp4", SIZE, 1)
