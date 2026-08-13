from pathlib import Path

import pytest

from fisheye_handpose.errors import SyncError, TimestampError
from fisheye_handpose.sync import TimestampSeries, match_timestamps, read_timestamp_csv


def series(name: str, values_us: list[int]) -> TimestampSeries:
    return TimestampSeries(
        Path(name),
        tuple(value * 1_000 for value in values_us),
        "us",
        "timestamp_us",
    )


def test_pairing_uses_timestamps_not_frame_indices():
    left = series("left", [0, 33_333, 66_666, 99_999, 133_332])
    right = series("right", [66_670, 100_003, 133_336])
    result = match_timestamps(left, right, max_skew_ns=1_000_000)
    assert [(m.left_index, m.right_index) for m in result.matches] == [(2, 0), (3, 1), (4, 2)]
    assert result.left_tail_indices == (0, 1)
    # The last left timestamp is inside the strict common interval, while its
    # matched right timestamp is just outside it. It remains a global match but
    # is intentionally not credited to the strict-overlap quality metric.
    assert result.overlap_unmatched_left_indices == (4,)


def test_middle_drop_does_not_shift_following_pairs():
    left = series("left", [0, 33_333, 66_666, 99_999])
    right = series("right", [4, 33_337, 100_003])
    result = match_timestamps(left, right, max_skew_ns=1_000_000)
    assert [(m.left_index, m.right_index) for m in result.matches] == [(0, 0), (1, 1), (3, 2)]
    assert result.overlap_unmatched_left_indices == (2, 3)


def test_tolerance_must_be_unambiguous():
    left = series("left", [0, 10_000, 20_000])
    right = series("right", [0, 10_000, 20_000])
    with pytest.raises(SyncError, match="less than half"):
        match_timestamps(left, right, max_skew_ns=5_000_000)


def test_csv_rejects_non_monotonic_and_header_only(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp_us\n10\n10\n", encoding="utf-8")
    with pytest.raises(TimestampError, match="strictly increasing"):
        read_timestamp_csv(bad)
    empty = tmp_path / "empty.csv"
    empty.write_text("timestamp_us\n", encoding="utf-8")
    with pytest.raises(TimestampError, match="empty"):
        read_timestamp_csv(empty)


def test_unambiguous_gate_uses_minimum_local_interval_not_median():
    left = series("left", [0, 100, 10_100, 20_100])
    right = series("right", [50, 10_050, 20_050, 30_050])
    with pytest.raises(SyncError, match="minimum local interval"):
        match_timestamps(left, right, max_skew_ns=60_000)


def test_clock_offset_provenance_preserves_raw_and_corrected_time():
    left = series("left", [100, 200])
    right = series("right", [90, 190])
    result = match_timestamps(
        left,
        right,
        max_skew_ns=20_000,
        clock_offset_ns=10_000,
    )
    match = result.matches[0]
    assert match.right_timestamp_raw_ns == 90_000
    assert match.right_timestamp_corrected_ns == 100_000
    assert match.skew_ns == 0
    report = result.to_dict(include_matches=True)
    assert report["clock_offset_ns"] == 10_000
    assert report["matches"][0]["right_timestamp_raw_ns"] == 90_000
    assert report["matches"][0]["right_timestamp_corrected_ns"] == 100_000


def test_overlap_rates_count_only_pairs_fully_inside_common_interval():
    left = series("left", [0, 100])
    right = series("right", [5, 105])
    result = match_timestamps(left, right, max_skew_ns=10_000)
    report = result.to_dict()
    assert report["pair_count"] == 2
    assert report["overlap_pair_count"] == 0
    assert report["left_overlap_frame_count"] == 1
    assert report["right_overlap_frame_count"] == 1
    assert report["left_overlap_match_rate"] == 0.0
    assert report["right_overlap_match_rate"] == 0.0


def test_gap_statistics_report_drop_location():
    left = series("left", [0, 10_000, 20_000, 40_000])
    right = series("right", [0, 10_000, 20_000, 30_000, 40_000])
    result = match_timestamps(left, right, max_skew_ns=1_000_000)
    left_report = left.to_dict()
    sync_report = result.to_dict()
    assert left_report["minimum_interval_ns"] == 10_000_000
    assert left_report["maximum_interval_ns"] == 20_000_000
    assert left_report["gap_after_indices"] == [2]
    assert sync_report["left_gap_count"] == 1
    assert sync_report["left_gap_after_indices"] == [2]
    assert sync_report["right_gap_count"] == 0
