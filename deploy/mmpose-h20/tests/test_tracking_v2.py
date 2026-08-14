from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.contracts import WorkerError  # noqa: E402
from fisheye_h20_worker.tracking import SequenceTracker  # noqa: E402

PALM_INDICES = (0, 5, 9, 13, 17)


def _observation(
    identifier: str,
    *,
    palm_x: float,
    other_x: float | None = None,
    invalid_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    fill_x = palm_x if other_x is None else other_x
    points = [[fill_x, 0.0, 1.0] for _ in range(21)]
    for index in PALM_INDICES:
        points[index] = [palm_x, 0.0, 1.0]
    validity = ["VALID"] * 21
    for index in invalid_indices:
        validity[index] = "LOW_KEYPOINT_SCORE"
    return {
        "observation_id": identifier,
        "landmarks_xyz_m": points,
        "validity": validity,
    }


def test_invalid_wrist_does_not_change_palm_anchor_semantics_or_split_track() -> None:
    tracker = SequenceTracker(max_root_distance_m=0.1, max_gap_ms=250.0)

    first = tracker.assign(
        [_observation("first", palm_x=0.0)],
        timestamp_ns=1_000_000_000,
    )[0]
    second = tracker.assign(
        [
            _observation(
                "second",
                palm_x=0.01,
                other_x=1.0,
                invalid_indices=(0,),
            )
        ],
        timestamp_ns=1_100_000_000,
    )[0]

    assert second["track_id"] == first["track_id"]
    assert second["decision"] == "MATCHED"
    assert second["anchor_method"] == "fhp21_palm_coordinate_median_v1"
    assert second["anchor_support"] == 4
    assert second["anchor_xyz_m"] == [0.01, 0.0, 1.0]


def test_palm_coordinate_median_rejects_one_extreme_palm_outlier() -> None:
    observation = _observation("outlier", palm_x=0.0)
    for index, x in zip(PALM_INDICES, (-0.02, -0.01, 0.0, 0.01, 4.0), strict=True):
        observation["landmarks_xyz_m"][index] = [x, 0.0, 1.0]

    tracked = SequenceTracker(max_root_distance_m=0.1, max_gap_ms=250.0).assign(
        [observation],
        timestamp_ns=1_000_000_000,
    )[0]

    assert tracked["anchor_xyz_m"] == [0.0, 0.0, 1.0]


def test_constant_velocity_keeps_identity_when_hands_cross_in_image_order() -> None:
    tracker = SequenceTracker(max_root_distance_m=0.25, max_gap_ms=250.0)

    first = tracker.assign(
        [
            _observation("a-0", palm_x=-0.30),
            _observation("b-0", palm_x=0.30),
        ],
        timestamp_ns=1_000_000_000,
    )
    tracker.assign(
        [
            _observation("a-1", palm_x=-0.10),
            _observation("b-1", palm_x=0.15),
        ],
        timestamp_ns=1_100_000_000,
    )
    crossed = tracker.assign(
        [
            _observation("b-2", palm_x=0.01),
            _observation("a-2", palm_x=0.09),
        ],
        timestamp_ns=1_200_000_000,
    )

    track_a = first[0]["track_id"]
    track_b = first[1]["track_id"]
    by_observation = {value["observation_id"]: value for value in crossed}
    assert by_observation["a-2"]["track_id"] == track_a
    assert by_observation["b-2"]["track_id"] == track_b
    assert by_observation["a-2"]["predicted_anchor_xyz_m"] == pytest.approx([0.10, 0.0, 1.0])
    assert by_observation["a-2"]["motion_method"] == "constant_velocity_metric_v1"


def test_short_unmatched_gap_recovers_track_but_gap_past_ttl_starts_new_track() -> None:
    tracker = SequenceTracker(max_root_distance_m=0.1, max_gap_ms=250.0)

    initial = tracker.assign(
        [_observation("moving-0", palm_x=0.00)],
        timestamp_ns=1_000_000_000,
    )[0]
    tracker.assign(
        [_observation("moving-1", palm_x=0.08)],
        timestamp_ns=1_100_000_000,
    )
    assert tracker.assign([], timestamp_ns=1_200_000_000) == []

    recovered = tracker.assign(
        [_observation("moving-2", palm_x=0.24)],
        timestamp_ns=1_300_000_000,
    )[0]
    assert recovered["track_id"] == initial["track_id"]
    assert recovered["decision"] == "MATCHED"
    assert recovered["recovered"] is True
    assert recovered["delta_ms"] == 200.0

    assert tracker.assign([], timestamp_ns=1_600_000_001) == []
    after_ttl = tracker.assign(
        [_observation("moving-3", palm_x=0.32)],
        timestamp_ns=1_610_000_000,
    )[0]
    assert after_ttl["decision"] == "NEW"
    assert after_ttl["track_id"] != initial["track_id"]


def test_out_of_order_timestamp_is_rejected_without_destroying_track_state() -> None:
    tracker = SequenceTracker(max_root_distance_m=0.1, max_gap_ms=250.0)
    initial = tracker.assign(
        [_observation("ordered-0", palm_x=0.0)],
        timestamp_ns=1_000_000_000,
    )[0]

    with pytest.raises(WorkerError, match="monotonically non-decreasing"):
        tracker.assign(
            [_observation("out-of-order", palm_x=0.01)],
            timestamp_ns=999_999_999,
        )

    resumed = tracker.assign(
        [_observation("ordered-1", palm_x=0.01)],
        timestamp_ns=1_100_000_000,
    )[0]
    assert resumed["track_id"] == initial["track_id"]
    assert resumed["decision"] == "MATCHED"


def test_non_palm_joints_do_not_substitute_for_missing_palm_support() -> None:
    tracker = SequenceTracker(max_root_distance_m=0.1, max_gap_ms=250.0)
    initial = tracker.assign(
        [_observation("supported-0", palm_x=0.0)],
        timestamp_ns=1_000_000_000,
    )[0]

    unsupported = tracker.assign(
        [
            _observation(
                "unsupported",
                palm_x=9.0,
                other_x=0.01,
                invalid_indices=PALM_INDICES,
            )
        ],
        timestamp_ns=1_100_000_000,
    )[0]

    assert unsupported["trackable"] is False
    assert unsupported["anchor_support"] == 0
    assert unsupported["anchor_xyz_m"] is None
    assert unsupported["anchor_indices"] == [0, 5, 9, 13, 17]
    assert unsupported["decision_reason"] == "INSUFFICIENT_PALM_SUPPORT"
    assert unsupported["track_id"] != initial["track_id"]

    recovered = tracker.assign(
        [_observation("supported-1", palm_x=0.01)],
        timestamp_ns=1_200_000_000,
    )[0]
    assert recovered["track_id"] == initial["track_id"]
    assert recovered["recovered"] is True
