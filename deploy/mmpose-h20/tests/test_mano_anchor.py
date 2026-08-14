from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.mano_anchor import (  # noqa: E402
    ManoAnchorSelector,
    ManoFrameEvidence,
)


def _evidence(
    *,
    track_id: str,
    frame_id: str,
    timestamp_ns: int,
    valid_joint_count: int = 21,
    score_2d: float = 0.9,
    reprojection_px: float = 0.5,
    bone_outlier_fraction: float = 0.0,
    triangulation_quality: float = 0.9,
) -> ManoFrameEvidence:
    validity = tuple(index < valid_joint_count for index in range(21))
    return ManoFrameEvidence(
        track_id=track_id,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        raw_validity=validity,
        keypoint_scores_2d=tuple(score_2d if valid else None for valid in validity),
        reprojection_errors_px=tuple(reprojection_px if valid else None for valid in validity),
        bone_outlier_fraction=bone_outlier_fraction,
        triangulation_quality=triangulation_quality,
    )


def test_poor_first_frame_is_not_used_as_the_track_anchor() -> None:
    selector = ManoAnchorSelector(top_k=1, min_time_separation_ns=0)

    result = selector.select(
        [
            _evidence(
                track_id="track-a",
                frame_id="frame-000",
                timestamp_ns=0,
                valid_joint_count=8,
            ),
            _evidence(
                track_id="track-a",
                frame_id="frame-001",
                timestamp_ns=33_000_000,
            ),
        ]
    )

    track = result.for_track("track-a")
    assert tuple(anchor.frame_id for anchor in track.anchors) == ("frame-001",)
    assert track.decision_for("frame-000").accepted is False
    assert track.decision_for("frame-000").reason == "INSUFFICIENT_RAW_SUPPORT"
    assert track.decision_for("frame-001").accepted is True
    assert track.decision_for("frame-001").reason == "SELECTED"


def test_anchor_score_has_no_flat_hand_or_pose_penalty() -> None:
    selector = ManoAnchorSelector(top_k=2, min_time_separation_ns=0)

    # These frame IDs describe different natural poses; anchor scoring only receives
    # observation quality, so the curled pose must not receive a hidden penalty.
    result = selector.select(
        [
            _evidence(
                track_id="track-a",
                frame_id="flat-natural-pose",
                timestamp_ns=0,
            ),
            _evidence(
                track_id="track-a",
                frame_id="curled-natural-pose",
                timestamp_ns=1,
            ),
        ]
    )
    track = result.for_track("track-a")
    flat = track.decision_for("flat-natural-pose")
    curled = track.decision_for("curled-natural-pose")

    assert flat.score == curled.score
    assert flat.score_components == curled.score_components
    assert flat.pose_prior_applied is False
    assert curled.pose_prior_applied is False
    assert tuple(component.name for component in curled.score_components) == (
        "raw_valid_joint_ratio",
        "mean_2d_score",
        "reprojection_quality",
        "bone_quality",
        "triangulation_quality",
    )


def test_higher_geometry_quality_wins_over_a_slightly_higher_2d_score() -> None:
    selector = ManoAnchorSelector(top_k=1, min_time_separation_ns=0)

    result = selector.select(
        [
            _evidence(
                track_id="track-a",
                frame_id="strong-geometry",
                timestamp_ns=10,
                score_2d=0.80,
                reprojection_px=0.2,
                bone_outlier_fraction=0.0,
                triangulation_quality=0.95,
            ),
            _evidence(
                track_id="track-a",
                frame_id="weak-geometry",
                timestamp_ns=20,
                score_2d=0.95,
                reprojection_px=4.0,
                bone_outlier_fraction=0.20,
                triangulation_quality=0.35,
            ),
        ]
    )
    track = result.for_track("track-a")

    assert tuple(anchor.frame_id for anchor in track.anchors) == ("strong-geometry",)
    assert track.decision_for("strong-geometry").score > track.decision_for("weak-geometry").score
    assert track.decision_for("weak-geometry").reason == "TOP_K_LIMIT"


def test_anchor_selection_is_independent_per_track() -> None:
    selector = ManoAnchorSelector(top_k=1, min_time_separation_ns=1_000_000_000)

    result = selector.select(
        [
            _evidence(
                track_id="track-b",
                frame_id="b-best",
                timestamp_ns=100,
                reprojection_px=0.2,
            ),
            _evidence(
                track_id="track-a",
                frame_id="a-best",
                timestamp_ns=100,
                reprojection_px=0.1,
            ),
        ]
    )

    assert tuple(track.track_id for track in result.tracks) == ("track-a", "track-b")
    assert result.for_track("track-a").anchors[0].frame_id == "a-best"
    assert result.for_track("track-b").anchors[0].frame_id == "b-best"


def test_nearby_high_score_frame_is_rejected_as_time_redundant() -> None:
    selector = ManoAnchorSelector(top_k=2, min_time_separation_ns=100)

    result = selector.select(
        [
            _evidence(
                track_id="track-a",
                frame_id="best",
                timestamp_ns=0,
                reprojection_px=0.1,
            ),
            _evidence(
                track_id="track-a",
                frame_id="near-duplicate",
                timestamp_ns=50,
                reprojection_px=0.2,
            ),
            _evidence(
                track_id="track-a",
                frame_id="diverse",
                timestamp_ns=200,
                reprojection_px=0.3,
            ),
        ]
    )
    track = result.for_track("track-a")

    assert tuple(anchor.frame_id for anchor in track.anchors) == ("best", "diverse")
    redundant = track.decision_for("near-duplicate")
    assert redundant.accepted is False
    assert redundant.quality_accepted is True
    assert redundant.reason == "TIME_REDUNDANT"
    assert redundant.nearest_selected_delta_ns == 50


def test_track_with_only_bad_evidence_fails_closed_with_frame_reasons() -> None:
    selector = ManoAnchorSelector(top_k=3, min_time_separation_ns=0)

    result = selector.select(
        [
            _evidence(
                track_id="track-a",
                frame_id="low-2d",
                timestamp_ns=0,
                score_2d=0.10,
            ),
            _evidence(
                track_id="track-a",
                frame_id="bad-reprojection",
                timestamp_ns=1,
                reprojection_px=8.0,
            ),
            _evidence(
                track_id="track-a",
                frame_id="bone-outlier",
                timestamp_ns=2,
                bone_outlier_fraction=0.5,
            ),
            _evidence(
                track_id="track-a",
                frame_id="weak-triangulation",
                timestamp_ns=3,
                triangulation_quality=0.1,
            ),
        ]
    )
    track = result.for_track("track-a")

    assert track.anchors == ()
    assert track.status == "NO_ACCEPTABLE_ANCHOR"
    assert track.reason == "NO_ACCEPTABLE_ANCHOR"
    assert track.decision_for("low-2d").reason == "LOW_2D_SCORE"
    assert track.decision_for("bad-reprojection").reason == "HIGH_REPROJECTION_ERROR"
    assert track.decision_for("bone-outlier").reason == "BONE_LENGTH_OUTLIER"
    assert track.decision_for("weak-triangulation").reason == "LOW_TRIANGULATION_QUALITY"


def test_equal_scores_have_deterministic_order_independent_of_input_order() -> None:
    selector = ManoAnchorSelector(top_k=2, min_time_separation_ns=0)
    frames = [
        _evidence(
            track_id="track-a",
            frame_id="frame-b",
            timestamp_ns=100,
        ),
        _evidence(
            track_id="track-a",
            frame_id="frame-a",
            timestamp_ns=100,
        ),
    ]

    forward = selector.select(frames)
    reversed_input = selector.select(list(reversed(frames)))

    assert forward == reversed_input
    assert forward.method_id == "mano_anchor_quality_v1"
    assert tuple(anchor.frame_id for anchor in forward.for_track("track-a").anchors) == (
        "frame-a",
        "frame-b",
    )
    assert tuple(anchor.selection_rank for anchor in forward.for_track("track-a").anchors) == (1, 2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"score_2d": float("nan")},
        {"reprojection_px": float("inf")},
        {"bone_outlier_fraction": float("nan")},
        {"triangulation_quality": float("inf")},
    ],
)
def test_non_finite_frame_evidence_is_rejected(overrides: dict[str, float]) -> None:
    selector = ManoAnchorSelector(top_k=1, min_time_separation_ns=0)

    result = selector.select(
        [
            _evidence(
                track_id="track-a",
                frame_id="bad-frame",
                timestamp_ns=0,
                **overrides,
            )
        ]
    )
    track = result.for_track("track-a")
    decision = track.decision_for("bad-frame")

    assert track.status == "NO_ACCEPTABLE_ANCHOR"
    assert decision.accepted is False
    assert decision.quality_accepted is False
    assert decision.reason == "NON_FINITE_EVIDENCE"
    assert decision.score is None
    assert decision.score_components == ()
