from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.candidates import CandidatePolicy  # noqa: E402
from fisheye_h20_worker.contracts import WorkerError  # noqa: E402


def test_no_detector_instances_produces_an_empty_candidate_pool() -> None:
    batch = CandidatePolicy().classify(
        bboxes=[],
        scores=[],
        labels=[],
        category_id=0,
        view_id="left",
    )

    assert batch.decisions == ()
    assert batch.candidate_pool == ()


def test_seed_candidate_preserves_detector_identity_and_waits_for_association() -> None:
    batch = CandidatePolicy().classify(
        bboxes=[[11.0, 12.0, 41.0, 52.0]],
        scores=[0.86],
        labels=[7],
        category_id=7,
        view_id="right",
    )

    assert len(batch.decisions) == 1
    decision = batch.decisions[0]
    assert decision.candidate_id == "right-det-0000"
    assert decision.source_index == 0
    assert decision.bbox_xyxy == (11.0, 12.0, 41.0, 52.0)
    assert decision.score == 0.86
    assert decision.label == 7
    assert decision.classification == "SEED"
    assert decision.reason == "SCORE_MEETS_SEED_THRESHOLD"
    assert decision.eligible_for_association is True
    assert decision.final_selection is None
    assert batch.candidate_pool == (decision,)


def test_threshold_boundaries_are_seed_and_recovery_without_final_selection() -> None:
    batch = CandidatePolicy(seed_threshold=0.30, recovery_threshold=0.20).classify(
        bboxes=[[0.0, 1.0, 20.0, 31.0], [40.0, 2.0, 60.0, 32.0]],
        scores=[0.30, 0.20],
        labels=[0, 0],
        category_id=0,
        view_id="left",
    )

    assert [item.classification for item in batch.decisions] == ["SEED", "RECOVERY"]
    assert [item.reason for item in batch.decisions] == [
        "SCORE_MEETS_SEED_THRESHOLD",
        "SCORE_MEETS_RECOVERY_THRESHOLD",
    ]
    assert [item.source_index for item in batch.candidate_pool] == [0, 1]
    assert all(item.eligible_for_association for item in batch.candidate_pool)
    assert all(item.final_selection is None for item in batch.candidate_pool)


def test_high_score_from_the_wrong_detector_category_is_rejected() -> None:
    batch = CandidatePolicy().classify(
        bboxes=[[1.0, 2.0, 31.0, 42.0]],
        scores=[0.99],
        labels=[4],
        category_id=0,
        view_id="left",
    )

    decision = batch.decisions[0]
    assert decision.classification == "REJECTED"
    assert decision.reason == "CATEGORY_MISMATCH"
    assert decision.eligible_for_association is False
    assert batch.candidate_pool == ()


def test_four_candidate_pool_is_score_sorted_with_source_index_as_tie_breaker() -> None:
    batch = CandidatePolicy(max_candidates=4).classify(
        bboxes=[
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 0.0, 30.0, 10.0],
            [40.0, 0.0, 50.0, 10.0],
            [60.0, 0.0, 70.0, 10.0],
        ],
        scores=[0.21, 0.90, 0.90, 0.31],
        labels=[0, 0, 0, 0],
        category_id=0,
        view_id="right",
    )

    assert [item.source_index for item in batch.candidate_pool] == [1, 2, 3, 0]
    assert [item.classification for item in batch.candidate_pool] == [
        "SEED",
        "SEED",
        "SEED",
        "RECOVERY",
    ]


def test_pool_cap_rejects_lower_ranked_candidate_without_selecting_final_hands() -> None:
    batch = CandidatePolicy(max_candidates=4).classify(
        bboxes=[[float(index), 0.0, float(index + 1), 2.0] for index in range(5)],
        scores=[0.55, 0.91, 0.61, 0.81, 0.71],
        labels=[0, 0, 0, 0, 0],
        category_id=0,
        view_id="left",
    )

    assert [item.source_index for item in batch.candidate_pool] == [1, 3, 4, 2]
    overflow = batch.decisions[0]
    assert overflow.classification == "REJECTED"
    assert overflow.reason == "CANDIDATE_LIMIT_EXCEEDED"
    assert overflow.eligible_for_association is False
    assert all(item.final_selection is None for item in batch.decisions)


def test_duplicate_looking_boxes_are_left_for_geometry_and_temporal_gates() -> None:
    duplicate_box = [10.0, 12.0, 50.0, 62.0]
    batch = CandidatePolicy().classify(
        bboxes=[duplicate_box, duplicate_box],
        scores=[0.83, 0.77],
        labels=[0, 0],
        category_id=0,
        view_id="right",
    )

    assert [item.source_index for item in batch.candidate_pool] == [0, 1]
    assert [item.classification for item in batch.candidate_pool] == ["SEED", "SEED"]
    assert all(item.final_selection is None for item in batch.candidate_pool)


def test_score_just_below_recovery_threshold_is_rejected() -> None:
    batch = CandidatePolicy(recovery_threshold=0.20).classify(
        bboxes=[[1.0, 1.0, 9.0, 9.0]],
        scores=[0.199999],
        labels=[0],
        category_id=0,
        view_id="left",
    )

    decision = batch.decisions[0]
    assert decision.classification == "REJECTED"
    assert decision.reason == "SCORE_BELOW_RECOVERY_THRESHOLD"
    assert batch.candidate_pool == ()


def test_detector_instance_arrays_must_have_matching_counts() -> None:
    with pytest.raises(WorkerError, match="instance counts"):
        CandidatePolicy().classify(
            bboxes=[[0.0, 0.0, 10.0, 10.0]],
            scores=[],
            labels=[0],
            category_id=0,
            view_id="left",
        )


def test_each_detector_bbox_must_have_xyxy_shape() -> None:
    with pytest.raises(WorkerError, match="bbox shape"):
        CandidatePolicy().classify(
            bboxes=[[0.0, 1.0, 2.0]],
            scores=[0.8],
            labels=[0],
            category_id=0,
            view_id="left",
        )


@pytest.mark.parametrize(
    ("bboxes", "scores"),
    [
        ([[0.0, 1.0, float("nan"), 8.0]], [0.8]),
        ([[0.0, 1.0, 7.0, 8.0]], [float("inf")]),
    ],
)
def test_non_finite_detector_numbers_are_contract_errors(
    bboxes: list[list[float]], scores: list[float]
) -> None:
    with pytest.raises(WorkerError, match="finite"):
        CandidatePolicy().classify(
            bboxes=bboxes,
            scores=scores,
            labels=[0],
            category_id=0,
            view_id="left",
        )


@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        (np.asarray([[0.8]]), np.asarray([0])),
        (np.asarray([0.8]), np.asarray([[0]])),
    ],
)
def test_detector_scores_and_labels_must_be_flat_vectors(
    scores: np.ndarray, labels: np.ndarray
) -> None:
    with pytest.raises(WorkerError, match="shape"):
        CandidatePolicy().classify(
            bboxes=np.asarray([[0.0, 1.0, 7.0, 8.0]]),
            scores=scores,
            labels=labels,
            category_id=0,
            view_id="left",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed_threshold": 0.19, "recovery_threshold": 0.20},
        {"seed_threshold": float("nan")},
        {"max_candidates": 0},
        {"max_candidates": 5},
    ],
)
def test_policy_configuration_cannot_invert_thresholds_or_exceed_top_four(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError, match="candidate policy"):
        CandidatePolicy(**kwargs)


def test_detector_labels_must_be_integer_category_ids() -> None:
    with pytest.raises(WorkerError, match="label"):
        CandidatePolicy().classify(
            bboxes=[[0.0, 1.0, 7.0, 8.0]],
            scores=[0.8],
            labels=[0.5],
            category_id=0,
            view_id="left",
        )
