"""Deterministic detector-candidate policy before stereo association."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .contracts import WorkerError


@dataclass(frozen=True)
class CandidateDecision:
    """One traceable detector instance decision."""

    candidate_id: str
    source_index: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: int
    classification: str
    reason: str
    eligible_for_association: bool
    final_selection: None = None


@dataclass(frozen=True)
class CandidateBatch:
    """All detector decisions and the bounded pool offered to association."""

    decisions: tuple[CandidateDecision, ...]
    candidate_pool: tuple[CandidateDecision, ...]


@dataclass(frozen=True)
class CandidatePolicy:
    """Classify raw detector instances without making a final hand decision."""

    seed_threshold: float = 0.30
    recovery_threshold: float = 0.20
    max_candidates: int = 4

    def __post_init__(self) -> None:
        numeric_types = (int, float, np.integer, np.floating)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, numeric_types)
            for value in (self.seed_threshold, self.recovery_threshold)
        ):
            raise ValueError("candidate policy thresholds must be finite numbers")
        seed = float(self.seed_threshold)
        recovery = float(self.recovery_threshold)
        if not (math.isfinite(seed) and math.isfinite(recovery) and 0.0 <= recovery <= seed <= 1.0):
            raise ValueError("candidate policy thresholds must satisfy 0 <= recovery <= seed <= 1")
        if (
            isinstance(self.max_candidates, (bool, np.bool_))
            or not isinstance(self.max_candidates, (int, np.integer))
            or not 1 <= int(self.max_candidates) <= 4
        ):
            raise ValueError("candidate policy max_candidates must be an integer in [1, 4]")
        object.__setattr__(self, "seed_threshold", seed)
        object.__setattr__(self, "recovery_threshold", recovery)
        object.__setattr__(self, "max_candidates", int(self.max_candidates))

    def classify(
        self,
        *,
        bboxes: Any,
        scores: Any,
        labels: Any,
        category_id: int,
        view_id: str,
    ) -> CandidateBatch:
        try:
            bbox_values = np.asarray(bboxes)
            score_values = np.asarray(scores)
            label_values = np.asarray(labels)
        except (TypeError, ValueError) as exc:
            raise WorkerError("detector output shape is invalid") from exc
        if bbox_values.size and (bbox_values.ndim != 2 or bbox_values.shape[1] != 4):
            raise WorkerError("detector bbox shape must be (N, 4)")
        if score_values.ndim != 1 or label_values.ndim != 1:
            raise WorkerError("detector score and label shape must be (N,)")
        if not (len(bbox_values) == len(score_values) == len(label_values)):
            raise WorkerError("detector outputs have inconsistent instance counts")
        if len(bbox_values) == 0:
            return CandidateBatch(decisions=(), candidate_pool=())
        decisions: list[CandidateDecision] = []
        for index, (bbox, score_value, label_value) in enumerate(
            zip(bbox_values, score_values, label_values, strict=True)
        ):
            try:
                bbox_size = len(bbox)
            except TypeError as exc:
                raise WorkerError("detector bbox shape must be (4,)") from exc
            if bbox_size != 4:
                raise WorkerError("detector bbox shape must be (4,)")
            bbox_xyxy = tuple(float(value) for value in bbox)
            score = float(score_value)
            if not all(math.isfinite(value) for value in (*bbox_xyxy, score)):
                raise WorkerError("detector bbox and score values must be finite")
            if isinstance(label_value, (bool, np.bool_)) or not isinstance(
                label_value, (int, np.integer)
            ):
                raise WorkerError("detector label values must be integers")
            label = int(label_value)
            if label != category_id:
                classification = "REJECTED"
                reason = "CATEGORY_MISMATCH"
                eligible = False
            elif score >= self.seed_threshold:
                classification = "SEED"
                reason = "SCORE_MEETS_SEED_THRESHOLD"
                eligible = True
            elif score >= self.recovery_threshold:
                classification = "RECOVERY"
                reason = "SCORE_MEETS_RECOVERY_THRESHOLD"
                eligible = True
            else:
                classification = "REJECTED"
                reason = "SCORE_BELOW_RECOVERY_THRESHOLD"
                eligible = False
            decisions.append(
                CandidateDecision(
                    candidate_id=f"{view_id}-det-{index:04d}",
                    source_index=index,
                    bbox_xyxy=bbox_xyxy,
                    score=score,
                    label=label,
                    classification=classification,
                    reason=reason,
                    eligible_for_association=eligible,
                )
            )
        ranked = sorted(
            (decision for decision in decisions if decision.eligible_for_association),
            key=lambda decision: (-decision.score, decision.source_index),
        )
        selected_indexes = {decision.source_index for decision in ranked[: self.max_candidates]}
        decisions = [
            decision
            if not decision.eligible_for_association or decision.source_index in selected_indexes
            else replace(
                decision,
                classification="REJECTED",
                reason="CANDIDATE_LIMIT_EXCEEDED",
                eligible_for_association=False,
            )
            for decision in decisions
        ]
        pool = sorted(
            (decision for decision in decisions if decision.eligible_for_association),
            key=lambda decision: (-decision.score, decision.source_index),
        )
        return CandidateBatch(decisions=tuple(decisions), candidate_pool=tuple(pool))


__all__ = ["CandidateBatch", "CandidateDecision", "CandidatePolicy"]
