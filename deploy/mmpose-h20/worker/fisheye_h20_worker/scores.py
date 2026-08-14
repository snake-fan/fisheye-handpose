"""Operational score semantics for the pinned RTMPose Hand5 backend.

The pinned SimCC codec is configured with ``normalize=False`` and decodes the
minimum x/y peak without softmax.  That raw response is useful provenance, but
it is not a probability and is not bounded by one.  The v1 stereo pipeline
therefore preserves the model response and derives a separate bounded quality
weight for thresholds and heuristic covariance weighting.
"""

from __future__ import annotations

import math

from .contracts import WorkerError

MODEL_SCORE_SEMANTICS = "RTMPOSE_SIMCC_MAX_RESPONSE_UNCALIBRATED"
QUALITY_WEIGHT_METHOD = "CLIP_0_1_V1"
QUALITY_WEIGHT_STATUS = "HEURISTIC_UNCALIBRATED"


def quality_weight(model_score: object) -> float:
    """Map one finite raw SimCC response to the v1 bounded quality weight."""

    if (
        isinstance(model_score, bool)
        or not isinstance(model_score, (int, float))
        or not math.isfinite(float(model_score))
    ):
        raise WorkerError("model keypoint score must be a finite number")
    return min(max(float(model_score), 0.0), 1.0)


__all__ = [
    "MODEL_SCORE_SEMANTICS",
    "QUALITY_WEIGHT_METHOD",
    "QUALITY_WEIGHT_STATUS",
    "quality_weight",
]
