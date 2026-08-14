from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.mano_fitting import (  # noqa: E402
    ROBUST_GATE_METHOD,
    ROBUST_GATE_STATUS,
    ManoTrackFitter,
)


def _target(*, valid_count: int = 21) -> tuple[list[list[float] | None], list[str]]:
    points: list[list[float] | None] = [[0.001 * index, -0.002 * index, 0.5] for index in range(21)]
    validity = ["VALID"] * valid_count + ["INVALID_MASK"] * (21 - valid_count)
    for index in range(valid_count, 21):
        points[index] = None
    return points, validity


def _fit_result(
    *,
    rmse_m: float,
    residuals_m: list[float | None],
    marker: float,
    side: str = "right",
) -> dict[str, Any]:
    return {
        "side": side,
        "mapping_id": "mano-v1.2-j16-tips-to-fhp21/v1",
        "landmarks_xyz_m": [[marker, 0.0, 0.5] for _ in range(21)],
        "validity": ["VALID"] * 21,
        "rmse_m": rmse_m,
        "global_orient": [marker, 0.0, 0.0],
        "hand_pose": [marker] * 45,
        "transl": [marker, marker + 0.1, marker + 0.2],
        "beta": [marker] * 10,
        "iterations_run": 17,
        "best_loss": marker + 0.01,
        "final_loss": marker + 0.02,
        "joint_residuals_m": residuals_m,
        "converged": True,
    }


class _RobustRuntime:
    def __init__(
        self,
        *,
        first: dict[str, Any],
        second: dict[str, Any] | Exception | None = None,
    ) -> None:
        self.first = first
        self.second = second
        self.calls: list[dict[str, Any]] = []

    def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
        del models
        self.calls.append(kwargs)
        if kwargs.get("joint_weights") is None:
            result = dict(self.first)
            result["side"] = kwargs["side"]
            return result
        if isinstance(self.second, Exception):
            raise self.second
        if self.second is None:
            raise AssertionError("unexpected robust refit")
        result = dict(self.second)
        result["side"] = kwargs["side"]
        return result


def _fitter(runtime: _RobustRuntime) -> ManoTrackFitter:
    return ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None},
        rmse_gate_m=0.02,
        max_gap_ms=100.0,
        device="cpu",
        iterations=200,
        learning_rate=0.01,
    )


def _fit_one(runtime: _RobustRuntime, *, valid_count: int = 21) -> dict[str, Any]:
    target, validity = _target(valid_count=valid_count)
    return _fitter(runtime).fit_frame(
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )


def test_fit_within_20mm_gate_does_not_run_a_second_fit() -> None:
    first = _fit_result(rmse_m=0.015, residuals_m=[0.015] * 21, marker=0.1)
    runtime = _RobustRuntime(first=first)

    decision = _fit_one(runtime)

    assert decision["status"] == "ACCEPTED"
    assert len(runtime.calls) == 2  # one normal fit for each cold-start handedness
    assert all(call.get("joint_weights") is None for call in runtime.calls)
    gate = decision["gate"]
    assert gate["method"] == ROBUST_GATE_METHOD
    assert gate["status"] == ROBUST_GATE_STATUS
    assert gate["triggered"] is False
    assert gate["raw_rmse_m"] == 0.015
    assert gate["full_rmse_m"] == 0.015
    assert gate["weighted_rmse_m"] == pytest.approx(0.015)
    assert gate["inlier_rmse_m"] == pytest.approx(0.015)
    assert gate["effective_joint_count"] == 21
    assert gate["stage_iterations"] == [{"stage": "FULL_HUBER", "iterations_run": 17}]


def test_one_or_two_residual_outliers_are_trimmed_then_refit_from_first_best_state() -> None:
    first = _fit_result(
        rmse_m=0.029,
        residuals_m=[0.010] * 19 + [0.090, 0.070],
        marker=0.1,
    )
    second = _fit_result(
        rmse_m=0.028,
        residuals_m=[0.009] * 19 + [0.088, 0.068],
        marker=0.2,
    )
    runtime = _RobustRuntime(first=first, second=second)

    decision = _fit_one(runtime)

    assert decision["status"] == "ACCEPTED"
    assert decision["fit"]["hand_pose"] == [0.2] * 45
    # left full + left weighted, then right full + right weighted
    assert len(runtime.calls) == 4
    for full_call, weighted_call in zip(runtime.calls[::2], runtime.calls[1::2], strict=True):
        assert full_call["joint_weights"] is None
        assert weighted_call["joint_weights"] == [1.0] * 19 + [0.0, 0.0]
        assert weighted_call["initial_parameters"] == {
            "global_orient": [0.1, 0.0, 0.0],
            "hand_pose": [0.1] * 45,
            "transl": [0.1, 0.2, 0.30000000000000004],
            "beta": [0.1] * 10,
        }
    gate = decision["gate"]
    assert gate["triggered"] is True
    assert gate["trimmed_joint_indices"] == [19, 20]
    assert gate["inlier_mask"] == [True] * 19 + [False, False]
    assert gate["effective_joint_count"] == 19
    assert gate["raw_rmse_m"] == 0.028
    assert gate["first_pass_rmse_m"] == 0.029
    assert gate["full_rmse_m"] == 0.028
    assert gate["inlier_rmse_m"] == pytest.approx(0.009)
    assert gate["weighted_rmse_m"] == pytest.approx(0.009)
    assert gate["reason"] == "ROBUST_INLIER_GATE_PASSED"
    assert gate["stage_iterations"] == [
        {"stage": "FULL_HUBER", "iterations_run": 17},
        {"stage": "WEIGHTED_REFIT", "iterations_run": 17},
    ]


def test_handedness_selection_compares_final_full_rmse_not_different_inlier_masks() -> None:
    class HandednessRuntime:
        def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
            del models
            side = kwargs["side"]
            if side == "left" and kwargs.get("joint_weights") is None:
                return _fit_result(
                    side="left",
                    rmse_m=0.030,
                    residuals_m=[0.010] * 19 + [0.090, 0.070],
                    marker=0.1,
                )
            if side == "left":
                return _fit_result(
                    side="left",
                    rmse_m=0.039,
                    residuals_m=[0.005] * 19 + [0.120, 0.110],
                    marker=0.2,
                )
            return _fit_result(
                side="right",
                rmse_m=0.019,
                residuals_m=[0.019] * 21,
                marker=0.3,
            )

    target, validity = _target()
    decision = _fitter(HandednessRuntime()).fit_frame(  # type: ignore[arg-type]
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )

    assert decision["status"] == "ACCEPTED"
    assert decision["fit"]["side"] == "right"
    assert decision["gate"]["triggered"] is False
    left_attempt = decision["attempts"][0]
    assert left_attempt["gate"]["inlier_rmse_m"] == pytest.approx(0.005)
    assert left_attempt["gate"]["full_rmse_m"] == 0.039
    assert decision["gate"]["full_rmse_m"] == 0.019


def test_a_single_residual_outlier_trims_only_that_joint() -> None:
    first = _fit_result(
        rmse_m=0.022,
        residuals_m=[0.010] * 20 + [0.090],
        marker=0.1,
    )
    second = _fit_result(
        rmse_m=0.019,
        residuals_m=[0.009] * 20 + [0.080],
        marker=0.2,
    )
    runtime = _RobustRuntime(first=first, second=second)

    decision = _fit_one(runtime)

    assert decision["status"] == "ACCEPTED"
    assert decision["gate"]["trimmed_joint_indices"] == [20]
    assert decision["gate"]["joint_weights"] == [1.0] * 20 + [0.0]
    assert decision["gate"]["effective_joint_count"] == 20


def test_fewer_than_15_valid_joints_are_not_trimmed() -> None:
    residuals: list[float | None] = [0.010] * 13 + [0.090] + [None] * 7
    runtime = _RobustRuntime(first=_fit_result(rmse_m=0.027, residuals_m=residuals, marker=0.1))

    decision = _fit_one(runtime, valid_count=14)

    assert decision["status"] == "REJECTED"
    assert len(runtime.calls) == 2
    assert all(call.get("joint_weights") is None for call in runtime.calls)
    assert decision["gate"]["reason"] == "INSUFFICIENT_EFFECTIVE_JOINTS"
    assert decision["gate"]["effective_joint_count"] == 14


def test_fewer_than_15_valid_joints_fail_support_even_when_rmse_is_low() -> None:
    residuals: list[float | None] = [0.010] * 14 + [None] * 7
    runtime = _RobustRuntime(first=_fit_result(rmse_m=0.010, residuals_m=residuals, marker=0.1))

    decision = _fit_one(runtime, valid_count=14)

    assert decision["status"] == "REJECTED"
    assert decision["gate"]["reason"] == "INSUFFICIENT_EFFECTIVE_JOINTS"
    assert decision["gate"]["effective_joint_count"] == 14


def test_inlier_fit_is_rejected_when_ordinary_full_rmse_exceeds_40mm_ceiling() -> None:
    first = _fit_result(
        rmse_m=0.060,
        residuals_m=[0.010] * 19 + [0.200, 0.180],
        marker=0.1,
    )
    second = _fit_result(
        rmse_m=0.041,
        residuals_m=[0.009] * 19 + [0.130, 0.120],
        marker=0.2,
    )
    runtime = _RobustRuntime(first=first, second=second)

    decision = _fit_one(runtime)

    assert decision["status"] == "REJECTED"
    assert decision["gate"]["inlier_rmse_m"] == pytest.approx(0.009)
    assert decision["gate"]["full_rmse_m"] == 0.041
    assert decision["gate"]["full_rmse_ceiling_m"] == 0.04
    assert decision["gate"]["reason"] == "FULL_RMSE_CEILING_EXCEEDED"


def test_weighted_refit_error_does_not_replace_the_last_accepted_state() -> None:
    good = _fit_result(rmse_m=0.010, residuals_m=[0.010] * 21, marker=0.1)
    bad = _fit_result(
        rmse_m=0.030,
        residuals_m=[0.010] * 19 + [0.090, 0.070],
        marker=0.9,
    )

    class StatefulRuntime:
        def __init__(self) -> None:
            self.phase = "good"
            self.calls: list[dict[str, Any]] = []

        def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
            del models
            self.calls.append(kwargs)
            if self.phase == "good":
                return good
            if kwargs.get("joint_weights") is not None:
                raise RuntimeError("simulated weighted optimizer failure")
            return bad

    runtime = StatefulRuntime()
    fitter = _fitter(runtime)  # type: ignore[arg-type]
    target, validity = _target()
    accepted = fitter.fit_frame(
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    assert accepted["status"] == "ACCEPTED"
    runtime.phase = "bad"

    failed = fitter.fit_frame(
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_030_000_000,
    )

    assert failed["status"] == "ERROR"
    assert failed["gate"]["reason"] == "ROBUST_REFIT_ERROR"
    runtime.phase = "good"
    runtime.calls.clear()
    recovered = fitter.fit_frame(
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_060_000_000,
    )
    assert recovered["status"] == "ACCEPTED"
    assert runtime.calls[0]["initial_parameters"]["hand_pose"] == [0.1] * 45


def test_finite_weighted_refit_rejection_does_not_replace_the_last_accepted_state() -> None:
    good = _fit_result(rmse_m=0.010, residuals_m=[0.010] * 21, marker=0.1)
    bad_first = _fit_result(
        rmse_m=0.030,
        residuals_m=[0.010] * 19 + [0.090, 0.070],
        marker=0.8,
    )
    bad_second = _fit_result(
        rmse_m=0.030,
        residuals_m=[0.025] * 19 + [0.080, 0.060],
        marker=0.9,
    )

    class StatefulRuntime:
        def __init__(self) -> None:
            self.phase = "good"
            self.calls: list[dict[str, Any]] = []

        def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
            del models
            self.calls.append(kwargs)
            if self.phase == "good":
                result = dict(good)
            elif kwargs.get("joint_weights") is None:
                result = dict(bad_first)
            else:
                result = dict(bad_second)
            result["side"] = kwargs["side"]
            return result

    runtime = StatefulRuntime()
    fitter = _fitter(runtime)  # type: ignore[arg-type]
    target, validity = _target()
    accepted = fitter.fit_frame(
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    assert accepted["status"] == "ACCEPTED"
    runtime.phase = "bad"

    rejected = fitter.fit_frame(
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_030_000_000,
    )

    assert rejected["status"] == "REJECTED"
    assert rejected["gate"]["reason"] == "INLIER_RMSE_GATE_EXCEEDED"
    runtime.phase = "good"
    runtime.calls.clear()
    recovered = fitter.fit_frame(
        track_id="track-0000",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_060_000_000,
    )
    assert recovered["status"] == "ACCEPTED"
    assert runtime.calls[0]["initial_parameters"]["hand_pose"] == [0.1] * 45
