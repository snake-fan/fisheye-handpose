from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.contracts import WorkerError  # noqa: E402
from fisheye_h20_worker.mano_fitting import ManoTrackFitter  # noqa: E402


def _target() -> tuple[list[list[float]], list[str]]:
    points = [[0.001 * index, -0.002 * index, 0.5] for index in range(21)]
    return points, ["VALID"] * 21


def _fit_result(*, side: str, rmse_m: float, marker: float) -> dict[str, Any]:
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
        "joint_residuals_m": [rmse_m] * 21,
        "converged": True,
    }


class _Runtime:
    def __init__(self, results: dict[tuple[str, str], dict[str, Any] | Exception]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def fit_mano(self, models: object, **kwargs: Any) -> dict[str, Any]:
        del models
        self.calls.append(kwargs)
        result = self.results[(kwargs["side"], kwargs["seed_id"])]
        if isinstance(result, Exception):
            raise result
        return result


def test_cold_start_tries_both_sides_and_seeds_then_accepts_lowest_passing_rmse() -> None:
    runtime = _Runtime(
        {
            ("left", "mean"): _fit_result(side="left", rmse_m=0.012, marker=0.11),
            ("left", "relaxed"): _fit_result(side="left", rmse_m=0.009, marker=0.12),
            ("right", "mean"): _fit_result(side="right", rmse_m=0.006, marker=0.21),
            ("right", "relaxed"): _fit_result(side="right", rmse_m=0.008, marker=0.22),
        }
    )
    relaxed_seed = {"hand_pose": [0.25] * 45}
    fitter = ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None, "relaxed": relaxed_seed},
        rmse_gate_m=0.01,
        max_gap_ms=100.0,
        device="cuda:0",
        iterations=200,
        learning_rate=0.01,
    )
    target, validity = _target()

    decision = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )

    assert decision["status"] == "ACCEPTED"
    assert decision["fit"] == _fit_result(side="right", rmse_m=0.006, marker=0.21)
    assert decision["selected_attempt_index"] == 2
    assert [
        (call["side"], call["seed_id"], call["initial_parameters"]) for call in runtime.calls
    ] == [
        ("left", "mean", None),
        ("left", "relaxed", relaxed_seed),
        ("right", "mean", None),
        ("right", "relaxed", relaxed_seed),
    ]
    assert all(call["fixed_beta"] is None for call in runtime.calls)
    assert decision["fit"]["iterations_run"] == 17
    assert decision["fit"]["joint_residuals_m"] == [0.006] * 21


def test_next_frame_warm_starts_only_locked_side_from_complete_accepted_state() -> None:
    runtime = _Runtime(
        {
            ("left", "mean"): _fit_result(side="left", rmse_m=0.02, marker=0.11),
            ("right", "mean"): _fit_result(side="right", rmse_m=0.006, marker=0.21),
        }
    )
    fitter = ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None},
        rmse_gate_m=0.01,
        max_gap_ms=100.0,
        device="cuda:0",
        iterations=200,
        learning_rate=0.01,
    )
    target, validity = _target()
    fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    runtime.results[("right", "accepted_state")] = _fit_result(
        side="right", rmse_m=0.004, marker=0.31
    )
    runtime.calls.clear()

    decision = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_033_000_000,
    )

    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["side"] == "right"
    assert call["seed_id"] == "accepted_state"
    assert call["fixed_beta"] == [0.21] * 10
    assert call["initial_parameters"] == {
        "global_orient": [0.21, 0.0, 0.0],
        "hand_pose": [0.21] * 45,
        "transl": [0.21, 0.31, 0.41000000000000003],
        "beta": [0.21] * 10,
    }
    assert decision["status"] == "ACCEPTED"
    assert decision["init_source"] == "ACCEPTED_STATE"
    assert decision["predecessor_timestamp_ns"] == 1_000_000_000
    assert decision["fit"] == _fit_result(side="right", rmse_m=0.004, marker=0.31)


def test_rejected_warm_start_preserves_best_diagnostics_without_polluting_state() -> None:
    runtime = _Runtime(
        {
            ("left", "mean"): _fit_result(side="left", rmse_m=0.02, marker=0.11),
            ("right", "mean"): _fit_result(side="right", rmse_m=0.006, marker=0.21),
        }
    )
    fitter = ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None},
        rmse_gate_m=0.01,
        max_gap_ms=100.0,
        device="cuda:0",
        iterations=200,
        learning_rate=0.01,
    )
    target, validity = _target()
    fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    rejected_fit = _fit_result(side="right", rmse_m=0.018, marker=0.91)
    runtime.results[("right", "accepted_state")] = rejected_fit
    runtime.results[("right", "mean")] = _fit_result(side="right", rmse_m=0.019, marker=0.92)

    rejected = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_030_000_000,
    )

    assert rejected["status"] == "REJECTED"
    assert rejected["fit"] is None
    assert rejected["best_attempt"]["result"] == rejected_fit
    assert rejected["best_attempt"]["result"]["best_loss"] == 0.92
    runtime.results[("right", "accepted_state")] = _fit_result(
        side="right", rmse_m=0.005, marker=0.31
    )
    runtime.calls.clear()

    recovered = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_060_000_000,
    )

    assert recovered["status"] == "ACCEPTED"
    assert runtime.calls[0]["initial_parameters"]["hand_pose"] == [0.21] * 45
    assert runtime.calls[0]["fixed_beta"] == [0.21] * 10
    assert recovered["predecessor_timestamp_ns"] == 1_000_000_000


def test_rejected_warm_start_retries_a_cold_seed_on_the_locked_side() -> None:
    runtime = _Runtime(
        {
            ("left", "mean"): _fit_result(side="left", rmse_m=0.02, marker=0.11),
            ("right", "mean"): _fit_result(side="right", rmse_m=0.006, marker=0.21),
        }
    )
    fitter = ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None},
        rmse_gate_m=0.01,
        max_gap_ms=100.0,
        device="cuda:0",
        iterations=200,
        learning_rate=0.01,
    )
    target, validity = _target()
    fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    runtime.results[("right", "accepted_state")] = _fit_result(
        side="right", rmse_m=0.018, marker=0.31
    )
    runtime.results[("right", "mean")] = _fit_result(side="right", rmse_m=0.007, marker=0.41)
    runtime.calls.clear()

    recovered = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_033_000_000,
    )

    assert recovered["status"] == "ACCEPTED"
    assert [(call["side"], call["seed_id"]) for call in runtime.calls] == [
        ("right", "accepted_state"),
        ("right", "mean"),
    ]
    assert runtime.calls[1]["initial_parameters"] is None
    assert runtime.calls[1]["fixed_beta"] == [0.21] * 10
    assert [attempt["init_source"] for attempt in recovered["attempts"]] == [
        "ACCEPTED_STATE",
        "COLD_RECOVERY",
    ]
    assert recovered["fit"]["hand_pose"] == [0.41] * 45


def test_runtime_error_is_reported_and_does_not_replace_accepted_state() -> None:
    runtime = _Runtime(
        {
            ("left", "mean"): _fit_result(side="left", rmse_m=0.02, marker=0.11),
            ("right", "mean"): _fit_result(side="right", rmse_m=0.006, marker=0.21),
        }
    )
    fitter = ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None},
        rmse_gate_m=0.01,
        max_gap_ms=100.0,
        device="cuda:0",
        iterations=200,
        learning_rate=0.01,
    )
    target, validity = _target()
    fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    runtime.results[("right", "accepted_state")] = RuntimeError("simulated optimizer OOM")
    runtime.results[("right", "mean")] = RuntimeError("simulated cold recovery OOM")

    failed = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_030_000_000,
    )

    assert failed["status"] == "ERROR"
    assert failed["fit"] is None
    assert failed["best_attempt"] is None
    assert failed["attempts"] == [
        {
            "side": "right",
            "seed_id": "accepted_state",
            "init_source": "ACCEPTED_STATE",
            "status": "ERROR",
            "rmse_m": None,
            "result": None,
            "error": {"type": "RuntimeError", "message": "simulated optimizer OOM"},
        },
        {
            "side": "right",
            "seed_id": "mean",
            "init_source": "COLD_RECOVERY",
            "status": "ERROR",
            "rmse_m": None,
            "result": None,
            "error": {
                "type": "RuntimeError",
                "message": "simulated cold recovery OOM",
            },
        },
    ]
    runtime.results[("right", "accepted_state")] = _fit_result(
        side="right", rmse_m=0.005, marker=0.31
    )
    runtime.calls.clear()

    recovered = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_060_000_000,
    )

    assert recovered["status"] == "ACCEPTED"
    assert runtime.calls[0]["initial_parameters"]["global_orient"] == [0.21, 0.0, 0.0]
    assert recovered["predecessor_timestamp_ns"] == 1_000_000_000


def test_gap_expiry_drops_side_lock_and_restarts_all_cold_hypotheses() -> None:
    runtime = _Runtime(
        {
            ("left", "mean"): _fit_result(side="left", rmse_m=0.02, marker=0.11),
            ("right", "mean"): _fit_result(side="right", rmse_m=0.006, marker=0.21),
        }
    )
    fitter = ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None},
        rmse_gate_m=0.01,
        max_gap_ms=50.0,
        device="cuda:0",
        iterations=200,
        learning_rate=0.01,
    )
    target, validity = _target()
    fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    runtime.results[("left", "mean")] = _fit_result(side="left", rmse_m=0.005, marker=0.41)
    runtime.results[("right", "mean")] = _fit_result(side="right", rmse_m=0.008, marker=0.42)
    runtime.calls.clear()

    restarted = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_060_000_000,
    )

    assert [(call["side"], call["seed_id"]) for call in runtime.calls] == [
        ("left", "mean"),
        ("right", "mean"),
    ]
    assert all(call["initial_parameters"] is None for call in runtime.calls)
    assert all(call["fixed_beta"] is None for call in runtime.calls)
    assert restarted["status"] == "ACCEPTED"
    assert restarted["fit"]["side"] == "left"
    assert restarted["init_source"] == "COLD_START"
    assert restarted["reset_reason"] == "GAP_EXCEEDED"
    assert restarted["predecessor_timestamp_ns"] == 1_000_000_000


def test_non_monotonic_time_is_rejected_until_track_is_explicitly_reset() -> None:
    runtime = _Runtime(
        {
            ("left", "mean"): _fit_result(side="left", rmse_m=0.02, marker=0.11),
            ("right", "mean"): _fit_result(side="right", rmse_m=0.006, marker=0.21),
        }
    )
    fitter = ManoTrackFitter(
        runtime=runtime,
        models=object(),
        cold_start_seeds={"mean": None},
        rmse_gate_m=0.01,
        max_gap_ms=100.0,
        device="cuda:0",
        iterations=200,
        learning_rate=0.01,
    )
    target, validity = _target()
    fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_000_000_000,
    )
    runtime.results[("right", "accepted_state")] = _fit_result(
        side="right", rmse_m=0.02, marker=0.91
    )
    runtime.results[("right", "mean")] = _fit_result(side="right", rmse_m=0.021, marker=0.92)
    fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=1_030_000_000,
    )
    call_count = len(runtime.calls)

    with pytest.raises(WorkerError, match="monotonically non-decreasing"):
        fitter.fit_frame(
            track_id="track-0007",
            target_xyz_m=target,
            validity=validity,
            timestamp_ns=1_020_000_000,
        )
    assert len(runtime.calls) == call_count

    fitter.reset("track-0007")
    runtime.results[("left", "mean")] = _fit_result(side="left", rmse_m=0.005, marker=0.41)
    runtime.results[("right", "mean")] = _fit_result(side="right", rmse_m=0.008, marker=0.42)
    runtime.calls.clear()
    restarted = fitter.fit_frame(
        track_id="track-0007",
        target_xyz_m=target,
        validity=validity,
        timestamp_ns=900_000_000,
    )

    assert [(call["side"], call["seed_id"]) for call in runtime.calls] == [
        ("left", "mean"),
        ("right", "mean"),
    ]
    assert restarted["reset_reason"] == "EXPLICIT_RESET"
    assert restarted["predecessor_timestamp_ns"] == 1_000_000_000
