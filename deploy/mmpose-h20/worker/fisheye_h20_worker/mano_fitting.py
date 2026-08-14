"""Track-local accepted-state control for frame-wise MANO fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .contracts import WorkerError


@dataclass(frozen=True)
class _AcceptedState:
    side: str
    timestamp_ns: int
    global_orient: tuple[float, ...]
    hand_pose: tuple[float, ...]
    transl: tuple[float, ...]
    beta: tuple[float, ...]


class ManoTrackFitter:
    """Choose MANO hypotheses and retain only gate-accepted parameters per track."""

    def __init__(
        self,
        *,
        runtime: Any,
        models: object,
        cold_start_seeds: Mapping[str, dict[str, Any] | None],
        rmse_gate_m: float,
        max_gap_ms: float,
        device: str,
        iterations: int,
        learning_rate: float,
    ) -> None:
        if not cold_start_seeds:
            raise WorkerError("MANO cold_start_seeds must not be empty")
        if not math.isfinite(rmse_gate_m) or rmse_gate_m <= 0.0:
            raise WorkerError("MANO rmse_gate_m must be positive")
        if not math.isfinite(max_gap_ms) or max_gap_ms <= 0.0:
            raise WorkerError("MANO max_gap_ms must be positive")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise WorkerError("MANO iterations must be a positive integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise WorkerError("MANO learning_rate must be positive")
        if not isinstance(device, str) or not device:
            raise WorkerError("MANO device must be non-empty")
        for seed_id in cold_start_seeds:
            if not isinstance(seed_id, str) or not seed_id:
                raise WorkerError("MANO seed_id must be non-empty")

        self._runtime = runtime
        self._models = models
        self._cold_start_seeds = tuple(
            (seed_id, deepcopy(parameters)) for seed_id, parameters in cold_start_seeds.items()
        )
        self._rmse_gate_m = float(rmse_gate_m)
        self._max_gap_ns = int(max_gap_ms * 1_000_000)
        self._device = device
        self._iterations = iterations
        self._learning_rate = float(learning_rate)
        self._accepted: dict[str, _AcceptedState] = {}
        self._last_timestamps: dict[str, int] = {}
        self._pending_resets: dict[str, int | None] = {}

    def reset(self, track_id: str) -> None:
        """Forget one track's chronology and accepted fit before a sequence restart."""
        if not isinstance(track_id, str) or not track_id:
            raise WorkerError("MANO track_id must be non-empty")
        accepted = self._accepted.pop(track_id, None)
        self._last_timestamps.pop(track_id, None)
        predecessor = accepted.timestamp_ns if accepted is not None else None
        self._pending_resets[track_id] = predecessor

    def fit_frame(
        self,
        *,
        track_id: str,
        target_xyz_m: list[list[float] | None],
        validity: list[str],
        timestamp_ns: int,
    ) -> dict[str, Any]:
        """Fit one track frame and update warm-start state only after the RMSE gate."""
        if not isinstance(track_id, str) or not track_id:
            raise WorkerError("MANO track_id must be non-empty")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise WorkerError("MANO timestamp_ns must be an integer")
        last_timestamp_ns = self._last_timestamps.get(track_id)
        if last_timestamp_ns is not None and timestamp_ns < last_timestamp_ns:
            raise WorkerError("MANO timestamps must be monotonically non-decreasing")
        self._last_timestamps[track_id] = timestamp_ns

        accepted_state = self._accepted.get(track_id)
        if track_id in self._pending_resets:
            reset_reason: str | None = "EXPLICIT_RESET"
            predecessor_timestamp_ns = self._pending_resets.pop(track_id)
        else:
            reset_reason = None
            predecessor_timestamp_ns = None
        if (
            accepted_state is not None
            and timestamp_ns - accepted_state.timestamp_ns > self._max_gap_ns
        ):
            reset_reason = "GAP_EXCEEDED"
            predecessor_timestamp_ns = accepted_state.timestamp_ns
            del self._accepted[track_id]
            accepted_state = None
        if accepted_state is None:
            init_source = "COLD_START"
            hypotheses = [
                (side, seed_id, parameters, None, "COLD_START")
                for side in ("left", "right")
                for seed_id, parameters in self._cold_start_seeds
            ]
        else:
            init_source = "ACCEPTED_STATE"
            predecessor_timestamp_ns = accepted_state.timestamp_ns
            initial_parameters = {
                "global_orient": list(accepted_state.global_orient),
                "hand_pose": list(accepted_state.hand_pose),
                "transl": list(accepted_state.transl),
                "beta": list(accepted_state.beta),
            }
            hypotheses = [
                (
                    accepted_state.side,
                    "accepted_state",
                    initial_parameters,
                    list(accepted_state.beta),
                    "ACCEPTED_STATE",
                )
            ]

        attempts: list[dict[str, Any]] = []
        passing: list[tuple[float, int, dict[str, Any]]] = []
        scored: list[tuple[float, int]] = []

        def run_hypotheses(values: list[tuple[str, str, Any, Any, str]]) -> None:
            for side, seed_id, initial_parameters, fixed_beta, attempt_source in values:
                try:
                    result = self._runtime.fit_mano(
                        self._models,
                        side=side,
                        target_xyz_m=target_xyz_m,
                        validity=validity,
                        fixed_beta=fixed_beta,
                        device=self._device,
                        iterations=self._iterations,
                        learning_rate=self._learning_rate,
                        initial_parameters=deepcopy(initial_parameters),
                        seed_id=seed_id,
                    )
                    rmse_m = self._validate_result(result, expected_side=side)
                except Exception as error:
                    attempts.append(
                        {
                            "side": side,
                            "seed_id": seed_id,
                            "init_source": attempt_source,
                            "status": "ERROR",
                            "rmse_m": None,
                            "result": None,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                    )
                    continue
                status = "ACCEPTED" if rmse_m <= self._rmse_gate_m else "REJECTED"
                attempts.append(
                    {
                        "side": side,
                        "seed_id": seed_id,
                        "init_source": attempt_source,
                        "status": status,
                        "rmse_m": rmse_m,
                        "result": result,
                        "error": None,
                    }
                )
                scored.append((rmse_m, len(attempts) - 1))
                if status == "ACCEPTED":
                    passing.append((rmse_m, len(attempts) - 1, result))

        run_hypotheses(hypotheses)
        if accepted_state is not None and not passing:
            run_hypotheses(
                [
                    (
                        accepted_state.side,
                        seed_id,
                        parameters,
                        list(accepted_state.beta),
                        "COLD_RECOVERY",
                    )
                    for seed_id, parameters in self._cold_start_seeds
                ]
            )

        if not scored:
            return {
                "track_id": track_id,
                "timestamp_ns": timestamp_ns,
                "status": "ERROR",
                "fit": None,
                "selected_attempt_index": None,
                "attempts": attempts,
                "best_attempt": None,
                "init_source": init_source,
                "predecessor_timestamp_ns": predecessor_timestamp_ns,
                "reset_reason": reset_reason,
            }
        if not passing:
            best_attempt = attempts[min(scored, key=lambda value: (value[0], value[1]))[1]]
            return {
                "track_id": track_id,
                "timestamp_ns": timestamp_ns,
                "status": "REJECTED",
                "fit": None,
                "selected_attempt_index": None,
                "attempts": attempts,
                "best_attempt": best_attempt,
                "init_source": init_source,
                "predecessor_timestamp_ns": predecessor_timestamp_ns,
                "reset_reason": reset_reason,
            }
        _, selected_index, selected = min(passing, key=lambda value: (value[0], value[1]))
        self._accepted[track_id] = self._state_from_result(selected, timestamp_ns=timestamp_ns)
        return {
            "track_id": track_id,
            "timestamp_ns": timestamp_ns,
            "status": "ACCEPTED",
            "fit": selected,
            "selected_attempt_index": selected_index,
            "attempts": attempts,
            "best_attempt": attempts[selected_index],
            "init_source": init_source,
            "predecessor_timestamp_ns": predecessor_timestamp_ns,
            "reset_reason": reset_reason,
        }

    @staticmethod
    def _finite_vector(result: dict[str, Any], field: str, length: int) -> tuple[float, ...]:
        value = result.get(field)
        if not isinstance(value, list) or len(value) != length:
            raise WorkerError(f"MANO fit result {field} must contain {length} values")
        vector = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in vector):
            raise WorkerError(f"MANO fit result {field} must be finite")
        return vector

    @classmethod
    def _validate_result(cls, result: Any, *, expected_side: str) -> float:
        if not isinstance(result, dict):
            raise WorkerError("MANO fit result must be an object")
        if result.get("side") != expected_side:
            raise WorkerError("MANO fit result side does not match the attempted side")
        rmse = result.get("rmse_m")
        if isinstance(rmse, bool) or not isinstance(rmse, (int, float)):
            raise WorkerError("MANO fit result rmse_m must be numeric")
        rmse_m = float(rmse)
        if not math.isfinite(rmse_m) or rmse_m < 0.0:
            raise WorkerError("MANO fit result rmse_m must be finite and non-negative")
        cls._finite_vector(result, "global_orient", 3)
        cls._finite_vector(result, "hand_pose", 45)
        cls._finite_vector(result, "transl", 3)
        cls._finite_vector(result, "beta", 10)
        iterations_run = result.get("iterations_run")
        if (
            isinstance(iterations_run, bool)
            or not isinstance(iterations_run, int)
            or iterations_run < 0
        ):
            raise WorkerError("MANO fit result iterations_run must be a non-negative integer")
        for field in ("best_loss", "final_loss"):
            value = result.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise WorkerError(f"MANO fit result {field} must be numeric")
            if not math.isfinite(float(value)):
                raise WorkerError(f"MANO fit result {field} must be finite")
        residuals = result.get("joint_residuals_m")
        if not isinstance(residuals, list) or len(residuals) != 21:
            raise WorkerError("MANO fit result joint_residuals_m must contain 21 values")
        for residual in residuals:
            if residual is None:
                continue
            if (
                isinstance(residual, bool)
                or not isinstance(residual, (int, float))
                or not math.isfinite(float(residual))
                or float(residual) < 0.0
            ):
                raise WorkerError(
                    "MANO fit result joint_residuals_m must be finite and non-negative"
                )
        if not isinstance(result.get("converged"), bool):
            raise WorkerError("MANO fit result converged must be boolean")
        return rmse_m

    @classmethod
    def _state_from_result(cls, result: dict[str, Any], *, timestamp_ns: int) -> _AcceptedState:
        return _AcceptedState(
            side=str(result["side"]),
            timestamp_ns=timestamp_ns,
            global_orient=cls._finite_vector(result, "global_orient", 3),
            hand_pose=cls._finite_vector(result, "hand_pose", 45),
            transl=cls._finite_vector(result, "transl", 3),
            beta=cls._finite_vector(result, "beta", 10),
        )


__all__ = ["ManoTrackFitter"]
