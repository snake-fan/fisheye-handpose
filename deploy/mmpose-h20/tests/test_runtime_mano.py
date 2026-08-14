from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.runtime import OpenMMLabRuntime  # noqa: E402


class _TranslationOnlyMano:
    def __call__(self, *, transl, **kwargs):
        del kwargs
        return SimpleNamespace(
            joints=transl[:, None, :].expand(1, 16, 3),
            vertices=transl[:, None, :].expand(1, 745, 3),
        )


class RuntimeManoTests(unittest.TestCase):
    def test_single_iteration_returns_the_updated_state_and_post_step_loss(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is unavailable in the lightweight local test environment")

        fixed_beta = [0.05 * index for index in range(10)]
        result = OpenMMLabRuntime().fit_mano(
            {"right": _TranslationOnlyMano()},
            side="right",
            target_xyz_m=[[1.0, 0.0, 0.0] for _ in range(21)],
            validity=["VALID"] * 21,
            fixed_beta=fixed_beta,
            device="cpu",
            iterations=1,
            learning_rate=0.1,
            initial_parameters={
                "hand_pose": [0.0] * 45,
                "global_orient": [0.0] * 3,
                "transl": [0.0] * 3,
            },
        )

        self.assertGreater(result["transl"][0], 0.09)
        self.assertEqual(result["iterations_run"], 1)
        self.assertTrue(
            all(
                math.isclose(actual, expected, abs_tol=1e-7)
                for actual, expected in zip(result["beta"], fixed_beta, strict=True)
            )
        )
        post_step_residual = abs(1.0 - result["transl"][0])
        expected_post_step_loss = 0.02 * post_step_residual - 0.0002
        self.assertTrue(math.isclose(result["final_loss"], expected_post_step_loss, abs_tol=1e-6))
        self.assertTrue(math.isclose(result["best_loss"], expected_post_step_loss, abs_tol=1e-6))

    def test_joint_weights_control_the_objective_but_preserve_full_rmse(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is unavailable in the lightweight local test environment")

        result = OpenMMLabRuntime().fit_mano(
            {"right": _TranslationOnlyMano()},
            side="right",
            target_xyz_m=[[1.0, 0.0, 0.0] for _ in range(20)] + [[-10.0, 0.0, 0.0]],
            validity=["VALID"] * 21,
            joint_weights=[1.0] * 20 + [0.0],
            fixed_beta=[0.0] * 10,
            device="cpu",
            iterations=80,
            learning_rate=0.1,
            initial_parameters={
                "hand_pose": [0.0] * 45,
                "global_orient": [0.0] * 3,
                "transl": [0.0] * 3,
            },
        )

        self.assertGreater(result["transl"][0], 0.8)
        self.assertLess(result["weighted_rmse_m"], 0.2)
        self.assertGreater(result["rmse_m"], 2.0)
        self.assertEqual(result["effective_joint_count"], 20)
        self.assertEqual(result["joint_weights"], [1.0] * 20 + [0.0])


if __name__ == "__main__":
    unittest.main()
