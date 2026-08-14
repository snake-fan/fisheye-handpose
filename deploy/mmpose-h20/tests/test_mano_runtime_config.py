from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = DEPLOY_ROOT / "worker"
sys.path.insert(0, str(WORKER_ROOT))

from fisheye_h20_worker.runtime import OpenMMLabRuntime  # noqa: E402


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> None:
        self.requires_grad = value


class _Model:
    def __init__(self) -> None:
        self.parameter = _Parameter()
        self.device: object | None = None
        self.training = True

    def to(self, device: object) -> _Model:
        self.device = device
        return self

    def eval(self) -> None:
        self.training = False

    def parameters(self) -> list[_Parameter]:
        return [self.parameter]


def test_mano_models_use_mean_pose_without_reducing_the_full_45d_space() -> None:
    calls: list[dict[str, object]] = []
    models: list[_Model] = []
    fake_smplx = ModuleType("smplx")
    fake_torch = ModuleType("torch")

    def create(path: str, **kwargs: object) -> _Model:
        calls.append({"path": path, **kwargs})
        model = _Model()
        models.append(model)
        return model

    fake_smplx.create = create  # type: ignore[attr-defined]
    fake_torch.float32 = "float32"  # type: ignore[attr-defined]
    fake_torch.device = lambda value: f"device:{value}"  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"smplx": fake_smplx, "torch": fake_torch}):
        loaded = OpenMMLabRuntime().load_mano_models(
            model_root=Path("/private/models"),
            device="cuda:0",
        )

    assert set(loaded) == {"left", "right"}
    assert [call["is_rhand"] for call in calls] == [False, True]
    assert all(call["use_pca"] is False for call in calls)
    assert all(call["flat_hand_mean"] is False for call in calls)
    assert all(call["batch_size"] == 1 for call in calls)
    assert all(model.device == "device:cuda:0" for model in models)
    assert all(model.training is False for model in models)
    assert all(model.parameter.requires_grad is False for model in models)
