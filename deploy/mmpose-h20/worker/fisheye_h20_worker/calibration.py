"""Orbbec KB4 normalization and point rectification for the worker wire protocol.

The transform convention and metre normalization intentionally mirror the core v1 JSON
contract, but this Python 3.10 module is process-isolated and imports no core code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .contracts import CalibrationRequest, WorkerError


@dataclass(frozen=True)
class RectifiedStereo:
    calibration_id: str
    image_size: tuple[int, int]
    output_size: tuple[int, int]
    left_k: Any
    left_d: Any
    right_k: Any
    right_d: Any
    right_from_left_rotation: Any
    right_from_left_translation_m: Any
    r1: Any
    r2: Any
    p1: Any
    p2: Any
    q: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "image_size": list(self.image_size),
            "output_size": list(self.output_size),
            "length_unit": "m",
            "coordinate_frame": "rectified_left_camera",
            "baseline_m": float(
                sum(float(value) ** 2 for value in self.right_from_left_translation_m) ** 0.5
            ),
            "R1": self.r1.tolist(),
            "R2": self.r2.tolist(),
            "P1": self.p1.tolist(),
            "P2": self.p2.tolist(),
            "Q": self.q.tolist(),
        }

    def rectify_points(self, side: str, points: list[list[float]]) -> list[list[float]]:
        import cv2
        import numpy as np

        if not points:
            return []
        array = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if not np.all(np.isfinite(array)):
            raise WorkerError("pose keypoints contain non-finite coordinates")
        if side == "left":
            output = cv2.fisheye.undistortPoints(
                array, self.left_k, self.left_d, R=self.r1, P=self.p1[:, :3]
            )
        elif side == "right":
            output = cv2.fisheye.undistortPoints(
                array, self.right_k, self.right_d, R=self.r2, P=self.p2[:, :3]
            )
        else:
            raise WorkerError(f"unknown camera side: {side}")
        return [[float(u), float(v)] for u, v in output.reshape(-1, 2)]


def _matrix(value: Any, label: str) -> Any:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise WorkerError(f"{label} must be a finite 3x3 matrix")
    return array


def _extrinsic(entry: dict[str, Any], factor: float, label: str) -> Any:
    import numpy as np

    raw = entry.get("extrinsics")
    if not isinstance(raw, dict):
        raise WorkerError(f"{label}: missing extrinsics")
    rotation = _matrix(raw.get("rotation"), f"{label}.rotation")
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-3 or abs(determinant - 1.0) > 1e-3:
        raise WorkerError(f"{label}.rotation must be an SO(3) matrix")
    translation = np.asarray(raw.get("translation"), dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise WorkerError(f"{label}.translation must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation * factor
    return transform


def _camera(entry: dict[str, Any]) -> tuple[Any, Any, tuple[int, int]]:
    import numpy as np

    camera_id = str(entry.get("id", ""))
    if str(entry.get("distortion_model", "")).upper() != "KB":
        raise WorkerError(f"camera {camera_id}: distortion_model must be KB")
    intrinsics = entry.get("intrinsics")
    distortion = entry.get("distortion")
    if not isinstance(intrinsics, dict) or not isinstance(distortion, dict):
        raise WorkerError(f"camera {camera_id}: intrinsics/distortion are required")
    values = [intrinsics.get(name) for name in ("fx", "fy", "cx", "cy")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise WorkerError(f"camera {camera_id}: invalid intrinsics")
    fx, fy, cx, cy = (float(value) for value in values)
    if fx <= 0 or fy <= 0 or not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise WorkerError(f"camera {camera_id}: invalid intrinsics")
    coefficients = [distortion.get(f"k{index}") for index in range(1, 5)]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in coefficients
    ):
        raise WorkerError(f"camera {camera_id}: KB4 coefficients are required")
    if not all(math.isfinite(float(value)) for value in coefficients):
        raise WorkerError(f"camera {camera_id}: KB4 coefficients must be finite")
    unsupported = [distortion.get(name, 0.0) for name in ("k5", "k6", "p1", "p2")]
    if any(abs(float(value)) > 1e-12 for value in unsupported):
        raise WorkerError(f"camera {camera_id}: unsupported non-zero distortion coefficient")
    width, height = entry.get("image_width"), entry.get("image_height")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (width, height)
    ):
        raise WorkerError(f"camera {camera_id}: invalid image size")
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    d = np.asarray(coefficients, dtype=np.float64).reshape(4, 1)
    return k, d, (width, height)


def load_rectified_stereo(config: CalibrationRequest) -> RectifiedStereo:
    import cv2
    import numpy as np
    import yaml

    if not config.path.is_file():
        raise WorkerError(f"calibration YAML is not a file: {config.path}")
    try:
        document = yaml.safe_load(config.path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise WorkerError(f"cannot parse calibration YAML: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise WorkerError("calibration root/cameras structure is invalid")
    entries: dict[str, dict[str, Any]] = {}
    for entry in document["cameras"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise WorkerError("every calibration camera must have an ID")
        if entry["id"] in entries:
            raise WorkerError(f"duplicate calibration camera ID: {entry['id']}")
        entries[entry["id"]] = entry
    try:
        left_entry = entries[config.left_camera_id]
        right_entry = entries[config.right_camera_id]
    except KeyError as exc:
        raise WorkerError(f"requested calibration camera is absent: {exc}") from exc
    info = document.get("calibration_info")
    if not isinstance(info, dict) or not isinstance(info.get("reference_camera"), str):
        raise WorkerError("calibration_info.reference_camera is required")
    reference_id = info["reference_camera"]
    if reference_id not in entries:
        raise WorkerError("calibration reference camera is absent")
    factor = 1e-3 if config.translation_unit == "mm" else 1.0
    camera_from_reference: dict[str, Any] = {}
    for camera_id, entry in entries.items():
        transform = _extrinsic(entry, factor, camera_id)
        if config.extrinsics_convention == "camera_to_reference":
            transform = np.linalg.inv(transform)
        camera_from_reference[camera_id] = transform
    reference_self = camera_from_reference[reference_id]
    if not np.allclose(reference_self, np.eye(4), rtol=0.0, atol=1e-6):
        raise WorkerError("reference camera self-extrinsics must be identity")
    right_from_left = camera_from_reference[config.right_camera_id] @ np.linalg.inv(
        camera_from_reference[config.left_camera_id]
    )
    rotation = right_from_left[:3, :3]
    translation = right_from_left[:3, 3]
    baseline = float(np.linalg.norm(translation))
    if not 0.005 <= baseline <= 0.5:
        raise WorkerError(f"stereo baseline is implausible: {baseline} m")
    left_k, left_d, left_size = _camera(left_entry)
    right_k, right_d, right_size = _camera(right_entry)
    if left_size != right_size:
        raise WorkerError("stereo cameras must share one image size")
    flags = cv2.CALIB_ZERO_DISPARITY
    try:
        r1, r2, p1, p2, q = cv2.fisheye.stereoRectify(
            left_k,
            left_d,
            right_k,
            right_d,
            left_size,
            rotation,
            translation,
            flags=flags,
            newImageSize=config.output_size,
            balance=config.balance,
            fov_scale=config.fov_scale,
        )
    except cv2.error as exc:
        raise WorkerError(f"OpenCV KB4 stereo rectification failed: {exc}") from exc
    arrays = (r1, r2, p1, p2, q)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise WorkerError("stereo rectification produced non-finite matrices")
    digest = hashlib.sha256(config.path.read_bytes())
    digest.update(
        (
            f"\0{config.left_camera_id}\0{config.right_camera_id}"
            f"\0{config.translation_unit}\0{config.extrinsics_convention}"
        ).encode()
    )
    return RectifiedStereo(
        calibration_id=f"sha256:{digest.hexdigest()}",
        image_size=left_size,
        output_size=config.output_size,
        left_k=left_k,
        left_d=left_d,
        right_k=right_k,
        right_d=right_d,
        right_from_left_rotation=rotation,
        right_from_left_translation_m=translation,
        r1=r1,
        r2=r2,
        p1=p1,
        p2=p2,
        q=q,
    )


__all__ = ["RectifiedStereo", "load_rectified_stereo"]
