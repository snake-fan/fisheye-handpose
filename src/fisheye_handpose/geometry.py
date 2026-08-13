"""True stereo rectification for calibrated Kannala-Brandt cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal

from .calibration import StereoCalibration
from .errors import GeometryError

MapType = Literal["float32", "fixed16"]


@dataclass(frozen=True, slots=True)
class RectificationConfig:
    output_size: tuple[int, int]
    balance: float
    fov_scale: float = 1.0
    zero_disparity: bool = True
    map_type: MapType = "float32"

    def __post_init__(self) -> None:
        if len(self.output_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in self.output_size
        ):
            raise GeometryError("rectification output size must contain two integers")
        if self.output_size[0] <= 0 or self.output_size[1] <= 0:
            raise GeometryError("rectification output size must be positive")
        if (
            isinstance(self.balance, bool)
            or not isinstance(self.balance, Real)
            or not math.isfinite(self.balance)
        ):
            raise GeometryError("rectification balance must be a finite real number")
        if not 0.0 <= self.balance <= 1.0:
            raise GeometryError("rectification balance must be in [0, 1]")
        if (
            isinstance(self.fov_scale, bool)
            or not isinstance(self.fov_scale, Real)
            or self.fov_scale <= 0
            or not math.isfinite(self.fov_scale)
        ):
            raise GeometryError("fov_scale must be finite and positive")
        if self.map_type not in ("float32", "fixed16"):
            raise GeometryError(f"unsupported map type {self.map_type!r}")


class StereoRectifier:
    """Rectification matrices and remap tables derived from one calibration."""

    def __init__(
        self,
        *,
        calibration: StereoCalibration,
        config: RectificationConfig,
        r1: Any,
        r2: Any,
        p1: Any,
        p2: Any,
        q: Any,
        left_maps: tuple[Any, Any],
        right_maps: tuple[Any, Any],
        left_valid_mask: Any,
        right_valid_mask: Any,
    ) -> None:
        self.calibration = calibration
        self.config = config
        self.r1 = r1
        self.r2 = r2
        self.p1 = p1
        self.p2 = p2
        self.q = q
        self.left_maps = left_maps
        self.right_maps = right_maps
        self.left_valid_mask = left_valid_mask
        self.right_valid_mask = right_valid_mask
        self.common_valid_mask = left_valid_mask & right_valid_mask

    @staticmethod
    def _validate_rectification_geometry(
        np: Any,
        *,
        r1: Any,
        r2: Any,
        p1: Any,
        p2: Any,
        q: Any,
        baseline_m: float,
        zero_disparity: bool,
    ) -> None:
        expected_shapes = ((3, 3), (3, 3), (3, 4), (3, 4), (4, 4))
        arrays = (r1, r2, p1, p2, q)
        if tuple(array.shape for array in arrays) != expected_shapes:
            raise GeometryError(
                "rectification returned unexpected matrix shapes: "
                f"{tuple(array.shape for array in arrays)}"
            )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise GeometryError("rectification produced non-finite matrices")
        identity = np.eye(3, dtype=np.float64)
        for label, rotation in (("R1", r1), ("R2", r2)):
            orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - identity))
            determinant = float(np.linalg.det(rotation))
            if orthogonality_error > 1e-6 or abs(determinant - 1.0) > 1e-6:
                raise GeometryError(
                    f"{label} is not SO(3): orthogonality_error="
                    f"{orthogonality_error:.6g}, det={determinant:.6g}"
                )

        pinhole_last_row = np.asarray([0.0, 0.0, 1.0, 0.0])
        for label, projection in (("P1", p1), ("P2", p2)):
            if projection[0, 0] <= 0 or projection[1, 1] <= 0:
                raise GeometryError(f"{label} has non-positive focal length")
            if not np.allclose(projection[2], pinhole_last_row, rtol=0.0, atol=1e-10):
                raise GeometryError(f"{label} does not have canonical pinhole form")
            if abs(float(projection[0, 1])) > 1e-10 or abs(float(projection[1, 0])) > 1e-10:
                raise GeometryError(f"{label} contains unsupported skew")
        if not np.isclose(p1[0, 0], p2[0, 0], rtol=1e-8, atol=1e-10):
            raise GeometryError("P1/P2 do not share a rectified horizontal focal length")
        if not np.isclose(p1[1, 1], p2[1, 1], rtol=1e-8, atol=1e-10):
            raise GeometryError("P1/P2 do not share a rectified vertical focal length")
        if not np.isclose(p1[1, 2], p2[1, 2], rtol=1e-8, atol=1e-10):
            raise GeometryError("P1/P2 do not share a rectified principal row")
        if zero_disparity and not np.isclose(p1[0, 2], p2[0, 2], rtol=1e-8, atol=1e-10):
            raise GeometryError("zero-disparity rectification produced different principal columns")

        tx = float(p2[0, 3] / p2[0, 0] - p1[0, 3] / p1[0, 0])
        ty = float(p2[1, 3] / p2[1, 1] - p1[1, 3] / p1[1, 1])
        if abs(tx) <= 1e-12 or abs(ty) > max(1e-10, abs(tx) * 1e-8):
            raise GeometryError(
                f"rectified stereo baseline is not horizontal: tx={tx:.9g}, ty={ty:.9g}"
            )
        if not math.isclose(abs(tx), baseline_m, rel_tol=1e-7, abs_tol=1e-9):
            raise GeometryError(
                f"P1/P2 metric baseline {abs(tx):.9g} m disagrees with calibration "
                f"{baseline_m:.9g} m"
            )

        expected_q = np.asarray(
            [
                [1.0, 0.0, 0.0, -float(p1[0, 2])],
                [0.0, 1.0, 0.0, -float(p1[1, 2])],
                [0.0, 0.0, 0.0, float(p1[0, 0])],
                [
                    0.0,
                    0.0,
                    -1.0 / tx,
                    float(p1[0, 2] - p2[0, 2]) / tx,
                ],
            ],
            dtype=np.float64,
        )
        if not np.allclose(q, expected_q, rtol=1e-7, atol=1e-8):
            raise GeometryError("Q is inconsistent with the rectified projection matrices")

    @classmethod
    def build(cls, calibration: StereoCalibration, config: RectificationConfig) -> StereoRectifier:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise GeometryError("NumPy and opencv-python-headless are required") from exc
        size = calibration.left.image_size
        output_size = config.output_size
        k1 = np.asarray(calibration.left.intrinsics, dtype=np.float64)
        d1 = np.asarray(calibration.left.distortion, dtype=np.float64).reshape(4, 1)
        k2 = np.asarray(calibration.right.intrinsics, dtype=np.float64)
        d2 = np.asarray(calibration.right.distortion, dtype=np.float64).reshape(4, 1)
        rotation = np.asarray(calibration.right_from_left.rotation, dtype=np.float64)
        translation = np.asarray(calibration.right_from_left.translation_m, dtype=np.float64)
        flags = cv2.CALIB_ZERO_DISPARITY if config.zero_disparity else 0
        try:
            r1, r2, p1, p2, q = cv2.fisheye.stereoRectify(
                k1,
                d1,
                k2,
                d2,
                size,
                rotation,
                translation,
                flags=flags,
                newImageSize=output_size,
                balance=config.balance,
                fov_scale=config.fov_scale,
            )
        except cv2.error as exc:
            raise GeometryError(f"OpenCV fisheye.stereoRectify failed: {exc}") from exc
        cls._validate_rectification_geometry(
            np,
            r1=r1,
            r2=r2,
            p1=p1,
            p2=p2,
            q=q,
            baseline_m=calibration.right_from_left.baseline_m,
            zero_disparity=config.zero_disparity,
        )
        map_code = cv2.CV_32FC1 if config.map_type == "float32" else cv2.CV_16SC2
        left_maps = cv2.fisheye.initUndistortRectifyMap(
            k1, d1, r1, p1[:, :3], output_size, map_code
        )
        right_maps = cv2.fisheye.initUndistortRectifyMap(
            k2, d2, r2, p2[:, :3], output_size, map_code
        )

        def valid_mask(maps: tuple[Any, Any]) -> Any:
            if config.map_type == "float32":
                map_x, map_y = maps
            else:
                map_x, map_y = cv2.convertMaps(maps[0], maps[1], cv2.CV_32FC1)
            expected_shape = (output_size[1], output_size[0])
            if map_x.shape != expected_shape or map_y.shape != expected_shape:
                raise GeometryError(
                    f"rectification maps have wrong shape {map_x.shape}/{map_y.shape}; "
                    f"expected {expected_shape}"
                )
            # INTER_LINEAR reads the four neighbours around each coordinate, so
            # coordinates on the last source row/column are not fully valid.
            return (
                np.isfinite(map_x)
                & np.isfinite(map_y)
                & (map_x >= 0)
                & (map_x < size[0] - 1)
                & (map_y >= 0)
                & (map_y < size[1] - 1)
            )

        return cls(
            calibration=calibration,
            config=config,
            r1=r1,
            r2=r2,
            p1=p1,
            p2=p2,
            q=q,
            left_maps=left_maps,
            right_maps=right_maps,
            left_valid_mask=valid_mask(left_maps),
            right_valid_mask=valid_mask(right_maps),
        )

    @property
    def common_valid_fraction(self) -> float:
        return float(self.common_valid_mask.mean())

    @property
    def left_valid_fraction(self) -> float:
        return float(self.left_valid_mask.mean())

    @property
    def right_valid_fraction(self) -> float:
        return float(self.right_valid_mask.mean())

    def effective_fov_degrees(self) -> tuple[float, float]:
        width, height = self.config.output_size
        fx, fy = float(self.p1[0, 0]), float(self.p1[1, 1])
        cx, cy = float(self.p1[0, 2]), float(self.p1[1, 2])
        return (
            math.degrees(math.atan2(cx, fx) + math.atan2(float(width - 1) - cx, fx)),
            math.degrees(math.atan2(cy, fy) + math.atan2(float(height - 1) - cy, fy)),
        )

    def apply(self, left_image: Any, right_image: Any) -> tuple[Any, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise GeometryError("opencv-python-headless is required") from exc
        expected_height = self.calibration.left.image_size[1]
        expected_width = self.calibration.left.image_size[0]
        for side, image in (("left", left_image), ("right", right_image)):
            if image is None or image.shape[:2] != (expected_height, expected_width):
                shape = None if image is None else image.shape
                raise GeometryError(
                    f"{side} input shape {shape} does not match calibration "
                    f"{(expected_height, expected_width)}"
                )
        return (
            cv2.remap(
                left_image,
                *self.left_maps,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            ),
            cv2.remap(
                right_image,
                *self.right_maps,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        horizontal_fov, vertical_fov = self.effective_fov_degrees()
        return {
            "calibration_id": self.calibration.calibration_id,
            "output_size": list(self.config.output_size),
            "balance": self.config.balance,
            "fov_scale": self.config.fov_scale,
            "zero_disparity": self.config.zero_disparity,
            "map_type": self.config.map_type,
            "R1": self.r1.tolist(),
            "R2": self.r2.tolist(),
            "P1": self.p1.tolist(),
            "P2": self.p2.tolist(),
            "Q": self.q.tolist(),
            "left_valid_fraction": self.left_valid_fraction,
            "right_valid_fraction": self.right_valid_fraction,
            "common_valid_fraction": self.common_valid_fraction,
            "effective_hfov_deg": horizontal_fov,
            "effective_vfov_deg": vertical_fov,
        }
