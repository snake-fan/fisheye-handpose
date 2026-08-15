"""Hand-centred virtual perspective crops for native KB4 fisheye frames.

This module is intentionally process-local to the Python 3.10 H20 worker.  It maps
virtual pinhole rays to the physical source fisheye camera; a planar homography is
not a valid model for this operation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import RectifiedStereo


class UnrepresentablePerspectiveCropError(ValueError):
    """A detector candidate cannot be covered by one perspective crop."""


def _uv_array(points: Any, *, label: str) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError(f"{label} must have shape [..., 2]")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain finite coordinates")
    return values.reshape(-1, 2), values.shape


def _kb4_project(rays: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    radial = np.linalg.norm(rays[:, :2], axis=1)
    theta = np.arctan2(radial, rays[:, 2])
    theta2 = theta * theta
    coefficients = np.asarray(d, dtype=np.float64).reshape(4)
    theta_distorted = theta * (
        1.0
        + coefficients[0] * theta2
        + coefficients[1] * theta2**2
        + coefficients[2] * theta2**3
        + coefficients[3] * theta2**4
    )
    scale = np.divide(
        theta_distorted,
        radial,
        out=np.ones_like(theta_distorted),
        where=radial > 1e-15,
    )
    return np.column_stack(
        (
            float(k[0, 0]) * rays[:, 0] * scale + float(k[0, 2]),
            float(k[1, 1]) * rays[:, 1] * scale + float(k[1, 2]),
        )
    )


def _kb4_unproject(points: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    distorted = np.column_stack(
        (
            (points[:, 0] - float(k[0, 2])) / float(k[0, 0]),
            (points[:, 1] - float(k[1, 2])) / float(k[1, 1]),
        )
    )
    theta_distorted = np.linalg.norm(distorted, axis=1)
    theta = theta_distorted.copy()
    coefficients = np.asarray(d, dtype=np.float64).reshape(4)
    # Newton inversion of theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8).
    for _ in range(12):
        theta2 = theta * theta
        polynomial = (
            1.0
            + coefficients[0] * theta2
            + coefficients[1] * theta2**2
            + coefficients[2] * theta2**3
            + coefficients[3] * theta2**4
        )
        derivative = (
            1.0
            + 3.0 * coefficients[0] * theta2
            + 5.0 * coefficients[1] * theta2**2
            + 7.0 * coefficients[2] * theta2**3
            + 9.0 * coefficients[3] * theta2**4
        )
        update = np.divide(
            theta * polynomial - theta_distorted,
            derivative,
            out=np.zeros_like(theta),
            where=np.abs(derivative) > 1e-12,
        )
        theta -= update
    azimuth = np.divide(
        distorted,
        theta_distorted[:, None],
        out=np.zeros_like(distorted),
        where=theta_distorted[:, None] > 1e-15,
    )
    sin_theta = np.sin(theta)
    return np.column_stack((azimuth[:, 0] * sin_theta, azimuth[:, 1] * sin_theta, np.cos(theta)))


def _camera_for_side(rectification: RectifiedStereo, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "left":
        return np.asarray(rectification.left_k), np.asarray(rectification.left_d)
    if side == "right":
        return np.asarray(rectification.right_k), np.asarray(rectification.right_d)
    raise ValueError(f"unknown camera side: {side}")


@dataclass(frozen=True)
class PerspectiveCrop:
    """A physical, non-mirrored virtual-pinhole crop and its ray transforms."""

    image: np.ndarray
    valid_mask: np.ndarray
    K_virtual: np.ndarray
    R_source_from_virtual: np.ndarray
    T_rig_from_virtual: np.ndarray
    policy_id: str
    crop_id: str
    side: str
    bbox: tuple[float, float, float, float]
    _source_k: np.ndarray
    _source_d: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.valid_mask))

    def crop_uv_to_source_uv(self, points: Any) -> np.ndarray:
        flattened, original_shape = _uv_array(points, label="crop UV")
        homogeneous = np.column_stack((flattened, np.ones(len(flattened))))
        rays_virtual = homogeneous @ np.linalg.inv(self.K_virtual).T
        rays_virtual /= np.linalg.norm(rays_virtual, axis=1, keepdims=True)
        rays_source = rays_virtual @ self.R_source_from_virtual.T
        result = _kb4_project(rays_source, self._source_k, self._source_d)
        return result.reshape(original_shape)

    def source_uv_to_crop_uv(self, points: Any) -> np.ndarray:
        flattened, original_shape = _uv_array(points, label="source UV")
        rays_source = _kb4_unproject(flattened, self._source_k, self._source_d)
        rays_virtual = rays_source @ self.R_source_from_virtual
        result = np.full((len(flattened), 2), np.nan, dtype=np.float64)
        visible = rays_virtual[:, 2] > 1e-12
        normalized = rays_virtual[visible, :2] / rays_virtual[visible, 2:3]
        result[visible, 0] = float(self.K_virtual[0, 0]) * normalized[:, 0] + float(
            self.K_virtual[0, 2]
        )
        result[visible, 1] = float(self.K_virtual[1, 1]) * normalized[:, 1] + float(
            self.K_virtual[1, 2]
        )
        return result.reshape(original_shape)


class VirtualPerspectiveCropper:
    """Create deterministic ray-warped crops from native distorted frames."""

    def __init__(
        self,
        *,
        output_size: tuple[int, int] = (256, 256),
        bbox_scale: float = 1.5,
    ) -> None:
        if (
            not isinstance(output_size, (tuple, list))
            or len(output_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in output_size
            )
        ):
            raise ValueError("output_size must contain two integers >= 2")
        if isinstance(bbox_scale, bool) or not isinstance(
            bbox_scale, (int, float, np.integer, np.floating)
        ):
            raise ValueError("bbox_scale must be finite and positive")
        scale = float(bbox_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("bbox_scale must be finite and positive")
        self.output_size = (output_size[0], output_size[1])
        self.bbox_scale = scale
        self.policy_id = (
            "virtual-perspective-kb4/v1"
            f":{output_size[0]}x{output_size[1]}:bbox-scale={self.bbox_scale.hex()}"
        )

    def create(
        self,
        frame: Any,
        side: str,
        bbox: Any,
        rectification: RectifiedStereo,
    ) -> PerspectiveCrop:
        import cv2

        source = np.asarray(frame)
        if (
            source.ndim != 3
            or source.shape[2] != 3
            or source.dtype != np.uint8
            or source.shape[1::-1] != rectification.image_size
        ):
            raise ValueError("frame must be uint8 BGR and match the calibrated source image size")
        source_k, source_d = _camera_for_side(rectification, side)
        raw_box = np.asarray(bbox, dtype=object)
        if raw_box.shape != (4,) or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in raw_box
        ):
            raise ValueError("bbox must contain four finite coordinates")
        box = raw_box.astype(np.float64)
        if not np.all(np.isfinite(box)):
            raise ValueError("bbox must contain four finite coordinates")
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")
        if x2 <= 0.0 or y2 <= 0.0 or x1 >= source.shape[1] or y1 >= source.shape[0]:
            raise ValueError("bbox must overlap the source image")

        center_uv = np.asarray([[(x1 + x2) * 0.5, (y1 + y2) * 0.5]])
        optical_axis = _kb4_unproject(center_uv, source_k, source_d)[0]
        source_down = np.asarray([0.0, 1.0, 0.0])
        virtual_x = np.cross(source_down, optical_axis)
        if np.linalg.norm(virtual_x) < 1e-8:
            virtual_x = np.cross(np.asarray([1.0, 0.0, 0.0]), optical_axis)
        virtual_x /= np.linalg.norm(virtual_x)
        virtual_y = np.cross(optical_axis, virtual_x)
        virtual_y /= np.linalg.norm(virtual_y)
        rotation = np.column_stack((virtual_x, virtual_y, optical_axis))
        rig_from_virtual = np.eye(4, dtype=np.float64)
        if side == "left":
            rig_from_virtual[:3, :3] = rotation
        else:
            right_from_left_rotation = np.asarray(
                rectification.right_from_left_rotation, dtype=np.float64
            )
            right_from_left_translation = np.asarray(
                rectification.right_from_left_translation_m, dtype=np.float64
            )
            left_from_right_rotation = right_from_left_rotation.T
            rig_from_virtual[:3, :3] = left_from_right_rotation @ rotation
            rig_from_virtual[:3, 3] = -left_from_right_rotation @ right_from_left_translation

        corners = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        corner_rays_source = _kb4_unproject(corners, source_k, source_d)
        corner_rays_virtual = corner_rays_source @ rotation
        if np.any(corner_rays_virtual[:, 2] <= 1e-8):
            raise UnrepresentablePerspectiveCropError(
                "bbox cannot be represented by one perspective crop"
            )
        normalized = corner_rays_virtual[:, :2] / corner_rays_virtual[:, 2:3]
        extent_x = float(np.max(np.abs(normalized[:, 0]))) * self.bbox_scale
        extent_y = float(np.max(np.abs(normalized[:, 1]))) * self.bbox_scale
        if extent_x <= 1e-12 or extent_y <= 1e-12:
            raise ValueError("bbox angular extent is degenerate")
        output_width, output_height = self.output_size
        center_x = (output_width - 1.0) * 0.5
        center_y = (output_height - 1.0) * 0.5
        focal = min(center_x / extent_x, center_y / extent_y)
        virtual_k = np.asarray(
            [[focal, 0.0, center_x], [0.0, focal, center_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        grid_u, grid_v = np.meshgrid(
            np.arange(output_width, dtype=np.float64),
            np.arange(output_height, dtype=np.float64),
        )
        grid = np.column_stack((grid_u.ravel(), grid_v.ravel()))
        homogeneous = np.column_stack((grid, np.ones(len(grid))))
        rays_virtual = homogeneous @ np.linalg.inv(virtual_k).T
        rays_virtual /= np.linalg.norm(rays_virtual, axis=1, keepdims=True)
        rays_source = rays_virtual @ rotation.T
        source_uv = _kb4_project(rays_source, source_k, source_d)
        valid = (
            np.all(np.isfinite(source_uv), axis=1)
            & (source_uv[:, 0] >= 0.0)
            & (source_uv[:, 0] <= source.shape[1] - 1.0)
            & (source_uv[:, 1] >= 0.0)
            & (source_uv[:, 1] <= source.shape[0] - 1.0)
        )
        remap_x = source_uv[:, 0].reshape(output_height, output_width).astype(np.float32)
        remap_y = source_uv[:, 1].reshape(output_height, output_width).astype(np.float32)
        image = cv2.remap(
            source,
            remap_x,
            remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = valid.reshape(output_height, output_width)
        image = image.copy()
        image[~valid_mask] = 0

        digest = hashlib.sha256()
        digest.update(rectification.calibration_id.encode())
        digest.update(side.encode())
        digest.update(box.tobytes())
        digest.update(self.policy_id.encode())
        digest.update(virtual_k.astype("<f8", copy=False).tobytes())
        digest.update(rotation.astype("<f8", copy=False).tobytes())
        digest.update(rig_from_virtual.astype("<f8", copy=False).tobytes())
        return PerspectiveCrop(
            image=image,
            valid_mask=valid_mask,
            K_virtual=virtual_k,
            R_source_from_virtual=rotation,
            T_rig_from_virtual=rig_from_virtual,
            policy_id=self.policy_id,
            crop_id=f"sha256:{digest.hexdigest()}",
            side=side,
            bbox=(x1, y1, x2, y2),
            _source_k=source_k.copy(),
            _source_d=source_d.copy(),
        )


__all__ = [
    "PerspectiveCrop",
    "UnrepresentablePerspectiveCropError",
    "VirtualPerspectiveCropper",
]
