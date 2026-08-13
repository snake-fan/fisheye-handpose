"""Strict Orbbec Kannala-Brandt stereo calibration normalization."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, TypeAlias

from .errors import CalibrationError

Mat3: TypeAlias = tuple[tuple[float, float, float], ...]
Vec3: TypeAlias = tuple[float, float, float]
TranslationUnit = Literal["mm", "m"]
ExtrinsicsConvention = Literal["reference_to_camera", "camera_to_reference"]

_ROTATION_TOLERANCE = 1e-3
_REFERENCE_ROTATION_TOLERANCE = 1e-6
_REFERENCE_TRANSLATION_TOLERANCE_M = 1e-6
_KB_DERIVATIVE_EPSILON = 1e-9


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CalibrationError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError(f"{label} must be finite, got {value!r}")
    return number


def _positive_int(value: Any, label: str) -> int:
    number = _finite_float(value, label)
    if not number.is_integer() or number <= 0:
        raise CalibrationError(f"{label} must be a positive integer, got {value!r}")
    return int(number)


def _mat3(value: Any, label: str) -> Mat3:
    if not isinstance(value, list) or len(value) != 3:
        raise CalibrationError(f"{label} must be a 3x3 list")
    rows: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise CalibrationError(f"{label}[{row_index}] must contain 3 values")
        rows.append(tuple(_finite_float(v, f"{label}[{row_index}]") for v in row))
    return tuple(rows)


def _vec3(value: Any, label: str, scale: float = 1.0) -> Vec3:
    if not isinstance(value, list) or len(value) != 3:
        raise CalibrationError(f"{label} must contain exactly 3 values")
    return tuple(_finite_float(v, label) * scale for v in value)


def _transpose(matrix: Mat3) -> Mat3:
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _matmul(left: Mat3, right: Mat3) -> Mat3:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _matvec(matrix: Mat3, vector: Vec3) -> Vec3:
    return tuple(sum(matrix[i][k] * vector[k] for k in range(3)) for i in range(3))


def _det3(matrix: Mat3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _validate_rotation(rotation: Mat3, label: str, tolerance: float = _ROTATION_TOLERANCE) -> None:
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise CalibrationError(f"{label} must be 3x3")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for row in rotation
        for value in row
    ):
        raise CalibrationError(f"{label} must contain only finite real numbers")
    product = _matmul(_transpose(rotation), rotation)
    frobenius = math.sqrt(
        sum((product[i][j] - (1.0 if i == j else 0.0)) ** 2 for i in range(3) for j in range(3))
    )
    determinant = _det3(rotation)
    if frobenius > tolerance or abs(determinant - 1.0) > tolerance:
        raise CalibrationError(
            f"{label} is not SO(3): orthogonality_error={frobenius:.6g}, det={determinant:.6g}"
        )


@dataclass(frozen=True, slots=True)
class Transform3D:
    """Rigid transform with the invariant ``X_to = R @ X_from + t``."""

    from_frame: str
    to_frame: str
    rotation: Mat3
    translation_m: Vec3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.from_frame, str)
            or not self.from_frame.strip()
            or not isinstance(self.to_frame, str)
            or not self.to_frame.strip()
        ):
            raise CalibrationError("transform frame names must be non-empty")
        _validate_rotation(self.rotation, f"T_{self.to_frame}_from_{self.from_frame}.rotation")
        if len(self.translation_m) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in self.translation_m
        ):
            raise CalibrationError("transform translation must contain three finite real numbers")

    @property
    def baseline_m(self) -> float:
        return math.sqrt(sum(value * value for value in self.translation_m))

    def inverse(self) -> Transform3D:
        rotation = _transpose(self.rotation)
        transformed = _matvec(rotation, self.translation_m)
        translation = tuple(-value for value in transformed)
        return Transform3D(self.to_frame, self.from_frame, rotation, translation)

    def to_matrix4(self) -> list[list[float]]:
        return [
            [*self.rotation[0], self.translation_m[0]],
            [*self.rotation[1], self.translation_m[1]],
            [*self.rotation[2], self.translation_m[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]


def compose_transforms(outer: Transform3D, inner: Transform3D) -> Transform3D:
    """Return ``outer(inner(X))``; frame names must meet at the composition boundary."""

    if inner.to_frame != outer.from_frame:
        raise CalibrationError(
            f"cannot compose {inner.from_frame}->{inner.to_frame} with "
            f"{outer.from_frame}->{outer.to_frame}"
        )
    rotation = _matmul(outer.rotation, inner.rotation)
    rotated_translation = _matvec(outer.rotation, inner.translation_m)
    translation = tuple(rotated_translation[i] + outer.translation_m[i] for i in range(3))
    return Transform3D(inner.from_frame, outer.to_frame, rotation, translation)


@dataclass(frozen=True, slots=True)
class CameraKB:
    camera_id: str
    name: str
    image_size: tuple[int, int]
    intrinsics: Mat3
    distortion: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise CalibrationError("camera id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise CalibrationError(f"camera {self.camera_id}: name must be a non-empty string")
        if len(self.image_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in self.image_size
        ):
            raise CalibrationError(f"camera {self.camera_id}: image size must contain two integers")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise CalibrationError(f"camera {self.camera_id}: image size must be positive")
        _validate_rotation_like_intrinsics(self.intrinsics, self.camera_id)
        fx, fy = self.intrinsics[0][0], self.intrinsics[1][1]
        cx, cy = self.intrinsics[0][2], self.intrinsics[1][2]
        if fx <= 0 or fy <= 0:
            raise CalibrationError(f"camera {self.camera_id}: focal lengths must be positive")
        if not (0 <= cx < width and 0 <= cy < height):
            raise CalibrationError(
                f"camera {self.camera_id}: principal point {(cx, cy)} is outside {self.image_size}"
            )
        if len(self.distortion) != 4 or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in self.distortion
        ):
            raise CalibrationError(
                f"camera {self.camera_id}: distortion must contain four finite real numbers"
            )
        _validate_kb_invertibility(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "model": "KB4",
            "image_size": list(self.image_size),
            "intrinsics": [list(row) for row in self.intrinsics],
            "distortion": list(self.distortion),
        }


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    calibration_id: str
    source_path: Path
    reference_camera_id: str
    left: CameraKB
    right: CameraKB
    right_from_left: Transform3D
    translation_unit_declared: TranslationUnit
    extrinsics_convention_declared: ExtrinsicsConvention
    baseline_range_m: tuple[float, float] | None = (0.02, 0.30)

    def __post_init__(self) -> None:
        if self.right_from_left.from_frame != self.left.camera_id:
            raise CalibrationError("right_from_left has the wrong source frame")
        if self.right_from_left.to_frame != self.right.camera_id:
            raise CalibrationError("right_from_left has the wrong target frame")
        if self.left.image_size != self.right.image_size:
            raise CalibrationError(
                f"stereo cameras must share a source size, got {self.left.image_size} and "
                f"{self.right.image_size}"
            )
        baseline = self.right_from_left.baseline_m
        if not math.isfinite(baseline) or baseline <= 0:
            raise CalibrationError("stereo baseline must be finite and non-zero")
        if self.baseline_range_m is not None:
            if len(self.baseline_range_m) != 2:
                raise CalibrationError("baseline_range_m must contain (minimum, maximum)")
            minimum = _finite_float(self.baseline_range_m[0], "baseline_range_m.minimum")
            maximum = _finite_float(self.baseline_range_m[1], "baseline_range_m.maximum")
            if minimum <= 0 or maximum < minimum:
                raise CalibrationError("baseline_range_m must satisfy 0 < minimum <= maximum")
            if not minimum <= baseline <= maximum:
                raise CalibrationError(
                    f"baseline {baseline:.6f} m is outside the configured safety range "
                    f"[{minimum:.6f}, {maximum:.6f}] m; check units/direction"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "source_path": str(self.source_path),
            "reference_camera_id": self.reference_camera_id,
            "axis_convention": "opencv_x_right_y_down_z_forward",
            "length_unit": "m",
            "translation_unit_declared": self.translation_unit_declared,
            "extrinsics_convention_declared": self.extrinsics_convention_declared,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "T_right_from_left": self.right_from_left.to_matrix4(),
            "baseline_m": self.right_from_left.baseline_m,
            "baseline_range_m": (
                list(self.baseline_range_m) if self.baseline_range_m is not None else None
            ),
        }


def _validate_rotation_like_intrinsics(intrinsics: Mat3, camera_id: str) -> None:
    if len(intrinsics) != 3 or any(len(row) != 3 for row in intrinsics):
        raise CalibrationError(f"camera {camera_id}: intrinsics must be 3x3")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for row in intrinsics
        for value in row
    ):
        raise CalibrationError(f"camera {camera_id}: intrinsics must be finite real numbers")
    expected_zero = (
        intrinsics[0][1],
        intrinsics[1][0],
        intrinsics[2][0],
        intrinsics[2][1],
    )
    if any(abs(value) > 1e-12 for value in expected_zero) or abs(intrinsics[2][2] - 1.0) > 1e-12:
        raise CalibrationError(
            f"camera {camera_id}: intrinsics must have pinhole form [[fx,0,cx],[0,fy,cy],[0,0,1]]"
        )


def _kb_radius(theta: float, coefficients: tuple[float, float, float, float]) -> float:
    theta2 = theta * theta
    polynomial = 1.0
    power = theta2
    for coefficient in coefficients:
        polynomial += coefficient * power
        power *= theta2
    return theta * polynomial


def _kb_derivative_u(u: float, coefficients: tuple[float, float, float, float]) -> float:
    return (
        1.0
        + 3.0 * coefficients[0] * u
        + 5.0 * coefficients[1] * u**2
        + 7.0 * coefficients[2] * u**3
        + 9.0 * coefficients[3] * u**4
    )


def _certify_kb_derivative_positive(
    lower_u: float,
    upper_u: float,
    coefficients: tuple[float, float, float, float],
    *,
    depth: int = 0,
) -> None:
    """Conservatively certify that the KB radial derivative is positive.

    Natural interval bounds are conservative but rigorous for each monomial on
    non-negative ``u = theta**2``. Ambiguous intervals are bisected rather than
    silently accepted.
    """

    derivative_coefficients = (
        3.0 * coefficients[0],
        5.0 * coefficients[1],
        7.0 * coefficients[2],
        9.0 * coefficients[3],
    )
    lower_bound = 1.0
    upper_bound = 1.0
    for power, coefficient in enumerate(derivative_coefficients, start=1):
        low_term = lower_u**power
        high_term = upper_u**power
        if coefficient >= 0:
            lower_bound += coefficient * low_term
            upper_bound += coefficient * high_term
        else:
            lower_bound += coefficient * high_term
            upper_bound += coefficient * low_term
    if lower_bound > _KB_DERIVATIVE_EPSILON:
        return
    midpoint = (lower_u + upper_u) / 2.0
    if (
        upper_bound <= _KB_DERIVATIVE_EPSILON
        or _kb_derivative_u(lower_u, coefficients) <= _KB_DERIVATIVE_EPSILON
        or _kb_derivative_u(midpoint, coefficients) <= _KB_DERIVATIVE_EPSILON
        or _kb_derivative_u(upper_u, coefficients) <= _KB_DERIVATIVE_EPSILON
    ):
        raise CalibrationError("KB radial mapping folds or becomes non-monotonic")
    if depth >= 40 or upper_u - lower_u <= 1e-14:
        raise CalibrationError("could not certify KB radial mapping as strictly monotonic")
    _certify_kb_derivative_positive(lower_u, midpoint, coefficients, depth=depth + 1)
    _certify_kb_derivative_positive(midpoint, upper_u, coefficients, depth=depth + 1)


def _validate_kb_invertibility(camera: CameraKB) -> None:
    """Require a one-to-one KB radial map over every source-image pixel."""

    width, height = camera.image_size
    fx, fy = camera.intrinsics[0][0], camera.intrinsics[1][1]
    cx, cy = camera.intrinsics[0][2], camera.intrinsics[1][2]
    maximum_distorted_radius = max(
        math.hypot((x - cx) / fx, (y - cy) / fy)
        for x in (0.0, float(width - 1))
        for y in (0.0, float(height - 1))
    )
    previous_theta = 0.0
    # A KB ray angle is physically bounded by pi. Validate successive intervals
    # until the polynomial reaches the most distant image corner.
    for step in range(1, 1025):
        theta = math.pi * step / 1024.0
        _certify_kb_derivative_positive(
            previous_theta * previous_theta,
            theta * theta,
            camera.distortion,
        )
        if _kb_radius(theta, camera.distortion) >= maximum_distorted_radius:
            return
        previous_theta = theta
    raise CalibrationError(
        f"camera {camera.camera_id}: KB mapping cannot invert the full image radius "
        f"{maximum_distorted_radius:.6f}"
    )


def _parse_camera(entry: dict[str, Any]) -> CameraKB:
    raw_camera_id = entry.get("id")
    if not isinstance(raw_camera_id, str) or not raw_camera_id.strip():
        raise CalibrationError("every camera entry must have a non-empty id")
    camera_id = raw_camera_id.strip()
    model = str(entry.get("distortion_model", "")).strip().upper()
    if model != "KB":
        raise CalibrationError(f"camera {camera_id}: expected distortion_model=KB, got {model!r}")
    width = _positive_int(entry.get("image_width"), f"{camera_id}.image_width")
    height = _positive_int(entry.get("image_height"), f"{camera_id}.image_height")
    intrinsics = entry.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise CalibrationError(f"camera {camera_id}: missing intrinsics mapping")
    fx = _finite_float(intrinsics.get("fx"), f"{camera_id}.fx")
    fy = _finite_float(intrinsics.get("fy"), f"{camera_id}.fy")
    cx = _finite_float(intrinsics.get("cx"), f"{camera_id}.cx")
    cy = _finite_float(intrinsics.get("cy"), f"{camera_id}.cy")
    distortion = entry.get("distortion")
    if not isinstance(distortion, dict):
        raise CalibrationError(f"camera {camera_id}: missing distortion mapping")
    coefficients = tuple(
        _finite_float(distortion.get(f"k{i}"), f"{camera_id}.k{i}") for i in range(1, 5)
    )
    unsupported = {
        name: _finite_float(distortion.get(name, 0.0), f"{camera_id}.{name}")
        for name in ("k5", "k6", "p1", "p2")
    }
    nonzero = {name: value for name, value in unsupported.items() if abs(value) > 1e-12}
    if nonzero:
        raise CalibrationError(
            f"camera {camera_id}: unsupported non-zero KB coefficients {nonzero}"
        )
    raw_name = entry.get("name", camera_id)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise CalibrationError(f"camera {camera_id}: name must be a non-empty string")
    return CameraKB(
        camera_id=camera_id,
        name=raw_name.strip(),
        image_size=(width, height),
        intrinsics=((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
        distortion=coefficients,
    )


def load_orbbec_stereo(
    path: str | Path,
    *,
    left_id: str,
    right_id: str,
    translation_unit: TranslationUnit,
    extrinsics_convention: ExtrinsicsConvention,
    baseline_range_m: tuple[float, float] | None = (0.02, 0.30),
) -> StereoCalibration:
    """Load a two-camera Orbbec KB YAML without guessing units or transform direction."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CalibrationError(f"calibration file does not exist: {source}")
    try:
        import yaml
    except ImportError as exc:
        raise CalibrationError("PyYAML is required to read calibration files") from exc
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise CalibrationError(f"failed to parse YAML {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise CalibrationError("calibration root must be a mapping")
    entries = document.get("cameras")
    if not isinstance(entries, list) or len(entries) < 2:
        raise CalibrationError("calibration must contain at least two camera entries")
    if not isinstance(left_id, str) or not left_id.strip():
        raise CalibrationError("left_id must be a non-empty string")
    if not isinstance(right_id, str) or not right_id.strip():
        raise CalibrationError("right_id must be a non-empty string")
    left_id = left_id.strip()
    right_id = right_id.strip()
    if left_id == right_id:
        raise CalibrationError("left_id and right_id must be different")

    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CalibrationError("every cameras[] element must be a mapping")
        raw_camera_id = entry.get("id")
        if not isinstance(raw_camera_id, str) or not raw_camera_id.strip():
            raise CalibrationError("every camera entry must have a non-empty id")
        camera_id = raw_camera_id.strip()
        if camera_id in by_id:
            raise CalibrationError(f"duplicate camera id {camera_id!r}")
        by_id[camera_id] = entry
    missing = [camera_id for camera_id in (left_id, right_id) if camera_id not in by_id]
    if missing:
        raise CalibrationError(
            f"requested camera ids are absent: {missing}; available={sorted(by_id)}"
        )

    info = document.get("calibration_info")
    if not isinstance(info, dict) or not str(info.get("reference_camera", "")).strip():
        raise CalibrationError("calibration_info.reference_camera is required")
    reference_id = str(info["reference_camera"]).strip()
    if reference_id not in by_id:
        raise CalibrationError(f"reference camera {reference_id!r} is not present")

    factor = {"mm": 1e-3, "m": 1.0}.get(translation_unit)
    if factor is None:
        raise CalibrationError(f"unsupported translation unit {translation_unit!r}")
    if extrinsics_convention not in ("reference_to_camera", "camera_to_reference"):
        raise CalibrationError(f"unsupported extrinsics convention {extrinsics_convention!r}")

    cam_from_ref: dict[str, Transform3D] = {}
    for camera_id, entry in by_id.items():
        extrinsics = entry.get("extrinsics")
        if not isinstance(extrinsics, dict):
            raise CalibrationError(f"camera {camera_id}: missing extrinsics")
        rotation = _mat3(extrinsics.get("rotation"), f"{camera_id}.extrinsics.rotation")
        translation = _vec3(
            extrinsics.get("translation"), f"{camera_id}.extrinsics.translation", factor
        )
        if extrinsics_convention == "reference_to_camera":
            normalized = Transform3D(reference_id, camera_id, rotation, translation)
        else:
            normalized = Transform3D(camera_id, reference_id, rotation, translation).inverse()
        cam_from_ref[camera_id] = normalized

    reference_transform = cam_from_ref[reference_id]
    identity_rotation_error = max(
        abs(reference_transform.rotation[i][j] - (1.0 if i == j else 0.0))
        for i in range(3)
        for j in range(3)
    )
    identity_translation_error = max(abs(value) for value in reference_transform.translation_m)
    if (
        identity_rotation_error > _REFERENCE_ROTATION_TOLERANCE
        or identity_translation_error > _REFERENCE_TRANSLATION_TOLERANCE_M
    ):
        raise CalibrationError(
            f"reference camera {reference_id!r} must have identity self-extrinsics; "
            f"rotation_error={identity_rotation_error:.6g}, "
            f"translation_error_m={identity_translation_error:.6g}"
        )

    left = _parse_camera(by_id[left_id])
    right = _parse_camera(by_id[right_id])
    ref_from_left = cam_from_ref[left_id].inverse()
    right_from_left = compose_transforms(cam_from_ref[right_id], ref_from_left)
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(f"\0{left_id}\0{right_id}\0{translation_unit}\0{extrinsics_convention}".encode())
    return StereoCalibration(
        calibration_id=f"sha256:{digest.hexdigest()}",
        source_path=source,
        reference_camera_id=reference_id,
        left=left,
        right=right,
        right_from_left=right_from_left,
        translation_unit_declared=translation_unit,
        extrinsics_convention_declared=extrinsics_convention,
        baseline_range_m=baseline_range_m,
    )
