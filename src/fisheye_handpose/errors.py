"""Domain-specific failures with actionable diagnostics."""


class FisheyeHandposeError(Exception):
    """Base class for expected input, calibration, or geometry failures."""


class CalibrationError(FisheyeHandposeError):
    """The calibration is missing, ambiguous, or physically invalid."""


class TimestampError(FisheyeHandposeError):
    """A timestamp stream violates the strict input contract."""


class SyncError(FisheyeHandposeError):
    """Two timestamp streams cannot be paired unambiguously."""


class DiscoveryError(FisheyeHandposeError):
    """A capture directory is incomplete or ambiguous."""


class GeometryError(FisheyeHandposeError):
    """Stereo rectification or projection geometry is invalid."""
