# fisheye-handpose

An independent, calibration-first project for extracting metric 21-landmark hand poses
from synchronized stereo fisheye video. It does **not** import or execute the existing
VLA pipeline.

The project is intentionally split into two layers:

1. A strict geometry foundation: session discovery, timestamp pairing, Orbbec KB4
   calibration normalization, full video decode audit, true stereo rectification/QA,
   timestamp-indexed stereo reading, and a versioned 21-landmark contract.
2. Pluggable perception: native-fisheye detection, virtual perspective hand crops,
   per-view pose evidence, calibrated fusion, optional kinematic refinement, and
   confidence-aware temporal refinement.

Only the first layer is implemented in the initial milestone. No model weights or
model-specific licenses are embedded in the core package.

## Coordinate and unit conventions

- 3D axes use the OpenCV camera convention: `+x right`, `+y down`, `+z forward`.
- Internal timestamps are integer nanoseconds.
- Internal lengths are metres.
- A transform named `T_B_from_A` means `X_B = R @ X_A + t`.
- The canonical hand landmark set is `fhp21/v1`; see
  [docs/output-schema.md](docs/output-schema.md).

The supplied Orbbec YAML does not declare translation units or the direction of its
extrinsics. Commands therefore require both values explicitly; they are never guessed.

## Install

```bash
uv python install 3.11
uv sync --locked --extra dev --no-editable
uv run --locked --no-editable fisheye-handpose schema
```

`uv.lock` is the source of truth for the cross-platform geometry environment. Use
`uv lock --check` in CI and `uv sync --locked --no-editable` when reproducing an existing
lock; do not replace these commands with an unconstrained `pip install`. The non-editable
install also tests the actual packaged artifact instead of letting source-tree imports
hide packaging mistakes. Keep both `--locked` and `--no-editable` on subsequent `uv run`
commands so uv neither changes the resolution nor switches the project back to an
editable install.

## Geometry CLI

Discover complete sessions:

```bash
uv run --locked --no-editable fisheye-handpose discover /path/to/fisheye_data
```

Inspect and normalize an Orbbec calibration:

```bash
uv run --locked --no-editable fisheye-handpose inspect-calibration calibration_camera.yaml \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera
```

Pair two hardware timestamp streams monotonically and one-to-one:

```bash
uv run --locked --no-editable fisheye-handpose pair-pts left_pts.csv right_pts.csv \
  --max-skew-us 1000 --output pairs.csv
```

Fully audit a capture session and build true stereo rectification geometry:

```bash
uv run --locked --no-editable fisheye-handpose audit-session /path/to/session \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera \
  --max-skew-us 1000 --output report.json
```

`audit-session` fully decodes both videos, requires decoded frame counts to equal their
hardware-timestamp counts, checks synchronization/geometry gates, and by default runs
empirical epipolar/disparity/positive-depth QA. Its JSON report is written atomically;
invalid or inconclusive physical geometry exits non-zero. Other commands write
machine-readable JSON to stdout unless `--output` is supplied; diagnostics go to stderr.

Full-frame stereo rectification is a geometry/QA utility, not the mandatory image fed
to a hand model. The perception path will detect in native fisheye pixels and resample
only hand-centred virtual perspective crops so that peripheral field of view is not
silently discarded.

## Tests

```bash
uv run --locked --extra dev --no-editable pytest
uv run --locked --extra dev --no-editable ruff check .
```

The tests include synthetic projection/rectification geometry, timestamp drops and
offset tails, full video decode and frame selection, CLI failure reports, calibration
direction/unit checks, deterministic discovery, and the `fhp21/v1` contract.

See [docs/architecture.md](docs/architecture.md) for the planned model-facing APIs and
the next implementation milestone. The attached engineering proposal has been evaluated
in [docs/design-review.md](docs/design-review.md); that review is the implementation
decision record for which parts are retained, revised, or rejected.

## H20 perception environment

The legacy RTMDet + RTMPose Hand5 compatibility stack is intentionally isolated in
[`deploy/mmpose-h20`](deploy/mmpose-h20/README.md). It uses its own Python 3.10 uv lock,
CUDA 12.1 binary sources, fail-closed runtime doctor, explicit detector/pose artifacts,
and separate RTMPose/MANO smoke commands. Do not add CUDA/OpenMMLab dependencies to this
core environment. That subproject was locked with uv 0.12.3 and accepts only
`uv>=0.12.3,<0.13`; follow its README rather than using the root environment commands.

The H20 profile is for controlled research reproduction. Model checkpoint rights,
MANO/SMPL-X terms, and the known security debt of the pinned legacy PyTorch stack require
separate approval before any production or commercial use.
