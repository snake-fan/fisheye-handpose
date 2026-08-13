# Architecture

## Stable boundary versus reference composition

The stable API consists of typed geometry/evidence artifacts and the `fhp21/v1` output
contract. It is **not** a promise that every future model must contain the same sequence
of neural networks. The initial reference composition is:

```text
FrameSet(sync + calibration)
  -> Detector(output in native distorted pixels)
  -> VirtualPerspectiveCrop
  -> NativeViewEvidence
  -> explicit LandmarkMapper
  -> CanonicalViewEvidence
  -> CrossViewAssociation / MultiViewHandGroup
  -> CalibratedFusion / immutable SpatialObservation
  -> optional KinematicRefiner / PoseEstimate
  -> TemporalRefiner / PoseEstimate
  -> fhp21/v1 output
```

A backend may internally tile or rectify an image, jointly detect and regress pose, consume
native fisheye input, or perform learned multi-view fusion. An adapter may therefore fold
several reference stages together, but it must return the corresponding typed artifact at
the boundary. In particular, detector boxes are returned in native source pixels and
per-view pose evidence is returned in physical crop pixels after undoing internal resize,
letterbox, or mirroring.

Core code must not import a detector, parameterized hand model, model repository, or
weights package. Model adapters live under `backends/`. `BackendManifest` declares API and
model versions, input spaces/joint sets, capabilities, source revision, code and weight
terms, artifact hash, and an explicit commercial-use classification. `UNKNOWN` commercial
status is not deployment approval. A parameterized hand model is an optional kinematic
backend, never the definition of `fhp21/v1`.

## Geometry and coordinate contracts

No naked pixel or 3D array crosses a module boundary. Important objects carry:

- a named `PixelSpace` or `CoordinateFrame3D`;
- the calibration ID from which geometry was derived;
- actual source-camera timestamps rather than only a frame index;
- source camera/view/crop identifiers and backend/mapping provenance.

Pixels are zero-based `(u, v)` locations at pixel centres; image sizes are `(width,
height)` and boxes are half-open `(x_min, y_min, x_max, y_max)`. Three-dimensional lengths
are metres. Transform names follow `T_B_from_A`, meaning `X_B = R @ X_A + t`.

`FrameSet` contains one sample per available named camera, a representative timestamp, a
rig coordinate frame, and the configured sync tolerance. Every `ImageView` retains its
actual timestamp; the observed skew is derived and checked. This supports one to many
available cameras without a fixed `[left_camera, right_camera]` tensor contract. The
current Orbbec stereo loader and full-frame `stereoRectify` utility are initial capture and
QA adapters, not the generic camera-rig or perception contract.

The current stable contract emits only camera- or rig-frame 3D and does not carry a world
transform. A future world-frame extension must use a typed trajectory/rig-pose context,
including the pose evaluation timestamp, clock/interpolation semantics, trajectory source,
validity and uncertainty. It must not add a bare dynamic transform to camera calibration.

### Virtual perspective crop

A `VirtualCamera` is a typed local pinhole camera with intrinsics, image size, calibration
and source-camera IDs, plus `T_rig_from_virtual`. It shares the source physical camera's
optical centre. The authoritative crop mapping is ray based:

1. unproject a virtual crop pixel with the virtual intrinsics;
2. rotate its ray into the rig frame with `T_rig_from_virtual`;
3. project that ray with the calibrated source fisheye model.

Fisheye-to-pinhole warping is not represented as a single homography. `PerspectiveCrop`
also carries a valid-pixel mask and a versioned crop-policy ID. The geometry layer never
mirrors a crop. A model adapter that performs left/right normalization must undo that
mirror before emitting evidence in the physical crop space.

## Canonical mapping boundary

`PoseEvidenceBackend` emits `NativeViewEvidence` in its declared native joint set.
`LandmarkMapper` is a mandatory boundary—even an identity mapping has a persisted mapping
record. It produces `CanonicalViewEvidence`, the only evidence type accepted by fusion.

A `LandmarkMappingRecord` covers all 21 targets in order and records source names/indices,
source definitions, a versioned source-construction ID, mapping method and one of:

- `EXACT`: source and target use the same operational construction and version;
- `REGRESSED`: a learned regressor produces the target;
- `DERIVED`: deterministic geometry combines or extrapolates source values;
- `MISSING`: no target value exists and fusion must ignore it.

Name, index, or broad anatomical meaning alone does not establish `EXACT`. Mapping IDs
change whenever a vertex choice, annotation rule, regressor, definition, or derivation
changes. Canonical wrist and fingertip targets are operational references, not claims that
unique anatomical centres/surface extrema are directly observable. Missing canonical
evidence is encoded as invalid coordinates, unbounded covariance and zero visibility; it
must never be consumed as a measurement.

## Association and calibrated fusion

`MultiViewHandGroup` is the self-contained input to `FusionBackend`. Each member links its
source `ImageView`, native-pixel detection, physical `PerspectiveCrop`, typed
`VirtualCamera`, and canonical evidence. The group carries the complete `FrameSet`,
sequence-local track ID, association confidence and optional anatomical handedness. It
contains at most one candidate per camera and supports any member count from one upward.
Fusion therefore requires no global crop-camera or calibration lookup.

The baseline fuser minimizes robust, uncertainty-weighted reprojection error through the
member virtual cameras. It checks visibility, valid crop pixels, cheirality, ray angle,
timestamp skew and residuals. A single view or near-parallel rays do not provide stereo
metric certainty. If a backend does not supply defensible monocular metric-depth evidence,
the affected raw 3D landmark is invalid. A prior may initialize or regularize fusion but
cannot turn a prior-only value into a current measurement.

Fusion emits an immutable `SpatialObservation`. Each valid landmark records covariance,
current evidence source, supporting view IDs and per-view reprojection residuals. A
kinematic or temporal stage creates a separate `PoseEstimate` linked by source observation
IDs; it never overwrites the raw observation.

For the current milestone, both raw and refined outputs are restricted to named `CAMERA`
or `RIG` coordinate frames.

## Output state is factored, not overloaded

The following dimensions are independent:

- `Validity`: whether a coordinate may be consumed;
- `EvidenceSource`: current `MULTIVIEW`, `MONOCULAR`, or `NONE` image evidence;
- `EstimateStage`: raw fusion, kinematic refinement, or temporal refinement;
- `EstimateKind`: copied measurement, adjusted/refined measurement, or prediction.

For example, a temporally adjusted stereo observation is valid, has `MULTIVIEW` evidence,
stage `TEMPORAL_REFINEMENT`, and kind `REFINED`. An occlusion extrapolation can be valid,
have evidence `NONE`, the same stage, and kind `PREDICTED`, with increasing covariance and
decaying confidence. Visibility, validity, uncertainty/confidence, handedness, and stage
are never aliases for one another.

## Temporal lifecycle

A `TemporalRefiner` instance owns at most one sequence-local track. It advertises one of:

- `CAUSAL`: zero declared latency and immediate emission;
- `FIXED_LAG`: finite frame or nanosecond lag and delayed emission;
- `OFFLINE`: unbounded latency and emission at `flush`/`reset`.

The caller opens a track, then pushes strictly increasing values: either raw
`SpatialObservation` records or `KINEMATIC_REFINEMENT` estimates. Every accepted input
yields exactly one temporal `PoseEstimate`, returned once by `push`, `flush`, or `reset`;
delayed outputs remain timestamp ordered. Both terminal operations emit pending values and
close the track. Re-use requires another `open_track`. Track-ID switches, scene cuts,
calibration changes and excessive time gaps are hard reset boundaries and may not share
temporal state.

Temporal backends consume real timestamps and per-landmark covariance/evidence state.
Advanced implementations may retain per-view reprojection evidence, optimize root motion
and local articulation separately, and use causal, fixed-lag, or bidirectional windows.
They must preserve source observation IDs and distinguish prediction from refinement.

## Implemented foundation and H20 compatibility baseline

Implemented now:

- deterministic multi-part session discovery;
- strict Orbbec KB4 stereo calibration parsing;
- explicit extrinsic convention and translation-unit normalization;
- monotonic one-to-one hardware timestamp pairing;
- full PyAV decode audit with frame-count/timestamp-count equality;
- a timestamp-indexed presentation-order stereo pair reader;
- true `cv2.fisheye.stereoRectify` geometry and remap construction for QA/debugging;
- empirical rectified epipolar, disparity-sign, and positive-depth QA;
- versioned `fhp21/v1` definitions and model-neutral contracts;
- atomic JSON/CSV CLI artifacts with calibration hashes and provenance;
- one immutable `runs/<item>/<run>/` trace shared by audit and worker stages;
- a process-isolated H20 baseline with RTMDet/RTMPose, calibrated point rectification,
  cross-view association, metric triangulation, track-local MANO, real-time-delta EMA,
  and FHP21 JSONL export;
- a multi-run read-only API and independent React inspector.

Not yet implemented:

- explicit hand-centred virtual-perspective crop generation (the compatibility worker
  currently detects on the raw fisheye frame and uses the top-down model crop);
- learned ray/feature fusion, calibrated 2D covariance, and a learned temporal prior;
- a production-safe replacement for the legacy OpenMMLab/PyTorch environment.

## Non-negotiable invariants

- No frame-index pairing without hardware timestamps.
- No implicit calibration, crop-camera, pixel-space, or coordinate-frame singleton.
- No fixed camera ordering or requirement that both stereo views contain a hand.
- No backend-native array reaches fusion without an explicit canonical mapping.
- No parameterized hand model defines the canonical landmark semantics.
- No prior-only value is labelled as a raw image measurement.
- No refined pose overwrites its raw observation.
- No temporal state crosses an explicit reset boundary.
