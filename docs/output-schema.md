# `fhp21/v1` output contract

`fhp21/v1` is an operational **landmark** contract, not a claim that unique anatomical
points are directly observable from images. The five tips are canonical surface references
and the wrist is a canonical kinematic reference. Names, ordering, target semantics and the
construction recorded by each mapping are normative. A shared name or broad anatomical
meaning alone never establishes an `EXACT` mapping.

## Normative landmark definitions

- `wrist_center`: the `fhp21/v1` canonical kinematic wrist reference at the proximal
  hand/wrist base. Its realization (native keypoint, annotation rule, mesh regressor, or
  derived point) is declared by the mapping record. It does not claim a unique, directly
  observable radiocarpal centre.
- Every `*_cmc`, `*_mcp`, `*_ip`, `*_pip`, and `*_dip`: a latent estimate of the named
  articulation's 3D centre of rotation, not a directly visible skin point. Its source
  construction remains part of mapping provenance.
- Every `*_tip`: the `fhp21/v1` canonical distal fingertip surface reference. Its
  realization (annotation rule, selected mesh vertex, or regressor) is declared by the
  mapping record. It does not claim a unique, directly observable distal-most soft-tissue
  point or the distal phalanx joint centre.

The table fixes ordering and kinematic visualization edges; parenthood does not by itself
define a bone-length or joint-angle prior.

| Index | Name | Kind | Parent |
|---:|---|---|---:|
| 0 | wrist_center | reference point | -1 |
| 1 | thumb_cmc | articulation centre | 0 |
| 2 | thumb_mcp | articulation centre | 1 |
| 3 | thumb_ip | articulation centre | 2 |
| 4 | thumb_tip | surface point | 3 |
| 5 | index_mcp | articulation centre | 0 |
| 6 | index_pip | articulation centre | 5 |
| 7 | index_dip | articulation centre | 6 |
| 8 | index_tip | surface point | 7 |
| 9 | middle_mcp | articulation centre | 0 |
| 10 | middle_pip | articulation centre | 9 |
| 11 | middle_dip | articulation centre | 10 |
| 12 | middle_tip | surface point | 11 |
| 13 | ring_mcp | articulation centre | 0 |
| 14 | ring_pip | articulation centre | 13 |
| 15 | ring_dip | articulation centre | 14 |
| 16 | ring_tip | surface point | 15 |
| 17 | little_mcp | articulation centre | 0 |
| 18 | little_pip | articulation centre | 17 |
| 19 | little_dip | articulation centre | 18 |
| 20 | little_tip | surface point | 19 |

## Mandatory mapping record

Every backend-native joint set passes through a `LandmarkMapper`, including an identity
mapping. Its persisted `LandmarkMappingRecord` contains:

- a stable `mapping_id`, source joint-set ID, target schema version, and provenance;
- ordered source landmark names;
- exactly one entry for every target index;
- source indices, the source landmark's geometric definition, a versioned
  `source_construction_id`, and mapping method;
- one mapping quality per target:
  - `EXACT`: the same operational construction and version as the canonical target;
  - `REGRESSED`: learned regressor, whose identity/version belongs in provenance;
  - `DERIVED`: deterministic combination or extrapolation, described by `method`;
  - `MISSING`: no value; source indices must be empty.

Every canonical target has its own versioned construction ID. For `EXACT`, the entry's
`source_construction_id` must equal that target ID; this is enforced by the protocol. A
missing entry has no source construction ID.

`mapping_id` changes when source definitions, annotation rules, selected mesh vertices,
regressor weights, or derivation logic change. Merely returning 21 values, matching
names/index order, or targeting a similar anatomical region is not an exact canonical
mapping. In particular, a native 16-joint parametric-hand output is not `fhp21/v1` until
wrist and all five surface tips have explicit mappings.

At the 2D evidence boundary, `MISSING` means invalid/NaN coordinates, unbounded covariance
and zero visibility. It cannot contribute to fusion.

## Record types and immutability

There are two distinct output records:

1. `SpatialObservation` is the immutable raw result of calibrated fusion. Its stage is
   always `RAW_FUSION`. A prior-only value is invalid rather than a measurement.
2. `PoseEstimate` is a kinematically or temporally processed value. It has a new estimate
   ID and one or more `source_observation_ids`; it never overwrites those observations.

Each record identifies:

- `schema_version = "fhp21/v1"`, sequence-local `track_id`, and timestamp in integer ns;
- `calibration_id`, `output_frame.frame_id`, camera/rig frame kind, axis convention, and
  metres;
- anatomical handedness probabilities `{left, right, unknown}` summing to one (never image
  left/right position);
- `landmarks_xyz_m[21,3]` and `covariance_m2[21,3,3]`;
- `validity[21]`, `evidence_source[21]`, visibility and thresholded confidence;
- `support_view_ids[21]` and per-landmark, per-view pixel reprojection residuals;
- all contributing `mapping_ids` and backend/version provenance.

`PoseEstimate` additionally identifies its refinement stage and `kind[21]`.

## Independent per-landmark state

These dimensions must not be collapsed into one score or enum:

| Dimension | Values / meaning |
|---|---|
| `validity` | `VALID` means the coordinate may be consumed; otherwise `INVALID`. |
| `evidence_source` | Current image support: `MULTIVIEW`, `MONOCULAR`, or `NONE`. |
| `stage` | `RAW_FUSION`, `KINEMATIC_REFINEMENT`, or `TEMPORAL_REFINEMENT`. |
| `kind` | `MEASURED`, `REFINED`, or `PREDICTED`; present on `PoseEstimate`. |
| `visibility_probability` | Probability the landmark is directly visible in at least one available source view. |
| `covariance_m2` | Positional uncertainty in the declared 3D output frame. |
| `confidence_probability` | Optional probability that Euclidean position error is no greater than the record's positive `confidence_radius_m`; both fields are absent when confidence is not calibrated. |

The confidence radius makes confidence comparable and testable; an undocumented model
score is not accepted as this probability. An implementation without calibrated
confidence emits both confidence fields as `null`. A consumer should primarily use
validity and covariance, and apply its own task threshold if calibrated confidence is
unavailable.

Examples:

- A raw, successfully triangulated point is `VALID + MULTIVIEW`, with stage implicitly
  `RAW_FUSION`.
- A temporal optimizer's adjusted version of it is `VALID + MULTIVIEW +
  TEMPORAL_REFINEMENT + REFINED`.
- A short-occlusion extrapolation can be `VALID + NONE + TEMPORAL_REFINEMENT + PREDICTED`,
  but its covariance must grow and confidence decay with the gap.
- An underconstrained raw monocular point is `INVALID + MONOCULAR` unless the active
  backend explicitly supplies and declares defensible metric-depth evidence.

## Serialization rules

- JSON encoders use enum string values exactly as documented and preserve 64-bit integer
  timestamps; non-JSON transports must not round timestamps through floating point.
- Non-finite internal values for invalid/missing landmarks are represented by explicit
  `null` coordinates/covariance in JSON, never non-standard `NaN` tokens. `validity` remains
  authoritative.
- View ID order is not semantically meaningful. Landmark order is exactly the table above.
- `track_id` is unique only within `sequence_id`; it is not a cross-session person ID.
- Current `fhp21/v1` records are limited to named `CAMERA` or `RIG` frames. World output is
  reserved for a future typed trajectory/rig-pose extension with time and uncertainty.
- A calibration change creates a temporal reset boundary and a new provenance segment.
