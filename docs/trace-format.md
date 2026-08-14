# Stage trace format

`fisheye-handpose` stores one pipeline run as an append-only, content-addressed directory.
The trace is designed for stage-by-stage diagnosis: a consumer can replay provenance,
compare raw and refined outputs, and verify that neither records nor attached artifacts
changed after the run.

## Directory layout

Completed data items are catalogued with two immutable identity levels:

```text
runs/
  SAFE_ITEM_ID/
    RUN_ID/
      run_manifest.json
      trace.jsonl
      run_summary.json
      .writer.lock
      blobs/sha256/...
```

`SAFE_ITEM_ID` identifies the source data item and `RUN_ID` identifies one execution of
the pipeline. An explicit ID is accepted only when it is one safe ASCII path component
matching `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. If the source session ID needs
normalization, the default item ID adds the first eight hexadecimal characters of its
SHA-256 digest so two distinct source IDs cannot silently collapse onto one slug. A
generated run ID combines UTC time with a random suffix. An existing run directory is
never reopened or overwritten by `run-item`.

The command that creates this catalog entry is:

```bash
fisheye-handpose run-item /path/to/session \
  --runs-root runs \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera \
  --max-skew-us 1000
```

`--item-id` and `--run-id` may be supplied for deterministic orchestration. The command
prints one JSON result containing `item_id`, `run_id`, `run_dir`, terminal `status`,
`audit_status`, and `output_status`.

Each leaf directory is one canonical stage trace:

```text
RUN_DIR/
  run_manifest.json       # immutable run identity, config, inputs, manifest hash
  trace.jsonl             # append-only ordered stage records and hash chain
  run_summary.json        # immutable terminal status and last-record hash
  .writer.lock            # single-writer advisory lock
  blobs/sha256/ab/
    abcdef...png           # content-addressed images, arrays, reports, and video clips
```

The current core runner uses one `RunArtifactWriter` for the complete leaf directory.
It appends the audit as the provenance prefix, then appends later stage records to that
same hash chain. External model workers attach through the minimal
`PipelineStageExecutor.execute(context, writer)` boundary and return a
`PipelineExecutionSummary`; the core does not import or pre-compose their model stacks.
When perception/MANO/temporal backends are absent, their stages are not
silently omitted: each has a `SKIPPED` record with `output_status: NOT_PRODUCED`. An
audit-gate failure still finalizes and retains the leaf with terminal status `FAILED`, so
the front end can inspect the failure instead of losing its intermediate evidence.

`run_manifest.json` starts the run with status `ACTIVE`. The manifest is never rewritten.
The terminal status (`COMPLETED` or `FAILED`) is written separately to
`run_summary.json`. Absence of the summary means that an ACTIVE run may be resumed with
`RunArtifactWriter.open()`.

## Trace record

Every line of `trace.jsonl` is one `fisheye-handpose/trace-record/v1` JSON object:

```json
{
  "schema_version": "fisheye-handpose/trace-record/v1",
  "ordinal": 12,
  "record_id": "session:000123:pose2d:left",
  "timestamp_utc": "2026-08-13T04:00:00.000000Z",
  "stage": "POSE_2D",
  "status": "SUCCEEDED",
  "event": "view_keypoints_inferred",
  "parent_ids": ["session:000123:crop:left"],
  "blobs": [
    {
      "sha256": "...64 lowercase hex characters...",
      "bytes": 18422,
      "role": "crop",
      "media_type": "image/png",
      "relative_path": "blobs/sha256/ab/abcdef...png"
    }
  ],
  "payload": {
    "frame_id": "frame/000123",
    "timestamp_ns": 5100000000,
    "view_id": "left",
    "track_id": "hand-0",
    "keypoints_uv": [[123.4, 98.7]],
    "keypoint_scores": [0.96]
  },
  "previous_hash": "...previous record hash or null...",
  "record_hash": "...hash of this record body..."
}
```

The required stage vocabulary is `SYSTEM`, `DISCOVERY`, `CALIBRATION`, `DECODE`,
`SYNCHRONIZATION`, `RECTIFICATION`, `DETECTION`, `POSE_2D`,
`CROSS_VIEW_ASSOCIATION`, `RAW_FUSION`, `KINEMATIC_REFINEMENT`,
`TEMPORAL_REFINEMENT`, `QA`, and `EXPORT`. A stage must record `SKIPPED`, `WARNING`, or
`FAILED` explicitly; absence of a record must not be interpreted as success.

`parent_ids` form the provenance DAG. For example, a stereo fusion record points to the
left/right per-view evidence and calibration record; a temporal estimate points to its
current kinematic estimate and, when applicable, the previous temporal estimate.

## Payload conventions

The viewer groups records by `payload.frame_id` and filters by `payload.track_id`.
Producers should also use the following stable evidence keys:

- detection: `view_id`, `detections[]`, each with `bbox_xyxy`, `score`, and `label`;
- per-view pose: `view_id`, `keypoints_uv[21][2]`, and `keypoint_scores[21]`;
- raw/refined 3D: `landmarks_xyz_m[21][3]`, `validity[21]`, plus contract provenance;
- ordering/time: stable `frame_id` association and integer hardware `timestamp_ns`;
- invalid JSON values: `null`, never the non-standard tokens `NaN` or `Infinity`.

Use `contract_to_trace_payload()` for `SpatialObservation` and `PoseEstimate`. It keeps
the immutable raw-observation link, converts backend arrays to JSON values, maps
non-finite sentinels to `null` only for explicitly invalid landmarks, and rejects a
non-finite value on a valid landmark.

Blob roles are record-local semantics, not inferred from file names. Recommended roles
include `source_left`, `source_right`, `rectified_left`, `rectified_right`, `crop`,
`heatmap`, `overlay`, `array`, and `audit_report`. A SHA-256 path still deduplicates
identical bytes even when the same content is referenced under different roles.

## Integrity and lifecycle

The writer enforces a single process at a time with an OS advisory lock. It rejects
duplicate record IDs, unknown or forward parent IDs, non-JSON payloads, non-finite JSON
numbers, and blob paths outside the run directory. Writes are flushed and blobs are
published atomically.

`trace-validate` verifies:

- manifest and summary self-hashes;
- record ordinals, IDs, parent ordering, and the complete SHA-256 chain;
- final record count and last-record hash;
- each referenced blob's path, byte count, and SHA-256 digest.

The integrity chain is tamper-evident, not a cryptographic signature. If traces cross a
trust boundary, sign or archive the completed directory with an external trusted system.

## Inspection frontends

`trace-serve` exposes a loopback-only, read-only HTTP server. It has no upload or mutation
endpoint. The UI displays run validation, global records, the frame timeline, stage and
track filters, artifact previews, 2D overlays, a FHP21 3D canvas, and the original JSON.
Artifacts are served only after their content hash and in-run path have been verified.

The deterministic `trace-demo` command exercises the full stage vocabulary for UI testing;
its records are synthetic. `audit-session --trace-output` emits only the real audit stages.
`run-item --h20-executor-config ...` keeps that audit evidence in the same writer and then
imports real worker detection, pose, association, fusion, optional MANO, temporal, export,
warning, and failure records through a validated process package.

The normal catalog UI is split into `backend/` (read-only FastAPI, recursive multi-run
catalog) and `frontend/` (React/TypeScript). It uses opaque `run_key` and `frame_key`
identities, paginates long sequences, and can download the content-addressed final JSONL.
The legacy `trace-serve` remains useful for one-run, dependency-free inspection.

## Per-frame stage comparison and diagnostic video

The React inspector reconstructs stage comparisons from immutable records rather than
storing a rendered PNG for every logical node. A saved synchronized pair may attach these
image roles:

- `source_left` / `source_right` in native fisheye pixels;
- `undistorted_left` / `undistorted_right` as a debug-only OpenCV QA branch;
- `rectified_left` / `rectified_right` in the calibrated stereo pixel space.

RTMDet always consumes the native fisheye frame. With `baseline_native_v1`, RTMPose also
uses that frame; with `virtual_perspective_kb4_v1`, each candidate instead records a
`virtual_crop` and `virtual_crop_valid_mask`, a typed virtual camera, crop-space keypoints,
and their inverse mapping to native pixels. Full-frame undistortion and rectification remain
inspection evidence, not a false claim about model input. Detection records preserve every
`candidate_decision` (`SEED`, `RECOVERY`, or `REJECTED`) and the bounded association pool;
a recovery proposal is not a final hand until downstream geometry accepts it. Detection,
pose, association, raw fusion, MANO, temporal, and export comparisons are drawn from their
structured payloads. Three-dimensional stages use
`projected_keypoints_uv.left/right`, each exactly 21 entries in rectified pixels. Invalid or
behind-camera landmarks are JSON `null`, never fabricated coordinates. A
`projected_keypoints_space` value identifies that pixel space.

With `artifacts.overlay_video=true`, one global `EXPORT` record named
`raw_vs_stable_overlay_video_exported` references:

- `overlay_video_raw_vs_stable_stereo_rectified`, a browser-seekable H.264/yuv420p MP4;
- `overlay_video_timeline`, a JSON mapping from each video PTS to the original `frame_id`,
  global `frame_index`, hardware `timestamp_ns`, and track IDs.

Every synchronized pair contributes exactly one video frame, including pairs with no hand.
All tracks are drawn simultaneously with stable colors. The video metadata records
`stable_input_stages` and `temporal_method`, so consumers can distinguish a real MANO input
from a raw-fusion fallback. Both worker and core writers import the MP4 in bounded chunks;
large videos are not materialized as one Python bytes object.
