# MMPose H20 compatibility environment

This directory defines a standalone, exact-version `uv` environment for reproducing the
legacy MMPose stack on a Linux x86_64 NVIDIA H20 server. It is deliberately separate from
the Python 3.11+ geometry package at the repository root.

> **Compatibility baseline, not a secure production profile.** These versions reproduce
> an older upstream binary matrix. Pinning them does not mean they are currently supported
> or free of known vulnerabilities. Do not expose this environment as a network service or
> treat a successful doctor report as a security approval.

## Frozen binary matrix

| Component | Exact target |
| --- | --- |
| OS / architecture | Linux x86_64 |
| Python | 3.10 |
| GPU / compute capability | NVIDIA H20 / SM90 (`9.0`) |
| PyTorch / TorchVision | `2.1.0` / `0.16.0`, official CUDA 12.1 index |
| MMCV | `2.1.0`, official `cp310` Linux CUDA 12.1 / Torch 2.1 wheel |
| MMEngine / MMDetection / MMPose | `0.10.3` / `3.2.0` / `1.3.2` |
| NumPy / SMPL-X | `1.26.4` / `0.1.28` |
| Chumpy compatibility | `0.71`, fixed Git SHA `2816a138…` |

`environment.json` is the machine-readable contract for runtime pins and constraints.
Model weights are intentionally managed by the adjacent asset manifest and fetch script;
they are not duplicated in the environment contract.

## Create a new environment on a new H20 host

Run commands from this directory:

```bash
cd deploy/mmpose-h20
uv --version  # must be >=0.12.3,<0.13
uv python install 3.10
uv sync --locked
```

This lock was generated and tested with uv 0.12.3. The project accepts
`uv>=0.12.3,<0.13`; older uv releases cannot apply the declared build/source constraints
safely. Upgrade uv using the same method that installed it;
`uv self update` only works for Astral's standalone installer and must not be assumed for
pipx, Conda, apt, Homebrew, or a system image. `uv sync --locked` verifies that
`pyproject.toml` still matches the committed lock and installs it without changing the
resolution. The lock also excludes packages published after 2024-08-01 so the 2024
OpenMMLab release is not silently mixed with 2026 transitive APIs. The MMCV dependency is
an exact wheel URL, so uv cannot fall back to `mmcv-lite` or an unreviewed source build.

MMPose 1.3.2 declares the abandoned `chumpy==0.70` package. That release cannot be built
cleanly under modern PEP 517 tooling and cannot import with NumPy 1.26. This environment
therefore pins the reviewed modernization branch at an immutable commit (`0.71`) instead
of silently applying runtime monkey-patches.

Before copying the environment to another host, validate the static contract on any
machine with Python 3.10 or newer, including macOS:

```bash
python3 doctor.py --mode manifest
```

This standard-library-only mode intentionally bypasses uv project resolution, because
the subproject itself is constrained to Linux x86_64. It validates `environment.json`,
the built-in pin contract, and `.python-version` without importing CUDA. The separate
`uv sync --locked` command is the gate that checks `pyproject.toml` against `uv.lock`.

After synchronization on the H20 server, run the fail-closed runtime check:

```bash
uv run --locked python doctor.py --mode runtime
```

The runtime doctor requires all checks to pass: Linux x86_64, Python 3.10, exact package
versions, PyTorch CUDA 12.1, an available SM90 GPU, a real CUDA invocation of
`mmcv.ops.nms`, and an FP16 CUDA matrix operation. It prints exactly one JSON report to
stdout and exits `0` only when every required check passes. A missing import, CPU-only
fallback, wrong CUDA build, wrong GPU architecture, or failed native operator exits `1`.

To additionally require one or more populated model directories:

```bash
uv run --locked python doctor.py --mode runtime \
  --model-dir /srv/fisheye-handpose/models
```

This optional directory check verifies presence only. Use the model asset tooling for
artifact identity and checksum validation.

## Existing configured H20: pull without replacing SM90 MMCV

The repository at `/mnt/workspace/zyf/fisheye/fisheye-handpose` already has a validated
Python 3.10 GPU environment with a locally compiled SM90 MMCV wheel. Its system uv is older
than this subproject's required uv, and synchronizing this subproject would replace that
working wheel with the upstream binary that lacks the required H20 kernel. After a pull,
update only the root Python 3.11 environment and invoke the GPU interpreter directly:

```bash
REPO=/mnt/workspace/zyf/fisheye/fisheye-handpose
git -C "$REPO" pull --ff-only
cd "$REPO"
uv sync --locked --no-editable \
  --refresh-package fisheye-handpose \
  --reinstall-package fisheye-handpose

GPU_PY="$REPO/deploy/mmpose-h20/.venv/bin/python"
"$GPU_PY" deploy/mmpose-h20/doctor.py --mode runtime \
  --model-dir "$REPO/models/openmmlab"
```

Do **not** run `uv sync`, `uv run --locked`, or an editable install inside
`deploy/mmpose-h20` on this configured host. The core launches `$GPU_PY` directly and
supplies the worker via `PYTHONPATH`, preserving the compiled wheel.

## Model assets and smoke tests

The detector and pose aliases are intentionally not used. Fetch the two explicit official
artifacts only after reviewing the checkpoint/training-data rights:

```bash
uv run --locked python model_assets.py fetch \
  --manifest ./model-assets.json \
  --output-dir /srv/fisheye-handpose/models/openmmlab \
  --acknowledge-license-risk
uv run --locked python model_assets.py verify \
  --manifest ./model-assets.json \
  --output-dir /srv/fisheye-handpose/models/openmmlab
```

The adjacent `model-assets.json` is the single checkpoint manifest used by fetch, verify,
and the RTMPose smoke. It stores full SHA-256 digests; the downloader writes atomically
and never deserializes a checkpoint. The PyPI package does not guarantee the complete
`configs/` and `demo/` trees, so obtain a clean, detached MMPose checkout at the exact
signed v1.3.2 commit:

```bash
git clone --no-checkout https://github.com/open-mmlab/mmpose.git \
  /srv/fisheye-handpose/vendor/mmpose
git -C /srv/fisheye-handpose/vendor/mmpose checkout --detach \
  5408bc76f5b848cf925a0d1857899011d8c5b497
git -C /srv/fisheye-handpose/vendor/mmpose status --short
```

The final command must print nothing; the smoke rejects a different commit or a dirty
checkout.

Run the explicit detector + low-level RTMPose smoke on a real IR frame. The smoke verifies
the checkout commit and clean config paths, then rechecks both checkpoint sizes and hashes
before any Torch deserialization:

```bash
uv run --locked python scripts/rtmpose_smoke.py \
  --model-dir /srv/fisheye-handpose/models/openmmlab \
  --mmpose-source /srv/fisheye-handpose/vendor/mmpose \
  --image /ABS/PATH/real_ir_frame.png
```

A no-detection result is a failed smoke (exit `1`) and never falls through to whole-image
pose inference. MANO is never downloaded. After separately accepting its terms, mount the
files as `MODEL_ROOT/mano/MANO_LEFT.pkl` and `MANO_RIGHT.pkl`. Copy
`mano-assets.example.json` to a private location, fill in the two byte counts/SHA-256
digests and the real provenance record, then set both acknowledgement fields only after
the applicable review. `--manifest` is mandatory and has no implicit default. The MANO
smoke refuses to deserialize a file unless that private manifest matches:

```bash
uv run --locked python scripts/mano_smoke.py \
  --model-dir /ABS/PATH/MODEL_ROOT \
  --manifest /ABS/PATH/mano-assets.json
```

This validates left/right forward passes plus finite Adam gradients and an LBFGS closure.

## Stereo perception worker

`worker/fisheye_h20_worker` is a Python 3.10 process boundary for the real stereo hand
pipeline:

```text
timestamp CSV + presentation-order stereo video
  -> RTMDet hand candidates (0, 1, or 2 per view)
  -> RTMPose Hand5 21-point evidence and confidence
  -> KB4 point rectification
  -> epipolar cross-view association with explicit unmatched candidates
  -> metric raw triangulation in the rectified-left camera frame
  -> sequence-local one-to-one 3D tracking
  -> optional framewise MANO fitting with per-track handedness and frozen beta
  -> timestamp-aware causal temporal baseline
  -> fhp21.jsonl export
```

The worker never imports the Python 3.11 core package. Its session names, timestamp units,
coordinate frame, stage names, and `fhp21/v1` labels mirror the core JSON protocol; the
copied minimal parsing semantics are documented in the worker modules. A validated bridge
imports the immutable worker package through the core's single `RunArtifactWriter`.

Video frames are decoded in presentation order with the locked OpenCV already installed
by the OpenMMLab stack. PyYAML is likewise already present in `uv.lock`; the worker adds no
new runtime dependency. Do **not** run `uv sync` on the configured H20 host because that
would replace the locally compiled SM90 MMCV wheel. Keep using its Python directly:

When `artifacts.overlay_video` is enabled, the worker additionally requires
`/usr/bin/ffmpeg` with the `libx264` encoder. The configured H20 provides FFmpeg 6.1.1.
OpenCV performs fisheye remapping and drawing; FFmpeg receives raw BGR frames through a
pipe and writes H.264/yuv420p MP4 with a short GOP and `faststart` metadata. Encoding stays
on CPU and does not consume the H20 CUDA device.

```bash
cd /mnt/workspace/zyf/fisheye/fisheye-handpose/deploy/mmpose-h20
PYTHONPATH=worker .venv/bin/python -m fisheye_h20_worker \
  /ABS/PATH/request.json /ABS/PATH/worker-result
```

The worker is invoked through `PYTHONPATH` so the already-built environment does not need
an editable reinstall. `pyproject.toml` and `uv.lock` remain unchanged and consistent; no
dependency resolution or environment mutation is required for this worker.

The request uses the strict `fisheye-handpose/h20-worker-request/v1` schema:

```json
{
  "schema_version": "fisheye-handpose/h20-worker-request/v1",
  "session": {
    "path": "/mnt/workspace/zyf/fisheye/data/SESSION",
    "timestamp_column": "timestamp_us",
    "timestamp_unit": "us",
    "max_skew_us": 1000,
    "max_pairs": 100
  },
  "calibration": {
    "path": "/mnt/workspace/zyf/fisheye/data/SESSION/capture_calibration_camera.yaml",
    "left_camera_id": "cam_0",
    "right_camera_id": "cam_1",
    "translation_unit": "mm",
    "extrinsics_convention": "reference_to_camera",
    "output_size": [1600, 1300],
    "balance": 0.8,
    "fov_scale": 1.0
  },
  "thresholds": {
    "bbox_score": 0.3,
    "keypoint_score": 0.2,
    "association_epipolar_px": 5.0,
    "max_reprojection_error_px": 3.0,
    "min_ray_angle_deg": 0.5
  },
  "models": {
    "manifest": "/ABS/PATH/model-assets.json",
    "model_dir": "/ABS/PATH/models/openmmlab",
    "mmpose_source": "/ABS/PATH/vendor/mmpose",
    "device": "cuda:0",
    "detector_category_id": 0,
    "license_risk_acknowledged": true
  },
  "artifacts": {
    "source_frames": "SAMPLED",
    "sample_every": 1,
    "image_format": "jpg",
    "overlay_video": true
  },
  "tracking": {
    "max_root_distance_m": 0.15,
    "max_gap_ms": 250
  },
  "mano": {
    "model_root": "/ABS/PATH/models",
    "manifest": "/ABS/PRIVATE/PATH/mano-assets.json",
    "min_valid_landmarks": 15,
    "max_fit_rmse_m": 0.02,
    "iterations": 40,
    "learning_rate": 0.03
  },
  "temporal": {
    "method": "causal_time_ema_v1",
    "time_constant_ms": 80,
    "gap_reset_ms": 250
  }
}
```

`tracking` and `temporal` may be omitted to use the shown defaults. The backward-compatible
`artifacts.overlay_video` field defaults to `false` when absent. `mano` is optional;
set it to `null` or omit it to run a raw-geometry temporal baseline. That path always emits
`KINEMATIC_REFINEMENT / SKIPPED / output_status=NOT_PRODUCED` instead of claiming a MANO
result. When configured, both MANO files are checked against the private manifest before
SMPL-X can deserialize either pickle. The manifest must contain the same explicit license,
provenance, filenames, byte counts, and full SHA-256 identities required by
`scripts/mano_smoke.py`.

Tracking uses a wrist (or valid-landmark centroid) anchor and deterministic
max-cardinality/min-distance one-to-one assignment. Every assignment states `MATCHED` or
`NEW`, its distance, and its actual time delta. The first high-quality MANO frame for each
track fits both left and right models and selects the lower accepted RMSE. That handedness
and the selected ten shape coefficients are then shared by the track; later fits optimize
only pose, global orientation, and translation with beta frozen. The explicit mapping
`mano-v1.2-j16-tips-to-fhp21/v1` converts the MANO 16 joints plus five topology fingertip
vertices to FHP21 order.

The temporal stage is deliberately labeled a baseline, not a learned smoother. Its causal
EMA coefficient is `1 - exp(-dt/tau)` using each record's real timestamp. It resets on a
configured time gap, non-monotonic timestamp, or switch between raw and MANO input; it
never substitutes a fixed per-frame alpha or silently blends two geometry sources.

`license_risk_acknowledged` does not bypass any identity check. Before deserializing a
checkpoint, the worker verifies the manifest schema, fixed config binding, size and full
SHA-256 of both weights. It also requires the clean MMPose v1.3.2 commit and records its
config Git blobs. Both detector and pose model are initialized exactly once per request.

Every previously absent result path becomes one immutable process package:

```text
worker-result/
  manifest.json              # request, calibration, model/source provenance
  events.jsonl               # complete stage DAG through EXPORT
  summary.json               # terminal status and aggregate counts
  fhp21.jsonl                # one final FHP21 record per produced track/frame
  blobs/sha256/ab/...json     # raw, MANO and temporal stage payloads
  blobs/sha256/ab/...jpg      # optional content-addressed source frames
  blobs/sha256/ab/...mp4      # optional raw-vs-stable rectified diagnostic video
```

`events.jsonl` retains native and rectified 2D points, all 21 confidence values, matched
and unmatched candidate IDs, metric `landmarks_xyz_m`, per-joint validity, epipolar error,
left/right reprojection error, ray angle, track decisions, MANO parameters and fit quality,
temporal reset state, and export provenance. Viewer records use top-level `track_id`,
`detections[]`, `keypoints_uv`, `keypoint_scores`, `landmarks_xyz_m`, and `validity`; the
aggregate pose evidence is retained separately. `NONE`, `ALL`, and `SAMPLED` source-frame
policies control source image storage without changing inference. Every saved source pair
also gets one per-frame `RECTIFICATION` event with `undistorted_left/right` and
`rectified_left/right` image roles. Model inference intentionally remains on the original
fisheye frame. Raw, framewise MANO, temporal, and export event payloads include
`projected_keypoints_uv.left/right`; each side is always length 21, and an invalid,
non-finite, null, or behind-camera landmark remains JSON `null` instead of acquiring an
invented pixel coordinate.

When enabled, the overlay is one seekable 2x2 rectified video per run: raw-left/raw-right
on the top row and temporal-left/temporal-right on the bottom row. Every synchronized pair
contributes exactly one frame, including zero-hand frames, and every track in that pair is
drawn with a deterministic color. A JSON timeline blob preserves `frame_id`, global
`frame_index`, source `timestamp_ns`, video PTS, duration, and track IDs. The final global
`EXPORT` event has no `frame_id` and references blob roles
`overlay_video_raw_vs_stable_stereo_rectified` and `overlay_video_timeline`.
An existing result
directory is never overwritten. A stage with no actual output uses `NOT_PRODUCED` and is
never summarized as produced. If a late failure occurs after earlier frames were written,
the summary hashes that file as `partial_fhp21_output`; the bridge imports it only as a
debug artifact and keeps the package output status `NOT_PRODUCED`.

Each `fhp21.jsonl` line uses `fisheye-handpose/fhp21-output/v1` and contains the global
integer `frame_index`, lossless source `frame_id`, `timestamp_ns`, `track_id`, raw evidence,
an optional MANO result, temporal result, `selected_output_stage`, final 21 metric XYZ
values, and per-joint validity. It is created only when at least one final record exists.

### Import into the canonical core trace

The worker package is staging output, not a second front-end catalog. The backend does not
scan `events.jsonl` directly. A core `PipelineStageExecutor` must run the worker in a
separate result directory and then import it through the standard-library-only bridge:

```python
from fisheye_h20_worker.bridge import load_import_bundle

bundle = load_import_bundle(worker_result_dir)
record_ids = []
for record in bundle.core_records(external_parent_id=audit_record_id):
    blobs = tuple(
        writer.put_blob_file(
            blob.source_path,
            role=blob.role,
            media_type=blob.media_type,
            suffix=blob.suffix,
        )
        for blob in record.blobs
    )
    persisted = writer.append(
        record_id=record.record_id,
        stage=TraceStage(record.stage),
        status=TraceStatus(record.status),
        event=record.event,
        payload=record.payload,
        parent_ids=record.parent_ids,
        blobs=blobs,
    )
    record_ids.append(persisted.record_id)
```

The bridge validates manifest/summary schemas, event ordinals and DAG parents, strict JSON,
blob paths, sizes, and SHA-256 digests. It attaches the complete worker manifest,
`events.jsonl`, summary, and (when produced) `fhp21.jsonl` to the first imported record,
then copies every referenced source/raw/MANO/temporal blob through the core writer.
Therefore the canonical `runs/<item>/<run>/` directory still has exactly one writer and one
hash chain. Both the worker and core file-backed blob paths hash and copy in bounded chunks,
so a long MP4 is never materialized as one Python `bytes` value.

`payload.frame_index` is a globally increasing integer across all video parts for backend
routing. `payload.frame_id` retains the lossless source identity such as
`part0002/pair000000`; the per-part `pair_index` may restart at zero and must not be used as
the catalog frame key. Event/record IDs use the URL-safe equivalent
`part0002:pair000000:...`, while the payload remains lossless. The core
`H20WorkerExecutor` provides this process/import adapter;
configure it through the core `run-item --h20-executor-config ...` option. Without that option,
`run-item` correctly records model stages as `SKIPPED / NOT_PRODUCED`.

### Canonical end-to-end invocation

[`h20-executor.example.json`](h20-executor.example.json) wraps the request above in the
required `fisheye-handpose/h20-executor/v1` process configuration. It contains the exact
paths of the existing H20 installation. The core replaces session path, calibration path,
camera IDs, units, output size, and timestamp settings from the audited item; `max_pairs`
and perception/MANO/temporal settings remain controlled by the template.

```bash
cd /mnt/workspace/zyf/fisheye/fisheye-handpose
.venv/bin/fisheye-handpose run-item \
  /mnt/workspace/zyf/fisheye/data/Orbbec_Ego_AZEL764000H_19700102_204253 \
  --runs-root /mnt/workspace/zyf/fisheye/fisheye-handpose/runs \
  --run-id h20-e2e-20260813 \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera \
  --max-skew-us 1000 \
  --h20-executor-config deploy/mmpose-h20/h20-executor.example.json
```

The worker staging directory is temporary. After validation, events and blobs enter the
canonical `runs/<item>/<run>/` hash chain. The final JSONL is retained under blob role
`worker_fhp21_output`; it is downloadable from the React inspector/API and is not
duplicated as a mutable `runs/<item>/<run>/fhp21.jsonl` file.

## Tests

The deployment contract tests use only the Python standard library, so manifest and CLI
validation can run before installing the Linux-only CUDA environment:

```bash
python3 -m unittest discover -s tests -v
```

The worker tests use a fake model/video runtime but real KB4 rectification and stereo
geometry, so they do not initialize CUDA or deserialize checkpoints. The video encoder
unit test replaces the FFmpeg process with an in-memory fake while asserting the exact
libx264/yuv420p/faststart command and timeline contract; it does not require system FFmpeg:

```bash
python3 -m unittest tests/test_worker.py -v
```

Do not run `uv sync` for this subproject on macOS: the pinned MMCV wheel intentionally
supports only CPython 3.10 on Linux x86_64.
