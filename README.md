# fisheye-handpose

An independent, calibration-first project for extracting metric 21-landmark hand poses
from synchronized stereo fisheye video. It does **not** import or execute the existing
VLA pipeline.

The project is intentionally split into three independently deployable layers:

1. A strict geometry foundation: session discovery, timestamp pairing, Orbbec KB4
   calibration normalization, full video decode audit, true stereo rectification/QA,
   timestamp-indexed stereo reading, and a versioned 21-landmark contract.
2. An isolated H20 worker: RTMDet + RTMPose evidence, calibrated stereo fusion,
   tracking, optional MANO fitting, temporal refinement, and FHP21 export.
3. A read-only FastAPI run catalog and an independent React/TypeScript inspector.

The worker keeps RTMDet on full native fisheye frames and supports two versioned RTMPose
profiles: the compatible native path and an opt-in hand-centred virtual-perspective crop
path. The active v2 accuracy work and its H20 gates are tracked in
[docs/pipeline-v2-iteration-plan.md](docs/pipeline-v2-iteration-plan.md). No weights, MANO
files, or private manifests are embedded here.

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
uv sync --locked --extra dev
uv run --locked fisheye-handpose schema
```

`uv.lock` is the source of truth for the cross-platform geometry environment. Use
`uv lock --check` in CI and `uv sync --locked` in development; do not replace these with
an unconstrained `pip install`. CI/release verification additionally uses
`uv sync --locked --no-editable` so source-tree imports cannot hide packaging mistakes.

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

## One immutable folder per data item

`run-item` creates one non-overwriting attempt under
`runs/<item_id>/<run_id>/`. The folder contains `run_manifest.json`, `trace.jsonl`,
`run_summary.json`, and content-addressed `blobs/sha256/...`. Audit, perception, MANO,
temporal, export, warning, and failure records share one hash chain; an absent stage is
recorded as `SKIPPED / NOT_PRODUCED`.

On the configured H20, run one real data item with the checked-in executor profile:

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

The checked-in debugging template processes at most 120 synchronized pairs, saves source,
undistorted, and stereo-rectified evidence for every processed pair, and exports a 2x2
H.264 raw-versus-stable skeleton video plus its hardware-timestamp timeline. Once a run has
been visually accepted, increase `artifacts.sample_every` to reduce still-image storage or
disable `artifacts.overlay_video`. The orchestrator replaces its session/calibration fields
with audited values.
The final `fhp21.jsonl` is imported as a verified content-addressed blob with role
`worker_fhp21_output`, not copied to a mutable top-level run file. Download it through the
React/API artifact link or locate it from the EXPORT provenance.

Freeze reproducible stage/quality metrics for any completed canonical run:

```bash
uv run --locked --no-editable fisheye-handpose trace-baseline \
  runs/ITEM_ID/RUN_ID --output runs/ITEM_ID/RUN_ID.baseline.json
```

The baseline extractor validates the trace first and records the applied configuration,
model/calibration provenance, detections, associations, Raw validity/bone statistics,
tracks, MANO attempts/RMSE, and the temporal stage's actual input source.

## Stage traces and inspection UIs

Every pipeline stage can write append-only records to a run artifact directory. Records
carry explicit parent IDs and form a SHA-256 chain; source images, overlays, arrays, and
reports are content-addressed blobs with explicit semantic roles. Finalization writes a separate immutable summary,
so raw evidence is never overwritten by MANO or temporal output and interrupted runs can
be reopened while they remain active.

The on-disk v1 protocol, payload conventions, lifecycle, and integrity guarantees are
defined in [docs/trace-format.md](docs/trace-format.md).

Generate a deterministic three-frame stereo example, validate it, then inspect it in the
legacy single-run read-only UI:

```bash
uv run --locked --no-editable fisheye-handpose trace-demo runs/demo
uv run --locked --no-editable fisheye-handpose trace-validate runs/demo
uv run --locked --no-editable fisheye-handpose trace-serve runs/demo \
  --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The example contains three frames and the complete planned
stage vocabulary, including stereo source SVGs, detections, virtual crops, per-view 2D
evidence, association, raw 3D fusion, kinematic/temporal refinement, QA, and export links.
It is synthetic inspection data and must not be interpreted as a model accuracy result.

For the normal multi-item workflow, start the independent API and React application:

```bash
cd backend
uv sync --locked --group dev
uv run --locked --no-editable fisheye-trace-api --catalog-root ../runs \
  --host 127.0.0.1 --port 8000

# another terminal
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Remote H20 inspection uses an SSH tunnel; see
[backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md).

For each selected frame, the React inspector exposes the ten logical nodes from source RGB
through stable FHP21 export. Every node has a before/after view; missing evidence remains
visible as `NOT_PRODUCED` with its recorded reason. When no track filter is selected, all
hands are drawn together with deterministic colors. The run-level player uses the global
`overlay_video_raw_vs_stable_stereo_rectified` artifact to compare raw triangulation (top)
with the actual temporal output (bottom) in both cameras. Its label is provenance-driven:
if MANO was rejected, it says `RAW_FUSION -> causal_time_ema_v1` rather than claiming a
temporal MANO result.

For the configured remote workflow, the Mac helper keeps both the SSH tunnel and local
React development server alive, verifies the API health contract before declaring success,
and reconnects automatically after a network drop:

```bash
uv run --locked --no-editable python scripts/remote_trace_viewer.py
```

It defaults to SSH alias `h20`, remote API port `18080`, local API port `18081`, and
frontend port `15174`. Press Ctrl-C once to terminate only the tunnel and Vite process that
the helper created. Run `python scripts/remote_trace_viewer.py --help` to override ports,
the SSH alias, or frontend directory; no remote path, password, or private key is embedded.

To persist the stages already executed by `audit-session`, add a trace directory:

```bash
uv run --locked --no-editable fisheye-handpose audit-session /path/to/session \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera \
  --max-skew-us 1000 \
  --output runs/session-audit.json \
  --trace-output runs/session-audit-trace
```

The audit trace records discovery, calibration, rectification, both timestamp streams,
pairing, full video decode reports, epipolar QA, warnings, and failures. The complete audit
JSON is attached as a verified blob. It deliberately does not create detector, pose,
fusion, MANO, or temporal records, because those stages have not run. It also omits image
previews instead of decoding the videos a second time; perception stages attach source
frames and overlays as they execute.

For a producer that will append records later, `trace-init RUN_DIR` creates an empty ACTIVE
run. `RunArtifactWriter.open(RUN_DIR)` resumes that run under a single-writer lock. Treat a
run directory as immutable after finalization; `trace-validate` checks both its record
chain and referenced blob hashes and exits non-zero on corruption.

Full-frame stereo rectification is a geometry/QA utility, not the mandatory model image.
The current compatibility worker detects in native fisheye pixels and uses RTMPose's
top-down crop. A future backend should add explicit hand-centred virtual-perspective crops.

## Tests

```bash
uv run --locked --extra dev --no-editable pytest
uv run --locked --extra dev --no-editable ruff check .
```

The tests include synthetic projection/rectification geometry, timestamp drops and
offset tails, full video decode and frame selection, CLI failure reports, calibration
direction/unit checks, deterministic discovery, and the `fhp21/v1` contract.
The H20 worker, process bridge, multi-run API, and React project have separate tests too.

See [docs/architecture.md](docs/architecture.md) for the planned model-facing APIs and
the next implementation milestone. The attached engineering proposal has been evaluated
in [docs/design-review.md](docs/design-review.md); that review is the implementation
decision record for which parts are retained, revised, or rejected. The evidence-backed
stage-by-stage diagnosis of the current H20 baseline is in
[docs/current-pipeline-problem-analysis.md](docs/current-pipeline-problem-analysis.md).

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
