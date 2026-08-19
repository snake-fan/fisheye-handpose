# fisheye-handpose

Calibration-first reconstruction of metric `fhp21/v1` hand landmarks from synchronized
stereo fisheye video, with an isolated NVIDIA H20 worker and evidence-first inspection.

## Status

This is a controlled research pipeline, not a production hand-tracking service. The core
geometry, trace, H20 process bridge, read-only API, and React inspector are implemented and
covered by synthetic/contract tests. Real H20 runs support two explicit RTMPose input profiles:

- `baseline_native_v1` — backward-compatible default; RTMDet and top-down RTMPose operate from
  native distorted fisheye frames/proposals;
- `virtual_perspective_kb4_v1` — opt-in hand-centred perspective crop generated with calibrated
  KB4 ray mapping.

The virtual profile has passed the recorded H20 pipeline gates, but neither profile is a claim
of target-domain ground-truth accuracy or production readiness. The legacy PyTorch/OpenMMLab
stack also has known security and support debt. Current results are summarized in the
[`v2 iteration report`](docs/pipeline-v2-iteration-plan.md).

This repository is independent of the existing VLA pipeline and does not import or execute it.

## Architecture

The deployable system has four layers with deliberate process and write boundaries:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Core geometry and orchestration | `src/fisheye_handpose/` | Session discovery, calibration, timestamp pairing, video audit, rectification QA, typed contracts, canonical trace writing, and worker orchestration. Python 3.11+. |
| H20 perception worker | `deploy/mmpose-h20/` | Python 3.10/CUDA 12.1 RTMDet + RTMPose, stereo association/fusion, tracking, optional MANO, temporal baseline, and staged FHP21 export. |
| Read-only Trace API | `backend/` | Validated multi-item run catalog and integrity-checked artifact delivery. It never writes a run. |
| React inspector | `frontend/` | Multi-run, frame/stage/track evidence inspection through the API. It never reads the server filesystem directly. |

```text
stereo video + hardware timestamps + explicit calibration semantics
  -> core audit and canonical ACTIVE trace
  -> versioned JSON request to isolated H20 Python 3.10 process
  -> validated worker staging package
  -> single core writer imports records and content-addressed blobs
  -> immutable COMPLETED/FAILED canonical run
  -> read-only FastAPI catalog -> React inspector
```

The embedded `trace-serve` viewer remains a small single-run utility for local demos. The
FastAPI/React pair is the normal operator interface. See the accepted decisions in
[`docs/adr/`](docs/adr/) and the shared terms in [`CONTEXT.md`](CONTEXT.md).

## Repository layout

```text
src/fisheye_handpose/          Python 3.11 core package and CLI
tests/                        core geometry, trace, CLI, and bridge tests
deploy/mmpose-h20/            isolated Python 3.10 H20 project and worker
backend/                      independent FastAPI Trace API project
frontend/                     independent React/TypeScript/Vite inspector
contracts/                    canonical cross-runtime and Trace API contracts
docs/                         contracts, architecture, ADRs, and dated reports
scripts/                      operational helpers
runs/                         generated canonical runs; ignored by Git
```

Generated `results/`, runs, captures, weights, MANO files, and private manifests are local
artifacts and are not source code.

## Quick start: deterministic local demo

Requirements: `uv`, Python 3.11, and a host that permits binding a loopback port.

```bash
uv python install 3.11
uv sync --locked --extra dev --no-editable
uv run --locked --no-editable fisheye-handpose schema
uv run --locked --no-editable fisheye-handpose trace-demo runs/demo
uv run --locked --no-editable fisheye-handpose trace-validate runs/demo
uv run --locked --no-editable fisheye-handpose trace-serve runs/demo \
  --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The three-frame trace is synthetic inspection data, not a
model-accuracy result. Run directories are intentionally non-overwriting; choose another demo
path for a second run.

`uv.lock` is the source of truth for the core environment. Use locked, non-editable installs
for verification so imports from the source tree cannot hide packaging failures.

## Core capture workflow

Every command emits machine-readable output. Diagnostics go to stderr; output files are
written atomically.

```bash
# Discover complete multi-part sessions.
uv run --locked --no-editable fisheye-handpose discover /path/to/fisheye-data

# Inspect calibration with semantics the source YAML does not declare.
uv run --locked --no-editable fisheye-handpose inspect-calibration \
  /path/to/calibration_camera.yaml \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera

# Fully decode, synchronize, calibrate, rectify, and gate one session.
uv run --locked --no-editable fisheye-handpose audit-session /path/to/session \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera \
  --max-skew-us 1000 \
  --output /path/to/audit-report.json
```

Frames are paired monotonically and one-to-one by hardware timestamp, never by frame index.
Full video decode must agree with timestamp counts. Calibration translation units and extrinsic
direction are required because the Orbbec YAML does not identify them.

Coordinate conventions are stable:

- OpenCV 3D axes: `+x` right, `+y` down, `+z` forward;
- internal time: integer nanoseconds;
- internal length: metres;
- `T_B_from_A`: `X_B = R @ X_A + t`;
- canonical landmarks: [`fhp21/v1`](docs/output-schema.md).

Use `fisheye-handpose --help` and the [architecture](docs/architecture.md) for the remaining
pairing, trace, baseline, and run options.

## Canonical runs and the H20 worker

`run-item` creates one non-overwriting attempt under `runs/<item_id>/<run_id>/`. Its manifest,
JSONL records, final summary, and `blobs/sha256/...` share one canonical integrity boundary.
Audit, inference, refinement, export, warnings, skipped stages, and failures remain explicit.
Only the core writer mutates this directory; worker output is temporary staging data until the
bridge validates and imports it.

The H20 environment is intentionally separate:

| Requirement | H20 compatibility target |
| --- | --- |
| Host | Linux x86_64, NVIDIA H20 / SM90 |
| Python | exactly 3.10 |
| CUDA stack | CUDA 12.1, PyTorch 2.1.0, TorchVision 0.16.0 |
| OpenMMLab | MMCV 2.1.0, MMDetection 3.2.0, MMPose 1.3.2 |
| uv | `>=0.12.3,<0.13` for the isolated subproject |

Follow [`deploy/mmpose-h20/README.md`](deploy/mmpose-h20/README.md) for environment creation,
the fail-closed runtime doctor, checkpoint verification, MANO terms/manifests, and the special
procedure for the configured H20 with its locally compiled SM90 MMCV wheel. Do not synchronize
that Linux/CUDA project on macOS, and do not casually replace the configured H20 environment.

After the core audit and H20 prerequisites are ready:

```bash
.venv/bin/fisheye-handpose run-item /path/to/session \
  --runs-root /path/to/repository/runs \
  --run-id h20-e2e-YYYYMMDD \
  --left-id cam_0 --right-id cam_1 \
  --translation-unit mm \
  --extrinsics-convention reference_to_camera \
  --max-skew-us 1000 \
  --h20-executor-config deploy/mmpose-h20/h20-executor.example.json
```

The executor template is site-specific and records model/source/MANO settings. The orchestrator
replaces its session and audited calibration fields. Final `fhp21.jsonl` is retained as the
verified blob role `worker_fhp21_output`, not as a mutable top-level run file.

## Inspect runs

For normal multi-item inspection, start the independent API and frontend in two terminals:

```bash
cd backend
uv sync --locked --group dev --no-editable
uv run --locked --no-editable fisheye-trace-api \
  --catalog-root ../runs --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The API has no authentication layer: keep it on loopback or
behind a trusted tunnel/reverse proxy. The backend verifies paths, byte counts, and hashes
before serving artifacts.

For the configured remote H20 workflow, run this from the repository root on the Mac:

```bash
uv run --locked --no-editable python scripts/remote_trace_viewer.py
```

The helper maintains the SSH tunnel, checks API health, runs/reuses the local Vite server, and
reconnects after a drop. See the [backend](backend/README.md) and
[frontend](frontend/README.md) guides for ports, CORS, and the manual equivalent.

To publish named copies of verified H20 overlay videos for downstream review, run from the
repository root:

```bash
uv run --locked --no-editable python scripts/export_h20_result_videos.py \
  /path/to/runs /path/to/new-mp4-export
```

The output directory must not already exist and must be outside the source runs root. The helper
requires `ffprobe` (by default `/usr/bin/ffprobe`; override it with `--ffprobe`), completes full
canonical manifest/trace/summary/blob validation before publishing anything, and materializes
independent file copies rather than hard links. Editing an exported MP4 must never mutate its
source run; do not replace this operation with `ln` or otherwise modify finalized runs in place.

## Validation matrix

The `Makefile` is the shared local/CI entry point. None of these targets synchronizes the
Linux/CUDA H20 subproject.

| Scope | Command | What it verifies |
| --- | --- | --- |
| Core | `make check-core` | root lock, non-editable install, pytest, Ruff lint/format, installed CLI schema smoke |
| Trace API | `make check-backend` | backend lock/install, pytest, Ruff lint/format |
| React inspector | `make check-frontend` | `npm ci`, Vitest, strict TypeScript, production Vite build |
| H20 static | `make check-h20-static` | standard-library manifest doctor plus fake-runtime/contract tests and Ruff, using the root dev environment |
| All local-safe checks | `make check` | all four rows above |

The exact commands are also visible with `make help`. Real CUDA, MMCV native-op, model,
RTMPose, MANO, and full-data gates require the configured H20 and are documented separately;
GitHub-hosted CI intentionally does not pretend to validate them.

Shared contract values are changed only in `contracts/project-contract-v1.json`. Regenerate the
checked-in core, H20 tooling/worker, and frontend views with
`uv run --locked --no-editable python scripts/generate_contracts.py`, or append `--check` to
verify that they and the bound H20 JSON manifests are current.

The GitHub Actions workflow mirrors these four scopes. The H20-static job only validates
manifests and lightweight worker behavior; it never runs `uv sync` inside
`deploy/mmpose-h20`.

## Data, model, and license boundaries

- Do not commit captures, `runs/`, `results/`, generated videos, model weights, TensorRT/ONNX
  artifacts, MANO pickle files, or private asset manifests.
- The repository does not download MANO. Its files require separate acceptance of the MANO
  terms and must match a private byte-count/SHA-256 manifest before deserialization.
- RTMDet/RTMPose checkpoints are fetched only after an explicit license-risk acknowledgement
  and are checked against the committed identity manifest.
- The pinned PyTorch/OpenMMLab environment is a reproducibility enclave, not a security or
  production approval.
- Dataset, checkpoint, MANO/SMPL-X, and repository-code rights are separate. No project
  `LICENSE` is currently declared; choose one before treating the GitHub repository as an
  open-source distribution.

## Documentation

Start with [`docs/index.md`](docs/index.md), which distinguishes normative contracts, accepted
ADRs, current implementation notes, and historical reports. The most important references are:

- [`CONTEXT.md`](CONTEXT.md) — stable vocabulary and invariants;
- [`docs/output-schema.md`](docs/output-schema.md) — normative `fhp21/v1` output;
- [`docs/trace-format.md`](docs/trace-format.md) — normative trace lifecycle and integrity;
- [`docs/architecture.md`](docs/architecture.md) — current boundaries and known gaps;
- [`docs/pipeline-v2-iteration-plan.md`](docs/pipeline-v2-iteration-plan.md) — current v2 H20
  iteration result;
- [`docs/current-pipeline-problem-analysis.md`](docs/current-pipeline-problem-analysis.md) —
  historical `5eacb7a` baseline diagnosis, retained for evidence rather than current status.

## Troubleshooting

- **A test cannot bind `127.0.0.1`:** legacy-viewer integration tests require a loopback socket.
  Run them in a normal local/CI environment that permits ephemeral loopback ports.
- **A run already exists:** runs are non-overwriting by design. Select a new `run_id` or output
  directory; do not repair a finalized run in place.
- **The React app cannot reach the API:** use the default Vite `/api` proxy, or build with a
  browser-visible `VITE_API_BASE_URL`; custom frontend ports also require matching API CORS
  origins.
- **H20 sync fails on macOS:** expected. Run only `doctor.py --mode manifest` and the root-env
  lightweight tests locally; create/synchronize the CUDA project only on its Linux x86_64 host.
- **The configured H20 has a working compiled MMCV wheel:** follow its protected update procedure
  instead of running subproject `uv sync`.
- **Calibration looks plausible but geometry fails:** verify camera IDs, translation unit,
  extrinsic convention, timestamp column/unit, and the full decode/timestamp-count gate; none is
  guessed.
- **Generated result files are absent from Git status:** `runs/` and `results/` are ignored on
  purpose. Publish only reviewed, explicitly packaged artifacts outside the source tree.
