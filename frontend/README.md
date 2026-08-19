# React trace inspector

This is the independent React 19 + TypeScript + Vite frontend for the read-only trace API
in `../backend`. It never imports the Python pipeline and never writes to a run directory.

## Requirements

- Node.js `^20.19.0` or `>=22.12.0`
- npm (the lock records the tested npm release)
- the backend listening on `127.0.0.1:8000`

## Development

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. With an empty `VITE_API_BASE_URL`, the Vite development
server proxies `/api` to `http://127.0.0.1:8000`, so no CORS configuration is needed.

For the existing H20 deployment, keep the API on the server loopback interface and make
it available to the Mac first. The recommended command persistently reconnects the SSH
tunnel, waits for `/api/v1/health`, and starts or reuses the local Vite server:

```bash
cd ..
uv run --locked --no-editable python scripts/remote_trace_viewer.py
```

Open `http://127.0.0.1:15174`. Defaults are SSH alias `h20`, remote API `18080`, local
forward `18081`, and Vite `15174`; every value has a CLI flag. The helper passes the
browser-visible local API URL to Vite, never embeds an H20 filesystem path or credential,
and cleans up its own child processes on Ctrl-C. It reuses a healthy inspector already on
the frontend port instead of terminating it. The H20 does not need Node.js merely to
execute the pipeline or API.

The manual equivalent for troubleshooting is:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o TCPKeepAlive=yes \
  -L 127.0.0.1:18081:127.0.0.1:18080 h20

# another Mac terminal
VITE_API_BASE_URL=http://127.0.0.1:18081 npm run dev -- \
  --host 127.0.0.1 --port 15174 --strictPort
```

## Build and preview

```bash
npm run typecheck
npm test
npm run build
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run build
npm run preview -- --host 127.0.0.1
```

The first build is appropriate when a production web server serves `/api` on the same
origin. The second embeds an explicit API origin. Vite's development proxy is not active
under `npm run preview`, so set the URL at build time or use a same-origin reverse proxy.
The backend defaults allow the documented `5173` dev, `4173` preview, and `15174` remote
viewer loopback origins; custom ports require matching repeated backend `--cors-origin`
arguments.

## What the inspector shows

- data-item/run catalog, including failed and invalid runs;
- paginated frame timeline with stage, track, and status filters;
- a fixed ten-node pipeline rail with same-frame before/after evidence, explicit
  `NOT_PRODUCED` reasons, and debug-only rectification labels;
- content-addressed source/undistorted/rectified stereo artifacts, detection boxes,
  candidate-to-track association, and all-hand 2D/3D overlays;
- a run-level H.264 raw-versus-stable stereo player and downloadable hardware-timestamp
  timeline when the worker exported those artifacts;
- interactive FHP21 3D skeleton with missing-landmark handling;
- QA failures, stage status, parent DAG, model/calibration provenance, final JSONL download,
  and raw trace records.

With no track filter, every hand remains visible with a deterministic color. Selecting a
track highlights it and dims the others rather than deleting evidence. Rectified panels
only consume `projected_keypoints_space=rectified`; native RTMPose coordinates are never
drawn on a rectified image. Older runs without the new image/video roles remain readable
and show an explicit unavailable state.

All artifact requests are scoped by the backend's opaque `run_key`; the browser cannot
read arbitrary server paths.

`FrameInspector` converts each API frame response once through the domain-level
`FrameEvidence` adapter. Pipeline state, stage comparison, stereo evidence, and the 3D canvas
share that normalized view instead of independently interpreting raw trace aliases. Shared
stage/status/schema/FHP21 constants come from the generated project contract; edit the
canonical JSON at the repository root rather than the generated TypeScript file.
