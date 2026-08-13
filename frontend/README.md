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
it available to the Mac first:

```bash
ssh -N -L 8000:127.0.0.1:8000 h20
```

Then run the frontend locally on the Mac with the commands above. The H20 currently does
not need Node.js merely to execute the pipeline or API.

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

## What the inspector shows

- data-item/run catalog, including failed and invalid runs;
- paginated frame timeline with stage, track, and status filters;
- content-addressed left/right source artifacts and 2D landmark overlays;
- interactive FHP21 3D skeleton with missing-landmark handling;
- QA failures, stage status, parent DAG, model/calibration provenance, final JSONL download,
  and raw trace records.

All artifact requests are scoped by the backend's opaque `run_key`; the browser cannot
read arbitrary server paths.
