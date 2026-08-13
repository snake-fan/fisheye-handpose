# Trace API backend

This directory is an independent `uv` project. It serves a read-only catalog of completed,
failed, still-active, or invalid pipeline trace folders. It reuses the core package's
filesystem-only `RunArtifactReader` so canonical v1 validation cannot drift; importing model
or GPU backends is not required.

## Run locally

```bash
cd backend
uv sync --locked --group dev
uv run --locked --no-editable fisheye-trace-api \
  --catalog-root ../runs \
  --host 127.0.0.1 \
  --port 8000
```

The default CORS origins are the React development server at `localhost:5173` and
`127.0.0.1:5173`. On a remote H20, keep the API on loopback and use an SSH tunnel:

```bash
uv run --locked --no-editable fisheye-trace-api \
  --catalog-root /mnt/workspace/zyf/fisheye/fisheye-handpose/runs \
  --host 127.0.0.1 \
  --port 8000
```

On the Mac, run `ssh -N -L 8000:127.0.0.1:8000 h20` and then start the React development
server described in [`../frontend/README.md`](../frontend/README.md). This diagnostic API
has no authentication layer; do not expose it directly on a public interface.

## API boundary

The versioned contract is [`../contracts/trace-api-v1.openapi.json`](../contracts/trace-api-v1.openapi.json).
The primary endpoints are:

- `GET /api/v1/runs` — paginated multi-run catalog;
- `GET /api/v1/runs/{run_key}` — manifest, summary, validation, evidence-backed provenance,
  and filter vocabulary;
- `GET /api/v1/runs/{run_key}/frames` — filtered frame timeline;
- `GET /api/v1/runs/{run_key}/frames/{frame_key}` — all stage records for one exact frame;
- `GET /api/v1/runs/{run_key}/record?record_id=...` — one original record by stable ID,
  including legacy IDs containing `/`;
- `GET /api/v1/runs/{run_key}/records/{stage}/{frame_key}` — one stage/frame slice;
- `GET|HEAD /api/v1/runs/{run_key}/artifacts/{relative_path}` — integrity-checked artifact,
  including a single HTTP byte range for video seeking.

All write methods under `/api/v1` return `405`. Artifact paths must be referenced by a trace,
remain within that run directory after symlink resolution, and match the recorded SHA-256 and
byte count before any bytes are served.

`run_id` is only unique inside one data item, so it is display metadata rather than an API key.
The catalog returns a stable opaque `run_key` (the first 16 hexadecimal characters of the
SHA-256 of the run's catalog-relative path) together with `item_id` and `run_id`. Every scoped
route uses `run_key`, preventing equal run IDs in different item folders from being confused.

Likewise, `frame_index` is display and ordering metadata only: it can be absent and is not
assumed unique. Each frame returns a full SHA-256 `frame_key` derived from its exact
`frame_id`; every frame-scoped route uses that key. One malformed or partially written run is
listed as `INVALID` without hiding healthy runs, and its detail response reports the catalog
error. Legacy traces remain readable but are explicitly `LEGACY_UNVERIFIED`, never reported as
passing canonical integrity validation.

Canonical details expose optional named provenance facets copied from the manifest's
`config`/`metadata`, the exact `SYSTEM/worker_inputs_verified` record, and the exact
`RECTIFICATION/worker_rectification_loaded` record. Missing evidence stays absent; legacy and
structurally unreadable runs return an empty object rather than placeholder values.

Canonical runs use `run_manifest.json`, `trace.jsonl`, and optional `run_summary.json`. The
reader also accepts the older `manifest.json`/`trace_manifest.json`, `records.jsonl`, and
`summary.json` file names plus flat record metadata.

## Verify

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```
