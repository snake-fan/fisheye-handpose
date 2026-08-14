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

On the Mac, prefer the persistent viewer helper from the repository root:

```bash
uv run --locked --no-editable python scripts/remote_trace_viewer.py
```

It forwards local `18081` to remote `18080`, validates this API's exact read-only health
response, starts or reuses Vite on `15174`, and automatically recreates a dropped SSH
tunnel with bounded exponential backoff. See [`../frontend/README.md`](../frontend/README.md)
for its manual equivalent and overrides. The default CORS list contains only loopback
origins used by the documented Vite dev (`5173`), preview (`4173`), and remote helper
(`15174`) workflows. A custom frontend port requires a matching repeated `--cors-origin`.
This diagnostic API has no authentication layer; keep it on H20 loopback and do not expose
it directly on a public interface.

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

Verified content-addressed artifacts whose path is
`blobs/sha256/<prefix>/<sha256>.<suffix>` are returned with
`Cache-Control: private, max-age=31536000, immutable` for full, byte-range, and `HEAD`
responses. Their URL identifies the verified bytes, so the browser can reuse frame, crop, mask,
and video content without downloading it again when the viewer changes stages. Referenced files
outside that content-addressed layout remain `no-store`.

For a finalized canonical run, the detail endpoint reuses a completed integrity-validation
snapshot for at most one second. This bounds repeated viewer requests to one blob scan per short
interaction while retaining periodic tamper detection: once the TTL expires, the next detail
request validates every referenced blob again. ACTIVE runs are never covered by this cache. The
artifact endpoint independently stats the file before serving and verifies its SHA-256 whenever
that fingerprint is new or changed. If it detects a missing or changed artifact, it rejects the
request and immediately invalidates the run's validation snapshot. Consequently, `validation`
describes a check performed no more than one second before the detail response, rather than a
filesystem lock or an indefinite seal.

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
