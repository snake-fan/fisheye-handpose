# Trace API backend

This directory is an independent `uv` project. It serves a read-only catalog of completed,
failed, still-active, or invalid pipeline trace folders. It reuses the core package's
filesystem-only `RunArtifactReader` so canonical v1 validation cannot drift; importing model
or GPU backends is not required.

## Run locally

```bash
cd backend
uv sync --locked --group dev --no-editable
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

For a finalized canonical run, the detail endpoint keeps a completed full-SHA integrity snapshot
for at most five minutes. Reusing that snapshot is conditional: every request compares each
referenced artifact's resolved in-run path and stat fingerprint (`device`, inode, byte count,
mtime, and ctime) with the fully validated snapshot. A missing file, changed path, or changed stat
fingerprint forces an immediate full validation instead of returning the cached result. The next
detail request after five minutes also recomputes every referenced SHA-256 even when all stat
fingerprints remain unchanged. This leaves the response schema unchanged while avoiding repeated
multi-gigabyte reads during one viewer session.

ACTIVE runs are never covered by the validation cache. The artifact endpoint also remains
independent: every `GET` or `HEAD` re-resolves the referenced path inside the run and checks its
current stat fingerprint, then verifies SHA-256 whenever that fingerprint is new or changed. If it
detects a missing or changed artifact, it rejects the request and immediately invalidates the
run's validation snapshot.

The consistency window is therefore explicit: a finalized run's `validation` is backed by a full
hash pass no more than five minutes old plus current stat fingerprints, not by a filesystem lock.
An out-of-band mutation engineered to preserve the resolved path and every stat field can remain
covered by the prior result until the periodic full pass; normal writes, replacements, moves, and
symlink changes invalidate it immediately. Catalog-directory discovery is cached for 30 seconds,
but top-level item-directory changes invalidate it sooner; a manifest added later inside an
already-existing nested directory can take up to 30 seconds to appear.

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

Run-list queries apply `q` (run ID or item ID) and any cheaply decidable non-`INVALID` status
against the small manifest/summary documents before opening trace files. Returned candidates are
still fully summarized and post-filtered, so a trace that advertises `COMPLETED` but fails
canonical validation cannot leak into a completed-only result. `status=INVALID` deliberately does
not use the cheap status shortcut because corruption is only knowable after reading the trace.

## Verify

```bash
uv run --locked --no-editable pytest -q
uv run --locked --no-editable ruff check .
uv run --locked --no-editable ruff format --check .
```
