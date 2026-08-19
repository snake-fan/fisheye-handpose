# ADR 0004: Keep the legacy viewer and React inspector in distinct roles

- Status: Accepted
- Date: 2026-08-19

## Context

The project needs both a dependency-light way to inspect one synthetic/local run and an
operator interface for browsing many real runs, frames, stages, tracks, provenance records,
images, and videos. Expanding the original embedded viewer into the primary application would
couple static assets to core releases and duplicate catalog behavior.

## Decision

Keep `fisheye-handpose trace-serve RUN_DIR` as a legacy, self-contained, read-only single-run
viewer. Use the standalone FastAPI catalog plus React/TypeScript application as the normal
multi-item inspector.

Both interfaces consume canonical traces and never write run data. The API is read-only,
integrity-checks artifacts, and uses opaque run/frame keys. It has no authentication layer and
must remain on loopback or behind a trusted tunnel/reverse proxy.

## Consequences

- The legacy viewer remains useful for `trace-demo`, packaging smoke tests, and minimal local
  diagnosis.
- New multi-run operator features belong in `backend/` and `frontend/`.
- Trace semantics remain the integration boundary; the React app does not import Python worker
  types or access server filesystem paths.
- Static viewer fixes may continue for correctness, but feature parity with React is not a goal.
