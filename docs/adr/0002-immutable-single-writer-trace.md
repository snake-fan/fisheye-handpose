# ADR 0002: Use an immutable, single-writer canonical trace

- Status: Accepted
- Date: 2026-08-19

## Context

Audit, model inference, fusion, tracking, MANO, temporal refinement, visualization, and export
must remain diagnosable after success, rejection, interruption, or late failure. Allowing the
core and worker to write the same run independently would create competing ordering, hashes,
and lifecycle states.

## Decision

Each attempt owns one non-overwriting directory under `runs/<item_id>/<run_id>/`. A single
core `RunArtifactWriter` appends records and content-addressed blobs. The H20 worker writes a
temporary staging package; the bridge validates its schemas, DAG, paths, byte counts, and
SHA-256 digests before importing it through the core writer.

An ACTIVE run may be resumed under its single-writer lock. Finalization creates a separate
summary and makes the run immutable. Later stages append derived records instead of replacing
Raw observations.

## Consequences

- Audit and perception provenance share one verifiable hash chain.
- Missing work is represented explicitly as `SKIPPED` or `NOT_PRODUCED`; partial output cannot
  masquerade as a successful final artifact.
- Large worker artifacts are copied and hashed in bounded chunks.
- Repair means creating a new run, not editing a finalized one in place.
