# Documentation map

Documentation is grouped by authority so an older investigation is not mistaken for the
current implementation. When documents disagree, normative contracts and accepted ADRs take
precedence over current descriptions; current descriptions take precedence over historical
reports.

## Normative contracts

- [`../contracts/project-contract-v1.json`](../contracts/project-contract-v1.json) — canonical
  cross-runtime stage/status, schema-ID, mapping-ID, and FHP21 constants; generated Python and
  TypeScript views must match it.
- [`output-schema.md`](output-schema.md) — `fhp21/v1` landmark, coordinate, validity, and
  provenance contract.
- [`trace-format.md`](trace-format.md) — canonical append-only run layout, lifecycle, hash
  chain, and artifact rules.
- [`../contracts/trace-api-v1.openapi.json`](../contracts/trace-api-v1.openapi.json) — checked-in
  read-only Trace API v1 schema, verified against the runtime by backend tests.

## Accepted architecture decisions

- [`adr/0001-h20-python-process-isolation.md`](adr/0001-h20-python-process-isolation.md) — keep
  the Python 3.10 CUDA/OpenMMLab environment behind a process boundary.
- [`adr/0002-immutable-single-writer-trace.md`](adr/0002-immutable-single-writer-trace.md) — use
  one canonical writer and immutable finalized runs.
- [`adr/0003-image-spaces-and-pose-profiles.md`](adr/0003-image-spaces-and-pose-profiles.md) — keep
  full-frame rectification in QA/debug and make virtual-perspective pose input an explicit
  profile.
- [`adr/0004-inspection-ui-boundaries.md`](adr/0004-inspection-ui-boundaries.md) — position the
  legacy viewer as a single-run utility and React as the normal multi-run inspector.
- [`adr/0005-generated-cross-runtime-project-contract.md`](adr/0005-generated-cross-runtime-project-contract.md)
  — generate shared constants for the independent Python and TypeScript runtimes from one
  language-neutral contract.

## Current implementation and operations

- [`architecture.md`](architecture.md) — stable boundaries, implemented foundation, and known
  gaps.
- [`pipeline-v2-iteration-plan.md`](pipeline-v2-iteration-plan.md) — current v2 algorithm
  iteration report and H20 validation outcome.
- [`../deploy/mmpose-h20/README.md`](../deploy/mmpose-h20/README.md) — pinned H20 environment,
  assets, worker request, runtime doctor, and smoke workflow.
- [`../backend/README.md`](../backend/README.md) — read-only catalog API and security boundary.
- [`../frontend/README.md`](../frontend/README.md) — React inspector development, build, and
  remote-tunnel workflow.
- [`../CONTEXT.md`](../CONTEXT.md) — concise stable vocabulary shared across components.

## Historical and superseded analysis

- [`current-pipeline-problem-analysis.md`](current-pipeline-problem-analysis.md) — historical
  diagnosis of commit `5eacb7a`; retained as evidence, not a description of the current v2
  worker.
- [`design-review.md`](design-review.md) — dated review of the original engineering proposal;
  retained for decision provenance.

Historical documents are intentionally not rewritten when the implementation changes. Their
status banner and recorded commit/run identify the evidence they describe.
