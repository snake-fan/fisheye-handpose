# ADR 0005: Generate shared constants from one cross-runtime project contract

- Status: Accepted
- Date: 2026-08-19

## Context

The Python 3.11 core, isolated Python 3.10 worker, and TypeScript inspector all need the same
trace stages and statuses, run statuses, schema and mapping identities, and FHP21 landmark
topology. Copying these values into each runtime made ordinary schema changes easy to apply
partially and hard to review as one compatibility decision.

The process-isolation boundary still matters: the worker must not import the core package, and
the frontend must not depend on Python modules.

## Decision

Keep the language-neutral source in `contracts/project-contract-v1.json`. Generate
dependency-free Python constants separately into the core, worker, and standalone H20 tooling,
plus TypeScript constants and literal types for the inspector, with
`scripts/generate_contracts.py`. The same check verifies the schema IDs of committed H20 JSON
manifests that require deliberate, format-specific migration rather than blind regeneration.

Generated files are committed so each runtime remains independently buildable. They are not
edited by hand. `uv run --locked --no-editable python scripts/generate_contracts.py --check`
and the root test suite fail when a checked-in output has drifted from the canonical JSON.

The project contract contains stable shared vocabulary, not every runtime-specific DTO or
algorithm setting. Adding a value is a compatibility change and requires regenerating all
outputs and updating the relevant contract tests.

## Consequences

- Cross-runtime identifiers and FHP21 topology have one reviewable source of truth.
- The core, worker, standalone deployment tools, and inspector retain their dependency and
  process boundaries.
- Generated diffs make the effect of contract changes visible in every consumer language.
- Runtime-specific schemas remain local when sharing them would couple unrelated components.
