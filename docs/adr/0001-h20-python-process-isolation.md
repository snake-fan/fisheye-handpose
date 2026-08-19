# ADR 0001: Isolate the H20 worker behind a Python process boundary

- Status: Accepted
- Date: 2026-08-19

## Context

The cross-platform geometry, trace, and orchestration code requires Python 3.11 or newer.
The reproducible H20 perception stack is a legacy binary matrix: Python 3.10, CUDA 12.1,
PyTorch 2.1, MMCV 2.1, MMDetection 3.2, and MMPose 1.3.2. Importing both environments into
one interpreter would make dependency resolution, packaging, and failure isolation fragile.

## Decision

Keep the H20 implementation in `deploy/mmpose-h20` and launch it as a separate Python 3.10
process. The core sends a versioned JSON request and accepts only a validated result package.
The worker does not import the core package, and the core never imports CUDA, OpenMMLab,
Torch, or MANO runtime modules.

The configured H20 may use a locally compiled SM90 MMCV wheel. Core updates must not
implicitly synchronize or replace that environment.

## Consequences

- Core development and API/inspector operation remain portable and GPU-independent.
- Worker contracts, asset identities, exit status, and partial-failure semantics must be
  validated explicitly at the process boundary.
- Similar DTO semantics exist on both sides by design; they are checked through bridge and
  contract tests rather than shared imports.
- CUDA runtime verification remains an H20-only gate. Lightweight worker tests may run from
  the root dev environment without synchronizing the Linux-only CUDA project.
