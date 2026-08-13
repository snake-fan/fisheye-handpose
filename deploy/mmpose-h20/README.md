# MMPose H20 compatibility environment

This directory defines a standalone, exact-version `uv` environment for reproducing the
legacy MMPose stack on a Linux x86_64 NVIDIA H20 server. It is deliberately separate from
the Python 3.11+ geometry package at the repository root.

> **Compatibility baseline, not a secure production profile.** These versions reproduce
> an older upstream binary matrix. Pinning them does not mean they are currently supported
> or free of known vulnerabilities. Do not expose this environment as a network service or
> treat a successful doctor report as a security approval.

## Frozen binary matrix

| Component | Exact target |
| --- | --- |
| OS / architecture | Linux x86_64 |
| Python | 3.10 |
| GPU / compute capability | NVIDIA H20 / SM90 (`9.0`) |
| PyTorch / TorchVision | `2.1.0` / `0.16.0`, official CUDA 12.1 index |
| MMCV | `2.1.0`, official `cp310` Linux CUDA 12.1 / Torch 2.1 wheel |
| MMEngine / MMDetection / MMPose | `0.10.3` / `3.2.0` / `1.3.2` |
| NumPy / SMPL-X | `1.26.4` / `0.1.28` |
| Chumpy compatibility | `0.71`, fixed Git SHA `2816a138…` |

`environment.json` is the machine-readable contract for runtime pins and constraints.
Model weights are intentionally managed by the adjacent asset manifest and fetch script;
they are not duplicated in the environment contract.

## Create the environment on H20

Run commands from this directory:

```bash
cd deploy/mmpose-h20
uv --version  # must be >=0.12.3,<0.13
uv python install 3.10
uv sync --locked
```

This lock was generated and tested with uv 0.12.3. The project accepts
`uv>=0.12.3,<0.13`; older uv releases cannot apply the declared build/source constraints
safely. Upgrade uv using the same method that installed it;
`uv self update` only works for Astral's standalone installer and must not be assumed for
pipx, Conda, apt, Homebrew, or a system image. `uv sync --locked` verifies that
`pyproject.toml` still matches the committed lock and installs it without changing the
resolution. The lock also excludes packages published after 2024-08-01 so the 2024
OpenMMLab release is not silently mixed with 2026 transitive APIs. The MMCV dependency is
an exact wheel URL, so uv cannot fall back to `mmcv-lite` or an unreviewed source build.

MMPose 1.3.2 declares the abandoned `chumpy==0.70` package. That release cannot be built
cleanly under modern PEP 517 tooling and cannot import with NumPy 1.26. This environment
therefore pins the reviewed modernization branch at an immutable commit (`0.71`) instead
of silently applying runtime monkey-patches.

Before copying the environment to another host, validate the static contract on any
machine with Python 3.10 or newer, including macOS:

```bash
python3 doctor.py --mode manifest
```

This standard-library-only mode intentionally bypasses uv project resolution, because
the subproject itself is constrained to Linux x86_64. It validates the committed
`pyproject.toml`, `uv.lock`, and `environment.json` without importing the CUDA stack.

After synchronization on the H20 server, run the fail-closed runtime check:

```bash
uv run --locked python doctor.py --mode runtime
```

The runtime doctor requires all checks to pass: Linux x86_64, Python 3.10, exact package
versions, PyTorch CUDA 12.1, an available SM90 GPU, a real CUDA invocation of
`mmcv.ops.nms`, and an FP16 CUDA matrix operation. It prints exactly one JSON report to
stdout and exits `0` only when every required check passes. A missing import, CPU-only
fallback, wrong CUDA build, wrong GPU architecture, or failed native operator exits `1`.

To additionally require one or more populated model directories:

```bash
uv run --locked python doctor.py --mode runtime \
  --model-dir /srv/fisheye-handpose/models
```

This optional directory check verifies presence only. Use the model asset tooling for
artifact identity and checksum validation.

## Model assets and smoke tests

The detector and pose aliases are intentionally not used. Fetch the two explicit official
artifacts only after reviewing the checkpoint/training-data rights:

```bash
uv run --locked python model_assets.py fetch \
  --manifest ./model-assets.json \
  --output-dir /srv/fisheye-handpose/models/openmmlab \
  --acknowledge-license-risk
uv run --locked python model_assets.py verify \
  --manifest ./model-assets.json \
  --output-dir /srv/fisheye-handpose/models/openmmlab
```

The adjacent `model-assets.json` is the single checkpoint manifest used by fetch, verify,
and the RTMPose smoke. It stores full SHA-256 digests; the downloader writes atomically
and never deserializes a checkpoint. The PyPI package does not guarantee the complete
`configs/` and `demo/` trees, so obtain a clean, detached MMPose checkout at the exact
signed v1.3.2 commit:

```bash
git clone --no-checkout https://github.com/open-mmlab/mmpose.git \
  /srv/fisheye-handpose/vendor/mmpose
git -C /srv/fisheye-handpose/vendor/mmpose checkout --detach \
  5408bc76f5b848cf925a0d1857899011d8c5b497
git -C /srv/fisheye-handpose/vendor/mmpose status --short
```

The final command must print nothing; the smoke rejects a different commit or a dirty
checkout.

Run the explicit detector + low-level RTMPose smoke on a real IR frame. The smoke verifies
the checkout commit and clean config paths, then rechecks both checkpoint sizes and hashes
before any Torch deserialization:

```bash
uv run --locked python scripts/rtmpose_smoke.py \
  --model-dir /srv/fisheye-handpose/models/openmmlab \
  --mmpose-source /srv/fisheye-handpose/vendor/mmpose \
  --image /ABS/PATH/real_ir_frame.png
```

A no-detection result is a failed smoke (exit `1`) and never falls through to whole-image
pose inference. MANO is never downloaded. After separately accepting its terms, mount the
files as `MODEL_ROOT/mano/MANO_LEFT.pkl` and `MANO_RIGHT.pkl`. Copy
`mano-assets.example.json` to a private location, fill in the two byte counts/SHA-256
digests and the real provenance record, then set both acknowledgement fields only after
the applicable review. `--manifest` is mandatory and has no implicit default. The MANO
smoke refuses to deserialize a file unless that private manifest matches:

```bash
uv run --locked python scripts/mano_smoke.py \
  --model-dir /ABS/PATH/MODEL_ROOT \
  --manifest /ABS/PATH/mano-assets.json
```

This validates left/right forward passes plus finite Adam gradients and an LBFGS closure.

## Tests

The deployment contract tests use only the Python standard library, so manifest and CLI
validation can run before installing the Linux-only CUDA environment:

```bash
python3 -m unittest discover -s tests -v
```

Do not run `uv sync` for this subproject on macOS: the pinned MMCV wheel intentionally
supports only CPython 3.10 on Linux x86_64.
