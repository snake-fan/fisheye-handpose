.PHONY: help sync-core check check-core check-backend check-frontend check-h20-static

UV ?= uv
PYTHON ?= python3
NPM ?= npm

help:
	@echo "make sync-core        Install the locked Python 3.11 core/dev environment"
	@echo "make check-core       Verify the core lock, tests, lint, format, and packaging smoke"
	@echo "make check-backend    Verify the independent Trace API project"
	@echo "make check-frontend   Install and verify the React inspector"
	@echo "make check-h20-static Validate H20 manifests and lightweight worker tests (no CUDA sync)"
	@echo "make check            Run every local/CI-safe check"

sync-core:
	$(UV) sync --locked --extra dev --no-editable

check-core:
	$(UV) lock --check
	$(UV) sync --locked --extra dev --no-editable
	$(UV) run --locked --extra dev --no-editable python scripts/generate_contracts.py --check
	$(UV) run --locked --extra dev --no-editable pytest -q
	$(UV) run --locked --extra dev --no-editable ruff check src tests scripts
	$(UV) run --locked --extra dev --no-editable ruff format --check src tests scripts
	$(UV) run --locked --no-editable fisheye-handpose schema >/dev/null

check-backend:
	cd backend && $(UV) lock --check
	cd backend && $(UV) sync --locked --group dev --no-editable
	cd backend && $(UV) run --locked --no-editable pytest -q
	cd backend && $(UV) run --locked --no-editable ruff check .
	cd backend && $(UV) run --locked --no-editable ruff format --check .

check-frontend:
	cd frontend && $(NPM) ci
	cd frontend && $(NPM) test
	cd frontend && $(NPM) run typecheck
	cd frontend && $(NPM) run build

check-h20-static:
	$(PYTHON) deploy/mmpose-h20/doctor.py --mode manifest >/dev/null
	$(UV) sync --locked --extra dev --no-editable
	$(UV) run --locked --extra dev --no-editable pytest -q deploy/mmpose-h20/tests
	$(UV) run --locked --extra dev --no-editable ruff check deploy/mmpose-h20
	$(UV) run --locked --extra dev --no-editable ruff format --check deploy/mmpose-h20

check: check-core check-backend check-frontend check-h20-static
