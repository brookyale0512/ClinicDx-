# =============================================================================
# ClinicDx — Makefile
# =============================================================================
# Engine-only (any EMR):
#   make up          GPU mode
#   make up-cpu      CPU mode
#
# Full stack (with nginx + OpenMRS proxy):
#   make up-full     GPU mode + nginx
#   make up-full-cpu CPU mode + nginx
# =============================================================================

COMPOSE_ENGINE = docker compose
COMPOSE_FULL   = docker compose -f docker-compose.yml -f docker-compose.full.yml

.PHONY: up up-cpu up-full up-full-cpu down restart \
        logs ps test test-unit test-int smoke lint help

# ── Engine-only (Component 1) ────────────────────────────────────────────────

up:
	$(COMPOSE_ENGINE) --profile gpu up -d
	@echo ""
	@echo "ClinicDx Engine started (GPU)."
	@echo "  API: http://localhost:$${ENGINE_PORT:-8321}"
	@echo "  Health: http://localhost:$${ENGINE_PORT:-8321}/api/health"
	@echo ""
	@echo "On first start, ~6.6 GB of artifacts will download from HuggingFace."

up-cpu:
	$(COMPOSE_ENGINE) --profile cpu up -d
	@echo ""
	@echo "ClinicDx Engine started (CPU)."
	@echo "  API: http://localhost:$${ENGINE_PORT:-8321}"

# ── Full stack (Component 1 + nginx) ─────────────────────────────────────────

up-full:
	$(COMPOSE_FULL) --profile gpu up -d
	@echo ""
	@echo "ClinicDx Full Stack started (GPU)."
	@echo "  HTTPS: https://localhost"
	@echo "  API:   https://localhost/clinicdx-api/api/health"

up-full-cpu:
	$(COMPOSE_FULL) --profile cpu up -d
	@echo ""
	@echo "ClinicDx Full Stack started (CPU)."

# ── Lifecycle ─────────────────────────────────────────────────────────────────

down:
	$(COMPOSE_ENGINE) --profile gpu --profile cpu down

restart: down up

logs:
	$(COMPOSE_ENGINE) --profile gpu --profile cpu logs -f --tail=100

ps:
	$(COMPOSE_ENGINE) --profile gpu --profile cpu ps

# ── Testing ───────────────────────────────────────────────────────────────────

test: test-unit test-int

test-unit:
	@echo "--- Running unit tests ---"
	python3 -m pytest tests/unit/ -v --tb=short

test-int:
	@echo "--- Running integration tests ---"
	python3 -m pytest tests/integration/ -v --tb=short

smoke:
	@bash scripts/smoke_test.sh

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	python3 -m ruff check services/ tests/
	python3 -m mypy services/ --ignore-missing-imports

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "ClinicDx — available targets:"
	@echo ""
	@echo "  Engine-only (any EMR):"
	@echo "    up            Start engine (GPU)"
	@echo "    up-cpu        Start engine (CPU)"
	@echo ""
	@echo "  Full stack (+ nginx for OpenMRS):"
	@echo "    up-full       Start full stack (GPU)"
	@echo "    up-full-cpu   Start full stack (CPU)"
	@echo ""
	@echo "  Lifecycle:"
	@echo "    down          Stop all containers"
	@echo "    restart       down + up"
	@echo "    logs          Follow all container logs"
	@echo "    ps            Show container status"
	@echo ""
	@echo "  Testing:"
	@echo "    test          unit + integration tests"
	@echo "    test-unit     unit tests only"
	@echo "    test-int      integration tests (stack must be running)"
	@echo "    smoke         end-to-end smoke test"
	@echo "    lint          ruff + mypy"
