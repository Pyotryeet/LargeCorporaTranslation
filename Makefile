# ---------------------------------------------------------------------------
# Turkish Corpus Translation Benchmark — Makefile v3.9
# ---------------------------------------------------------------------------
# Usage:
#   make help              Show this help
#   make setup             One-command full environment setup
#   make setup-quick       Minimal dev setup (skip model DL)
#   make test              Run unit tests
#   make lint              Run linter
#   make format            Run formatter
#   make run               Full benchmark (auto-detect platform)
#   make run-quick         5-minute evaluation
#   make run-dry           Smoke test (60s)
#   make run-diffusion     Diffusion model benchmark
#   make docker-build      Build Docker image
#   make docker-run        Run in Docker
#   make dashboard         Launch observability dashboard
#   make clean             Clean all artifacts + caches
# ---------------------------------------------------------------------------

PYTHON := python3.11

.PHONY: help setup setup-quick setup-full \
        test test-safe lint format \
        run run-safe run-quick run-dry run-diffusion \
        run-obs run-h200 run-macos \
        docker-build docker-run \
        dashboard \
        clean clean-all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ═══════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════

setup: ## Full environment setup (auto-detect platform)
	bash setup.sh --full

setup-quick: ## Minimal dev setup
	bash setup.sh --quick

setup-cuda: ## CUDA-only setup
	bash setup.sh --cuda --full

# ═══════════════════════════════════════════════════════════════════════════
# Code quality
# ═══════════════════════════════════════════════════════════════════════════

test: ## Run unit tests
	pytest tests/ -v --timeout=120 --ignore=tests/test_e2e.py

test-safe: ## Run unit tests with --safe-mode
	pytest tests/ -v --timeout=120 --ignore=tests/test_e2e.py --safe-mode

test-all: ## Run all tests including E2E
	pytest tests/ -v --timeout=900

lint: ## Run linter
	ruff check benchmark/ tests/

format: ## Run formatter
	ruff format benchmark/ tests/

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

run: ## Full benchmark (auto-detect platform)
	bash run.sh --full

run-safe: ## Full benchmark with --safe-mode
	bash run.sh --full --safe-mode

run-quick: ## 5-minute evaluation
	bash run.sh --quick

run-dry: ## Smoke test (60s)
	bash run.sh --dry-run

run-diffusion: ## Diffusion model benchmark
	bash run.sh --diffusion --quick

run-obs: ## Full benchmark with observability dashboard
	bash run.sh --full --observability

run-h200: ## H200-compatible production run (uses run.sh)
	bash run.sh --full

run-macos: ## macOS-compatible run (uses run.sh)
	bash run.sh --full

# ═══════════════════════════════════════════════════════════════════════════
# Docker
# ═══════════════════════════════════════════════════════════════════════════

docker-build: ## Build Docker image
	docker build -t tr-benchmark:3.9 .

docker-run: ## Run benchmark in Docker
	docker run --rm \
	  --gpus '"device=0,1"' \
	  --ipc=host --ulimit memlock=-1 \
	  -v $$(pwd)/data:/data \
	  tr-benchmark:3.9 --config /data/config.yaml

# ═══════════════════════════════════════════════════════════════════════════
# Observability
# ═══════════════════════════════════════════════════════════════════════════

dashboard: ## Launch Prometheus + Grafana monitoring stack
	docker-compose -f .github/docker-compose.obs.yaml up -d
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

# ═══════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════

clean: ## Remove build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/ 2>/dev/null || true

clean-all: clean ## Remove build artifacts + all caches
	rm -rf ~/.cache/tr_benchmark/ 2>/dev/null || true
	rm -rf .venv/ 2>/dev/null || true
	@echo "All caches cleared. Run 'make setup' to rebuild."
