.PHONY: up down rebuild restart logs status dev test lint clean dev-up dev-down dev-logs dev-build dev-restart

DEV := docker compose -f docker-compose.dev.yml

# Dev stack — hot reload, no rebuild on code changes
dev-up:
	$(DEV) up -d

# Stop dev stack
dev-down:
	$(DEV) down

# Follow dev logs
dev-logs:
	$(DEV) logs -f

# Rebuild dev images (only needed when deps change)
dev-build:
	$(DEV) build

# Restart dev containers without rebuild
dev-restart:
	$(DEV) restart

# Start all services
up:
	docker compose up -d --build

# Stop all services
down:
	docker compose down

# Force rebuild and restart
rebuild:
	docker compose down
	docker compose up -d --build

# Restart without rebuild
restart:
	docker compose restart

# Follow logs (server + dashboard)
logs:
	docker compose logs -f

# Container status
status:
	docker compose ps

# Development — backend with hot reload
dev-server:
	cd apps/server && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Development — frontend with hot reload
dev-dashboard:
	cd apps/dashboard && npm install && npm run dev

# Run backend tests
test-server:
	cd apps/server && python -m pytest -v

# Run frontend tests
test-dashboard:
	cd apps/dashboard && npm test

# Run all tests
test: test-server test-dashboard

# Lint backend
lint-server:
	cd apps/server && ruff check src/ tests/

# Typecheck backend
typecheck-server:
	cd apps/server && mypy src/

# Lint frontend
lint-dashboard:
	cd apps/dashboard && npm run lint

# Clean build artifacts
clean:
	rm -rf apps/dashboard/.next apps/dashboard/node_modules
	rm -rf apps/server/.venv apps/server/__pycache__ apps/server/.pytest_cache
	find apps/server -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
