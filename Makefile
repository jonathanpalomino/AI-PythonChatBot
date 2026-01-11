# =============================================================================
# Makefile
# Common commands for RAG Chatbot
# =============================================================================

.PHONY: help install dev test clean docker-up docker-down init-db run lint format

# Default target
help:
	@echo "RAG Chatbot - Available Commands"
	@echo "================================="
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Install dev dependencies"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"
	@echo "  make init-db     - Initialize database"
	@echo "  make run         - Run API server"
	@echo "  make test        - Run tests"
	@echo "  make test-api    - Test API endpoints"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean cache files"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev: install
	pip install pytest pytest-asyncio black flake8 mypy

# Start Docker services
docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "✅ Services are up!"

# Stop Docker services
docker-down:
	docker-compose down

# Initialize database
init-db:
	python scripts/init_db.py

# Run API server
run:
	python main.py

# Run in development mode with auto-reload
run-dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v

# Test API endpoints
test-api:
	python scripts/test_api.py

# Run linters
lint:
	flake8 src/ --max-line-length=100 --exclude=__pycache__,*.pyc
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/ scripts/ --line-length=100

# Clean cache and temp files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Setup complete environment
setup: docker-up install init-db
	@echo "✅ Setup complete! Run 'make run' to start the API"

# Restart everything
restart: docker-down clean docker-up
	@echo "✅ Restarted!"

# View logs
logs:
	docker-compose logs -f

# Check service status
status:
	docker-compose ps
	@echo ""
	@echo "API Status:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "API not running"