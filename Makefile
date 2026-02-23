.PHONY: install test test-unit test-integration test-slow clean lint format setup dev-setup help

# Default target
help:
	@echo "Insurance RAG AI - Available commands:"
	@echo "  install       Install dependencies"
	@echo "  setup         Complete project setup (install + build index)"
	@echo "  dev-setup     Development setup (install + dev tools)"
	@echo "  test          Run all tests (unit + integration, no slow tests)"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests (no slow tests)"
	@echo "  test-slow     Run slow integration tests"
	@echo "  test-all      Run all tests including slow ones"
	@echo "  lint          Run code quality checks"
	@echo "  format        Format code with black and isort"
	@echo "  build-index   Build FAISS vector index from data"
	@echo "  run           Start the development server"
	@echo "  clean         Clean generated files"

install:
	pip install -r requirements.txt

dev-setup: install
	pip install black isort flake8 mypy safety bandit

setup: install build-index
	@echo "Project setup complete!"

build-index:
	python -c "from app.ingest import build_index; build_index()"

test:
	pytest tests/ -v -m "not slow" --cov=app --cov-report=term-missing

test-unit:
	pytest tests/ -v -m "unit" --cov=app --cov-report=term-missing

test-integration:
	pytest tests/ -v -m "integration and not slow" --cov=app --cov-report=term-missing

test-slow:
	pytest tests/ -v -m "slow" 

test-all:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

lint:
	@echo "Running code quality checks..."
	flake8 app/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 app/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	mypy app/ --ignore-missing-imports --no-strict-optional || true
	safety check -r requirements.txt

format:
	black app/ tests/ conftest.py
	isort app/ tests/ conftest.py

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f .coverage
	rm -f coverage.xml
	rm -rf .mypy_cache
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

# Docker targets (if needed later)
docker-build:
	docker build -t insurance-rag-ai .

docker-run:
	docker run -p 8000:8000 --env-file .env insurance-rag-ai