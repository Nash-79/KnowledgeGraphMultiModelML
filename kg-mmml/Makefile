.PHONY: help install test lint format clean

.DEFAULT_GOAL := help

help:
	@echo "KG-MMML Makefile Commands:"
	@echo "  install      Install package"
	@echo "  install-dev  Install with dev dependencies"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linters"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  setup        Complete development setup"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

setup: install-dev
	cp .env.example .env
	mkdir -p logs outputs reports/{tables,figures,debug}
	@echo "✅ Development environment ready!"
	@echo "⚠️  Edit .env with your SEC User-Agent!"
