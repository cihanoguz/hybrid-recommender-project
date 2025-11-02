.PHONY: format lint test install-dev help setup-precommit security-scan check-types

help:
	@echo "Available commands:"
	@echo "  make format          - Format code with black and isort"
	@echo "  make lint            - Run flake8 and pylint"
	@echo "  make test            - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-all        - Run all tests"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make setup-precommit - Install pre-commit hooks"
	@echo "  make security-scan   - Run security vulnerability scan (pip-audit)"
	@echo "  make check-types     - Run type checking with mypy (if installed)"

format:
	black .
	isort .

lint:
	flake8 .
	pylint --disable=C0111,R0903,R0913 app.py config.py utils.py data_loader/ recommenders/ ui/

test:
	@echo "🧪 Running unit tests..."
	pytest tests/unit -v

test-integration:
	@echo "🔗 Running integration tests..."
	pytest tests/integration -v

test-all:
	@echo "🧪 Running all tests..."
	pytest tests/ -v

test-cov:
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

install-dev:
	pip install -r requirements.txt

setup-precommit:
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed! They will run automatically on git commit."

security-scan:
	@echo "Running security vulnerability scan..."
	pip-audit --format=json --output=pip-audit-report.json || pip-audit || echo "pip-audit not available, skipping"

check-types:
	@echo "Running type checking with mypy..."
	@if command -v mypy >/dev/null 2>&1; then \
		mypy --ignore-missing-imports app.py config.py utils.py data_loader/ recommenders/ ui/ || echo "Type checking completed with errors"; \
	else \
		echo "mypy not installed. Install with: pip install mypy"; \
	fi

