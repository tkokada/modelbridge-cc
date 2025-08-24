# Makefile for ModelBridge development

.PHONY: help install install-dev lint format type-check test test-cov clean build

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install the package"
	@echo "  install-dev    - Install the package with development dependencies"
	@echo "  lint           - Run ruff linter"
	@echo "  format         - Format code with ruff"
	@echo "  type-check     - Run mypy type checking"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-cov       - Run tests with coverage report"
	@echo "  test-fast      - Run fast tests (skip slow ones)"
	@echo "  test-parallel  - Run tests in parallel"
	@echo "  clean          - Clean build artifacts"
	@echo "  build          - Build the package"
	@echo "  check-all      - Run all checks (lint, format, type-check, test)"
	@echo "  clean-outputs  - Clean all output files"
	@echo "  clean-examples - Clean example outputs only"
	@echo "  clean-reports  - Clean development reports only"
	@echo "  run-neural-demo     - Run neural network demo"
	@echo "  run-neural-example  - Run neural network model bridge"

# Installation
install:
	uv sync -e .

install-dev:
	uv sync -e ".[dev]"

install-examples:
	uv sync -e ".[examples]"

install-all:
	uv sync -e ".[all]"

# Code quality
lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

type-check:
	uv run mypy modelbridge/

# Testing
test:
	uv run pytest

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-cov:
	uv run pytest --cov=modelbridge --cov-report=html --cov-report=term-missing --cov-report=xml

test-fast:
	uv run pytest -m "not slow"

test-slow:
	uv run pytest -m "slow"

test-parallel:
	uv run pytest -n auto

test-benchmark:
	uv run pytest --benchmark-only

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/  # Legacy coverage directory
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete

clean-outputs:
	@echo "Cleaning all output files..."
	rm -rf outputs/
	rm -rf demo_results/  # Legacy demo directory
	rm -rf results/       # Legacy results directory
	find . -name "*.db" -not -path "./outputs/*" -delete
	find . -name "*.sqlite*" -not -path "./outputs/*" -delete

clean-examples:
	@echo "Cleaning example outputs..."
	rm -rf outputs/examples/
	rm -rf outputs/databases/
	rm -rf outputs/plots/

clean-reports:
	@echo "Cleaning development reports..."
	rm -rf outputs/reports/
	rm -rf outputs/tests/

clean-all: clean clean-outputs

build:
	python -m build

# Combined checks
check-all: lint format-check type-check test

# Development setup
setup-dev:
	@echo "Setting up development environment..."
	uv sync -e ".[dev]"
	@echo "Development environment ready!"

# Run examples
run-simple-example:
	cd example/simple_benchmark && python hpopt_benchmark_refactored.py -c config_sample.toml

run-neural-demo:
	cd example/neural_network && python mnist_sklearn_bridge.py --demo

run-neural-example:
	cd example/neural_network && python mnist_sklearn_bridge.py --subset-size 500 --n-train 2 --n-test 1

run-mas-example:
	cd example/mas_bench && python hpopt_data_assimilation_refactored.py

run-demo:
	python example_usage.py