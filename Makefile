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
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

install-examples:
	uv pip install -e ".[examples]"

install-all:
	uv pip install -e ".[all]"

# Code quality
lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

type-check:
	mypy modelbridge/

# Testing
test:
	pytest

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest --cov=modelbridge --cov-report=html --cov-report=term-missing --cov-report=xml

test-fast:
	pytest -m "not slow"

test-slow:
	pytest -m "slow"

test-parallel:
	pytest -n auto

test-benchmark:
	pytest --benchmark-only

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
	uv pip install -e ".[dev]"
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