# Makefile for ModelBridge development

.PHONY: help install install-dev install-examples install-all sync sync-dev sync-examples sync-all lint format format-check type-check test test-unit test-integration test-cov test-fast test-parallel test-benchmark clean clean-outputs clean-examples clean-reports clean-all build check-all run-simple-example run-benchmark-example run-neural-demo run-neural-example run-neural-pytorch run-mas-demo run-mas-example run-demo pre-commit-install pre-commit-run pre-commit-update setup-dev

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "📦 Installation:"
	@echo "  install        - Install the package (editable)"
	@echo "  install-dev    - Install with development dependencies"
	@echo "  install-examples - Install with example dependencies (PyTorch)"
	@echo "  install-all    - Install with all dependencies"
	@echo "  sync           - Sync dependencies from lockfile"
	@echo "  sync-dev       - Sync with dev dependencies"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  lint           - Run ruff linter"
	@echo "  format         - Format code with ruff"
	@echo "  type-check     - Run mypy type checking"
	@echo "  check-all      - Run all checks (lint, format, type-check, test)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-cov       - Run tests with coverage report"
	@echo "  test-fast      - Run fast tests (skip slow ones)"
	@echo "  test-parallel  - Run tests in parallel"
	@echo ""
	@echo "🔀 Pre-commit:"
	@echo "  pre-commit-install - Install pre-commit hooks"
	@echo "  pre-commit-run     - Run pre-commit on all files"
	@echo "  pre-commit-update  - Update pre-commit hook versions"
	@echo ""
	@echo "🚀 Examples:"
	@echo "  run-simple-example    - Run simple benchmark"
	@echo "  run-benchmark-example - Run advanced benchmark with TOML config"
	@echo "  run-neural-demo       - Run neural network demo (sklearn)"
	@echo "  run-neural-example    - Run neural network model bridge (sklearn)"
	@echo "  run-neural-pytorch    - Run neural network model bridge (PyTorch)"
	@echo "  run-mas-demo          - Run MAS-Bench traffic demo with mock simulation"
	@echo "  run-mas-example       - Run MAS-Bench traffic optimization example"
	@echo ""
	@echo "🏗️ Development:"
	@echo "  setup-dev      - Set up complete development environment"
	@echo "  build          - Build the package"
	@echo "  clean          - Clean build artifacts"
	@echo "  clean-outputs  - Clean all output files"
	@echo "  clean-examples - Clean example output files only"
	@echo "  clean-all      - Clean everything"

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

install-examples:
	uv pip install -e ".[examples]"

install-all:
	uv pip install -e ".[all]"

# Sync dependencies (use this for lockfile-based installs)
sync:
	uv sync

sync-dev:
	uv sync --extra dev

sync-examples:
	uv sync --extra examples

sync-all:
	uv sync --extra dev --extra examples

# Code quality
lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

type-check:
	uv run mypy modelbridge/ --disable-error-code=unreachable --disable-error-code=unused-ignore

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
	rm -rf example/simple_benchmark/benchmark_results/
	rm -rf example/simple_benchmark/simple_benchmark_results/
	rm -rf example/neural_network/neural_network_results/
	rm -rf example/neural_network/pytorch_results/
	rm -rf example/mas_bench/mas_demo_results/
	rm -rf example/mas_bench/mas_bench_results/
	rm -rf example/mas_bench/results/
	find example/ -name "*.db" -delete
	find example/ -name "*.png" -delete
	find example/ -name "*.csv" -delete

clean-reports:
	@echo "Cleaning development reports..."
	rm -rf outputs/reports/
	rm -rf outputs/tests/

clean-all: clean clean-outputs

build:
	uv build

# Combined checks
check-all: lint format-check type-check test

# Pre-commit hooks
pre-commit-install:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

pre-commit-update:
	uv run pre-commit autoupdate

# Development setup
setup-dev:
	@echo "Setting up development environment..."
	uv pip install -e ".[dev]"
	uv run pre-commit install
	@echo "Development environment ready!"
	@echo "Pre-commit hooks installed! Code quality checks will run automatically on commit."

# Run examples
run-simple-example:
	cd example/simple_benchmark && uv run python simple_example.py

run-benchmark-example:
	cd example/simple_benchmark && uv run python hpopt_benchmark.py -c config_sample.toml

run-neural-demo:
	cd example/neural_network && uv run python mnist_sklearn_bridge.py --demo

run-neural-example:
	cd example/neural_network && uv run python mnist_sklearn_bridge.py --subset-size 500 --n-train 2 --n-test 1

run-neural-pytorch:
	cd example/neural_network && uv run python mnist_model_bridge.py --subset-size 200 --n-train 2 --n-test 1

run-mas-demo:
	cd example/mas_bench && uv run python mas_demo.py

run-mas-example:
	cd example/mas_bench && uv run python simple_mas_example.py

run-demo:
	python example_usage.py
