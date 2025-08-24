"""Pytest configuration and fixtures for ModelBridge tests."""

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pytest

from modelbridge.types import (
    EvaluationMetrics,
    FloatArray,
    ParamConfig,
    ParamConfigValue,
    ParamDict,
    ParamList,
)


@pytest.fixture
def sample_param_config() -> ParamConfig:
    """Sample parameter configuration for testing."""
    return {
        "x_1": ParamConfigValue(type="float", low=-5.0, high=5.0),
        "x_2": ParamConfigValue(type="float", low=-5.0, high=5.0),
        "p_1": ParamConfigValue(type="float", low=0.1, high=2.0),
        "p_2": ParamConfigValue(type="float", low=0.1, high=2.0),
    }


@pytest.fixture
def sample_param_dict() -> ParamDict:
    """Sample parameter dictionary for testing."""
    return {
        "x_1": 1.5,
        "x_2": -0.5,
        "p_1": 1.0,
        "p_2": 0.8,
    }


@pytest.fixture
def sample_param_list() -> ParamList:
    """Sample parameter list for testing."""
    return [
        {"x_1": 1.0, "x_2": 2.0, "p_1": 0.5, "p_2": 1.5},
        {"x_1": -1.0, "x_2": 1.5, "p_1": 0.8, "p_2": 1.2},
        {"x_1": 0.5, "x_2": -2.0, "p_1": 1.0, "p_2": 0.6},
    ]


@pytest.fixture
def sample_float_array() -> FloatArray:
    """Sample float array for testing."""
    return np.array(
        [
            [1.0, 2.0, 0.5, 1.5],
            [-1.0, 1.5, 0.8, 1.2],
            [0.5, -2.0, 1.0, 0.6],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_evaluation_metrics() -> EvaluationMetrics:
    """Sample evaluation metrics for testing."""
    return EvaluationMetrics(mse=0.1, mae=0.05, r2=0.95)


@pytest.fixture
def temp_directory() -> Path:
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_config_file(temp_directory: Path) -> Path:
    """Temporary TOML configuration file for testing."""
    config_content = """
[generic]
debug = false
seed = 42

[dataset]
max_x = 5.0
min_x = -5.0
max_x_dim = 2
nx_train = 5
nx_test = 3
n_train = 3
n_test = 2

[micro_model]
micro_function_name = "sphere"
micro_max_param = 1.0
micro_min_param = 0.0
micro_param_prefix = "b"

[macro_model]
macro_function_name = "sphere"
macro_max_param = 1.0
macro_min_param = 0.0
macro_param_prefix = "a"

[optuna]
storage = "sqlite:///test.db"
direction = "minimize"
sampler_name = "tpe"

[regression]
regression_model_name = "polynomial"
"""
    config_file = temp_directory / "test_config.toml"
    config_file.write_text(config_content)
    return config_file


def sphere_function(params: ParamDict) -> float:
    """Simple sphere function for testing objective functions."""
    x = np.array([params["x_1"], params["x_2"]])
    p = np.array([params["p_1"], params["p_2"]])
    return float(np.sum(p * x**2))


def linear_function(params: ParamDict, target_value: float) -> float:
    """Simple linear function for testing macro objectives."""
    x = np.array([params["x_1"], params["x_2"]])
    p = np.array([params["p_1"], params["p_2"]])
    return float(np.sum(p * x))


@pytest.fixture
def micro_objective() -> Any:
    """Sample micro objective function."""
    return sphere_function


@pytest.fixture
def macro_objective() -> Any:
    """Sample macro objective function."""
    return linear_function


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def simple_micro_objective() -> Any:
    """Simple quadratic micro objective for testing."""

    def objective(params: ParamDict) -> float:
        x = params["x"]
        p = params["p"]
        return float(p * x**2)

    return objective


@pytest.fixture
def simple_macro_objective() -> Any:
    """Simple linear macro objective for testing."""

    def objective(params: ParamDict, target_value: float) -> float:
        x = params["x"]
        p = params["p"]
        return float(p * x)

    return objective


@pytest.fixture
def simple_param_config() -> ParamConfig:
    """Simple parameter configuration for testing."""
    return {
        "x": {"type": "float", "low": -2.0, "high": 2.0},
        "p": {"type": "float", "low": 0.1, "high": 2.0},
    }


@pytest.fixture(autouse=True)
def set_random_seed(random_seed: int) -> None:
    """Set random seed for all tests."""
    np.random.seed(random_seed)
