"""Test utilities and helper functions."""

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import optuna

from modelbridge.types import FloatArray, ParamConfig, ParamDict


class MockObjectiveFunction:
    """Mock objective function for testing."""

    def __init__(self, function_type: str = "quadratic", noise_level: float = 0.0):
        """Initialize mock objective function.
        
        Args:
            function_type: Type of function ("quadratic", "linear", "sine")
            noise_level: Amount of noise to add to function output
        """
        self.function_type = function_type
        self.noise_level = noise_level
        self.call_count = 0

    def __call__(self, params: ParamDict) -> float:
        """Evaluate the mock function."""
        self.call_count += 1

        # Extract x and p parameters
        x_params = [v for k, v in params.items() if k.startswith("x")]
        p_params = [v for k, v in params.items() if k.startswith("p")]

        x = np.array(x_params, dtype=np.float64)
        p = np.array(p_params, dtype=np.float64)

        if self.function_type == "quadratic":
            result = np.sum(p * x**2)
        elif self.function_type == "linear":
            result = np.sum(p * x)
        elif self.function_type == "sine":
            result = np.sum(p * np.sin(x))
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")

        # Add noise if specified
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level)
            result += noise

        return float(result)


class MockMacroObjectiveFunction:
    """Mock macro objective function for testing."""

    def __init__(self, approximation_type: str = "linear"):
        """Initialize mock macro objective function.
        
        Args:
            approximation_type: Type of approximation ("linear", "constant")
        """
        self.approximation_type = approximation_type
        self.call_count = 0

    def __call__(self, params: ParamDict, target_value: float) -> float:
        """Evaluate the mock macro function."""
        self.call_count += 1

        x_params = [v for k, v in params.items() if k.startswith("x")]
        p_params = [v for k, v in params.items() if k.startswith("p")]

        x = np.array(x_params, dtype=np.float64)
        p = np.array(p_params, dtype=np.float64)

        if self.approximation_type == "linear":
            result = np.sum(p * x)
        elif self.approximation_type == "constant":
            result = np.sum(p)
        else:
            raise ValueError(f"Unknown approximation type: {self.approximation_type}")

        return float(result)


def create_test_param_config(n_params: int) -> ParamConfig:
    """Create a test parameter configuration with specified number of parameters."""
    param_names = [f"x_{i}" for i in range(n_params // 2)] + [f"p_{i}" for i in range(n_params // 2)]
    if n_params % 2 == 1:
        param_names.append("extra_param")

    param_types = ["float"] * len(param_names)
    param_ranges = [(-2.0, 2.0)] * (n_params // 2) + [(0.1, 2.0)] * (n_params // 2)
    if n_params % 2 == 1:
        param_ranges.append((0.0, 1.0))

    return create_param_config(param_names, param_types, param_ranges)


def create_test_data(n_samples: int, n_features: int, seed: int = 42) -> FloatArray:
    """Create test data for regression testing."""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features).astype(np.float64)


def assert_valid_metrics(metrics: dict[str, Any]) -> None:
    """Assert that evaluation metrics are valid."""
    required_keys = {"mse", "mae", "r2"}
    assert set(metrics.keys()) >= required_keys

    # MSE and MAE should be non-negative
    assert metrics["mse"] >= 0
    assert metrics["mae"] >= 0

    # All values should be finite
    assert all(np.isfinite(v) for v in metrics.values())


def create_temporary_optuna_storage() -> str:
    """Create temporary storage for Optuna studies."""
    temp_dir = tempfile.mkdtemp()
    return f"sqlite:///{temp_dir}/test_optuna.db"


class StudyMocker:
    """Utility class for mocking Optuna studies."""

    @staticmethod
    def create_mock_study(n_trials: int, param_config: ParamConfig) -> Any:
        """Create a mock study with specified number of trials."""
        study = optuna.create_study(direction="minimize")

        for i in range(n_trials):
            trial = study.ask()

            # Mock parameter suggestions based on config
            for param_name, config in param_config.items():
                if config["type"] == "float":
                    trial.suggest_float(param_name, config["low"], config["high"])
                else:
                    trial.suggest_int(param_name, int(config["low"]), int(config["high"]))

            # Mock objective value
            objective_value = float(i + np.random.random())
            study.tell(trial, objective_value)

        return study


def validate_file_structure(directory: Path, expected_files: list[str]) -> None:
    """Validate that expected files exist in directory."""
    for filename in expected_files:
        file_path = directory / filename
        assert file_path.exists(), f"Expected file {filename} not found in {directory}"
        assert file_path.stat().st_size > 0, f"File {filename} is empty"


class ParameterValidator:
    """Utility for validating parameter structures."""

    @staticmethod
    def validate_param_dict(params: ParamDict, param_config: ParamConfig) -> None:
        """Validate that parameter dictionary conforms to configuration."""
        # Check all required parameters are present
        assert set(params.keys()) == set(param_config.keys())

        # Check parameter bounds
        for param_name, value in params.items():
            config = param_config[param_name]
            assert config["low"] <= value <= config["high"], (
                f"Parameter {param_name} value {value} outside bounds "
                f"[{config['low']}, {config['high']}]"
            )

    @staticmethod
    def validate_param_list(param_list: list[ParamDict], param_config: ParamConfig) -> None:
        """Validate that parameter list conforms to configuration."""
        assert len(param_list) > 0, "Parameter list should not be empty"

        for params in param_list:
            ParameterValidator.validate_param_dict(params, param_config)


class ArrayValidator:
    """Utility for validating numpy arrays."""

    @staticmethod
    def validate_float_array(array: FloatArray, expected_shape: tuple[int, ...] | None = None) -> None:
        """Validate that array is a proper float array."""
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert np.all(np.isfinite(array)), "Array contains non-finite values"

        if expected_shape is not None:
            assert array.shape == expected_shape

    @staticmethod
    def validate_bounds(array: FloatArray, min_val: float, max_val: float) -> None:
        """Validate that array values are within specified bounds."""
        assert np.all(array >= min_val), f"Some values below minimum {min_val}"
        assert np.all(array <= max_val), f"Some values above maximum {max_val}"
