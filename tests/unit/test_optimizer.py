"""Unit tests for OptunaOptimizer."""

import tempfile
from typing import Any

import optuna
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
import pytest

from modelbridge.core.optimizer import OptunaOptimizer
from modelbridge.types import ParamConfig


class TestOptunaOptimizer:
    """Test cases for OptunaOptimizer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        optimizer = OptunaOptimizer()
        assert optimizer.direction == "minimize"
        assert optimizer.storage == "sqlite:///optuna.db"
        assert isinstance(optimizer.sampler, TPESampler)

    def test_init_with_string_sampler(self) -> None:
        """Test initialization with string sampler names."""
        # Test random sampler
        optimizer = OptunaOptimizer(sampler="random", seed=42)
        assert isinstance(optimizer.sampler, RandomSampler)

        # Test TPE sampler
        optimizer = OptunaOptimizer(sampler="tpe", seed=42)
        assert isinstance(optimizer.sampler, TPESampler)

        # Test CMA-ES sampler
        optimizer = OptunaOptimizer(sampler="cmaes", seed=42)
        assert isinstance(optimizer.sampler, CmaEsSampler)

    def test_init_with_custom_sampler(self) -> None:
        """Test initialization with custom sampler object."""
        custom_sampler = RandomSampler(seed=123)
        optimizer = OptunaOptimizer(sampler=custom_sampler)
        assert optimizer.sampler is custom_sampler

    def test_invalid_sampler_name(self) -> None:
        """Test invalid sampler name raises error."""
        with pytest.raises(ValueError, match="Unknown sampler 'invalid'"):
            OptunaOptimizer(sampler="invalid")

    def test_create_study(self, sample_param_config: ParamConfig) -> None:
        """Test study creation and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = f"sqlite:///{temp_dir}/test.db"
            optimizer = OptunaOptimizer(storage=storage, seed=42)

            # Create new study
            study, is_existing = optimizer.create_or_load_study("test_study", load_if_exists=False)
            assert not is_existing
            assert study.study_name == "test_study"

            # Load existing study
            study2, is_existing2 = optimizer.create_or_load_study("test_study", load_if_exists=True)
            assert is_existing2
            assert study2.study_name == "test_study"

    def test_suggest_parameters(self, sample_param_config: ParamConfig) -> None:
        """Test parameter suggestion."""
        optimizer = OptunaOptimizer(seed=42)
        study = optuna.create_study()
        trial = study.ask()

        params = optimizer.suggest_parameters(trial, sample_param_config)

        # Check that all parameters are suggested
        assert set(params.keys()) == set(sample_param_config.keys())

        # Check parameter bounds
        for param_name, value in params.items():
            config = sample_param_config[param_name]
            assert config["low"] <= value <= config["high"]

    def test_suggest_parameters_invalid_type(self) -> None:
        """Test parameter suggestion with invalid type."""
        optimizer = OptunaOptimizer()
        study = optuna.create_study()
        trial = study.ask()

        invalid_config = {
            "param1": {"type": "invalid", "low": 0, "high": 1}
        }

        with pytest.raises(ValueError, match="Unknown parameter type 'invalid'"):
            optimizer.suggest_parameters(trial, invalid_config)

    def test_optimize_batch(self, sample_param_config: ParamConfig) -> None:
        """Test batch optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = f"sqlite:///{temp_dir}/test.db"
            optimizer = OptunaOptimizer(storage=storage, seed=42)

            def objective(trial: Any) -> float:
                params = optimizer.suggest_parameters(trial, sample_param_config)
                # Simple quadratic objective
                return sum(v**2 for v in params.values())

            study = optimizer.optimize_batch(
                objective, "test_optimization", n_trials=5
            )

            assert len(study.trials) == 5
            assert study.best_trial is not None
            assert study.best_value is not None

    def test_get_best_params(self, sample_param_config: ParamConfig) -> None:
        """Test getting best parameters from completed study."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = f"sqlite:///{temp_dir}/test.db"
            optimizer = OptunaOptimizer(storage=storage, seed=42)

            def objective(trial: Any) -> float:
                params = optimizer.suggest_parameters(trial, sample_param_config)
                return sum(v**2 for v in params.values())

            study_name = "test_best_params"
            optimizer.optimize_batch(objective, study_name, n_trials=3)

            best_params = optimizer.get_best_params(study_name)
            assert isinstance(best_params, dict)
            assert set(best_params.keys()) == set(sample_param_config.keys())

    def test_get_all_completed_params(self, sample_param_config: ParamConfig) -> None:
        """Test getting all completed parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = f"sqlite:///{temp_dir}/test.db"
            optimizer = OptunaOptimizer(storage=storage, seed=42)

            def objective(trial: Any) -> float:
                params = optimizer.suggest_parameters(trial, sample_param_config)
                return sum(v**2 for v in params.values())

            study_name = "test_all_params"
            optimizer.optimize_batch(objective, study_name, n_trials=3)

            all_params = optimizer.get_all_completed_params(study_name)
            assert len(all_params) == 3
            assert all(isinstance(params, dict) for params in all_params)
            assert all(
                set(params.keys()) == set(sample_param_config.keys())
                for params in all_params
            )
