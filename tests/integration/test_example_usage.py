"""Integration tests for example usage scenarios."""

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pytest

from modelbridge import ModelBridge, ParamConfig, ParamDict
from modelbridge.utils.config_loader import create_param_config


class TestExampleUsageScenarios:
    """Test real-world usage scenarios similar to examples."""

    @pytest.fixture
    def benchmark_functions(self) -> dict[str, Any]:
        """Benchmark functions for testing."""

        def sphere(params: ParamDict) -> float:
            """Sphere function."""
            x = np.array([params["x_0"], params["x_1"]])
            p = np.array([params["p_0"], params["p_1"]])
            return float(np.sum(p * x**2))

        def rastrigin(params: ParamDict) -> float:
            """Rastrigin function."""
            x = np.array([params["x_0"], params["x_1"]])
            p = np.array([params["p_0"], params["p_1"]])
            return float(10 * len(x) + np.sum(p * x**2 - 10.0 * np.cos(2 * np.pi * x)))

        def linear_macro(params: ParamDict, target_value: float) -> float:
            """Linear macro approximation."""
            x = np.array([params["x_0"], params["x_1"]])
            p = np.array([params["p_0"], params["p_1"]])
            return float(np.sum(p * x))

        return {
            "sphere": sphere,
            "rastrigin": rastrigin,
            "linear_macro": linear_macro,
        }

    @pytest.fixture
    def benchmark_param_config(self) -> ParamConfig:
        """Parameter configuration for benchmark functions."""
        param_names = ["x_0", "x_1", "p_0", "p_1"]
        param_types = ["float", "float", "float", "float"]
        param_ranges = [(-2.0, 2.0), (-2.0, 2.0), (0.1, 1.0), (0.1, 1.0)]

        return create_param_config(param_names, param_types, param_ranges)

    def test_sphere_to_linear_bridge(
        self,
        benchmark_functions: dict[str, Any],
        benchmark_param_config: ParamConfig,
    ) -> None:
        """Test bridging from sphere to linear function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bridge = ModelBridge(
                micro_objective=benchmark_functions["sphere"],
                macro_objective=benchmark_functions["linear_macro"],
                micro_param_config=benchmark_param_config,
                macro_param_config=benchmark_param_config,
                regression_type="polynomial",
                optimizer_config={
                    "storage": f"sqlite:///{temp_dir}/sphere_linear.db",
                    "direction": "minimize",
                    "sampler": "tpe",
                    "seed": 42,
                },
                regression_config={"degree": 2},
            )

            metrics = bridge.run_full_pipeline(
                n_train=3,
                n_test=2,
                micro_trials_per_dataset=15,
                macro_trials_per_dataset=15,
                visualize=True,
                output_dir=temp_dir,
            )

            # Verify results
            assert isinstance(metrics, dict)
            assert "mse" in metrics
            assert "mae" in metrics
            assert "r2" in metrics

            # Check that files were created
            output_path = Path(temp_dir)
            assert (output_path / "parameter_relationships.png").exists()
            assert (output_path / "prediction_accuracy.png").exists()

    def test_rastrigin_optimization(
        self,
        benchmark_functions: dict[str, Any],
        benchmark_param_config: ParamConfig,
    ) -> None:
        """Test optimization with more complex Rastrigin function."""
        bridge = ModelBridge(
            micro_objective=benchmark_functions["rastrigin"],
            macro_objective=benchmark_functions["linear_macro"],
            micro_param_config=benchmark_param_config,
            macro_param_config=benchmark_param_config,
            regression_type="linear",
            optimizer_config={
                "storage": "sqlite:///rastrigin_test.db",
                "direction": "minimize",
                "sampler": "random",
                "seed": 123,
            },
        )

        metrics = bridge.run_full_pipeline(
            n_train=2,
            n_test=1,
            micro_trials_per_dataset=10,
            macro_trials_per_dataset=10,
            visualize=False,
        )

        assert isinstance(metrics, dict)
        assert bridge._is_trained

    @pytest.mark.parametrize("n_train,n_test", [(2, 1), (3, 2), (1, 1)])
    def test_different_train_test_splits(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
        n_train: int,
        n_test: int,
    ) -> None:
        """Test pipeline with different train/test splits."""
        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="linear",
            optimizer_config={"seed": 42},
        )

        metrics = bridge.run_full_pipeline(
            n_train=n_train,
            n_test=n_test,
            micro_trials_per_dataset=5,
            macro_trials_per_dataset=5,
            visualize=False,
        )

        assert len(bridge.train_micro_params) == n_train
        assert len(bridge.train_macro_params) == n_train
        assert len(bridge.test_micro_params) == n_test
        assert len(bridge.test_macro_params) == n_test

    def test_zero_test_datasets(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test pipeline with zero test datasets."""
        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="linear",
            optimizer_config={"seed": 42},
        )

        metrics = bridge.run_full_pipeline(
            n_train=2,
            n_test=0,
            micro_trials_per_dataset=5,
            macro_trials_per_dataset=5,
            visualize=False,
        )

        # Should return zero metrics when no test data
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 0.0

    def test_high_dimensional_parameters(self) -> None:
        """Test with higher dimensional parameter space."""
        # Create high-dimensional parameter config
        param_names = [f"x_{i}" for i in range(5)] + [f"p_{i}" for i in range(5)]
        param_types = ["float"] * 10
        param_ranges = [(-1.0, 1.0)] * 5 + [(0.1, 1.0)] * 5

        param_config = create_param_config(param_names, param_types, param_ranges)

        def high_dim_micro(params: ParamDict) -> float:
            x = np.array([params[f"x_{i}"] for i in range(5)])
            p = np.array([params[f"p_{i}"] for i in range(5)])
            return float(np.sum(p * x**2))

        def high_dim_macro(params: ParamDict, target_value: float) -> float:
            x = np.array([params[f"x_{i}"] for i in range(5)])
            p = np.array([params[f"p_{i}"] for i in range(5)])
            return float(np.sum(p * x))

        bridge = ModelBridge(
            micro_objective=high_dim_micro,
            macro_objective=high_dim_macro,
            micro_param_config=param_config,
            macro_param_config=param_config,
            regression_type="linear",
            optimizer_config={"seed": 42},
        )

        metrics = bridge.run_full_pipeline(
            n_train=2,
            n_test=1,
            micro_trials_per_dataset=8,
            macro_trials_per_dataset=8,
            visualize=False,
        )

        assert isinstance(metrics, dict)
        assert bridge.predicted_micro_params is not None
        assert bridge.predicted_micro_params.shape[1] == 10  # 10 parameters

    @pytest.mark.slow
    def test_performance_benchmark(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Performance benchmark test (marked as slow)."""
        import time

        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="linear",
            optimizer_config={"seed": 42},
        )

        start_time = time.time()

        metrics = bridge.run_full_pipeline(
            n_train=5,
            n_test=3,
            micro_trials_per_dataset=20,
            macro_trials_per_dataset=20,
            visualize=False,
        )

        elapsed_time = time.time() - start_time

        # Basic performance check (should complete within reasonable time)
        assert elapsed_time < 60.0  # Should complete within 1 minute
        assert isinstance(metrics, dict)
