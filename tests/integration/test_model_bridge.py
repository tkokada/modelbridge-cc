"""Integration tests for ModelBridge complete pipeline."""

from pathlib import Path
import tempfile
from typing import Any

import pytest

from modelbridge import ModelBridge
from modelbridge.types import (
    ParamConfig,
    ParamDict,
)


class TestModelBridgeIntegration:
    """Integration tests for the complete ModelBridge pipeline."""

    @pytest.fixture
    def simple_micro_objective(self) -> Any:
        """Simple quadratic micro objective for testing."""

        def objective(params: ParamDict) -> float:
            x = params["x"]
            p = params["p"]
            return float(p * x**2)

        return objective

    @pytest.fixture
    def simple_macro_objective(self) -> Any:
        """Simple linear macro objective for testing."""

        def objective(params: ParamDict, target_value: float) -> float:
            x = params["x"]
            p = params["p"]
            return float(p * x)

        return objective

    @pytest.fixture
    def simple_param_config(self) -> ParamConfig:
        """Simple parameter configuration for testing."""
        return {
            "x": {"type": "float", "low": -2.0, "high": 2.0},
            "p": {"type": "float", "low": 0.1, "high": 2.0},
        }

    def test_full_pipeline_linear_regression(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test complete pipeline with linear regression."""
        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="linear",
            optimizer_config={
                "storage": "sqlite:///test_linear.db",
                "direction": "minimize",
                "sampler": "random",
                "seed": 42,
            },
        )

        metrics = bridge.run_full_pipeline(
            n_train=3,
            n_test=2,
            micro_trials_per_dataset=10,
            macro_trials_per_dataset=10,
            visualize=False,
        )

        # Check that we got valid metrics
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

        # Check that training was completed
        assert bridge._is_trained
        assert len(bridge.train_micro_params) == 3
        assert len(bridge.train_macro_params) == 3
        assert len(bridge.test_micro_params) == 2
        assert len(bridge.test_macro_params) == 2
        assert bridge.predicted_micro_params is not None

    def test_full_pipeline_polynomial_regression(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test complete pipeline with polynomial regression."""
        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="polynomial",
            optimizer_config={
                "storage": "sqlite:///test_poly.db",
                "direction": "minimize",
                "sampler": "tpe",
                "seed": 42,
            },
            regression_config={"degree": 2},
        )

        metrics = bridge.run_full_pipeline(
            n_train=2,
            n_test=2,
            micro_trials_per_dataset=5,
            macro_trials_per_dataset=5,
            visualize=False,
        )

        assert isinstance(metrics, dict)
        assert bridge._is_trained

    def test_pipeline_with_visualization(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test pipeline with visualization enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bridge = ModelBridge(
                micro_objective=simple_micro_objective,
                macro_objective=simple_macro_objective,
                micro_param_config=simple_param_config,
                macro_param_config=simple_param_config,
                regression_type="linear",
                optimizer_config={
                    "storage": "sqlite:///test_viz.db",
                    "seed": 42,
                },
            )

            _ = bridge.run_full_pipeline(
                n_train=3,
                n_test=2,
                micro_trials_per_dataset=5,
                macro_trials_per_dataset=5,
                visualize=True,
                output_dir=temp_dir,
            )

            # Check plots were created
            output_path = Path(temp_dir)
            assert (output_path / "parameter_relationships.png").exists()
            assert (output_path / "prediction_accuracy.png").exists()

            # Check CSV files were created
            assert (output_path / "train_macro_params.csv").exists()
            assert (output_path / "train_micro_params.csv").exists()
            assert (output_path / "test_macro_params.csv").exists()
            assert (output_path / "test_micro_params.csv").exists()
            assert (output_path / "predicted_micro_params.csv").exists()

    def test_train_phase_only(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test training phase only."""
        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="linear",
            optimizer_config={
                "storage": "sqlite:///test_train.db",
                "seed": 42,
            },
        )

        bridge.train_phase(
            n_train=2,
            micro_trials_per_dataset=5,
            macro_trials_per_dataset=5,
        )

        assert bridge._is_trained
        assert len(bridge.train_micro_params) == 2
        assert len(bridge.train_macro_params) == 2
        assert len(bridge.test_micro_params) == 0
        assert len(bridge.test_macro_params) == 0

    def test_test_phase_before_training_fails(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test that test phase fails if training hasn't been completed."""
        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="linear",
        )

        with pytest.raises(
            ValueError, match="Must complete training phase before testing"
        ):
            bridge.test_phase(n_test=1)

    def test_visualize_before_training_fails(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test that visualization fails if training/testing hasn't been completed."""
        bridge = ModelBridge(
            micro_objective=simple_micro_objective,
            macro_objective=simple_macro_objective,
            micro_param_config=simple_param_config,
            macro_param_config=simple_param_config,
            regression_type="linear",
        )

        with pytest.raises(
            ValueError, match="Must complete training and testing before visualization"
        ):
            bridge.visualize_results()

    @pytest.mark.parametrize("regression_type", ["linear", "polynomial"])
    def test_different_regression_types(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
        regression_type: str,
    ) -> None:
        """Test pipeline with different regression types."""
        regression_config = {"degree": 2} if regression_type == "polynomial" else {}

        import tempfile
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = f"sqlite:///{temp_dir}/test_regtype_{regression_type}_{int(time.time()*1000)}.db"

            bridge = ModelBridge(
                micro_objective=simple_micro_objective,
                macro_objective=simple_macro_objective,
                micro_param_config=simple_param_config,
                macro_param_config=simple_param_config,
                regression_type=regression_type,  # type: ignore[arg-type]
                optimizer_config={"seed": 42, "storage": storage_path},
                regression_config=regression_config,
            )

            metrics = bridge.run_full_pipeline(
                n_train=2,
                n_test=1,
                micro_trials_per_dataset=5,
                macro_trials_per_dataset=5,
                visualize=False,
            )

            assert isinstance(metrics, dict)
            assert bridge._is_trained

    def test_invalid_regression_type(
        self,
        simple_micro_objective: Any,
        simple_macro_objective: Any,
        simple_param_config: ParamConfig,
    ) -> None:
        """Test invalid regression type raises error."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelBridge(
                micro_objective=simple_micro_objective,
                macro_objective=simple_macro_objective,
                micro_param_config=simple_param_config,
                macro_param_config=simple_param_config,
                regression_type="invalid",  # type: ignore[arg-type]
            )
