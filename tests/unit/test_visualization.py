"""Unit tests for Visualizer."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from modelbridge.types import FloatArray
from modelbridge.utils.visualization import Visualizer


class TestVisualizer:
    """Test cases for Visualizer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        viz = Visualizer()
        assert viz.figsize == (12, 8)
        assert viz.dpi == 100

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        viz = Visualizer(figsize=(10, 6), dpi=150)
        assert viz.figsize == (10, 6)
        assert viz.dpi == 150

    @pytest.mark.parametrize("save_to_file", [True, False])
    def test_plot_parameter_relationship(
        self,
        sample_float_array: FloatArray,
        temp_directory: Path,
        save_to_file: bool
    ) -> None:
        """Test parameter relationship plotting."""
        viz = Visualizer()

        # Split array into macro and micro parameters
        macro_params = sample_float_array[:, :2]
        micro_params = sample_float_array[:, 2:]

        macro_names = ["macro_1", "macro_2"]
        micro_names = ["micro_1", "micro_2"]

        output_dir = str(temp_directory) if save_to_file else None

        with patch('matplotlib.pyplot.show') as mock_show:
            viz.plot_parameter_relationship(
                macro_params,
                micro_params,
                macro_names,
                micro_names,
                title="Test Relationships",
                output_dir=output_dir,
                filename="test_relationships.png"
            )

        if save_to_file:
            # Check file was created
            expected_file = temp_directory / "test_relationships.png"
            assert expected_file.exists()
            mock_show.assert_not_called()
        else:
            # Check show was called when not saving
            mock_show.assert_called_once()

    @pytest.mark.parametrize("save_to_file", [True, False])
    def test_plot_prediction_accuracy(
        self,
        sample_float_array: FloatArray,
        temp_directory: Path,
        save_to_file: bool
    ) -> None:
        """Test prediction accuracy plotting."""
        viz = Visualizer()

        # Create true and predicted values (add small noise to predictions)
        true_values = sample_float_array[:, :2]
        predicted_values = true_values + np.random.normal(0, 0.1, true_values.shape)

        param_names = ["param_1", "param_2"]
        output_dir = str(temp_directory) if save_to_file else None

        with patch('matplotlib.pyplot.show') as mock_show:
            viz.plot_prediction_accuracy(
                true_values,
                predicted_values,
                param_names,
                title="Test Accuracy",
                output_dir=output_dir,
                filename="test_accuracy.png"
            )

        if save_to_file:
            expected_file = temp_directory / "test_accuracy.png"
            assert expected_file.exists()
            mock_show.assert_not_called()
        else:
            mock_show.assert_called_once()

    def test_plot_optimization_history(self, temp_directory: Path) -> None:
        """Test optimization history plotting."""
        viz = Visualizer()

        # Create mock study with trials
        mock_study = Mock()
        mock_study.direction.name = "MINIMIZE"

        # Create mock trials
        mock_trials = []
        for i in range(5):
            trial = Mock()
            trial.number = i
            trial.value = 10.0 - i  # Decreasing values
            mock_trials.append(trial)

        mock_study.trials = mock_trials

        with patch('matplotlib.pyplot.show'):
            viz.plot_optimization_history(
                mock_study,
                title="Test History",
                output_dir=str(temp_directory),
                filename="test_history.png"
            )

        expected_file = temp_directory / "test_history.png"
        assert expected_file.exists()

    def test_plot_regression_model_performance(
        self,
        sample_float_array: FloatArray,
        temp_directory: Path
    ) -> None:
        """Test regression model performance plotting."""
        viz = Visualizer()

        # Create mock regression model
        mock_model = Mock()
        mock_model.predict.return_value = sample_float_array + 0.1

        train_input = sample_float_array[:, :2]
        train_target = sample_float_array[:, 2:]
        test_input = sample_float_array[:, :2]
        test_target = sample_float_array[:, 2:]

        param_names = ["param_1", "param_2"]

        with patch('matplotlib.pyplot.show'):
            viz.plot_regression_model_performance(
                mock_model,
                train_input,
                train_target,
                test_input,
                test_target,
                param_names,
                title="Test Performance",
                output_dir=str(temp_directory),
                filename="test_performance.png"
            )

        expected_file = temp_directory / "test_performance.png"
        assert expected_file.exists()

        # Verify model.predict was called
        assert mock_model.predict.call_count == 2

    def test_plot_single_parameter(self, temp_directory: Path) -> None:
        """Test plotting with single parameter."""
        viz = Visualizer()

        # Single parameter data
        macro_params = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        micro_params = np.array([[0.5], [1.0], [1.5]], dtype=np.float64)

        with patch('matplotlib.pyplot.show'):
            viz.plot_parameter_relationship(
                macro_params,
                micro_params,
                ["macro_1"],
                ["micro_1"],
                output_dir=str(temp_directory)
            )

        expected_file = temp_directory / "parameter_relationships.png"
        assert expected_file.exists()

    def test_plot_creates_directories(self, temp_directory: Path) -> None:
        """Test that plotting creates necessary directories."""
        viz = Visualizer()

        # Use nested directory that doesn't exist
        nested_dir = temp_directory / "nested" / "plots"

        macro_params = np.array([[1.0], [2.0]], dtype=np.float64)
        micro_params = np.array([[0.5], [1.0]], dtype=np.float64)

        with patch('matplotlib.pyplot.show'):
            viz.plot_parameter_relationship(
                macro_params,
                micro_params,
                ["macro_1"],
                ["micro_1"],
                output_dir=str(nested_dir),
                filename="test.png"
            )

        expected_file = nested_dir / "test.png"
        assert expected_file.exists()
        assert nested_dir.exists()
