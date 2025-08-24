"""Unit tests for regression models."""


import numpy as np
import pytest

from modelbridge.core.regression import (
    LinearRegressionModel,
    PolynomialRegressionModel,
    RegressionModel,
)
from modelbridge.types import FloatArray, ParamList


class TestLinearRegressionModel:
    """Test cases for LinearRegressionModel."""

    def test_init(self) -> None:
        """Test initialization."""
        model = LinearRegressionModel()
        assert model.model is not None

    def test_fit_and_predict(self, sample_float_array: FloatArray) -> None:
        """Test fitting and prediction."""
        model = LinearRegressionModel()

        # Use first 2 columns as input, last 2 as target
        X = sample_float_array[:, :2]
        y = sample_float_array[:, 2:]

        # Fit model
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert isinstance(predictions, np.ndarray)


class TestPolynomialRegressionModel:
    """Test cases for PolynomialRegressionModel."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        model = PolynomialRegressionModel()
        assert model.degree == 2
        assert model.include_bias is False

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        model = PolynomialRegressionModel(degree=3, include_bias=True)
        assert model.degree == 3
        assert model.include_bias is True

    def test_fit_and_predict(self, sample_float_array: FloatArray) -> None:
        """Test fitting and prediction."""
        model = PolynomialRegressionModel(degree=2)

        X = sample_float_array[:, :2]
        y = sample_float_array[:, 2:]

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert isinstance(predictions, np.ndarray)


class TestRegressionModel:
    """Test cases for unified RegressionModel."""

    def test_init_linear(self) -> None:
        """Test initialization with linear model."""
        model = RegressionModel("linear")
        assert model.model_type == "linear"
        assert isinstance(model.model, LinearRegressionModel)
        assert not model._is_fitted

    def test_init_polynomial(self) -> None:
        """Test initialization with polynomial model."""
        model = RegressionModel("polynomial", degree=3)
        assert model.model_type == "polynomial"
        assert isinstance(model.model, PolynomialRegressionModel)
        assert model.model.degree == 3

    def test_init_invalid_type(self) -> None:
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type: invalid"):
            RegressionModel("invalid")

    def test_fit_with_param_list(
        self,
        sample_param_list: ParamList,
        sample_float_array: FloatArray
    ) -> None:
        """Test fitting with parameter list input."""
        model = RegressionModel("linear")

        # Use param list as macro params, array as micro params
        micro_params = sample_float_array[:, 2:].tolist()
        micro_param_list = [
            {"p_1": row[0], "p_2": row[1]}
            for row in micro_params
        ]

        model.fit(
            sample_param_list,
            micro_param_list,
            macro_param_names=["x_1", "x_2", "p_1", "p_2"],
            micro_param_names=["p_1", "p_2"]
        )

        assert model._is_fitted

    def test_fit_with_arrays(self, sample_float_array: FloatArray) -> None:
        """Test fitting with numpy array input."""
        model = RegressionModel("linear")

        X = sample_float_array[:, :2]
        y = sample_float_array[:, 2:]

        model.fit(X, y)
        assert model._is_fitted

    def test_predict_before_fit(self, sample_float_array: FloatArray) -> None:
        """Test prediction before fitting raises error."""
        model = RegressionModel("linear")

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model.predict(sample_float_array[:, :2])

    def test_predict_after_fit(self, sample_float_array: FloatArray) -> None:
        """Test prediction after fitting."""
        model = RegressionModel("linear")

        X = sample_float_array[:, :2]
        y = sample_float_array[:, 2:]

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert isinstance(predictions, np.ndarray)

    def test_evaluate(self, sample_float_array: FloatArray) -> None:
        """Test evaluation metrics calculation."""
        model = RegressionModel("linear")

        true_values = sample_float_array[:, :2]
        pred_values = sample_float_array[:, :2] + 0.1  # Add small error

        metrics = model.evaluate(true_values, pred_values)

        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_convert_to_array_with_param_list(self, sample_param_list: ParamList) -> None:
        """Test conversion from parameter list to array."""
        model = RegressionModel("linear")

        param_names = ["x_1", "x_2", "p_1", "p_2"]
        array = model._convert_to_array(sample_param_list, param_names)

        assert array.shape == (3, 4)
        assert isinstance(array, np.ndarray)

        # Check values are correctly ordered
        expected_first_row = [1.0, 2.0, 0.5, 1.5]
        np.testing.assert_array_almost_equal(array[0], expected_first_row)

    def test_convert_to_array_with_numpy(self, sample_float_array: FloatArray) -> None:
        """Test conversion with numpy array input (should return as-is)."""
        model = RegressionModel("linear")

        result = model._convert_to_array(sample_float_array)
        assert result is sample_float_array

    def test_convert_to_array_no_param_names_empty_list(self) -> None:
        """Test conversion with empty parameter list."""
        model = RegressionModel("linear")

        with pytest.raises(ValueError, match="param_names must be provided"):
            model._convert_to_array([], None)


@pytest.mark.skipif(
    condition=True,  # Skip GPy tests by default since it's optional
    reason="GPy is optional dependency"
)
class TestGaussianProcessModel:
    """Test cases for GaussianProcessModel (requires GPy)."""

    def test_init_without_gpy(self) -> None:
        """Test initialization without GPy raises ImportError."""
        # This would require mocking GPY_AVAILABLE
        pass
