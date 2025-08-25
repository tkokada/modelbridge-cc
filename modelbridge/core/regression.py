"""Regression models for model bridging."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, override

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..types import (
    EvaluationMetrics,
    FloatArray,
    ParamList,
    RegressionModelType,
)

try:
    import GPy

    GPY_AVAILABLE = True
except ImportError:
    GPY_AVAILABLE = False


class BaseRegression(ABC):
    """Abstract base class for regression models."""

    @abstractmethod
    def fit(self, input_data: FloatArray, target_data: FloatArray) -> None:
        """Fit the regression model."""
        pass

    @abstractmethod
    def predict(self, input_data: FloatArray) -> FloatArray:
        """Make predictions."""
        pass


class LinearRegressionModel(BaseRegression):
    """Linear regression model."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    @override
    def fit(self, input_data: FloatArray, target_data: FloatArray) -> None:
        """Fit linear regression model."""
        self.model.fit(input_data, target_data)

    @override
    def predict(self, input_data: FloatArray) -> FloatArray:
        """Predict using linear regression."""
        return self.model.predict(input_data).astype(np.float64)  # type: ignore[no-any-return]


class PolynomialRegressionModel(BaseRegression):
    """Polynomial regression model."""

    def __init__(self, degree: int = 2, include_bias: bool = False) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=include_bias),
            LinearRegression(),
        )

    @override
    def fit(self, input_data: FloatArray, target_data: FloatArray) -> None:
        """Fit polynomial regression model."""
        self.model.fit(input_data, target_data)

    @override
    def predict(self, input_data: FloatArray) -> FloatArray:
        """Predict using polynomial regression."""
        return self.model.predict(input_data).astype(np.float64)  # type: ignore[no-any-return]


class GaussianProcessModel(BaseRegression):
    """Gaussian Process regression model."""

    def __init__(
        self,
        kernel: Any | None = None,
        variance: float = 1.0,
        lengthscale: float = 1.0,
    ) -> None:
        if not GPY_AVAILABLE:
            raise ImportError("GPy is required for Gaussian Process regression")

        self.variance = variance
        self.lengthscale = lengthscale
        self.kernel = kernel
        self.model: Any | None = None

    @override
    def fit(self, input_data: FloatArray, target_data: FloatArray) -> None:
        """Fit Gaussian Process model."""
        input_dim = input_data.shape[1]

        if self.kernel is None:
            kernel = GPy.kern.RBF(
                input_dim=input_dim,
                variance=self.variance,
                lengthscale=self.lengthscale,
            )
        else:
            kernel = self.kernel

        self.model = GPy.models.GPRegression(input_data, target_data, kernel)
        self.model.optimize()

    @override
    def predict(self, input_data: FloatArray) -> FloatArray:
        """Predict using Gaussian Process."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        predictions, _ = self.model.predict(input_data)
        return predictions.astype(np.float64)  # type: ignore[no-any-return]


class RegressionModel:
    """Unified regression model interface."""

    SUPPORTED_MODELS: ClassVar[dict[RegressionModelType, type[BaseRegression]]] = {
        "linear": LinearRegressionModel,
        "polynomial": PolynomialRegressionModel,
        "gaussian_process": GaussianProcessModel,
    }

    def __init__(self, model_type: RegressionModelType, **kwargs: Any):
        """Initialize regression model.

        Args:
            model_type: Type of regression model
            **kwargs: Additional arguments for the specific model

        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.SUPPORTED_MODELS.keys())}"
            )

        if model_type == "gaussian_process" and not GPY_AVAILABLE:
            raise ImportError("GPy is required for Gaussian Process regression")

        self.model_type = model_type
        self.model = self.SUPPORTED_MODELS[model_type](**kwargs)
        self._is_fitted = False

    def fit(
        self,
        macro_params: ParamList | FloatArray,
        micro_params: ParamList | FloatArray,
        macro_param_names: list[str] | None = None,
        micro_param_names: list[str] | None = None,
    ) -> None:
        """Fit regression model to map macro to micro parameters.

        Trains the regression model to learn the mapping from macro model parameters
        to micro model parameters using the provided training data.

        Args:
            macro_params (ParamList | FloatArray): Macro model parameters as input data.
                Can be list of parameter dictionaries or numpy array.
            micro_params (ParamList | FloatArray): Micro model parameters as target data.
                Can be list of parameter dictionaries or numpy array.
            macro_param_names (list[str] | None, optional): Names of macro parameters in
                specific order when input is list of dictionaries. Defaults to None.
            micro_param_names (list[str] | None, optional): Names of micro parameters in
                specific order when input is list of dictionaries. Defaults to None.

        Raises:
            ValueError: If parameter names are required but not provided, or if input
                data shapes are incompatible.

        """
        input_data = self._convert_to_array(macro_params, macro_param_names)
        target_data = self._convert_to_array(micro_params, micro_param_names)

        self.model.fit(input_data, target_data)
        self._is_fitted = True

    def predict(
        self,
        macro_params: ParamList | FloatArray,
        macro_param_names: list[str] | None = None,
    ) -> FloatArray:
        """Predict micro parameters from macro parameters.

        Args:
            macro_params: Macro model parameters to predict from
            macro_param_names: Names of macro parameters (for dict input)

        Returns:
            Predicted micro parameters

        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet")

        input_data = self._convert_to_array(macro_params, macro_param_names)
        return self.model.predict(input_data)

    def evaluate(
        self, true_micro: FloatArray, pred_micro: FloatArray
    ) -> EvaluationMetrics:
        """Evaluate prediction performance.

        Args:
            true_micro: True micro parameters
            pred_micro: Predicted micro parameters

        Returns:
            Dictionary of evaluation metrics

        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r2 = r2_score(true_micro, pred_micro)
            except ValueError:
                # Handle case where R² cannot be calculated
                r2 = 0.0

        return EvaluationMetrics(
            mse=mean_squared_error(true_micro, pred_micro),
            mae=mean_absolute_error(true_micro, pred_micro),
            r2=r2,
        )

    def _convert_to_array(
        self, params: ParamList | FloatArray, param_names: list[str] | None = None
    ) -> FloatArray:
        """Convert parameter input to numpy array."""
        if isinstance(params, np.ndarray):
            return params

        if param_names is None:
            if len(params) > 0:
                param_names = sorted(params[0].keys())
            else:
                raise ValueError("param_names must be provided for dict input")

        return np.array(
            [[param_dict[name] for name in param_names] for param_dict in params]
        )
