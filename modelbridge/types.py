"""Type definitions for the ModelBridge library using PEP 695 and modern typing."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Self, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import optuna

# Basic type aliases
type ParamValue = float | int | str | bool
type ParamDict = dict[str, ParamValue]
type ParamList = list[ParamDict]

# Specialized NumPy array types
type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.integer[Any]]
type NumPyArray = FloatArray  # Default to float64


# Configuration types with TypedDict for structure validation
class ParamConfigValue(TypedDict):
    """Typed structure for parameter configuration values."""

    type: Literal["float", "int"]
    low: float
    high: float


class StudyConfigDict(TypedDict, total=False):
    """Typed structure for optimization study configuration."""

    storage: str
    direction: Literal["minimize", "maximize"]
    sampler: str
    seed: int


class PlotConfigDict(TypedDict, total=False):
    """Typed structure for plot configuration."""

    figsize: tuple[int, int]
    dpi: int
    output_dir: str
    filename: str


# Configuration type aliases
type ConfigDict = dict[str, Any]
type ValidationRules = dict[str, type[Any]]
type ScalingConfig = dict[str, dict[str, float]]
type ParamConfig = dict[str, ParamConfigValue]


# Evaluation metrics with precise structure
class EvaluationMetrics(TypedDict):
    """Typed structure for evaluation metrics."""

    mse: float
    mae: float
    r2: float


# Function type aliases
type ObjectiveFunction = Callable[[ParamDict], float]
type MacroObjectiveFunction = Callable[[ParamDict, float], float]
type OptimizationStudy = tuple[optuna.Study, bool]

# Literal types for better IDE support and validation
type RegressionModelType = Literal["linear", "polynomial", "gaussian_process"]
type OptimizerSamplerType = Literal["random", "tpe", "cmaes"]
type OptimizationDirection = Literal["minimize", "maximize"]

# Generic type variables with proper constraints
T = TypeVar("T")
P = TypeVar("P", bound=ParamDict)
ArrayT = TypeVar("ArrayT", bound=FloatArray)


# Generic Protocol classes using PEP 695 features
class ObjectiveFunctionProtocol(Protocol):
    """Protocol for objective functions."""

    def __call__(self, params: ParamDict) -> float:
        """Call the objective function with parameters."""
        ...


class MacroObjectiveFunctionProtocol(Protocol):
    """Protocol for macro objective functions."""

    def __call__(self, params: ParamDict, target_value: float) -> float:
        """Call the macro objective function with parameters and target."""
        ...


class RegressionModelProtocol(Protocol):
    """Protocol for regression models."""

    def fit(self, input_data: FloatArray, target_data: FloatArray) -> None:
        """Fit the regression model."""
        ...

    def predict(self, input_data: FloatArray) -> FloatArray:
        """Make predictions with the regression model."""
        ...


class ConfigurableProtocol(Protocol):
    """Protocol for configurable objects with Self return type."""

    def configure(self, config: ConfigDict) -> Self:
        """Configure the object with a configuration dictionary."""
        ...


class VisualizableProtocol(Protocol):
    """Protocol for objects that can be visualized."""

    def visualize(self, output_dir: str | None = None) -> None:
        """Create visualizations of the object."""
        ...


class OptimizableProtocol(Protocol):
    """Protocol for optimizable functions."""

    def __call__(self, params: ParamDict) -> float:
        """Evaluate the function with given parameters."""
        ...

    def get_param_bounds(self) -> ParamConfig:
        """Get parameter bounds for optimization."""
        ...


# Specialized array types for different use cases
type ParameterArray = FloatArray
type FeatureArray = FloatArray
type TargetArray = FloatArray
type PredictionArray = FloatArray

# File path types
type FilePath = str | Path


# Enhanced plotting configuration with TypedDict
class PlotStyle(TypedDict, total=False):
    """Plot styling configuration."""

    color: str | tuple[float, ...]
    marker: str
    alpha: float
    linewidth: float


type ColorType = str | tuple[float, ...]
type MarkerType = str

# Study configuration
type StudyConfig = StudyConfigDict

# Result types with generic constraints
type OptimizationResult = dict[str, ParamValue | EvaluationMetrics]
type TrainingResult = dict[str, ParamList | FloatArray | EvaluationMetrics]

# Union types for flexible input
type FlexibleParamInput = ParamDict | ParamList | FloatArray
type FlexibleArrayInput = FloatArray | Sequence[Sequence[float]]

# Enhanced callback function types
type ProgressCallback = Callable[[int, int], None]
type ValidationCallback = Callable[[ParamDict], bool]
type TransformCallback = Callable[[FloatArray], FloatArray]
type ErrorHandler = Callable[[Exception], None]
type RetryStrategy = Callable[[int], bool]


# Advanced type constraints for machine learning workflows
class ModelMetrics(TypedDict):
    """Comprehensive model evaluation metrics."""

    mse: float
    mae: float
    r2: float
    rmse: float
    mape: float
    training_time: float
    prediction_time: float


class OptimizationMetrics(TypedDict):
    """Optimization process metrics."""

    n_trials: int
    best_value: float
    convergence_iter: int
    total_time: float


# Factory function types
type ModelFactory = Callable[[RegressionModelType], Any]
type OptimizerFactory = Callable[[StudyConfig], Any]


# PEP 695: Generic container types with type parameters
class ParameterCollection[T]:
    """Generic container for parameters with type safety."""

    def __init__(self, params: list[T]) -> None:
        self.params = params

    def add(self, param: T) -> Self:
        """Add a parameter to the collection."""
        self.params.append(param)
        return self

    def get_all(self) -> list[T]:
        """Get all parameters."""
        return self.params.copy()


class ResultCollection[T]:
    """Generic container for results with type safety."""

    def __init__(self) -> None:
        self.results: list[T] = []

    def add_result(self, result: T) -> Self:
        """Add a result to the collection."""
        self.results.append(result)
        return self

    def get_latest(self) -> T | None:
        """Get the latest result."""
        return self.results[-1] if self.results else None

    def get_all(self) -> list[T]:
        """Get all results."""
        return self.results.copy()
