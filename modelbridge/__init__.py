"""ModelBridge Library for Hyperparameter Optimization and Model Bridging.

This library provides a general framework for model bridging between micro and macro models
using hyperparameter optimization and regression models.
"""

from .core.bridge import ModelBridge
from .core.optimizer import OptunaOptimizer
from .core.regression import RegressionModel

# Export commonly used types
from .types import (
    EvaluationMetrics,
    MacroObjectiveFunctionProtocol,
    NumPyArray,
    ObjectiveFunctionProtocol,
    ParamConfig,
    ParamDict,
    ParamList,
    RegressionModelType,
)
from .utils.config_loader import load_toml_config
from .utils.data_manager import DataManager
from .utils.visualization import Visualizer

__version__ = "0.1.0"
__all__ = [
    "DataManager",
    "EvaluationMetrics",
    "MacroObjectiveFunctionProtocol",
    "ModelBridge",
    "NumPyArray",
    "ObjectiveFunctionProtocol",
    "OptunaOptimizer",
    "ParamConfig",
    "ParamDict",
    "ParamList",
    "RegressionModel",
    "RegressionModelType",
    "Visualizer",
    "load_toml_config",
]
