"""Main type stub file for ModelBridge library."""

from typing import Any

from .types import (
    EvaluationMetrics as EvaluationMetrics,
)
from .types import (
    FloatArray,
)
from .types import (
    MacroObjectiveFunctionProtocol as MacroObjectiveFunctionProtocol,
)
from .types import (
    NumPyArray as NumPyArray,
)
from .types import (
    ObjectiveFunctionProtocol as ObjectiveFunctionProtocol,
)
from .types import (
    ParamConfig as ParamConfig,
)
from .types import (
    ParamDict as ParamDict,
)
from .types import (
    ParamList as ParamList,
)
from .types import (
    RegressionModelType as RegressionModelType,
)

class ModelBridge:
    """Main model bridging class."""

    def __init__(
        self,
        micro_objective: ObjectiveFunctionProtocol,
        macro_objective: MacroObjectiveFunctionProtocol,
        micro_param_config: ParamConfig,
        macro_param_config: ParamConfig,
        regression_type: RegressionModelType = ...,
        optimizer_config: dict[str, Any] | None = ...,
        regression_config: dict[str, Any] | None = ...,
    ) -> None: ...
    def run_full_pipeline(
        self,
        n_train: int,
        n_test: int,
        micro_trials_per_dataset: int = ...,
        macro_trials_per_dataset: int = ...,
        visualize: bool = ...,
        output_dir: str | None = ...,
    ) -> EvaluationMetrics: ...

class OptunaOptimizer:
    """Optuna optimizer wrapper."""

    def __init__(
        self,
        storage: str | None = ...,
        direction: str = ...,
        sampler: str | Any | None = ...,
        seed: int | None = ...,
    ) -> None: ...

class RegressionModel:
    """Unified regression model interface."""

    def __init__(self, model_type: RegressionModelType, **kwargs: Any) -> None: ...
    def fit(
        self,
        macro_params: ParamList | FloatArray,
        micro_params: ParamList | FloatArray,
        macro_param_names: list[str] | None = ...,
        micro_param_names: list[str] | None = ...,
    ) -> None: ...
    def predict(
        self,
        macro_params: ParamList | FloatArray,
        macro_param_names: list[str] | None = ...,
    ) -> FloatArray: ...

class DataManager:
    """Data management utilities."""

    def convert_params_to_array(
        self, params_list: ParamList, param_names: list[str]
    ) -> FloatArray: ...

class Visualizer:
    """Visualization utilities."""

    def __init__(self, figsize: tuple = ..., dpi: int = ...) -> None: ...
    def plot_parameter_relationship(
        self,
        macro_params: FloatArray,
        micro_params: FloatArray,
        macro_param_names: list[str],
        micro_param_names: list[str],
        title: str = ...,
        output_dir: str | None = ...,
        filename: str = ...,
    ) -> None: ...

def load_toml_config(config_path: str | Any) -> dict[str, Any]: ...

__version__: str
__all__: list[str]
