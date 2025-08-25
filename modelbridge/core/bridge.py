"""Main ModelBridge class for coordinating the bridging process."""

from pathlib import Path
from typing import Any

import numpy as np

from ..types import (
    EvaluationMetrics,
    FloatArray,
    MacroObjectiveFunctionProtocol,
    ObjectiveFunctionProtocol,
    ParamConfig,
    ParamList,
    RegressionModelType,
)
from ..utils.data_manager import DataManager
from ..utils.visualization import Visualizer
from .optimizer import OptunaOptimizer
from .regression import RegressionModel


class ModelBridge:
    """Main class for model bridging workflow.

    Coordinates the complete model bridging process including training phase optimization,
    testing phase evaluation, and regression model training to map macro parameters
    to micro parameters. This is the primary interface for ModelBridge functionality.

    The workflow consists of:
    1. Training phase: Optimize both micro and macro models on multiple datasets
    2. Regression training: Learn mapping from macro to micro parameters
    3. Testing phase: Evaluate prediction accuracy on new datasets
    4. Visualization: Generate plots and analysis of results

    Attributes:
        micro_objective (ObjectiveFunctionProtocol): Micro model objective function.
        macro_objective (MacroObjectiveFunctionProtocol): Macro model objective function.
        optimizer (OptunaOptimizer): Optuna optimization wrapper.
        regression (RegressionModel): Regression model for parameter mapping.
        train_micro_params (ParamList): Training micro parameters.
        train_macro_params (ParamList): Training macro parameters.
        test_micro_params (ParamList): Test micro parameters.
        test_macro_params (ParamList): Test macro parameters.
        predicted_micro_params (FloatArray | None): Predicted micro parameters.

    Example:
        >>> bridge = ModelBridge(
        ...     micro_objective=complex_function,
        ...     macro_objective=simple_function,
        ...     micro_param_config=config,
        ...     macro_param_config=config
        ... )
        >>> metrics = bridge.run_full_pipeline(n_train=5, n_test=3)
    """

    def __init__(
        self,
        micro_objective: ObjectiveFunctionProtocol,
        macro_objective: MacroObjectiveFunctionProtocol,
        micro_param_config: ParamConfig,
        macro_param_config: ParamConfig,
        regression_type: RegressionModelType = "polynomial",
        optimizer_config: dict[str, Any] | None = None,
        regression_config: dict[str, Any] | None = None,
    ):
        """Initialize ModelBridge.

        Args:
            micro_objective (ObjectiveFunctionProtocol): Micro model objective function that takes
                parameters and returns a float objective value.
            macro_objective (MacroObjectiveFunctionProtocol): Macro model objective function that
                takes parameters and a target value, returning a float objective value.
            micro_param_config (ParamConfig): Parameter configuration dictionary for micro model
                with parameter names as keys and type/range specs as values.
            macro_param_config (ParamConfig): Parameter configuration dictionary for macro model
                with parameter names as keys and type/range specs as values.
            regression_type (RegressionModelType, optional): Type of regression model to use for
                parameter mapping. Defaults to "polynomial".
            optimizer_config (dict[str, Any] | None, optional): Configuration dictionary for
                Optuna optimizer including storage, sampler, and seed settings. Defaults to None.
            regression_config (dict[str, Any] | None, optional): Configuration dictionary for
                regression model with model-specific parameters. Defaults to None.

        """
        self.micro_objective = micro_objective
        self.macro_objective = macro_objective
        self.micro_param_config = micro_param_config
        self.macro_param_config = macro_param_config

        # Initialize optimizer
        optimizer_config = optimizer_config or {}
        self.optimizer = OptunaOptimizer(**optimizer_config)

        # Initialize regression model
        regression_config = regression_config or {}
        self.regression = RegressionModel(regression_type, **regression_config)

        # Initialize utilities
        self.data_manager = DataManager()
        self.visualizer = Visualizer()

        # Storage for results with proper type annotations
        self.train_micro_params: ParamList = []
        self.train_macro_params: ParamList = []
        self.test_micro_params: ParamList = []
        self.test_macro_params: ParamList = []
        self.predicted_micro_params: FloatArray | None = None

        self._is_trained = False

    def train_phase(
        self,
        n_train: int,
        micro_trials_per_dataset: int = 100,
        macro_trials_per_dataset: int = 100,
        study_prefix: str = "bridge_train",
    ) -> None:
        """Execute training phase of model bridging.

        This phase optimizes both micro and macro models on multiple training datasets,
        then trains a regression model to map macro parameters to micro parameters.

        Args:
            n_train (int): Number of training datasets to generate and optimize.
            micro_trials_per_dataset (int, optional): Number of optimization trials to run
                for each micro model dataset. Defaults to 100.
            macro_trials_per_dataset (int, optional): Number of optimization trials to run
                for each macro model dataset. Defaults to 100.
            study_prefix (str, optional): Prefix string for Optuna study names to avoid
                conflicts between different runs. Defaults to "bridge_train".

        Raises:
            ValueError: If optimization fails to find valid solutions.

        """
        print(f"Starting training phase with {n_train} datasets...")

        for i in range(n_train):
            print(f"Processing training dataset {i + 1}/{n_train}")

            # Optimize micro model
            micro_study_name = f"{study_prefix}_micro_{i}"

            def micro_objective_wrapper(trial: Any) -> float:
                params = self.optimizer.suggest_parameters(
                    trial, self.micro_param_config
                )
                return self.micro_objective(params)

            micro_study = self.optimizer.optimize_batch(
                micro_objective_wrapper, micro_study_name, micro_trials_per_dataset
            )

            best_micro_params = micro_study.best_trial.params
            best_micro_value = micro_study.best_trial.value
            if best_micro_value is None:
                raise ValueError("Optimization failed to find a valid solution")
            self.train_micro_params.append(best_micro_params)

            # Optimize macro model (minimize difference with micro result)
            macro_study_name = f"{study_prefix}_macro_{i}"

            # Create closure that captures best_micro_value
            def create_macro_wrapper(target_value: float) -> Any:
                def macro_objective_wrapper(trial: Any) -> float:
                    params = self.optimizer.suggest_parameters(
                        trial, self.macro_param_config
                    )
                    macro_value = self.macro_objective(params, target_value)
                    return abs(macro_value - target_value)

                return macro_objective_wrapper

            macro_study = self.optimizer.optimize_batch(
                create_macro_wrapper(best_micro_value),
                macro_study_name,
                macro_trials_per_dataset,
            )

            best_macro_params = macro_study.best_trial.params
            self.train_macro_params.append(best_macro_params)

        # Train regression model
        print("Training regression model...")
        self.regression.fit(
            self.train_macro_params,
            self.train_micro_params,
            macro_param_names=list(self.macro_param_config.keys()),
            micro_param_names=list(self.micro_param_config.keys()),
        )

        self._is_trained = True
        print("Training phase completed!")

    def test_phase(
        self,
        n_test: int,
        micro_trials_per_dataset: int = 100,
        macro_trials_per_dataset: int = 100,
        study_prefix: str = "bridge_test",
    ) -> EvaluationMetrics:
        """Execute testing phase of model bridging.

        Args:
            n_test: Number of test datasets
            micro_trials_per_dataset: Number of trials for micro optimization
            macro_trials_per_dataset: Number of trials for macro optimization
            study_prefix: Prefix for study names

        Returns:
            Evaluation metrics

        """
        if not self._is_trained:
            raise ValueError("Must complete training phase before testing")

        print(f"Starting test phase with {n_test} datasets...")

        for i in range(n_test):
            print(f"Processing test dataset {i + 1}/{n_test}")

            # Optimize micro model (ground truth)
            micro_study_name = f"{study_prefix}_micro_{i}"

            def micro_objective_wrapper(trial: Any) -> float:
                params = self.optimizer.suggest_parameters(
                    trial, self.micro_param_config
                )
                return self.micro_objective(params)

            micro_study = self.optimizer.optimize_batch(
                micro_objective_wrapper, micro_study_name, micro_trials_per_dataset
            )

            best_micro_params = micro_study.best_trial.params
            best_micro_value = micro_study.best_trial.value
            if best_micro_value is None:
                raise ValueError("Test optimization failed to find a valid solution")
            self.test_micro_params.append(best_micro_params)

            # Optimize macro model
            macro_study_name = f"{study_prefix}_macro_{i}"

            # Create closure that captures best_micro_value
            def create_test_macro_wrapper(target_value: float) -> Any:
                def macro_objective_wrapper(trial: Any) -> float:
                    params = self.optimizer.suggest_parameters(
                        trial, self.macro_param_config
                    )
                    macro_value = self.macro_objective(params, target_value)
                    return abs(macro_value - target_value)

                return macro_objective_wrapper

            macro_study = self.optimizer.optimize_batch(
                create_test_macro_wrapper(best_micro_value),
                macro_study_name,
                macro_trials_per_dataset,
            )

            best_macro_params = macro_study.best_trial.params
            self.test_macro_params.append(best_macro_params)

        # Predict micro parameters using regression model
        if self.test_macro_params:
            self.predicted_micro_params = self.regression.predict(
                self.test_macro_params,
                macro_param_names=list(self.macro_param_config.keys()),
            )
        else:
            # No test data, create empty array with correct shape
            n_micro_params = len(self.micro_param_config)
            self.predicted_micro_params = np.empty((0, n_micro_params))

        # Evaluate predictions
        if (
            self.test_micro_params
            and self.predicted_micro_params is not None
            and self.predicted_micro_params.size > 0
        ):
            true_micro_array = self.data_manager.convert_params_to_array(
                self.test_micro_params, list(self.micro_param_config.keys())
            )

            metrics = self.regression.evaluate(
                true_micro_array, self.predicted_micro_params
            )
        else:
            # No test data, return zero metrics
            metrics = EvaluationMetrics(mse=0.0, mae=0.0, r2=0.0)

        print("Test phase completed!")
        print(f"Evaluation metrics: {metrics}")

        return metrics

    def run_full_pipeline(
        self,
        n_train: int,
        n_test: int,
        micro_trials_per_dataset: int = 100,
        macro_trials_per_dataset: int = 100,
        visualize: bool = True,
        output_dir: str | None = None,
    ) -> EvaluationMetrics:
        """Run complete model bridging pipeline.

        Executes the full model bridging workflow including training phase, testing phase,
        and optional visualization. This is the main entry point for model bridging.

        Args:
            n_train (int): Number of training datasets to generate and optimize on.
            n_test (int): Number of test datasets for evaluation and validation.
            micro_trials_per_dataset (int, optional): Number of optimization trials to run
                for each micro model dataset. Defaults to 100.
            macro_trials_per_dataset (int, optional): Number of optimization trials to run
                for each macro model dataset. Defaults to 100.
            visualize (bool, optional): Whether to generate parameter relationship plots
                and prediction accuracy visualizations. Defaults to True.
            output_dir (str | None, optional): Directory path to save results, plots, and
                data files. If None, no files are saved. Defaults to None.

        Returns:
            EvaluationMetrics: Dictionary containing evaluation metrics including MSE,
                MAE, and R² values comparing predicted vs actual micro parameters.

        Raises:
            ValueError: If training phase has not been completed before testing phase.

        """
        # Training phase
        self.train_phase(n_train, micro_trials_per_dataset, macro_trials_per_dataset)

        # Testing phase
        metrics = self.test_phase(
            n_test, micro_trials_per_dataset, macro_trials_per_dataset
        )

        # Visualization
        if visualize:
            self.visualize_results(output_dir)

        # Save results
        if output_dir:
            self.save_results(output_dir)

        return metrics

    def visualize_results(self, output_dir: str | None = None) -> None:
        """Generate visualizations of results."""
        if not self._is_trained or self.predicted_micro_params is None:
            raise ValueError("Must complete training and testing before visualization")

        # Convert data for visualization
        train_macro_array = self.data_manager.convert_params_to_array(
            self.train_macro_params, list(self.macro_param_config.keys())
        )
        train_micro_array = self.data_manager.convert_params_to_array(
            self.train_micro_params, list(self.micro_param_config.keys())
        )
        # test_macro_array = self.data_manager.convert_params_to_array(
        #     self.test_macro_params, list(self.macro_param_config.keys())
        # )
        test_micro_array = self.data_manager.convert_params_to_array(
            self.test_micro_params, list(self.micro_param_config.keys())
        )

        # Generate plots
        self.visualizer.plot_parameter_relationship(
            train_macro_array,
            train_micro_array,
            list(self.macro_param_config.keys()),
            list(self.micro_param_config.keys()),
            "Training Data - Macro vs Micro Parameters",
            output_dir,
        )

        self.visualizer.plot_prediction_accuracy(
            test_micro_array,
            self.predicted_micro_params,
            list(self.micro_param_config.keys()),
            "Prediction Accuracy",
            output_dir,
        )

    def save_results(self, output_dir: str) -> None:
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save parameter data
        self.data_manager.save_params_csv(
            self.train_macro_params,
            output_path / "train_macro_params.csv",
            list(self.macro_param_config.keys()),
        )
        self.data_manager.save_params_csv(
            self.train_micro_params,
            output_path / "train_micro_params.csv",
            list(self.micro_param_config.keys()),
        )
        self.data_manager.save_params_csv(
            self.test_macro_params,
            output_path / "test_macro_params.csv",
            list(self.macro_param_config.keys()),
        )
        self.data_manager.save_params_csv(
            self.test_micro_params,
            output_path / "test_micro_params.csv",
            list(self.micro_param_config.keys()),
        )

        # Save predictions
        if self.predicted_micro_params is not None:
            self.data_manager.save_array_csv(
                self.predicted_micro_params,
                output_path / "predicted_micro_params.csv",
                list(self.micro_param_config.keys()),
            )
