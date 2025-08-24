"""Visualization utilities for model bridging results."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..types import (
    FilePath,
    NumPyArray,
)


class Visualizer:
    """Visualization utility for model bridging results."""

    def __init__(self, figsize: tuple = (12, 8), dpi: int = 100):
        """Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: Default DPI for plots
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use("default")

    def plot_parameter_relationship(
        self,
        macro_params: NumPyArray,
        micro_params: NumPyArray,
        macro_param_names: list[str],
        micro_param_names: list[str],
        title: str = "Parameter Relationships",
        output_dir: FilePath | None = None,
        filename: str = "parameter_relationships.png",
    ) -> None:
        """Plot relationships between macro and micro parameters.

        Args:
            macro_params: Macro parameters array
            micro_params: Micro parameters array
            macro_param_names: Names of macro parameters
            micro_param_names: Names of micro parameters
            title: Plot title
            output_dir: Directory to save plot (if None, displays plot)
            filename: Filename for saved plot
        """
        n_micro_params = micro_params.shape[1]
        n_cols = min(3, n_micro_params)
        n_rows = (n_micro_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figsize[0], self.figsize[1] * n_rows / 2),
            dpi=self.dpi,
        )

        if n_micro_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each micro parameter against first macro parameter
        macro_param_idx = 0  # Use first macro parameter for x-axis

        for i in range(n_micro_params):
            ax = axes[i]

            ax.scatter(
                macro_params[:, macro_param_idx], micro_params[:, i], alpha=0.6, s=30
            )

            ax.set_xlabel(f"{macro_param_names[macro_param_idx]}")
            ax.set_ylabel(f"{micro_param_names[i]}")
            ax.set_title(
                f"{macro_param_names[macro_param_idx]} vs {micro_param_names[i]}"
            )
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_micro_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_prediction_accuracy(
        self,
        true_values: NumPyArray,
        predicted_values: NumPyArray,
        param_names: list[str],
        title: str = "Prediction Accuracy",
        output_dir: FilePath | None = None,
        filename: str = "prediction_accuracy.png",
    ) -> None:
        """Plot prediction accuracy for each parameter.

        Args:
            true_values: True parameter values
            predicted_values: Predicted parameter values
            param_names: Parameter names
            title: Plot title
            output_dir: Directory to save plot
            filename: Filename for saved plot
        """
        n_params = true_values.shape[1]
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figsize[0], self.figsize[1] * n_rows / 2),
            dpi=self.dpi,
        )

        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(n_params):
            ax = axes[i]

            # Scatter plot: predicted vs true
            ax.scatter(
                predicted_values[:, i],
                true_values[:, i],
                alpha=0.6,
                s=30,
                label="Predictions",
            )

            # Perfect prediction line
            min_val = min(true_values[:, i].min(), predicted_values[:, i].min())
            max_val = max(true_values[:, i].max(), predicted_values[:, i].max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                alpha=0.8,
                label="Perfect",
            )

            ax.set_xlabel(f"Predicted {param_names[i]}")
            ax.set_ylabel(f"True {param_names[i]}")
            ax.set_title(f"{param_names[i]} Prediction")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Calculate and display R²
            correlation = np.corrcoef(true_values[:, i], predicted_values[:, i])[0, 1]
            r_squared = correlation**2
            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_optimization_history(
        self,
        study: Any,  # optuna.Study, but avoiding import dependency
        title: str = "Optimization History",
        output_dir: FilePath | None = None,
        filename: str = "optimization_history.png",
    ) -> None:
        """Plot optimization history from Optuna study.

        Args:
            study: Optuna study object
            title: Plot title
            output_dir: Directory to save plot
            filename: Filename for saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # Plot objective values over trials
        trial_numbers = [trial.number for trial in study.trials]
        objective_values = [
            trial.value for trial in study.trials if trial.value is not None
        ]

        if objective_values:
            ax1.plot(
                trial_numbers[: len(objective_values)],
                objective_values,
                "b-",
                alpha=0.7,
            )
            ax1.set_xlabel("Trial")
            ax1.set_ylabel("Objective Value")
            ax1.set_title("Objective Value History")
            ax1.grid(True, alpha=0.3)

            # Plot best value so far
            best_values = []
            current_best = (
                float("inf") if study.direction.name == "MINIMIZE" else float("-inf")
            )

            for value in objective_values:
                if study.direction.name == "MINIMIZE":
                    current_best = min(current_best, value)
                else:
                    current_best = max(current_best, value)
                best_values.append(current_best)

            ax2.plot(trial_numbers[: len(best_values)], best_values, "g-", linewidth=2)
            ax2.set_xlabel("Trial")
            ax2.set_ylabel("Best Objective Value")
            ax2.set_title("Best Value History")
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_regression_model_performance(
        self,
        regression_model: Any,  # Regression model with predict method
        train_input: NumPyArray,
        train_target: NumPyArray,
        test_input: NumPyArray,
        test_target: NumPyArray,
        param_names: list[str],
        title: str = "Regression Model Performance",
        output_dir: FilePath | None = None,
        filename: str = "regression_performance.png",
    ) -> None:
        """Plot regression model performance on train and test data.

        Args:
            regression_model: Trained regression model
            train_input: Training input data
            train_target: Training output data
            test_input: Test input data
            test_target: Test output data
            param_names: Output parameter names
            title: Plot title
            output_dir: Directory to save plot
            filename: Filename for saved plot
        """
        # Make predictions
        train_pred = regression_model.predict(train_input)
        test_pred = regression_model.predict(test_input)

        n_params = train_target.shape[1]
        fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8), dpi=self.dpi)

        if n_params == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_params):
            # Training data
            axes[0, i].scatter(
                train_pred[:, i],
                train_target[:, i],
                alpha=0.6,
                s=30,
                c="blue",
                label="Train",
            )

            # Test data
            axes[1, i].scatter(
                test_pred[:, i],
                test_target[:, i],
                alpha=0.6,
                s=30,
                c="red",
                label="Test",
            )

            for row in range(2):
                ax = axes[row, i]

                # Perfect prediction line
                if row == 0:
                    min_val = min(train_target[:, i].min(), train_pred[:, i].min())
                    max_val = max(train_target[:, i].max(), train_pred[:, i].max())
                else:
                    min_val = min(test_target[:, i].min(), test_pred[:, i].min())
                    max_val = max(test_target[:, i].max(), test_pred[:, i].max())

                ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.8)
                ax.set_xlabel(f"Predicted {param_names[i]}")
                ax.set_ylabel(f"True {param_names[i]}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add R² score
                if row == 0:
                    r2 = np.corrcoef(train_target[:, i], train_pred[:, i])[0, 1] ** 2
                    ax.set_title(f"Train - {param_names[i]} (R² = {r2:.3f})")
                else:
                    r2 = np.corrcoef(test_target[:, i], test_pred[:, i])[0, 1] ** 2
                    ax.set_title(f"Test - {param_names[i]} (R² = {r2:.3f})")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()
