"""Refactored simple benchmark using ModelBridge library."""

import argparse
from pathlib import Path
from pprint import pprint
import random
from typing import Any

import numpy as np

from modelbridge import ModelBridge
from modelbridge.utils.config_loader import create_param_config, load_toml_config
from modelbridge.utils.data_manager import DataManager


class BenchmarkFunctions:
    """Collection of benchmark optimization functions."""

    @staticmethod
    def sphere(x: np.ndarray, p: np.ndarray) -> float:
        """Sphere function: sum(p * x^2)"""
        return float(np.sum(p * x**2))

    @staticmethod
    def rastrigin(x: np.ndarray, p: np.ndarray) -> float:
        """Rastrigin function."""
        return float(10 * len(x) + np.sum(p * x**2 - 10.0 * np.cos(2 * np.pi * x)))

    @staticmethod
    def griewank(x: np.ndarray, p: np.ndarray) -> float:
        """Griewank function."""
        sum1 = np.sum(p * x**2)
        prod1 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return float(1 + sum1 / 4000 - prod1)

    @classmethod
    def get_function_by_name(cls, function_name: str) -> Any:
        """Get function by name."""
        functions = {
            "sphere": cls.sphere,
            "rastrigin": cls.rastrigin,
            "griewank": cls.griewank,
        }
        if function_name not in functions:
            raise ValueError(f"Unknown function name: {function_name}")
        return functions[function_name]


class SimpleBenchmarkBridge:
    """Simple benchmark model bridge implementation."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize benchmark bridge with configuration."""
        self.config = config
        self.data_manager = DataManager()

        # Extract configuration
        self.max_x_dim = config["dataset"]["max_x_dim"]
        self.max_x = config["dataset"]["max_x"]
        self.min_x = config["dataset"]["min_x"]
        self.nx_train = config["dataset"]["nx_train"]
        self.nx_test = config["dataset"]["nx_test"]
        self.n_train = config["dataset"]["n_train"]
        self.n_test = config["dataset"]["n_test"]

        # Get function objects
        self.micro_function = BenchmarkFunctions.get_function_by_name(
            config["micro_model"]["micro_function_name"]
        )
        self.macro_function = BenchmarkFunctions.get_function_by_name(
            config["macro_model"]["macro_function_name"]
        )

        # Create parameter configurations
        self.micro_param_config = create_param_config(
            param_names=[
                f"{config['micro_model']['micro_param_prefix']}_{i + 1}"
                for i in range(self.max_x_dim)
            ],
            param_types=["float"] * self.max_x_dim,
            param_ranges=[
                (
                    config["micro_model"]["micro_min_param"],
                    config["micro_model"]["micro_max_param"],
                )
            ]
            * self.max_x_dim,
        )

        self.macro_param_config = create_param_config(
            param_names=[
                f"{config['macro_model']['macro_param_prefix']}_{i + 1}"
                for i in range(self.max_x_dim)
            ],
            param_types=["float"] * self.max_x_dim,
            param_ranges=[
                (
                    config["macro_model"]["macro_min_param"],
                    config["macro_model"]["macro_max_param"],
                )
            ]
            * self.max_x_dim,
        )

        # Generate datasets
        self.generate_datasets()

    def generate_datasets(self) -> None:
        """Generate training and test datasets."""
        seed = self.config["generic"]["seed"]
        total_samples = (self.nx_train + self.nx_test) * (self.n_train + self.n_test)

        # Generate variable dataset
        self.x_dataset = self.data_manager.generate_variable_dataset(
            dim=self.max_x_dim,
            max_value=self.max_x,
            min_value=self.min_x,
            num_samples=total_samples,
            seed=seed,
        )

        self.x_train = self.x_dataset[: self.nx_train * self.n_train]
        self.x_test = self.x_dataset[self.nx_train * self.n_train :]

    def create_micro_objective(self, dataset_idx: int, is_train: bool = True) -> Any:
        """Create micro objective function for specific dataset."""
        x_data = self.x_train if is_train else self.x_test
        nx = self.nx_train if is_train else self.nx_test

        def micro_objective(params: dict[str, Any]) -> float:
            # Convert parameters to array
            param_array = np.array(
                [
                    params[
                        f"{self.config['micro_model']['micro_param_prefix']}_{i + 1}"
                    ]
                    for i in range(self.max_x_dim)
                ]
            )

            # Evaluate function on all data points for this dataset
            total_value = 0.0
            start_idx = dataset_idx * nx
            end_idx = (dataset_idx + 1) * nx

            for j in range(start_idx, end_idx):
                x_j = x_data[j - (self.nx_train * self.n_train if not is_train else 0)]
                value = self.micro_function(x_j, param_array)
                total_value += value

            return total_value / nx  # Average over dataset

        return micro_objective

    def create_macro_objective(
        self, dataset_idx: int, target_value: float, is_train: bool = True
    ) -> Any:
        """Create macro objective function for specific dataset."""
        x_data = self.x_train if is_train else self.x_test
        nx = self.nx_train if is_train else self.nx_test

        def macro_objective(params: dict[str, Any], target: float) -> float:
            # Convert parameters to array
            param_array = np.array(
                [
                    params[
                        f"{self.config['macro_model']['macro_param_prefix']}_{i + 1}"
                    ]
                    for i in range(self.max_x_dim)
                ]
            )

            # Evaluate function on all data points for this dataset
            total_value = 0.0
            start_idx = dataset_idx * nx
            end_idx = (dataset_idx + 1) * nx

            for j in range(start_idx, end_idx):
                x_j = x_data[j - (self.nx_train * self.n_train if not is_train else 0)]
                value = self.macro_function(x_j, param_array)
                total_value += value

            return total_value / nx  # Average over dataset

        return macro_objective

    def run_traditional_approach(self) -> Any:
        """Run traditional individual optimization for comparison."""
        print("Running traditional approach...")

        # Training phase - optimize each dataset individually
        train_micro_results = []
        train_macro_results = []

        for i in range(self.n_train):
            print(f"Processing training dataset {i + 1}/{self.n_train}")

            # Create micro objective for this dataset
            micro_obj = self.create_micro_objective(i, is_train=True)

            # Create simple bridge for this dataset
            bridge = ModelBridge(
                micro_objective=micro_obj,
                macro_objective=self.create_macro_objective(i, 0.0, is_train=True),
                micro_param_config=self.micro_param_config,
                macro_param_config=self.macro_param_config,
                regression_type=self.config["regression"]["regression_model_name"],
                optimizer_config={
                    "storage": self.config["optuna"]["storage"],
                    "direction": self.config["optuna"]["direction"],
                    "sampler": self.config["optuna"]["sampler_name"],
                    "seed": self.config["generic"]["seed"],
                },
            )

            # Run single dataset optimization
            _ = bridge.run_full_pipeline(
                n_train=1,
                n_test=0,
                micro_trials_per_dataset=100,
                macro_trials_per_dataset=100,
                visualize=False,
            )

            train_micro_results.append(bridge.train_micro_params[0])
            train_macro_results.append(bridge.train_macro_params[0])

        # Test phase
        test_micro_results = []
        test_macro_results = []

        for i in range(self.n_test):
            print(f"Processing test dataset {i + 1}/{self.n_test}")

            micro_obj = self.create_micro_objective(i, is_train=False)

            bridge = ModelBridge(
                micro_objective=micro_obj,
                macro_objective=self.create_macro_objective(i, 0.0, is_train=False),
                micro_param_config=self.micro_param_config,
                macro_param_config=self.macro_param_config,
                regression_type=self.config["regression"]["regression_model_name"],
                optimizer_config={
                    "storage": self.config["optuna"]["storage"],
                    "direction": self.config["optuna"]["direction"],
                    "sampler": self.config["optuna"]["sampler_name"],
                    "seed": self.config["generic"]["seed"],
                },
            )

            _ = bridge.run_full_pipeline(
                n_train=1,
                n_test=0,
                micro_trials_per_dataset=100,
                macro_trials_per_dataset=100,
                visualize=False,
            )

            test_micro_results.append(bridge.train_micro_params[0])
            test_macro_results.append(bridge.train_macro_params[0])

        return (
            train_micro_results,
            train_macro_results,
            test_micro_results,
            test_macro_results,
        )


def main():
    """Main function for simple benchmark."""
    parser = argparse.ArgumentParser(
        description="Simple benchmark with ModelBridge library"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config file", required=True
    )
    args = parser.parse_args()

    # Load configuration
    config_file = Path(args.config).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file '{args.config}' not found.")

    config = load_toml_config(config_file)
    if args.debug:
        pprint(config)

    # Set random seeds
    seed = config["generic"]["seed"]
    random.seed(seed)
    np.random.seed(seed)

    # Create benchmark bridge
    benchmark = SimpleBenchmarkBridge(config)

    # Run traditional approach (each dataset optimized individually)
    train_micro, train_macro, test_micro, test_macro = (
        benchmark.run_traditional_approach()
    )

    # Now create unified bridge model for comparison
    print("\\nRunning model bridge approach...")

    # Create combined objective functions
    def combined_micro_objective(params: dict[str, Any]) -> float:
        """Evaluate micro objective on random dataset."""
        dataset_idx = np.random.randint(0, benchmark.n_train)
        return benchmark.create_micro_objective(dataset_idx, is_train=True)(params)

    def combined_macro_objective(params: dict[str, Any], target: float) -> float:
        """Evaluate macro objective on random dataset."""
        dataset_idx = np.random.randint(0, benchmark.n_train)
        return benchmark.create_macro_objective(dataset_idx, target, is_train=True)(
            params
        )

    # Create model bridge
    bridge = ModelBridge(
        micro_objective=combined_micro_objective,
        macro_objective=combined_macro_objective,
        micro_param_config=benchmark.micro_param_config,
        macro_param_config=benchmark.macro_param_config,
        regression_type=config["regression"]["regression_model_name"],
        optimizer_config={
            "storage": "sqlite:///outputs/databases/simple_benchmark.db",
            "direction": config["optuna"]["direction"],
            "sampler": config["optuna"]["sampler_name"],
            "seed": seed,
        },
        regression_config={"degree": benchmark.max_x_dim}
        if config["regression"]["regression_model_name"] == "polynomial"
        else {},
    )

    # Run model bridge
    metrics = bridge.run_full_pipeline(
        n_train=config["dataset"]["n_train"],
        n_test=config["dataset"]["n_test"],
        micro_trials_per_dataset=100,
        macro_trials_per_dataset=100,
        visualize=True,
        output_dir="outputs/examples/simple_benchmark",
    )

    print("\\nModel Bridge Results:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")

    print("\\nSimple benchmark completed successfully!")


if __name__ == "__main__":
    main()
