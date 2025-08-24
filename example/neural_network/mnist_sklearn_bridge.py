"""Neural Network Model Bridge example using sklearn and MNIST."""

import argparse
import time
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from modelbridge import ModelBridge
from modelbridge.types import ParamDict
from modelbridge.utils.config_loader import create_param_config


class SklearnNeuralObjectives:
    """Neural network objectives using sklearn MLPClassifier."""

    def __init__(self, subset_size: int = 1000, test_size: float = 0.2):
        """Initialize with MNIST data.

        Args:
            subset_size: Size of dataset subset for fast execution
            test_size: Fraction of data to use for testing
        """
        print("Loading MNIST data...")
        self.subset_size = subset_size

        # Load MNIST data
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data, mnist.target.astype(int)  # type: ignore[attr-defined]

        # Use subset for speed
        indices = np.random.choice(len(X), size=subset_size, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_subset, y_subset, test_size=test_size, random_state=42, stratify=y_subset
        )

        # Normalize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(
            f"Loaded MNIST subset: {len(self.X_train)} training, {len(self.X_test)} test samples"
        )

    def micro_objective(self, params: ParamDict) -> float:
        """Micro model: More complex MLP (higher accuracy, slower)."""
        # Extract hyperparameters
        hidden_layer_sizes = (
            int(params["hidden_size_1"]),
            int(params["hidden_size_2"]),
        )
        alpha = float(params["alpha"])  # L2 regularization
        learning_rate_init = float(params["learning_rate"])

        # Create detailed model
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=50,  # Limited iterations for speed
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )

        start_time = time.time()

        try:
            # Train model
            model.fit(self.X_train, self.y_train)

            # Evaluate on test set
            accuracy = model.score(self.X_test, self.y_test)
            training_time = time.time() - start_time

            # Objective: minimize (1 - accuracy) with time penalty
            objective_value = (1.0 - accuracy) + 0.001 * training_time

            print(
                f"Micro MLP {hidden_layer_sizes} - Accuracy: {accuracy:.4f}, "
                f"Time: {training_time:.2f}s, Objective: {objective_value:.4f}"
            )

            return objective_value

        except Exception as e:
            print(f"Micro model failed: {e}")
            return 1.0  # High penalty for failed training

    def macro_objective(self, params: ParamDict, target_value: float) -> float:
        """Macro model: Simple MLP (lower accuracy, faster)."""
        # Extract hyperparameters
        hidden_size = int(params["macro_hidden_size"])
        alpha = float(params.get("macro_alpha", 0.01))
        learning_rate_init = float(params["learning_rate"])

        # Create simple model
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_size,),  # Single hidden layer
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=20,  # Very limited iterations
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=3,
        )

        start_time = time.time()

        try:
            # Train model
            model.fit(self.X_train, self.y_train)

            # Evaluate on test set
            accuracy = model.score(self.X_test, self.y_test)
            training_time = time.time() - start_time

            # Objective: minimize (1 - accuracy) with time penalty
            objective_value = (1.0 - accuracy) + 0.002 * training_time

            print(
                f"Macro MLP ({hidden_size},) - Accuracy: {accuracy:.4f}, "
                f"Time: {training_time:.2f}s, Objective: {objective_value:.4f}"
            )

            return objective_value

        except Exception as e:
            print(f"Macro model failed: {e}")
            return 1.0  # High penalty for failed training


class MNISTSklearnBridge:
    """MNIST neural network model bridge using sklearn."""

    def __init__(self, subset_size: int = 1000):
        """Initialize MNIST sklearn bridge.

        Args:
            subset_size: Size of MNIST subset for fast execution
        """
        self.subset_size = subset_size

        # Create neural network objectives
        print("Setting up neural network objectives...")
        self.nn_objectives = SklearnNeuralObjectives(subset_size=subset_size)

        # Create parameter configurations
        self.micro_param_config = create_param_config(
            param_names=[
                "hidden_size_1",  # First hidden layer size
                "hidden_size_2",  # Second hidden layer size
                "alpha",  # L2 regularization
                "learning_rate",  # Learning rate
            ],
            param_types=["int", "int", "float", "float"],
            param_ranges=[
                (32, 128),  # First layer: 32-128 neurons
                (16, 64),  # Second layer: 16-64 neurons
                (0.0001, 0.1),  # Alpha: 0.0001-0.1
                (0.0001, 0.01),  # Learning rate: 0.0001-0.01
            ],
        )

        self.macro_param_config = create_param_config(
            param_names=[
                "macro_hidden_size",  # Single hidden layer size
                "macro_alpha",  # L2 regularization
                "learning_rate",  # Learning rate (shared)
            ],
            param_types=["int", "float", "float"],
            param_ranges=[
                (16, 64),  # Hidden size: 16-64 neurons
                (0.001, 0.1),  # Alpha: 0.001-0.1
                (0.0001, 0.01),  # Learning rate: 0.0001-0.01
            ],
        )

    def run_model_bridge(self, n_train: int = 3, n_test: int = 2) -> Any:
        """Run the neural network model bridge."""
        print("\n🧠 Starting Neural Network Model Bridge (sklearn)")
        print(f"Training scenarios: {n_train}, Test scenarios: {n_test}")

        # Create model bridge
        bridge = ModelBridge(
            micro_objective=self.nn_objectives.micro_objective,
            macro_objective=self.nn_objectives.macro_objective,
            micro_param_config=self.micro_param_config,
            macro_param_config=self.macro_param_config,
            regression_type="polynomial",
            optimizer_config={
                "storage": "sqlite:///outputs/databases/mnist_sklearn_bridge.db",
                "direction": "minimize",
                "sampler": "tpe",
                "seed": 42,
            },
            regression_config={"degree": 2},
        )

        # Run bridge pipeline
        start_time = time.time()

        metrics = bridge.run_full_pipeline(
            n_train=n_train,
            n_test=n_test,
            micro_trials_per_dataset=4,  # Fast execution
            macro_trials_per_dataset=4,
            visualize=True,
            output_dir="outputs/examples/neural_network",
        )

        elapsed_time = time.time() - start_time

        print("\n🎯 Neural Network Bridge Results:")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print("Parameter prediction quality:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")

        # Analyze best parameters found
        if bridge.train_micro_params:  # type: ignore[attr-defined]
            print("\n📊 Optimal Neural Network Parameters:")

            # Find best micro parameters
            print("Best Micro Model (2-layer MLP) parameters:")
            for i, params in enumerate(bridge.train_micro_params):  # type: ignore[attr-defined]
                h1, h2 = params["hidden_size_1"], params["hidden_size_2"]
                alpha, lr = params["alpha"], params["learning_rate"]
                print(
                    f"  Scenario {i + 1}: [{h1}, {h2}] neurons, "
                    f"alpha={alpha:.4f}, lr={lr:.5f}"
                )

            # Find best macro parameters
            print("Best Macro Model (1-layer MLP) parameters:")
            for i, params in enumerate(bridge.train_macro_params):  # type: ignore[attr-defined]
                h = params["macro_hidden_size"]
                alpha, lr = params["macro_alpha"], params["learning_rate"]
                print(
                    f"  Scenario {i + 1}: [{h}] neurons, alpha={alpha:.4f}, lr={lr:.5f}"
                )

        return metrics


def demo_individual_models() -> None:
    """Demo individual model training for comparison."""
    print("\n🔬 Individual Neural Network Models Demo")
    print("=" * 50)

    objectives = SklearnNeuralObjectives(subset_size=500)

    # Test micro model
    print("\n1. Testing Micro Model (2-layer MLP):")
    micro_params = {
        "hidden_size_1": 64,
        "hidden_size_2": 32,
        "alpha": 0.001,
        "learning_rate": 0.001,
    }
    micro_result = objectives.micro_objective(micro_params)

    # Test macro model
    print("\n2. Testing Macro Model (1-layer MLP):")
    macro_params = {
        "macro_hidden_size": 32,
        "macro_alpha": 0.01,
        "learning_rate": 0.001,
    }
    macro_result = objectives.macro_objective(macro_params, target_value=0.5)

    print("\nComparison:")
    print(f"Micro model objective: {micro_result:.4f}")
    print(f"Macro model objective: {macro_result:.4f}")


def main() -> None:
    """Main function for neural network model bridge example."""
    parser = argparse.ArgumentParser(
        description="Neural Network Model Bridge with MNIST (sklearn)"
    )
    parser.add_argument("--demo", action="store_true", help="Run individual model demo")
    parser.add_argument(
        "--subset-size", type=int, default=1000, help="MNIST subset size"
    )
    parser.add_argument(
        "--n-train", type=int, default=3, help="Number of training scenarios"
    )
    parser.add_argument(
        "--n-test", type=int, default=2, help="Number of test scenarios"
    )

    args = parser.parse_args()

    print("🧠 Neural Network Model Bridge Example")
    print("Using sklearn MLPClassifier and MNIST dataset")
    print("=" * 60)

    if args.demo:
        demo_individual_models()
        return

    # Run model bridge
    bridge = MNISTSklearnBridge(subset_size=args.subset_size)

    total_start_time = time.time()

    try:
        metrics = bridge.run_model_bridge(n_train=args.n_train, n_test=args.n_test)

        total_time = time.time() - total_start_time

        print("\n🏆 Final Results:")
        print(f"Total execution time: {total_time:.2f} seconds")
        print("Successfully bridged 2-layer MLP → 1-layer MLP")
        print(
            f"Bridge prediction quality: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}"
        )

        print("\n📁 Results saved to: outputs/examples/neural_network/")
        print("Check the plots to see neural network hyperparameter relationships!")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
