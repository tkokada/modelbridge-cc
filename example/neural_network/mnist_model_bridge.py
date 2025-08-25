"""Neural Network Model Bridge example using PyTorch and MNIST."""

import argparse
import time

from neural_models import NeuralNetworkObjectives
import torch

from modelbridge import ModelBridge
from modelbridge.utils.config_loader import create_param_config


class MNISTModelBridge:
    """MNIST neural network model bridge implementation."""

    def __init__(self, subset_size: int = 500, fast_mode: bool = True):
        """Initialize MNIST model bridge.

        Args:
            subset_size: Size of MNIST subset to use (smaller = faster)
            fast_mode: Whether to use fast training settings

        """
        self.subset_size = subset_size
        self.fast_mode = fast_mode

        # Create neural network objectives
        print("Initializing neural network objectives...")
        self.nn_objectives = NeuralNetworkObjectives(subset_size=subset_size)

        # Create parameter configurations
        self.micro_param_config = create_param_config(
            param_names=[
                "dropout_rate",  # CNN dropout rate
                "hidden_size",  # CNN hidden layer size
                "learning_rate",  # Learning rate
            ],
            param_types=["float", "int", "float"],
            param_ranges=[
                (0.1, 0.8),  # dropout: 10% to 80%
                (64, 256),  # hidden: 64 to 256 neurons
                (0.0001, 0.01),  # lr: 0.0001 to 0.01
            ],
        )

        self.macro_param_config = create_param_config(
            param_names=[
                "macro_hidden_size",  # MLP hidden layer size
                "macro_dropout_rate",  # MLP dropout rate
                "learning_rate",  # Learning rate (shared)
            ],
            param_types=["int", "float", "float"],
            param_ranges=[
                (16, 128),  # macro hidden: 16 to 128 neurons
                (0.1, 0.6),  # macro dropout: 10% to 60%
                (0.0001, 0.01),  # lr: 0.0001 to 0.01
            ],
        )

        print(f"Micro parameters: {list(self.micro_param_config.keys())}")
        print(f"Macro parameters: {list(self.macro_param_config.keys())}")

    def run_model_bridge(self, n_train: int = 3, n_test: int = 2) -> dict[str, float]:
        """Run the complete neural network model bridge."""
        print("\n🚀 Starting Neural Network Model Bridge")
        print(f"Training scenarios: {n_train}, Test scenarios: {n_test}")
        print(f"MNIST subset size: {self.subset_size}")

        # Determine trials per dataset based on mode
        if self.fast_mode:
            micro_trials = 3  # Very few trials for speed
            macro_trials = 3
        else:
            micro_trials = 5
            macro_trials = 5

        print(f"Trials per dataset: Micro={micro_trials}, Macro={macro_trials}")

        # Create local output directory
        from pathlib import Path

        output_dir = Path("pytorch_results")
        output_dir.mkdir(exist_ok=True)

        # Create model bridge
        bridge = ModelBridge(
            micro_objective=self.nn_objectives.micro_objective,
            macro_objective=self.nn_objectives.macro_objective,
            micro_param_config=self.micro_param_config,
            macro_param_config=self.macro_param_config,
            regression_type="polynomial",
            optimizer_config={
                "storage": f"sqlite:///{output_dir}/neural_bridge.db",
                "direction": "minimize",
                "sampler": "tpe",
                "seed": 42,
            },
            regression_config={"degree": 2},
        )

        # Run model bridge pipeline
        start_time = time.time()

        try:
            metrics = bridge.run_full_pipeline(
                n_train=n_train,
                n_test=n_test,
                micro_trials_per_dataset=micro_trials,
                macro_trials_per_dataset=macro_trials,
                visualize=True,
                output_dir=str(output_dir),
            )

            elapsed_time = time.time() - start_time

            print("\n🎯 Neural Network Model Bridge Results:")
            print(f"Total execution time: {elapsed_time:.2f} seconds")
            print(f"MSE: {metrics['mse']:.6f}")
            print(f"MAE: {metrics['mae']:.6f}")
            print(f"R²: {metrics['r2']:.6f}")

            # Print parameter analysis
            print("\n📊 Parameter Analysis:")
            print(
                f"Micro parameters (CNN): {len(bridge.train_micro_params)} training sets"
            )
            print(
                f"Macro parameters (MLP): {len(bridge.train_macro_params)} training sets"
            )

            if bridge.train_micro_params:
                avg_micro_dropout = sum(
                    p["dropout_rate"] for p in bridge.train_micro_params
                ) / len(bridge.train_micro_params)
                avg_micro_hidden = sum(
                    p["hidden_size"] for p in bridge.train_micro_params
                ) / len(bridge.train_micro_params)
                avg_micro_lr = sum(
                    p["learning_rate"] for p in bridge.train_micro_params
                ) / len(bridge.train_micro_params)

                print("Average Micro CNN parameters:")
                print(f"  Dropout rate: {avg_micro_dropout:.3f}")
                print(f"  Hidden size: {avg_micro_hidden:.1f}")
                print(f"  Learning rate: {avg_micro_lr:.5f}")

            if bridge.train_macro_params:
                avg_macro_dropout = sum(
                    p["macro_dropout_rate"] for p in bridge.train_macro_params
                ) / len(bridge.train_macro_params)
                avg_macro_hidden = sum(
                    p["macro_hidden_size"] for p in bridge.train_macro_params
                ) / len(bridge.train_macro_params)
                avg_macro_lr = sum(
                    p["learning_rate"] for p in bridge.train_macro_params
                ) / len(bridge.train_macro_params)

                print("Average Macro MLP parameters:")
                print(f"  Dropout rate: {avg_macro_dropout:.3f}")
                print(f"  Hidden size: {avg_macro_hidden:.1f}")
                print(f"  Learning rate: {avg_macro_lr:.5f}")

            return metrics

        except Exception as e:
            print(f"❌ Error during model bridge execution: {e}")
            raise


def demo_individual_models() -> None:
    """Demonstrate individual model training for comparison."""
    print("\n🔬 Individual Model Demonstration")
    print("=" * 50)

    objectives = NeuralNetworkObjectives(subset_size=300)

    # Test micro model (CNN)
    print("\n1. Testing Micro Model (CNN):")
    micro_params = {
        "dropout_rate": 0.5,
        "hidden_size": 128,
        "learning_rate": 0.001,
    }
    micro_result = objectives.micro_objective(micro_params)
    print(f"Micro model result: {micro_result:.4f}")

    # Test macro model (MLP)
    print("\n2. Testing Macro Model (MLP):")
    macro_params = {
        "macro_hidden_size": 64,
        "macro_dropout_rate": 0.3,
        "learning_rate": 0.001,
    }
    macro_result = objectives.macro_objective(macro_params, target_value=0.5)
    print(f"Macro model result: {macro_result:.4f}")


def main() -> None:
    """Main function for neural network model bridge example."""
    parser = argparse.ArgumentParser(
        description="Neural Network Model Bridge with MNIST"
    )
    parser.add_argument("--demo", action="store_true", help="Run individual model demo")
    parser.add_argument(
        "--subset-size", type=int, default=500, help="MNIST subset size"
    )
    parser.add_argument(
        "--n-train", type=int, default=3, help="Number of training scenarios"
    )
    parser.add_argument(
        "--n-test", type=int, default=2, help="Number of test scenarios"
    )
    parser.add_argument(
        "--slow", action="store_true", help="Use slower but more accurate settings"
    )

    args = parser.parse_args()

    print("🧠 Neural Network Model Bridge Example")
    print("Using PyTorch and MNIST dataset")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    if args.demo:
        demo_individual_models()
        return

    # Run model bridge
    bridge = MNISTModelBridge(subset_size=args.subset_size, fast_mode=not args.slow)

    total_start_time = time.time()

    try:
        metrics = bridge.run_model_bridge(n_train=args.n_train, n_test=args.n_test)

        total_time = time.time() - total_start_time

        print("\n🏆 Overall Results:")
        print(f"Total execution time: {total_time:.2f} seconds")
        print("Prediction quality:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")

        print("\n📁 Results saved to: outputs/examples/neural_network_pytorch/")
        print("Check the generated plots for neural network parameter relationships!")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
