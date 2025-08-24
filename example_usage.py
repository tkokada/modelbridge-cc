"""Simple example demonstrating ModelBridge library usage."""

import numpy as np

from modelbridge import (
    ModelBridge,
    ParamConfig,
    ParamDict,
)
from modelbridge.utils.config_loader import create_param_config


def demo_micro_objective(params: ParamDict) -> float:
    """Demo micro model - expensive but accurate (sphere function)."""
    x = np.array([params[f"x_{i}"] for i in range(3)])
    p = np.array([params[f"p_{i}"] for i in range(3)])
    return float(np.sum(p * x**2))


def demo_macro_objective(params: ParamDict, target_value: float) -> float:
    """Demo macro model - fast but approximate (simplified linear model)."""
    x = np.array([params[f"x_{i}"] for i in range(3)])
    p = np.array([params[f"p_{i}"] for i in range(3)])
    # Simplified approximation (target_value used for compatibility but not in this simple demo)
    _ = target_value  # Acknowledge parameter to avoid warnings
    return float(np.sum(p * x) * 2.0)


def main() -> None:
    """Demonstrate basic ModelBridge usage."""
    print("ModelBridge Library Demo")
    print("=" * 50)

    # Create parameter configurations
    param_names = [f"x_{i}" for i in range(3)] + [f"p_{i}" for i in range(3)]
    param_types = ["float"] * 6
    param_ranges = [(-2.0, 2.0)] * 3 + [(0.1, 2.0)] * 3  # x in [-2,2], p in [0.1,2]

    param_config: ParamConfig = create_param_config(
        param_names, param_types, param_ranges
    )

    print(f"Parameter configuration: {len(param_config)} parameters")
    print("Parameters:", list(param_config.keys()))

    # Create ModelBridge
    bridge = ModelBridge(
        micro_objective=demo_micro_objective,
        macro_objective=demo_macro_objective,
        micro_param_config=param_config,
        macro_param_config=param_config,  # Same params for this demo
        regression_type="polynomial",
        optimizer_config={
            "storage": "sqlite:///outputs/databases/demo.db",
            "direction": "minimize",
            "sampler": "tpe",
            "seed": 42,
        },
        regression_config={"degree": 2},
    )

    print("\\nRunning model bridge pipeline...")

    # Run the complete pipeline
    metrics = bridge.run_full_pipeline(
        n_train=5,  # 5 training scenarios
        n_test=3,  # 3 test scenarios
        micro_trials_per_dataset=20,  # 20 trials per micro optimization
        macro_trials_per_dataset=20,  # 20 trials per macro optimization
        visualize=True,  # Generate plots
        output_dir="outputs/demos/demo_results",
    )

    print("\\nModel Bridge Results:")
    print("=" * 30)
    print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
    print(f"R-squared (R²): {metrics['r2']:.6f}")

    print("\\nResults saved to 'outputs/demos/demo_results/' directory")
    print("Check the generated plots for visualization of:")
    print("- Parameter relationships")
    print("- Prediction accuracy")

    print("\\nDemo completed successfully!")


if __name__ == "__main__":
    main()
