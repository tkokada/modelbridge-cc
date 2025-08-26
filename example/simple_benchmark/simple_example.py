"""Simple benchmark example using ModelBridge library."""

from pathlib import Path

import numpy as np

from modelbridge import ModelBridge
from modelbridge.utils.config_loader import create_param_config


def sphere_function(params: dict) -> float:
    """Sphere function - simple quadratic optimization benchmark.

    A simple, convex optimization function commonly used for testing optimization
    algorithms. The global minimum is at the origin.

    Args:
        params (dict): Parameter dictionary containing:
            - x1, x2 (float): Input variables to optimize
            - p1, p2 (float): Scaling coefficients

    Returns:
        float: Function value computed as sum(p * x²)

    Note:
        This is a unimodal function with a single global minimum, making it
        relatively easy to optimize and suitable as a macro model approximation.

    """
    x = np.array([params["x1"], params["x2"]])
    p = np.array([params["p1"], params["p2"]])
    return float(np.sum(p * x**2))


def rastrigin_function(params: dict) -> float:
    """Rastrigin function - complex multimodal optimization benchmark.

    A highly multimodal function with many local minima, making it challenging
    for optimization algorithms. Used to test micro model complexity.

    Args:
        params (dict): Parameter dictionary containing:
            - x1, x2 (float): Input variables to optimize
            - p1, p2 (float): Scaling coefficients

    Returns:
        float: Function value computed as 10*n + sum(p*(x² - 10*cos(2π*x)))

    Note:
        This function has many local minima but a single global minimum at the origin.
        The cosine term creates the multimodal landscape that makes optimization difficult.

    """
    x = np.array([params["x1"], params["x2"]])
    p = np.array([params["p1"], params["p2"]])
    return float(10 * len(x) + np.sum(p * x**2 - 10.0 * np.cos(2 * np.pi * x)))


def linear_macro(params: dict, target_value: float) -> float:
    """Linear macro approximation for fast optimization.

    A simplified linear function that approximates more complex micro models
    with much faster evaluation time. Used as the macro model in bridging.

    Args:
        params (dict): Parameter dictionary containing:
            - x1, x2 (float): Input variables
            - p1, p2 (float): Linear coefficients
        target_value (float): Target value from micro model (unused in this implementation)

    Returns:
        float: Linear function value computed as sum(p * x)

    Note:
        This linear approximation trades accuracy for speed, making it suitable
        as a fast macro model that can be optimized quickly.

    """
    _ = target_value  # Suppress unused parameter warning
    x = np.array([params["x1"], params["x2"]])
    p = np.array([params["p1"], params["p2"]])
    return float(np.sum(p * x))


def main() -> None:
    """Run simple benchmark example demonstrating ModelBridge.

    Executes two model bridging scenarios:
    1. Sphere (micro) → Linear (macro) bridging with polynomial regression
    2. Rastrigin (micro) → Linear (macro) bridging with linear regression

    The example shows how to bridge complex optimization functions with simpler
    approximations, demonstrating the core ModelBridge functionality with
    mathematical benchmark functions.

    Output:
        Creates visualization plots and CSV data files in simple_benchmark_results/
        directory, including parameter relationships and prediction accuracy analysis.

    Example:
        >>> main()
        ModelBridge Simple Benchmark Example
        =====================================
        1. Sphere (micro) → Linear (macro) Bridge
        --------------------------------------------------
        Results - MSE: 0.3717, MAE: 0.4608, R²: -2109.4738
        ...
        ✅ Simple Benchmark Complete!

    """
    print("ModelBridge Simple Benchmark Example")
    print("=====================================")

    # Create parameter configuration
    param_config = create_param_config(
        param_names=["x1", "x2", "p1", "p2"],
        param_types=["float", "float", "float", "float"],
        param_ranges=[(-2.0, 2.0), (-2.0, 2.0), (0.1, 1.0), (0.1, 1.0)],
    )

    # Create output directory
    output_dir = Path("simple_benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\\n1. Sphere (micro) → Linear (macro) Bridge")
    print("-" * 50)

    # Create ModelBridge for sphere → linear
    bridge_sphere = ModelBridge(
        micro_objective=sphere_function,
        macro_objective=linear_macro,
        micro_param_config=param_config,
        macro_param_config=param_config,
        regression_type="polynomial",
        optimizer_config={
            "storage": f"sqlite:///{output_dir}/sphere_linear.db",
            "seed": 42,
            "direction": "minimize",
            "sampler": "tpe",
        },
        regression_config={"degree": 2},
    )

    # Run pipeline
    sphere_metrics = bridge_sphere.run_full_pipeline(
        n_train=3,
        n_test=2,
        micro_trials_per_dataset=10,
        macro_trials_per_dataset=10,
        visualize=True,
        output_dir=str(output_dir),
    )

    print(
        f"Results - MSE: {sphere_metrics['mse']:.4f}, MAE: {sphere_metrics['mae']:.4f}, R²: {sphere_metrics['r2']:.4f}"
    )

    print("\\n2. Rastrigin (micro) → Linear (macro) Bridge")
    print("-" * 50)

    # Create ModelBridge for rastrigin → linear
    bridge_rastrigin = ModelBridge(
        micro_objective=rastrigin_function,
        macro_objective=linear_macro,
        micro_param_config=param_config,
        macro_param_config=param_config,
        regression_type="linear",
        optimizer_config={
            "storage": f"sqlite:///{output_dir}/rastrigin_linear.db",
            "seed": 123,
            "direction": "minimize",
            "sampler": "random",
        },
    )

    # Run pipeline
    rastrigin_metrics = bridge_rastrigin.run_full_pipeline(
        n_train=2,
        n_test=2,
        micro_trials_per_dataset=8,
        macro_trials_per_dataset=8,
        visualize=True,
        output_dir=str(output_dir),
    )

    print(
        f"Results - MSE: {rastrigin_metrics['mse']:.4f}, MAE: {rastrigin_metrics['mae']:.4f}, R²: {rastrigin_metrics['r2']:.4f}"
    )

    print("\\n✅ Simple Benchmark Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("🖼️  Plots: parameter_relationships.png, prediction_accuracy.png")
    print("💾 Data: *.csv files with parameter datasets")


if __name__ == "__main__":
    main()
