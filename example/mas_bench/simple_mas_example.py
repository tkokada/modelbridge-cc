"""Simple MAS-Bench-inspired data assimilation example using ModelBridge."""

from pathlib import Path
import time

import numpy as np

from modelbridge import ModelBridge
from modelbridge.utils.config_loader import create_param_config


def traffic_micro_simulation(params: dict) -> float:
    """Detailed traffic micro-simulation (mocked).

    Simulates a detailed agent-based traffic model.
    In reality, this would call MAS-Bench.jar with specific parameters.
    """
    # Extract traffic parameters
    sigma_1 = params["sigma_1"]  # Flow variance
    mu_1 = params["mu_1"]  # Mean flow rate
    sigma_2 = params["sigma_2"]  # Density variance
    mu_2 = params["mu_2"]  # Mean density

    # Mock complex traffic simulation
    # In reality: agent interactions, lane changing, signal timing, etc.
    flow_efficiency = 1.0 / (1.0 + sigma_1 * 0.1)
    capacity_usage = min(mu_1 / 300.0, 1.0)
    density_impact = 1.0 - min(mu_2 / 100.0, 1.0) * sigma_2

    # Complex nonlinear traffic dynamics (simplified)
    congestion = (
        np.exp(-((mu_1 - 150) ** 2) / (2 * sigma_1**2 * 100)) if sigma_1 > 0.01 else 0
    )
    bottleneck = np.sin(mu_2 * np.pi / 50) * sigma_2 * 0.5

    # Total travel time (objective to minimize)
    travel_time = (
        100.0  # Base travel time
        + 50.0 * (1.0 - flow_efficiency)  # Flow inefficiency penalty
        + 30.0 * (1.0 - capacity_usage)  # Under-utilization penalty
        + 40.0 * (1.0 - density_impact)  # High density penalty
        + 20.0 * congestion  # Congestion penalty
        + 10.0 * bottleneck  # Bottleneck penalty
    )

    # Add some noise for realism
    noise = np.random.normal(0, 2.0)

    return float(travel_time + noise)


def traffic_macro_model(params: dict, target_value: float) -> float:
    """Fast traffic macro-model approximation.

    Uses simplified flow-density relationships.
    """
    _ = target_value  # Suppress unused parameter warning

    sigma_1 = params["sigma_1"]
    mu_1 = params["mu_1"]
    sigma_2 = params["sigma_2"]
    mu_2 = params["mu_2"]

    # Simple linear traffic flow model
    base_time = 100.0
    flow_factor = mu_1 / 300.0
    density_factor = mu_2 / 100.0
    variance_penalty = (sigma_1 + sigma_2) * 10.0

    # Simplified travel time estimation
    travel_time = (
        base_time + 50.0 * flow_factor + 30.0 * density_factor + variance_penalty
    )

    return float(travel_time)


def main():
    """Run traffic simulation data assimilation example."""
    print("🚗 MAS-Bench-Inspired Traffic Data Assimilation")
    print("==============================================")
    print("Micro: Detailed agent-based traffic simulation")
    print("Macro: Fast flow-density traffic model")
    print()

    # Traffic parameter configuration
    param_config = create_param_config(
        param_names=["sigma_1", "mu_1", "sigma_2", "mu_2"],
        param_types=["float", "float", "float", "float"],
        param_ranges=[
            (0.0, 1.0),  # sigma_1: Flow variance
            (0.0, 300.0),  # mu_1: Mean flow rate (vehicles/hour)
            (0.0, 1.0),  # sigma_2: Density variance
            (0.0, 100.0),  # mu_2: Mean density (vehicles/km)
        ],
    )

    # Create output directory
    output_dir = Path("mas_bench_results")
    output_dir.mkdir(exist_ok=True)

    print("🏗️ Setting up ModelBridge for traffic optimization...")

    # Create ModelBridge for traffic data assimilation
    bridge = ModelBridge(
        micro_objective=traffic_micro_simulation,
        macro_objective=traffic_macro_model,
        micro_param_config=param_config,
        macro_param_config=param_config,
        regression_type="polynomial",
        optimizer_config={
            "storage": f"sqlite:///{output_dir}/traffic_optimization.db",
            "direction": "minimize",
            "sampler": "tpe",
            "seed": 42,
        },
        regression_config={"degree": 2},
    )

    print("🚦 Running traffic data assimilation pipeline...")
    start_time = time.time()

    # Run data assimilation pipeline
    metrics = bridge.run_full_pipeline(
        n_train=3,  # 3 traffic scenarios for training
        n_test=2,  # 2 scenarios for testing
        micro_trials_per_dataset=8,  # Limited trials for demo
        macro_trials_per_dataset=8,
        visualize=True,
        output_dir=str(output_dir),
    )

    elapsed_time = time.time() - start_time

    print("\\n🎯 Traffic Data Assimilation Results:")
    print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")
    print("📊 Parameter prediction quality:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   R²:  {metrics['r2']:.6f}")

    print("\\n🚗 Traffic Optimization Summary:")
    print("Successfully bridged detailed traffic simulation → flow model")
    print("Bridge can predict optimal traffic parameters from fast flow calculations")

    print(f"\\n📁 Results saved to: {output_dir}/")
    print("🖼️  Visualizations: parameter_relationships.png, prediction_accuracy.png")
    print("💾 Data: CSV files with traffic optimization parameters")
    print("🗄️  Database: traffic_optimization.db with optimization history")

    print("\\n🏆 MAS-Bench-inspired data assimilation completed successfully!")


if __name__ == "__main__":
    main()
