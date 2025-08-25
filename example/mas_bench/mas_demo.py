"""MAS-Bench data assimilation demo using ModelBridge (with mock simulation)."""

from pathlib import Path
import time
from typing import Any

import numpy as np

from modelbridge import ModelBridge
from modelbridge.utils.config_loader import create_param_config


class MockMASBenchSimulator:
    """Mock MAS-Bench simulator for demonstration purposes."""

    def __init__(self):
        """Initialize mock simulator."""
        self.simulation_count = 0

    def run_traffic_simulation(self, params: dict[str, float]) -> float:
        """Mock traffic simulation that mimics MAS-Bench behavior.

        Args:
            params: Traffic parameters (flow rates, densities, signal timing)

        Returns:
            Total travel time (objective to minimize)

        """
        self.simulation_count += 1

        # Extract traffic parameters
        sigma_1 = params.get("sigma_1", 0.5)  # Flow variance
        mu_1 = params.get("mu_1", 150.0)  # Mean flow rate
        sigma_2 = params.get("sigma_2", 0.3)  # Density variance
        mu_2 = params.get("mu_2", 50.0)  # Mean density

        # Mock complex traffic dynamics
        # This simulates the behavior of a detailed agent-based model

        # Flow efficiency (variance affects smoothness)
        flow_efficiency = np.exp(-sigma_1 * 2.0)

        # Congestion based on flow rate
        optimal_flow = 180.0
        congestion_penalty = abs(mu_1 - optimal_flow) / optimal_flow

        # Density impact (high density = more interactions)
        density_factor = min(mu_2 / 80.0, 1.0)
        variability_penalty = sigma_2 * density_factor

        # Complex interactions (simplified from agent-based model)
        intersection_delay = np.sin(mu_1 * np.pi / 300.0) * sigma_1 * 10.0
        lane_changing = density_factor * variability_penalty * 20.0
        signal_coordination = (1.0 - flow_efficiency) * 15.0

        # Base travel time + penalties
        base_time = 120.0  # Base travel time in seconds
        total_time = (
            base_time
            + congestion_penalty * 50.0
            + variability_penalty * 30.0
            + intersection_delay
            + lane_changing
            + signal_coordination
        )

        # Add realistic noise
        noise = np.random.normal(0, 3.0)

        # Simulate computation delay (like real MAS-Bench)
        time.sleep(0.05)  # Mock simulation time

        return float(total_time + noise)


def traffic_micro_objective(params: dict[str, Any]) -> float:
    """Detailed traffic micro-simulation (agent-based model)."""
    simulator = MockMASBenchSimulator()
    return simulator.run_traffic_simulation(params)


def traffic_macro_objective(params: dict[str, Any], target_value: float) -> float:
    """Fast traffic macro-model (flow-density relationships)."""
    _ = target_value  # Suppress unused parameter warning

    # Fast analytical traffic model
    sigma_1 = params.get("sigma_1", 0.5)
    mu_1 = params.get("mu_1", 150.0)
    sigma_2 = params.get("sigma_2", 0.3)
    mu_2 = params.get("mu_2", 50.0)

    # Simplified traffic flow equations
    base_time = 120.0
    flow_factor = mu_1 / 300.0
    density_factor = mu_2 / 100.0
    variance_penalty = (sigma_1 + sigma_2) * 15.0

    # Linear approximation of traffic dynamics
    travel_time = (
        base_time + flow_factor * 40.0 + density_factor * 25.0 + variance_penalty
    )

    return float(travel_time)


def main():
    """Run MAS-Bench data assimilation demo."""
    print("🚗 MAS-Bench Data Assimilation Demo")
    print("===================================")
    print("Micro: Agent-based traffic simulation (mocked)")
    print("Macro: Flow-density traffic model")
    print("Goal: Optimize traffic parameters through data assimilation")
    print()

    # Traffic parameter configuration (realistic ranges)
    param_config = create_param_config(
        param_names=["sigma_1", "mu_1", "sigma_2", "mu_2"],
        param_types=["float", "float", "float", "float"],
        param_ranges=[
            (0.0, 1.0),  # sigma_1: Traffic flow variance
            (50.0, 300.0),  # mu_1: Mean flow rate (vehicles/hour)
            (0.0, 1.0),  # sigma_2: Density variance
            (10.0, 100.0),  # mu_2: Mean density (vehicles/km)
        ],
    )

    # Create output directory
    output_dir = Path("mas_demo_results")
    output_dir.mkdir(exist_ok=True)

    print("🏗️ Setting up ModelBridge for traffic data assimilation...")
    print("📊 Parameter space:")
    print("   σ₁: Traffic flow variance [0.0, 1.0]")
    print("   μ₁: Mean flow rate [50, 300] vehicles/hour")
    print("   σ₂: Density variance [0.0, 1.0]")
    print("   μ₂: Mean density [10, 100] vehicles/km")
    print()

    # Create ModelBridge for traffic optimization
    bridge = ModelBridge(
        micro_objective=traffic_micro_objective,
        macro_objective=traffic_macro_objective,
        micro_param_config=param_config,
        macro_param_config=param_config,
        regression_type="polynomial",
        optimizer_config={
            "storage": f"sqlite:///{output_dir}/traffic_data_assimilation.db",
            "direction": "minimize",
            "sampler": "tpe",
            "seed": 42,
        },
        regression_config={"degree": 2},
    )

    print("🚦 Running traffic data assimilation pipeline...")
    print("Phase 1: Training on multiple traffic scenarios...")
    start_time = time.time()

    # Run data assimilation pipeline
    metrics = bridge.run_full_pipeline(
        n_train=4,  # 4 traffic scenarios for training
        n_test=2,  # 2 scenarios for testing
        micro_trials_per_dataset=10,  # Limited trials for demo
        macro_trials_per_dataset=10,
        visualize=True,
        output_dir=str(output_dir),
    )

    elapsed_time = time.time() - start_time

    print("\\n🎯 Traffic Data Assimilation Results:")
    print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")
    print(f"🔄 Simulations run: ~{(4 + 2) * (10 + 10)} mock traffic scenarios")
    print("📊 Parameter prediction quality:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   R²:  {metrics['r2']:.6f}")

    print("\\n🚗 Traffic Optimization Insights:")
    print("✓ Successfully bridged agent-based simulation → flow model")
    print("✓ Can predict optimal traffic parameters from fast calculations")
    print("✓ Data assimilation enables real-time traffic optimization")

    print(f"\\n📁 Results saved to: {output_dir}/")
    print(
        "🖼️  Traffic visualizations: parameter_relationships.png, prediction_accuracy.png"
    )
    print("💾 Traffic data: CSV files with optimal traffic parameters")
    print("🗄️  Optimization database: traffic_data_assimilation.db")

    print("\\n🏆 MAS-Bench data assimilation demo completed successfully!")
    print("📝 For full MAS-Bench integration, see: hpopt_data_assimilation.py")


if __name__ == "__main__":
    main()
