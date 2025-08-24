"""Refactored MAS-Bench data assimilation using ModelBridge library."""

import csv
import os
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd

from modelbridge import ModelBridge
from modelbridge.utils.data_manager import DataManager
from modelbridge.utils.visualization import Visualizer


class MASBenchSimulator:
    """Wrapper for MAS-Bench traffic simulation."""

    def __init__(self, jar_path: str = "MAS-Bench.jar"):
        """Initialize MAS-Bench simulator.

        Args:
            jar_path: Path to MAS-Bench.jar file
        """
        self.jar_path = jar_path

    def load_agent_size(self, model: str) -> tuple[int, int, int]:
        """Load agent size configuration from shell script.

        Args:
            model: Model name

        Returns:
            Tuple of (naive_agents, rational_agents, ruby_agents)
        """
        script_path = f"masbench-resources/Dataset/{model}/agent_size.sh"
        if not os.path.exists(script_path):
            print(f"❌ ERROR: agent_size.sh not found at: {script_path}")
            print("💡 This example requires MAS-Bench simulation resources.")
            print("📚 For a working example, try: python simple_mas_example.py")
            sys.exit(1)

        shell_script = (
            f"source {script_path} && echo $NAIVE_AGENT $RATIONAL_AGENT $RUBY_AGENT"
        )
        result = subprocess.check_output(["bash", "-c", shell_script], text=True)
        naive, rational, ruby = map(int, result.strip().split())
        return naive, rational, ruby

    def run_simulation(self, model: str, result_path: str, input_csv: str) -> None:
        """Run MAS-Bench simulation.

        Args:
            model: Model name
            result_path: Path to store results
            input_csv: Path to input CSV file
        """
        try:
            subprocess.run(
                ["java", "-jar", self.jar_path, model, result_path, input_csv],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("Simulation timed out after 30 seconds")
            return False
        except Exception as e:
            print(f"Unexpected simulation error: {e}")
            return False

    def save_input_parameters(
        self,
        trial_number: int,
        total_agents: int,
        sigma: list[float],
        mu: list[float],
        pi: list[float],
        header: list[str],
        input_path: str,
    ) -> None:
        """Save input parameters to CSV file."""
        os.makedirs(input_path, exist_ok=True)
        input_csv = os.path.join(input_path, f"input_parameter_{trial_number}.csv")

        row = []
        for i in range(total_agents):
            row.extend([sigma[i], mu[i], pi[i]])

        with open(input_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # csv.writer doesn't have writeheader(), so we write header manually
            header = []
            for i in range(total_agents):
                header.extend([f"sigma_{i}", f"mu_{i}", f"pi_{i}"])
            writer.writerow(header)
            writer.writerow(row)

    def extract_fitness_score(
        self, result_path: str, trial_number: int, output_path: str
    ) -> float:
        """Extract fitness score from simulation results.

        Args:
            result_path: Path to simulation results
            trial_number: Trial number
            output_path: Path to save output error CSV

        Returns:
            Fitness score (AllError)
        """
        os.makedirs(output_path, exist_ok=True)
        source_path = os.path.join(result_path, "analyze", "Fitness.csv")
        target_path = os.path.join(output_path, f"output_error_{trial_number}.csv")

        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
        else:
            print(f"[Warning] Fitness file not found: {source_path}")
            return float("inf")

        try:
            with open(target_path) as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                row = next(reader, None)  # Read data row
                if row and len(row) > 0:
                    return float(row[0])  # AllError
                else:
                    print(f"[Warning] {target_path} is empty or malformed.")
                    return float("inf")
        except Exception as e:
            print(f"[Error] Failed to read {target_path}: {e}")
            return float("inf")


class MASBenchBridge:
    """MAS-Bench model bridge implementation."""

    def __init__(self, model: str):
        """Initialize MAS-Bench bridge.

        Args:
            model: Model name for simulation
        """
        self.model = model
        self.simulator = MASBenchSimulator()
        self.data_manager = DataManager()
        self.visualizer = Visualizer()

        # Load agent configuration
        self.naive_agents, self.rational_agents, self.ruby_agents = (
            self.simulator.load_agent_size(model)
        )
        self.total_agents = self.naive_agents + self.rational_agents + self.ruby_agents

        # Create parameter configuration
        self.param_config = self._create_parameter_config()

    def _create_parameter_config(self) -> dict[str, dict[str, Any]]:
        """Create parameter configuration for optimization."""
        param_config = {}

        # Add parameters for each agent type
        for i in range(self.naive_agents):
            param_config[f"sigma_naive{i}"] = {"type": "float", "low": 0.0, "high": 1.0}
            param_config[f"mu_naive{i}"] = {"type": "float", "low": 0.0, "high": 1.0}
            if i < self.total_agents - 1:  # Don't add last pi (will be computed)
                param_config[f"pi_naive{i}"] = {
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0,
                }

        for i in range(self.rational_agents):
            param_config[f"sigma_rational{i}"] = {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
            }
            param_config[f"mu_rational{i}"] = {"type": "float", "low": 0.0, "high": 1.0}
            if len(param_config) // 3 < self.total_agents - 1:
                param_config[f"pi_rational{i}"] = {
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0,
                }

        for i in range(self.ruby_agents):
            param_config[f"sigma_ruby{i}"] = {"type": "float", "low": 0.0, "high": 1.0}
            param_config[f"mu_ruby{i}"] = {"type": "float", "low": 0.0, "high": 1.0}
            if len(param_config) // 3 < self.total_agents - 1:
                param_config[f"pi_ruby{i}"] = {"type": "float", "low": 0.0, "high": 1.0}

        return param_config

    def _convert_params_to_simulation_format(
        self, params: dict[str, Any]
    ) -> tuple[list[float], list[float], list[float], list[str]]:
        """Convert optimization parameters to simulation format."""
        sigma, mu, pi = [], [], []
        header = []

        # Extract parameters by agent type
        for i in range(self.naive_agents):
            sigma.append(params[f"sigma_naive{i}"])
            mu.append(params[f"mu_naive{i}"])
            if f"pi_naive{i}" in params:
                pi.append(params[f"pi_naive{i}"])
            header.extend([f"sigma_naive{i}", f"mu_naive{i}", f"pi_naive{i}"])

        for i in range(self.rational_agents):
            sigma.append(params[f"sigma_rational{i}"])
            mu.append(params[f"mu_rational{i}"])
            if f"pi_rational{i}" in params:
                pi.append(params[f"pi_rational{i}"])
            header.extend([f"sigma_rational{i}", f"mu_rational{i}", f"pi_rational{i}"])

        for i in range(self.ruby_agents):
            sigma.append(params[f"sigma_ruby{i}"])
            mu.append(params[f"mu_ruby{i}"])
            if f"pi_ruby{i}" in params:
                pi.append(params[f"pi_ruby{i}"])
            header.extend([f"sigma_ruby{i}", f"mu_ruby{i}", f"pi_ruby{i}"])

        # Calculate last pi to ensure sum = 1
        if self.total_agents > 1:
            pi_sum = sum(pi)
            pi_last = max(0.0, min(1.0, 1.0 - pi_sum))
            pi.append(pi_last)
        else:
            pi = [1.0]

        # Apply scaling
        sigma = [s * 97.0 + 3.0 for s in sigma]  # [3, 100]
        mu = [m * 300.0 for m in mu]  # [0, 300]

        return sigma, mu, pi, header

    def create_micro_objective(self, study_name: str, max_trials: int, seed: int):
        """Create micro model objective function."""

        def objective(params: dict[str, Any]) -> float:
            """Micro model objective function."""
            # Convert parameters to simulation format
            sigma, mu, pi, header = self._convert_params_to_simulation_format(params)

            # Create unique trial identifier
            trial_id = hash(frozenset(params.items())) % 1000000

            # Set up paths
            result_path = f"results/{self.model}/{study_name}/{trial_id}"
            input_path = f"results/{self.model}/{study_name}/input_parameters"
            input_csv = f"{input_path}/input_parameter_{trial_id}.csv"
            output_path = f"results/{self.model}/{study_name}/output_errors"

            # Save input parameters
            self.simulator.save_input_parameters(
                trial_id, self.total_agents, sigma, mu, pi, header, input_path
            )

            # Run simulation
            try:
                simulation_success = self.simulator.run_simulation(
                    self.model, result_path, input_csv
                )
                if simulation_success:
                    fitness = self.simulator.extract_fitness_score(
                        result_path, trial_id, output_path
                    )
                    return fitness
                else:
                    print(f"Simulation failed for trial {trial_id}")
                    return float("inf")
            except Exception as e:
                print(f"Simulation failed: {e}")
                return float("inf")

        return objective

    def create_macro_objective(self, study_name: str, max_trials: int, seed: int):
        """Create macro model objective function (simplified model)."""

        def objective(params: dict[str, Any], target_value: float) -> float:
            """Macro model objective function - simplified approximation."""
            # Simple approximation: weighted sum of parameters
            # In practice, this would be a faster surrogate model

            sigma_sum = sum(params[k] for k in params if k.startswith("sigma"))
            mu_sum = sum(params[k] for k in params if k.startswith("mu"))
            pi_sum = sum(params[k] for k in params if k.startswith("pi"))

            # Simple approximation formula
            approx_value = sigma_sum * 50 + mu_sum * 150 + pi_sum * 100

            return approx_value

        return objective

    def create_training_ground_truth(
        self,
        training_result_path: str,
        training_macro_model: str,
        macro_model: str,
        n_scenarios: int,
    ) -> None:
        """Create training ground truth datasets."""
        for i in range(n_scenarios):
            training_result = f"{training_result_path}/{i}"
            training_macro_gt = f"masbench-resources/Dataset/{training_macro_model}-{i}"
            macro_settings = f"masbench-resources/Dataset/{macro_model}"

            print(f"Creating ground truth for scenario {i}")

            os.makedirs(training_macro_gt, exist_ok=True)

            # Copy configuration files
            for filename in [
                "agent_size.sh",
                "model.properties",
                "Scenario_Guidance.json",
            ]:
                src = f"{macro_settings}/{filename}"
                dst = f"{training_macro_gt}/{filename}"
                if os.path.exists(src):
                    shutil.copy(src, dst)

            # Copy and scale simulation results
            goal_flow_src = f"{training_result}/analyze/simulationGoalFlow.csv"
            goal_flow_dst = f"{training_macro_gt}/observationGoalFlow.csv"

            if os.path.exists(goal_flow_src):
                # Scale goal flow to 1/10
                df = pd.read_csv(goal_flow_src)
                last_col = df.columns[-1]
                df[last_col] = df[last_col] / 10
                df.to_csv(goal_flow_dst, index=False)

            # Copy start flow
            start_flow_src = f"{training_result}/property/simulationStartFlow.csv"
            start_flow_dst = f"{training_macro_gt}/observationStartFlow.csv"

            if os.path.exists(start_flow_src):
                df = pd.read_csv(start_flow_src)
                df.to_csv(start_flow_dst, index=False)

            # Copy agent GPS data (scaled to 1/10)
            for j in range(30):
                for r in range(3):
                    src = f"{training_result}/analyze/simulationAgent_R{r + 1}_{j * 10}.csv"
                    dst = f"{training_macro_gt}/observationAgent_R{r + 1}_{j * 10}.csv"
                    if os.path.exists(src):
                        shutil.copy(src, dst)

    def run_data_assimilation_pipeline(
        self,
        function_id: str = "1-2",
        n_micro_scenarios: int = 4,
        n_macro_trials: int = 5,
        study_method: str = "cmaes",
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run complete data assimilation pipeline."""

        # Phase 1: Create training data (micro model optimization)
        print("Phase 1: Creating training data...")
        micro_model = f"FL{function_id}"
        macro_model = f"FS{function_id}"

        # Create model bridge for training data generation
        micro_study_name = f"{micro_model}_rs_{n_micro_scenarios}_{seed}"

        bridge = ModelBridge(
            micro_objective=self.create_micro_objective(
                micro_study_name, n_micro_scenarios, seed
            ),
            macro_objective=self.create_macro_objective(
                micro_study_name, n_micro_scenarios, seed
            ),
            micro_param_config=self.param_config,
            macro_param_config=self.param_config,
            regression_type="polynomial",
            optimizer_config={
                "storage": "sqlite:///mas_bench_results.db",
                "direction": "minimize",
                "sampler": "random",
                "seed": seed,
            },
        )

        # Phase 2: Data assimilation for macro model training
        print("Phase 2: Training macro models...")

        # Phase 3: Model bridge training
        print("Phase 3: Training model bridge...")

        metrics = bridge.run_full_pipeline(
            n_train=n_micro_scenarios,
            n_test=2,
            micro_trials_per_dataset=50,
            macro_trials_per_dataset=50,
            visualize=True,
            output_dir=f"mas_bench_results/Bridge-{micro_model}-{macro_model}",
        )

        return metrics


def main():
    """Main function for MAS-Bench data assimilation."""
    # Configuration
    function_id = "1-2"
    model = f"FL{function_id}"

    # Create MAS-Bench bridge
    mas_bridge = MASBenchBridge(model)

    # Run data assimilation pipeline
    results = mas_bridge.run_data_assimilation_pipeline(
        function_id=function_id,
        n_micro_scenarios=4,
        n_macro_trials=5,
        study_method="cmaes",
        seed=42,
    )

    print("\\nMAS-Bench Data Assimilation Results:")
    print(f"MSE: {results['mse']:.6f}")
    print(f"MAE: {results['mae']:.6f}")
    print(f"R²: {results['r2']:.6f}")

    print("\\nMAS-Bench data assimilation completed successfully!")


if __name__ == "__main__":
    main()
