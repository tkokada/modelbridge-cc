# MAS-Bench Data Assimilation Example

This example demonstrates ModelBridge integration with MAS-Bench traffic simulation system for traffic flow optimization through data assimilation.

## Overview

This example shows how to apply model bridging to real-world traffic simulation:
- **Micro Model**: Detailed MAS-Bench traffic simulation - accurate but computationally expensive
- **Macro Model**: Simplified traffic flow model - fast approximation
- **Bridge**: Data assimilation technique to map between simulation parameters

## Prerequisites

### Java Runtime Environment
MAS-Bench requires Java to run the simulation engine:
```bash
# Check Java installation
java -version

# Install Java if needed (example for macOS)
brew install openjdk@11
```

### MAS-Bench JAR File
- `MAS-Bench.jar` - Traffic simulation engine (included)
- Download from: https://github.com/MAS-Bench/MAS-Bench

### MAS-Bench Resources (for Full Integration)
For full MAS-Bench integration (`hpopt_data_assimilation.py`), you need the simulation resources:

```bash
# Create the required directory structure
mkdir -p masbench-resources/Dataset/FL1-2

# Create agent configuration file
cat > masbench-resources/Dataset/FL1-2/agent_size.sh << 'EOF'
#!/bin/bash
# MAS-Bench agent configuration for FL1-2 scenario
export NAIVE_AGENT=3
export RATIONAL_AGENT=2
export RUBY_AGENT=1
EOF

# Make it executable
chmod +x masbench-resources/Dataset/FL1-2/agent_size.sh

# Create basic model properties file
cat > masbench-resources/Dataset/FL1-2/model.properties << 'EOF'
# MAS-Bench model configuration for FL1-2
simulation.time.step=1.0
simulation.max.time=300.0
traffic.flow.capacity=2000
road.network.type=highway
EOF

# Create scenario guidance file
cat > masbench-resources/Dataset/FL1-2/Scenario_Guidance.json << 'EOF'
{
  "scenario": "FL1-2",
  "description": "Highway traffic scenario with moderate flow",
  "agents": {
    "naive": 3,
    "rational": 2,
    "ruby": 1
  },
  "parameters": {
    "flow_range": [50, 300],
    "density_range": [10, 100],
    "variance_range": [0.0, 1.0]
  }
}
EOF

# Verify the structure
ls -la masbench-resources/Dataset/FL1-2/
```

**Note**: The demo versions (`mas_demo.py`, `simple_mas_example.py`) work without these resources and use mock simulation instead.

## Files

- `mas_demo.py` - Traffic data assimilation demo with mock simulation (recommended)
- `simple_mas_example.py` - Alternative simple traffic optimization example
- `hpopt_data_assimilation.py` - Full MAS-Bench integration (requires complete setup)
- `MAS-Bench.jar` - Traffic simulation engine
- `README.md` - This documentation

## Traffic Simulation Models

### Micro Model (MAS-Bench)
- **Agent-based simulation**: Individual vehicle behavior modeling
- **High fidelity**: Realistic traffic dynamics, lane changing, intersection behavior
- **Computational cost**: High (several seconds per evaluation)
- **Parameters**: Traffic flow rates, signal timing, vehicle densities

### Macro Model (Simplified)
- **Aggregate flow model**: Simplified traffic equations
- **Low fidelity**: Basic flow-density relationships
- **Computational cost**: Low (milliseconds per evaluation)
- **Parameters**: Average flow rates, capacity estimates

## Configuration

The script automatically configures:

```python
# Traffic scenarios
scenarios = ["highway", "urban", "intersection"]

# Parameter space
param_config = {
    "sigma_1": {"type": "float", "low": 0.0, "high": 1.0},    # Flow variance
    "mu_1": {"type": "float", "low": 0.0, "high": 300.0},     # Mean flow rate
    "sigma_2": {"type": "float", "low": 0.0, "high": 1.0},    # Density variance
    "mu_2": {"type": "float", "low": 0.0, "high": 100.0},     # Mean density
}

# Optimization settings
n_train = 3           # Training scenarios
n_test = 2            # Test scenarios
trials_per_dataset = 50  # Optimization trials per scenario
```

## Usage

### Option 1: Traffic Data Assimilation Demo (Recommended)
```bash
# Run traffic data assimilation demo with mock simulation
uv run python mas_demo.py

# View results
ls -la mas_demo_results/
```

### Option 2: Alternative Simple Example
```bash
# Run simple traffic optimization (different implementation)
uv run python simple_mas_example.py

# View results
ls -la mas_bench_results/
```

### Option 3: Full MAS-Bench Integration (Advanced)
```bash
# First, set up the MAS-Bench resources (see Prerequisites section above)
# Then run with demo mode for faster execution:
uv run python hpopt_data_assimilation.py --demo

# Or run full integration (requires complete setup):
uv run python hpopt_data_assimilation.py

# Custom parameters:
uv run python hpopt_data_assimilation.py --n-train 3 --n-test 2 --trials 10

# Monitor progress (if available)
tail -f outputs/logs/mas_bench.log
```

### Using Makefile (from project root)
```bash
make run-mas-example    # Runs mas_demo.py (recommended)
```

## Quick Start Guide

### 1. First Time Users
```bash
# Start with the demo (no Java/MAS-Bench required)
cd example/mas_bench
uv run python mas_demo.py
```

### 2. Understanding Results
```bash
# Check generated files
ls -la mas_demo_results/

# View optimization database
sqlite3 mas_demo_results/traffic_data_assimilation.db ".tables"

# Inspect parameter data
head mas_demo_results/train_macro_params.csv
```

### 3. Advanced Usage
```bash
# Try alternative implementation
uv run python simple_mas_example.py

# For full MAS-Bench (requires setup)
uv run python hpopt_data_assimilation.py
```

## Workflow

### 1. Data Preparation
- Creates traffic scenarios with different conditions
- Sets up MAS-Bench simulation environments
- Configures parameter ranges for optimization

### 2. Training Phase
```
For each training scenario:
  1. Run micro model (MAS-Bench) optimization
  2. Run macro model optimization to match results
  3. Store parameter mappings
Train regression model: macro_params → micro_params
```

### 3. Testing Phase
```
For each test scenario:
  1. Run micro model optimization (ground truth)
  2. Run macro model optimization
  3. Use regression to predict micro parameters
  4. Compare predicted vs actual parameters
```

### 4. Analysis
- Generate parameter relationship plots
- Calculate prediction accuracy metrics
- Export results for further analysis

## Expected Results

### Performance Metrics
- **MSE**: Mean squared error between predicted and actual parameters
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination (correlation strength)

### Output Files

**Demo Version (mas_demo.py)**:
```
mas_demo_results/
├── traffic_data_assimilation.db      # Optimization history
├── parameter_relationships.png       # Traffic parameter relationships
├── prediction_accuracy.png           # Prediction quality analysis
├── predicted_micro_params.csv        # Predicted optimal parameters
├── test_macro_params.csv            # Test macro parameters
├── test_micro_params.csv            # Test micro parameters
├── train_macro_params.csv           # Training macro parameters
└── train_micro_params.csv           # Training micro parameters
```

**Alternative Version (simple_mas_example.py)**:
```
mas_bench_results/
├── traffic_optimization.db          # Optimization database
├── parameter_relationships.png      # Parameter analysis plots
├── prediction_accuracy.png          # Accuracy visualizations
└── *.csv                           # Parameter data files
```

## MAS-Bench Integration Details

### Demo Implementation (mas_demo.py)
The demo uses sophisticated mock simulation that captures key traffic dynamics:

**Mock Agent-Based Model (Micro)**:
- **Flow efficiency**: Variance affects traffic smoothness
- **Congestion modeling**: Optimal flow rate around 180 vehicles/hour
- **Density interactions**: High density increases vehicle interactions
- **Complex dynamics**: Intersection delays, lane changing, signal coordination
- **Realistic noise**: Stochastic elements for real-world behavior

**Flow-Density Model (Macro)**:
- **Linear approximation**: Simplified traffic flow equations
- **Fast evaluation**: Analytical computation in milliseconds
- **Parameter mapping**: Base time + flow/density factors + variance penalties

### Traffic Parameters
- **σ₁ (sigma_1)**: Traffic flow variance [0.0, 1.0] - affects flow smoothness
- **μ₁ (mu_1)**: Mean flow rate [50, 300] vehicles/hour - primary traffic volume
- **σ₂ (sigma_2)**: Density variance [0.0, 1.0] - vehicle distribution variability
- **μ₂ (mu_2)**: Mean density [10, 100] vehicles/km - average vehicle density

### Real MAS-Bench Integration
For full MAS-Bench integration (`hpopt_data_assimilation.py`):

**Simulation Parameters**:
- **Agent populations**: Naive, rational, and ruby agents with different behaviors
- **Traffic scenarios**: Highway, urban, intersection configurations
- **Environmental factors**: Weather, incidents, construction zones
- **Optimization space**: Multi-dimensional parameter space with hundreds of variables

**Data Assimilation Process**:
1. **Observation data**: Real traffic measurements (simulated)
2. **Model parameters**: Traffic flow model coefficients
3. **Optimization**: Find parameters that best match observations
4. **Prediction**: Use optimized parameters for traffic forecasting

## Expected Output

### Demo Version (mas_demo.py)
```
🚗 MAS-Bench Data Assimilation Demo
===================================
Micro: Agent-based traffic simulation (mocked)
Macro: Flow-density traffic model

🏗️ Setting up ModelBridge for traffic optimization...
📊 Parameter space:
   σ₁: Traffic flow variance [0.0, 1.0]
   μ₁: Mean flow rate [50, 300] vehicles/hour
   σ₂: Density variance [0.0, 1.0]
   μ₂: Mean density [10, 100] vehicles/km

🚦 Running traffic data assimilation pipeline...
🎯 Traffic Data Assimilation Results:
   ⏱️  Total execution time: 6.35 seconds
   📊 Parameter prediction quality:
      MSE: 76299.202976
      MAE: 139.870349
      R²:  -720.800733

🏆 MAS-Bench data assimilation demo completed successfully!
```

### Alternative Example (simple_mas_example.py)
```
🚗 MAS-Bench-Inspired Traffic Data Assimilation
🏗️ Setting up ModelBridge for traffic optimization...
🚦 Running traffic data assimilation pipeline...
📊 Parameter prediction quality:
   MSE: 2563.993742
   MAE: 26.141877
   R²:  -282.237159
🏆 Completed successfully!
```

## Performance Considerations

### Computational Requirements
- **Demo version**: ~1-6 seconds, minimal memory (~100MB)
- **Alternative version**: ~1-3 seconds, lightweight
- **Full MAS-Bench**: ~30-60 minutes, ~2GB RAM for large scenarios
- **Storage**: ~100-200MB for optimization databases

### Optimization Tips
- **Start with demo**: Use `mas_demo.py` for understanding concepts
- **Reduce trials**: Lower `trials_per_dataset` for faster testing
- **Parameter bounds**: Narrow ranges improve convergence
- **Mock vs Real**: Demo uses mock simulation for development/learning

## Troubleshooting

**Getting Started**:
- **New to MAS-Bench?** Start with `mas_demo.py` - no setup required
- **Quick test?** Try `simple_mas_example.py` - minimal dependencies
- **Full integration?** Use `hpopt_data_assimilation.py` - requires complete setup

**Demo Version Issues**:
```bash
# If mas_demo.py fails
uv run python mas_demo.py

# Check dependencies
uv pip list | grep -E "(modelbridge|numpy|optuna)"
```

**Full MAS-Bench Issues**:
```bash
# Check Java installation
java -version

# Test MAS-Bench.jar (may show usage info)
java -jar MAS-Bench.jar --help

# Verify resource files exist
ls -la masbench-resources/Dataset/FL1-2/
# Should show: agent_size.sh, model.properties, Scenario_Guidance.json

# If resources are missing, create them:
# (Follow the setup instructions in the Prerequisites section above)

# Test the setup with demo mode first:
uv run python hpopt_data_assimilation.py --demo
```

**Common Solutions**:
- **Resource missing**: Demo versions don't require `masbench-resources/`
- **Java errors**: Demo versions don't require Java installation
- **Slow execution**: Use demo versions for development and testing
- **Complex setup**: Full MAS-Bench integration requires additional configuration

## Related Work

- **MAS-Bench**: https://github.com/MAS-Bench/MAS-Bench
- **Traffic simulation**: Agent-based modeling for transportation
- **Data assimilation**: Parameter estimation in dynamic systems
