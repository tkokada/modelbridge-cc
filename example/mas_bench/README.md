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

## Files

- `hpopt_data_assimilation.py` - Main script using ModelBridge library
- `MAS-Bench.jar` - Traffic simulation engine

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

### Option 2: Full MAS-Bench Integration (Advanced)
```bash
# Requires full MAS-Bench setup with simulation resources
uv run python hpopt_data_assimilation.py

# Monitor progress (if available)
tail -f outputs/logs/mas_bench.log
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
```
outputs/
├── databases/
│   └── mas_bench.db          # Optuna optimization history
├── plots/
│   ├── parameter_relationships.png
│   ├── prediction_accuracy.png
│   └── optimization_history.png
└── data/
    ├── train_micro_params.csv
    ├── train_macro_params.csv
    ├── test_micro_params.csv
    ├── test_macro_params.csv
    └── predicted_micro_params.csv
```

## MAS-Bench Integration Details

### Simulation Parameters
- **Traffic flow rates**: Vehicles per hour at entry points
- **Signal timing**: Traffic light cycle durations
- **Capacity settings**: Maximum flow rates per lane
- **Weather/incident factors**: Environmental impact parameters

### Data Assimilation Process
1. **Observation data**: Real traffic measurements (simulated)
2. **Model parameters**: Traffic flow model coefficients
3. **Optimization**: Find parameters that best match observations
4. **Prediction**: Use optimized parameters for traffic forecasting

## Performance Considerations

### Computational Requirements
- **Memory**: ~2GB RAM for large scenarios
- **CPU**: Multi-core recommended for parallel trials
- **Storage**: ~100MB for optimization databases
- **Time**: 30-60 minutes for full optimization

### Optimization Tips
- **Reduce trials**: Lower `trials_per_dataset` for faster testing
- **Parallel execution**: Use multiple CPU cores when available
- **Scenario selection**: Start with simpler traffic patterns
- **Parameter bounds**: Narrow ranges improve convergence

## Troubleshooting

**Java Issues**:
```bash
# Check Java installation
java -version

# Run MAS-Bench directly to test
java -jar MAS-Bench.jar --help
```

**Simulation Errors**:
- Check `masbench-resources/` directory exists
- Verify traffic scenario files are properly formatted
- Ensure sufficient disk space for temporary files

**Optimization Issues**:
- Increase optimization trials if convergence is poor
- Check parameter bounds are reasonable for traffic scenarios
- Monitor memory usage during long optimizations

## Related Work

- **MAS-Bench**: https://github.com/MAS-Bench/MAS-Bench
- **Traffic simulation**: Agent-based modeling for transportation
- **Data assimilation**: Parameter estimation in dynamic systems
