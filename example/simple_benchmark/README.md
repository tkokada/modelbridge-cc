# Simple Benchmark Example

This example demonstrates ModelBridge using mathematical optimization benchmark functions.

## Overview

This example showcases model bridging between different mathematical functions:
- **Micro Model**: Complex functions (Rastrigin, Griewank) - computationally expensive but accurate
- **Macro Model**: Simple functions (Sphere) - fast but less accurate
- **Bridge**: Regression model maps macro parameters to micro parameters

## Files

- `simple_example.py` - Simple, reliable example using ModelBridge library (recommended)
- `hpopt_benchmark.py` - Advanced example with TOML configuration support
- `config_sample.toml` - Configuration file with optimization parameters

## Benchmark Functions

### Micro Functions (High-accuracy, expensive)
- **Rastrigin**: `10n + Σ(x²ᵢ - 10cos(2πxᵢ))` - highly multimodal
- **Griewank**: `1 + Σ(x²ᵢ)/4000 - Π(cos(xᵢ/√i))` - many local minima

### Macro Functions (Fast, approximate)
- **Sphere**: `Σ(x²ᵢ)` - simple quadratic function

## Configuration

Edit `config_sample.toml` to customize:

```toml
[dataset]
max_x = 5.0           # Parameter bounds
min_x = -5.0
max_x_dim = 3         # Number of x parameters
n_train = 5           # Training datasets
n_test = 3            # Test datasets

[micro_model]
micro_function_name = "rastrigin"  # or "griewank"

[macro_model]
macro_function_name = "sphere"

[regression]
regression_model_name = "polynomial"  # "linear", "polynomial", "gp"
```

## Usage

### Option 1: Simple Example (Recommended)
```bash
# Run simple, reliable example (outputs to local simple_benchmark_results/)
uv run python simple_example.py

# View results
ls -la simple_benchmark_results/
```

### Option 2: Advanced Configuration
```bash
# Run advanced version with TOML configuration support
uv run python hpopt_benchmark.py -c config_sample.toml

# View results
ls -la benchmark_results/
```

### Option 3: Using Makefile (from project root)
```bash
# Run simple example through Makefile (recommended for automation)
make run-simple-example

# Run advanced benchmark example
make run-benchmark-example

# These automatically handle paths and dependencies
```

## Results

The script generates:

### Optimization Results
- **Training data**: Micro and macro parameter optimizations
- **Test data**: Ground truth micro parameters vs predicted parameters
- **Regression model**: Trained mapping between macro → micro parameters

### Visualizations
- `train_plot.png` - Training data parameter relationships
- `test_plot.png` - Test data parameter relationships
- `bridge_plot.png` - Prediction accuracy analysis

### Data Files
- `sphere_linear.db`, `rastrigin_linear.db` - Individual optimization databases (simple example)
- `benchmark.db` - Optuna optimization database (advanced config)
- CSV files with parameter datasets and predictions

## Expected Output

### Simple Example
```
ModelBridge Simple Benchmark Example
=====================================
1. Sphere (micro) → Linear (macro) Bridge
--------------------------------------------------
Starting training phase with 3 datasets...
Training phase completed!
Starting test phase with 2 datasets...
Results - MSE: 0.3717, MAE: 0.4608, R²: -2109.4738

2. Rastrigin (micro) → Linear (macro) Bridge
--------------------------------------------------
Results - MSE: 0.6500, MAE: 0.5927, R²: -489.2426
✅ Simple Benchmark Complete!
```

### Advanced Configuration
```
Running benchmark with config: config_sample.toml
Function: rastrigin -> sphere bridge
Starting training phase with 5 datasets...
Training phase completed!
Starting test phase with 3 datasets...
Test phase completed!
Model Bridge Results:
MSE: 0.012750
MAE: 0.104549
R²: -1.226585
✅ Benchmark completed successfully!
Results saved to: benchmark_results/
```

## Performance Notes

- **Training datasets**: More datasets improve regression accuracy
- **Optimization trials**: More trials find better parameters but take longer
- **Function choice**: Rastrigin vs Griewank affects difficulty and results
- **Regression model**: Polynomial often works best for these functions

## Troubleshooting

**Issue**: Poor prediction accuracy (low R²)
- **Solution**: Increase `n_train` or `nx_train/nx_test` in config

**Issue**: Slow execution
- **Solution**: Reduce number of optimization trials in config

**Issue**: Database conflicts
- **Solution**:
  - For simple example: Delete local `simple_benchmark_results/` directory or specific `.db` files
  - For advanced config: Delete local `benchmark_results/` directory or specific `.db` files

**Issue**: Module import errors
- **Solution**: Ensure you're in the correct directory and have installed dependencies with `uv pip install -e ".[dev]"` from the project root

**Issue**: Missing dependencies
- **Solution**: Install with `make install-all` or `uv pip install -e ".[all]"` to get all optional dependencies
