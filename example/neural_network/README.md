# Neural Network Model Bridge Example

This example demonstrates model bridging between different neural network architectures using the MNIST dataset.

## Overview

- **Micro Model**: 2-layer MLP with more neurons (higher accuracy, slower training)
- **Macro Model**: 1-layer MLP with fewer neurons (lower accuracy, faster training) 
- **Dataset**: MNIST handwritten digits (subset for fast execution)
- **Goal**: Predict optimal micro model hyperparameters using macro model optimization

## Files

- `mnist_sklearn_bridge.py` - Main example using sklearn MLPClassifier (recommended)
- `neural_models.py` - PyTorch implementation (requires PyTorch installation)
- `config_mnist.toml` - Configuration file for PyTorch version
- `README.md` - This documentation

## Quick Start (sklearn version)

```bash
# Run the sklearn-based example (fast, no extra dependencies)
cd example/neural_network
python mnist_sklearn_bridge.py

# Run individual model demo
python mnist_sklearn_bridge.py --demo

# Customize parameters
python mnist_sklearn_bridge.py --subset-size 500 --n-train 2 --n-test 1
```

## PyTorch Version (Optional)

```bash
# Install PyTorch dependencies
uv pip install -e ".[examples]"

# Test PyTorch models
python neural_models.py

# Run PyTorch model bridge
python mnist_model_bridge.py --demo
```

## Model Architecture

### Micro Model (2-layer MLP)
- **Hidden layers**: 2 layers with configurable sizes
- **Parameters**: `hidden_size_1`, `hidden_size_2`, `alpha`, `learning_rate`
- **Range**: 32-128 neurons (layer 1), 16-64 neurons (layer 2)
- **Training**: More epochs, early stopping enabled

### Macro Model (1-layer MLP)  
- **Hidden layers**: 1 layer with configurable size
- **Parameters**: `macro_hidden_size`, `macro_alpha`, `learning_rate`
- **Range**: 16-64 neurons, faster training
- **Training**: Fewer epochs, simplified architecture

## Model Bridge Process

1. **Training Phase**: 
   - Train micro model (2-layer MLP) on different MNIST subsets
   - Train macro model (1-layer MLP) on same subsets
   - Learn regression mapping: macro params → micro params

2. **Testing Phase**:
   - Optimize macro model on new test subsets
   - Predict micro model params using learned regression
   - Compare predicted vs actual micro model params

3. **Evaluation**:
   - MSE, MAE, R² between predicted and actual parameters
   - Visualization of parameter relationships

## Expected Results

- **Execution time**: ~30-60 seconds (sklearn), ~2-3 minutes (PyTorch)
- **Parameter prediction**: Should show correlation between model complexities
- **Visualizations**: Plots showing hyperparameter relationships between architectures

## Performance Tips

- **Subset size**: Smaller = faster (500-1000 samples recommended)
- **Trials per dataset**: 3-5 trials for demonstration
- **Early stopping**: Models stop training early to save time
- **Simplified architectures**: Small networks for speed

## Interpretation

This example shows how model bridging can help:
- **Transfer knowledge** between different neural architectures
- **Speed up hyperparameter optimization** by using simpler proxy models
- **Understand relationships** between different model configurations
- **Predict optimal complex model parameters** from simple model optimization