# Neural Network Model Bridge Example

This example demonstrates model bridging between different neural network architectures using the MNIST dataset.

## Overview

- **Micro Model**: 2-layer MLP with more neurons (higher accuracy, slower training)
- **Macro Model**: 1-layer MLP with fewer neurons (lower accuracy, faster training)
- **Dataset**: MNIST handwritten digits (subset for fast execution)
- **Goal**: Predict optimal micro model hyperparameters using macro model optimization

## Files

- `mnist_sklearn_bridge.py` - Main example using sklearn MLPClassifier (recommended)
- `mnist_model_bridge.py` - PyTorch implementation with CNN/MLP comparison
- `neural_models.py` - PyTorch neural network implementations
- `config_mnist.toml` - Configuration file for PyTorch version
- `README.md` - This documentation

## Quick Start

### sklearn Implementation (Recommended)
```bash
# Individual model demonstration
uv run python mnist_sklearn_bridge.py --demo

# Full model bridge example
uv run python mnist_sklearn_bridge.py --subset-size 500 --n-train 2 --n-test 1
```

### PyTorch Implementation
```bash
# Individual model demonstration (CNN vs MLP)
uv run python mnist_model_bridge.py --demo

# Full model bridge example
uv run python mnist_model_bridge.py --subset-size 200 --n-train 2 --n-test 1
```

### Using Makefile (from project root)
```bash
make run-neural-demo       # sklearn demo
make run-neural-example    # sklearn full example
make run-neural-pytorch    # PyTorch full example
```

## PyTorch Implementation Details

The PyTorch version (`mnist_model_bridge.py`) offers:

- **CNN Micro Model**: Convolutional neural network with dropout and configurable layers
- **MLP Macro Model**: Multi-layer perceptron with variable architecture
- **Advanced Features**: Dropout optimization, learning rate tuning, GPU support
- **Realistic Training**: Actual neural network training with PyTorch

### Installation
```bash
# Install with PyTorch support
uv pip install -e ".[examples]"
# or for all dependencies
uv pip install -e ".[all]"
```

### Usage Options
```bash
# Quick demo
uv run python mnist_model_bridge.py --demo

# Fast example
uv run python mnist_model_bridge.py --subset-size 200 --n-train 2 --n-test 1

# Thorough optimization (slower)
uv run python mnist_model_bridge.py --subset-size 1000 --n-train 3 --n-test 2 --slow
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
