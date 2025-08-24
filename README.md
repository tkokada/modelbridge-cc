# ModelBridge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-blue.svg)](https://mypy.readthedocs.io/)

**A Python framework for hyperparameter optimization and model bridging between micro and macro models**

ModelBridge enables efficient hyperparameter optimization by bridging computationally expensive micro models with fast macro models through regression-based parameter mapping.

## 🔧 Key Features

- **Model Bridging**: Connect high-accuracy micro models with fast macro models
- **Multiple Optimization Backends**: Optuna with TPE, CMA-ES, and Random samplers
- **Flexible Regression Models**: Linear, Polynomial, and Gaussian Process regression
- **Comprehensive Visualization**: Automated plotting and analysis tools
- **Type Safety**: Full type hints and mypy compatibility
- **Modern Python**: Built for Python 3.12+ with latest language features

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/tkokada/modelbridge.git
cd modelbridge
uv pip install -e .

# With development dependencies
uv pip install -e ".[dev]"

# With all optional dependencies
uv pip install -e ".[all]"
```

### Basic Usage

```python
from modelbridge import ModelBridge

# Define objective functions
def micro_objective(params):
    """Expensive, accurate model"""
    return expensive_evaluation(params)

def macro_objective(params, target_value):
    """Fast, approximate model"""
    return fast_approximation(params)

# Configure parameters
param_config = {
    "x1": {"type": "float", "low": -5.0, "high": 5.0},
    "x2": {"type": "float", "low": -5.0, "high": 5.0}
}

# Create bridge
bridge = ModelBridge(
    micro_objective=micro_objective,
    macro_objective=macro_objective,
    micro_param_config=param_config,
    macro_param_config=param_config,
    regression_type="polynomial"
)

# Run complete pipeline
results = bridge.run_full_pipeline(
    n_train=10, n_test=5,
    visualize=True, output_dir="results"
)
```

## 📚 Examples

### 1. Mathematical Benchmark Functions
```bash
cd example/simple_benchmark
python hpopt_benchmark_refactored.py -c config_sample.toml
```

### 2. Neural Network Optimization (MNIST)
```bash
cd example/neural_network
python mnist_sklearn_bridge.py --demo  # Individual models
python mnist_sklearn_bridge.py         # Full model bridge
```

### 3. Traffic Simulation (MAS-Bench)
```bash
cd example/mas_bench
python hpopt_data_assimilation_refactored.py
```

## 🔬 How It Works

### Model Bridge Workflow

1. **Training Phase**
   - Optimize micro model parameters on n_train datasets
   - Optimize macro model parameters to match micro results
   - Train regression model to map macro → micro parameters

2. **Testing Phase**
   - Optimize micro model on n_test datasets (ground truth)
   - Optimize macro model on same datasets
   - Use trained regression to predict micro parameters
   - Evaluate prediction accuracy

3. **Analysis**
   - Compare predicted vs actual micro parameters
   - Generate visualizations and performance metrics
   - Export results for further analysis

## 🏗️ Architecture

```
📦 ModelBridge
├── 🧠 modelbridge/           # Core library
│   ├── core/                 # Core modules
│   │   ├── optimizer.py      # Optuna wrapper
│   │   ├── regression.py     # ML regression models
│   │   └── bridge.py         # Main coordination
│   ├── utils/                # Utilities
│   │   ├── config_loader.py  # Configuration handling
│   │   ├── data_manager.py   # Data I/O and conversion
│   │   └── visualization.py  # Plotting utilities
│   └── types.py              # Type definitions
├── 📊 example/               # Example implementations
│   ├── simple_benchmark/     # Mathematical functions
│   ├── neural_network/       # Neural network bridging
│   └── mas_bench/           # Traffic simulation
├── 🧪 tests/                # Test suite
└── 📋 pyproject.toml        # Project configuration
```

## 🛠️ Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/tkokada/modelbridge.git
cd modelbridge

# Install with development dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks (optional)
pre-commit install
```

### Development Commands

| Command | Purpose |
|---------|---------|
| `make lint` | Run ruff linter |
| `make format` | Format code with ruff |
| `make type-check` | Run mypy type checking |
| `make test` | Run pytest test suite |
| `make test-cov` | Run tests with coverage |
| `make check-all` | Run all quality checks |
| `make build` | Build distribution packages |

```bash
# Quality assurance
make check-all

# Test with coverage
make test-cov

# Build package
make build
```

## 📊 Supported Configurations

### Optimization Backends
- **Optuna**: TPE, CMA-ES, Random samplers
- **Storage**: SQLite, In-memory
- **Direction**: Minimize/Maximize

### Regression Models
- **Linear**: Fast, interpretable
- **Polynomial**: Higher-order relationships
- **Gaussian Process**: Uncertainty quantification (requires GPy)

### Visualization
- Parameter relationship plots
- Prediction accuracy analysis
- Optimization history tracking
- Regression performance metrics

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`make check-all`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏛️ Citation

If you use ModelBridge in your research, please consider citing:

```bibtex
@software{modelbridge2024,
  title={ModelBridge: A Python Framework for Model Bridging and Hyperparameter Optimization},
  author={ModelBridge Contributors},
  year={2024},
  url={https://github.com/tkokada/modelbridge}
}
```

## 🆘 Support

- 📖 **Documentation**: [GitHub README](https://github.com/tkokada/modelbridge#readme)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/tkokada/modelbridge/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/tkokada/modelbridge/discussions)

## 🌟 Acknowledgments

- Built with [Optuna](https://optuna.org/) for optimization
- Powered by [scikit-learn](https://scikit-learn.org/) for regression
- Enhanced with [GPy](https://sheffieldml.github.io/GPy/) for Gaussian Processes
- Developed with modern Python tooling: [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [mypy](https://mypy.readthedocs.io/)

---

<div align="center">
  <strong>Made with ❤️ by the ModelBridge Contributors</strong>
</div>
