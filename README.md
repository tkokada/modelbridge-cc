# ModelBridge

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
# Install the package (editable)
make install
# or manually: uv pip install -e .

# With development dependencies
make install-dev
# or manually: uv pip install -e ".[dev]"

# With example dependencies (PyTorch for neural_network example)
make install-examples
# or manually: uv pip install -e ".[examples]"

# With all dependencies
make install-all
# or manually: uv pip install -e ".[all]"
```

### Alternative: Sync from Lockfile
```bash
# Sync dependencies from uv.lock (faster, reproducible)
make sync-dev
# or manually: uv sync --extra dev
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
# Install with development dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
make pre-commit-install
```

### Development Commands

#### Code Quality
| Command | Purpose |
|---------|---------|
| `make lint` | Run ruff linter |
| `make format` | Format code with ruff |
| `make type-check` | Run mypy type checking |
| `make check-all` | Run all quality checks |

#### Testing
| Command | Purpose |
|---------|---------|
| `make test` | Run pytest test suite |
| `make test-cov` | Run tests with coverage |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |

#### Pre-commit Hooks
| Command | Purpose |
|---------|---------|
| `make pre-commit-install` | Install pre-commit hooks |
| `make pre-commit-run` | Run pre-commit on all files |
| `make setup-dev` | Complete development setup |

#### Examples
| Command | Purpose |
|---------|---------|
| `make run-simple-example` | Run simple benchmark |
| `make run-neural-demo` | Run neural network demo |
| `make run-neural-pytorch` | Run PyTorch neural example |
| `make run-mas-example` | Run traffic simulation demo |

```bash
# Complete development workflow
make install-dev         # Install with dev dependencies
make setup-dev           # Set up pre-commit hooks
make check-all           # Run all quality checks
make test-cov            # Test with coverage
make build               # Build distribution
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

## 🤝 Development

For development setup and contribution guidelines, see the project documentation.

## 🌟 Acknowledgments

- Built with [Optuna](https://optuna.org/) for optimization
- Powered by [scikit-learn](https://scikit-learn.org/) for regression
- Enhanced with [GPy](https://sheffieldml.github.io/GPy/) for Gaussian Processes
- Developed with modern Python tooling: [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [mypy](https://mypy.readthedocs.io/)
