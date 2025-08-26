# Contributing to ModelBridge

Thank you for your interest in contributing to ModelBridge! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/modelbridge.git
   cd modelbridge
   ```

2. **Set up development environment**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Complete development setup (recommended)
   make setup-dev
   # This runs: install-dev + pre-commit setup + environment verification

   # Or manually step by step:
   make install-dev         # Install with development dependencies
   make pre-commit-install  # Install pre-commit hooks

   # Alternative: sync from lockfile (faster, reproducible)
   make sync-dev           # Uses uv.lock for exact dependency versions
   ```

3. **Verify setup**
   ```bash
   make check-all
   ```

## 🛠️ Development Workflow

### Before You Start

1. **Check existing issues** - Look for existing issues or discussions related to your contribution
2. **Create an issue** - For substantial changes, create an issue to discuss your approach first
3. **Create a feature branch** - Always work on a feature branch, never directly on `main`

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Make your changes**
   - Write clear, well-documented code
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all quality checks
   make check-all

   # Run tests with coverage
   make test-cov
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"

   # Follow conventional commit format:
   # feat: new features
   # fix: bug fixes
   # docs: documentation changes
   # style: formatting, missing semi colons, etc
   # refactor: code refactoring
   # test: adding tests
   # chore: maintenance tasks
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request through the GitHub interface.

## 🏗️ Code Standards

### Code Quality

We maintain high code quality standards using modern Python tooling:

- **Linting**: [ruff](https://docs.astral.sh/ruff/) for fast linting and formatting
- **Type Checking**: [mypy](https://mypy.readthedocs.io/) for static type analysis
- **Testing**: [pytest](https://pytest.org/) with comprehensive test coverage
- **Documentation**: Clear docstrings following Google/NumPy style

### Quality Checks

All code must pass these checks before being merged:

```bash
make check-all  # Runs all checks below: lint + format + type-check + test
make lint       # ruff linting
make format     # ruff formatting
make type-check # mypy type checking
make test       # pytest test suite
```

#### Individual Testing Options
```bash
make test-unit        # Unit tests only (fast)
make test-integration # Integration tests only
make test-cov         # Tests with coverage report
make test-fast        # Skip slow tests
```

### Pre-commit Hooks

We use pre-commit hooks to automatically run quality checks on every commit:

```bash
# Complete setup (recommended)
make setup-dev        # Installs dependencies + pre-commit hooks

# Manual setup
make pre-commit-install  # Install hooks (one-time setup)
make pre-commit-run      # Run on all files manually
make pre-commit-update   # Update hook versions
```

**Pre-commit hooks automatically run on every commit and include:**
- **Code formatting** (ruff)
- **Linting** (ruff)
- **Type checking** (mypy)
- **Security scanning** (bandit)
- **File validation** (trailing whitespace, file size, etc.)

If pre-commit hooks fail, the commit will be blocked until issues are fixed.

### Code Style

- **Python Version**: Requires Python 3.12+
- **Type Hints**: All public APIs must have complete type hints
- **Docstrings**: All public functions/classes must have docstrings
- **Line Length**: 88 characters (ruff default)
- **Import Sorting**: Handled automatically by ruff

### Example Code Style

```python
from typing import Any

from modelbridge.types import ParamDict, FloatArray


class ExampleClass:
    """Example class demonstrating code style.

    This class shows the preferred code style for ModelBridge,
    including type hints, docstrings, and formatting.
    """

    def __init__(self, param_config: ParamDict) -> None:
        """Initialize the example class.

        Args:
            param_config: Parameter configuration dictionary
        """
        self.param_config = param_config

    def process_data(self, data: FloatArray) -> dict[str, Any]:
        """Process input data and return results.

        Args:
            data: Input data array

        Returns:
            Dictionary containing processed results

        Raises:
            ValueError: If data is empty
        """
        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        # Process data here
        return {"result": data.mean()}
```

## 🧪 Testing

### Writing Tests

- **Location**: Place tests in the `tests/` directory
- **Structure**: Mirror the source code structure in tests
- **Naming**: Test files should be named `test_*.py`
- **Coverage**: Aim for high test coverage, especially for core functionality

### Test Categories

- **Unit Tests** (`tests/unit/`): Test individual functions/classes
- **Integration Tests** (`tests/integration/`): Test component interactions
- **Property Tests**: Use Hypothesis for property-based testing where applicable

### Test Examples

```python
import pytest
from modelbridge.core.optimizer import OptunaOptimizer


class TestOptunaOptimizer:
    """Test OptunaOptimizer functionality."""

    def test_optimizer_creation(self) -> None:
        """Test that optimizer can be created successfully."""
        optimizer = OptunaOptimizer()
        assert optimizer is not None

    def test_invalid_configuration(self) -> None:
        """Test that invalid configuration raises appropriate error."""
        optimizer = OptunaOptimizer()

        with pytest.raises(ValueError, match="Invalid parameter"):
            optimizer.suggest_parameters(None, {})
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_optimizer.py

# Run tests matching pattern
pytest -k "test_optimizer"

# Run tests with verbose output
pytest -v
```

## 📚 Documentation

### Docstring Style

Use Google-style docstrings for all public APIs:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.

    Longer description providing more details about what the function
    does, its behavior, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 0.

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        RuntimeError: When operation fails

    Example:
        ```python
        result = example_function("hello", 42)
        assert result is True
        ```
    """
```

### README Updates

When adding new features:
- Update the main README.md if the feature affects the public API
- Add examples for new functionality
- Update installation instructions if needed

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Minimal reproduction case**
3. **Expected vs actual behavior**
4. **Environment information** (Python version, OS, dependency versions)
5. **Stack trace** if applicable

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug

## To Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
Clear description of what you expected to happen

## Environment
- OS: [e.g. macOS 14.0]
- Python: [e.g. 3.12.0]
- ModelBridge: [e.g. 0.1.0]
- Dependencies: [paste relevant versions]

## Additional Context
Any other context about the problem
```

## 💡 Feature Requests

When proposing new features:

1. **Check existing issues** first
2. **Describe the use case** - why is this feature needed?
3. **Propose a solution** - how should it work?
4. **Consider alternatives** - what other approaches could work?
5. **Breaking changes** - will this break existing code?

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass (`make check-all`)
- [ ] New tests added for new functionality
- [ ] Documentation updated as needed
- [ ] Commit messages follow conventional format
- [ ] Changes are backwards compatible (or breaking changes are clearly documented)

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] New tests added
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes are backwards compatible
```

### Review Process

1. **Automated checks** must pass (GitHub Actions)
2. **Code review** by maintainers
3. **Discussion** of any feedback
4. **Approval** by at least one maintainer
5. **Merge** after approval

## 🏷️ Release Process

Releases are managed by maintainers following semantic versioning:

- **Patch** (0.1.1): Bug fixes, minor improvements
- **Minor** (0.2.0): New features, backwards compatible
- **Major** (1.0.0): Breaking changes

## 🆘 Getting Help

If you need help:

1. **Check documentation** - README, docstrings, examples
2. **Search issues** - existing and closed issues
3. **Ask questions** - create a GitHub Discussion
4. **Join community** - participate in discussions

## 🙏 Recognition

Contributors will be recognized in:
- Repository contributors list
- Release notes for significant contributions
- README acknowledgments

Thank you for contributing to ModelBridge! 🚀
