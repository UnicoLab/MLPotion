# Contributing to MLPotion 🤝

Thank you for your interest in contributing to MLPotion! We love community contributions and want to make it as easy as possible.

## Ways to Contribute 🌟

### 1. Report Bugs 🐛

Found a bug? Please [open an issue](https://github.com/UnicoLab/MLPotion/issues) with:

- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, framework versions)
- Minimal reproducible example

### 2. Suggest Features 💡

Have an idea? [Start a discussion](https://github.com/UnicoLab/MLPotion/discussions) or open an issue with:

- Clear description of the feature
- Use case and motivation
- Example API (if applicable)

### 3. Submit Pull Requests 🔨

We welcome code contributions! See below for guidelines.

### 4. Improve Documentation 📖

Documentation improvements are always appreciated:

- Fix typos or unclear explanations
- Add examples
- Improve API documentation
- Write tutorials

### 5. Help Others 🙋

- Answer questions in [Discussions](https://github.com/UnicoLab/MLPotion/discussions)
- Help review pull requests
- Share your MLPotion projects

## Development Setup 🛠️

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/MLPotion.git
cd MLPotion
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Development Dependencies

```bash
# Install with all extras for development
pip install -e ".[all]"

# Install development tools
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## Code Style 🎨

We use:

- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking

Run before committing:

```bash
# Format code
black mlpotion tests

# Sort imports
isort mlpotion tests

# Lint
ruff check mlpotion tests

# Type check
mypy mlpotion
```

Or just let pre-commit handle it:

```bash
pre-commit run --all-files
```

## Testing 🧪

### Run All Tests

```bash
pytest
```

### Run Specific Framework Tests

```bash
# TensorFlow only
pytest -m tensorflow

# PyTorch only
pytest -m pytorch

# Keras only
pytest -m keras
```

### Run with Coverage

```bash
pytest --cov=mlpotion --cov-report=html
```

### Write Tests

All new features need tests! Place tests in `tests/` with the same structure as `mlpotion/`.

Example:

```python
# tests/frameworks/tensorflow/test_loaders.py
import pytest
from mlpotion.frameworks.tensorflow import TFCSVDataLoader

def test_csv_loader_basic():
    """Test basic CSV loading."""
    loader = TFCSVDataLoader("test_data.csv", label_name="target")
    dataset = loader.load()

    assert dataset is not None
    # More assertions...
```

## Pull Request Process 🔄

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, documented code
- Follow existing patterns
- Add tests for new features
- Update documentation

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add awesome feature"
```

Commit message format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `style:` - Code style changes
- `chore:` - Maintenance

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:

- Clear description of changes
- Link to related issue (if any)
- Screenshots (if UI changes)
- Checklist of completed items

## Project Structure 📁

```
mlpotion/
├── core/                    # Framework-agnostic core
│   ├── protocols.py        # Protocol definitions
│   ├── results.py          # Result types
│   ├── config.py           # Configuration classes
│   └── exceptions.py       # Custom exceptions
├── frameworks/              # Framework implementations
│   ├── tensorflow/         # TensorFlow components
│   ├── pytorch/            # PyTorch components
│   └── keras/              # Keras components
├── integrations/            # Third-party integrations
│   └── zenml/              # ZenML integration
└── utils/                   # Utility functions
```

## Adding a New Component 🆕

### 1. Define Protocol (if new)

```python
# mlpotion/core/protocols.py
from typing import Protocol, TypeVar

DatasetT = TypeVar("DatasetT")

class NewComponent(Protocol[DatasetT]):
    """Protocol for new component type."""

    def do_something(self, dataset: DatasetT) -> DatasetT:
        """Do something with dataset."""
        ...
```

### 2. Implement for Each Framework

```python
# mlpotion/frameworks/tensorflow/new_module.py
import tensorflow as tf

class TFNewComponent:
    """TensorFlow implementation."""

    def do_something(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Implementation for TensorFlow."""
        # Your code here
        return dataset
```

### 3. Add Tests

```python
# tests/frameworks/tensorflow/test_new_module.py
def test_new_component():
    """Test new component."""
    component = TFNewComponent()
    result = component.do_something(dataset)
    assert result is not None
```

### 4. Document

```python
# Add docstrings
class TFNewComponent:
    """TensorFlow implementation of NewComponent.

    This component does X, Y, and Z.

    Args:
        param1: Description
        param2: Description

    Example:
        ```python
        component = TFNewComponent(param1="value")
        result = component.do_something(dataset)
        ```
    """
```

## Code Review Process 👀

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer approval
3. **Documentation**: Must be updated if needed
4. **Tests**: Must pass and have good coverage
5. **Breaking Changes**: Require discussion

## Release Process 🚀

1. Version bump in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. GitHub Actions publishes to PyPI

## Questions? 💬

- [GitHub Discussions](https://github.com/UnicoLab/MLPotion/discussions)
- [Issues](https://github.com/UnicoLab/MLPotion/issues)
- Email: piotr@unicolab.ai

## Code of Conduct 📜

Be respectful, inclusive, and collaborative. We're all here to make ML better!

---

<p align="center">
  <strong>Thank you for contributing to MLPotion!</strong> 🙏<br>
  <em>Built with ❤️ by the community for the community</em>
</p>
