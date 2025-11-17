# MLPotion 🧪

ML-Potion is all about crafting your own machine-learning elixir using the ingredients and frameworks you prefer. Think of the library as a set of building blocks—atoms, steps, and components—that you can mix, match, and sequence to build a fully customized training or inference pipeline.

You can choose from multiple supported frameworks (Keras, TensorFlow, PyTorch) or easily add your own if something is missing. Every block is designed to be composable, so you can assemble pipelines manually or plug them into an orchestration framework like ZenML to create reusable workflow steps backed by fully tested, scalable code.

Prefer a more “drag-and-drop” approach? The library also includes ready-made, framework-specific steps that cover most common use cases while still giving you full control over the underlying methods. And if you can’t find what you need, we encourage you to extend ML-Potion following our contribution guidelines—your addition may help others in the community as well.


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

---

## ✨ Features

- 🎯 Framework-agnostic core — Works even without installing any ML framework
- 🔧 Modular installation — Only install what you need (tensorflow, pytorch, zenml, etc.)
- 🛡️ Type-safe — Full Python 3.10+ typing and mypy-friendly design
- 🧪 Testable architecture — Protocols and abstractions make mocking trivial
- 📦 No framework lock-in — Use standalone or integrate with ZenML, Prefect, Airflow, etc.
- 🚀 Production-ready — Robust error handling, logging, and consistent interfaces
- 📖 Well-documented — Rich examples, docstrings, and guides to help you get started

---

## 📦 Installation

### Core Package (No Frameworks)

```bash
pip install mlpotion
```

### With TensorFlow

```bash
pip install mlpotion[tensorflow]
```

### With PyTorch

```bash
pip install mlpotion[pytorch]
```

### With Both Frameworks

```bash
pip install mlpotion[tensorflow,pytorch]
```

### With ZenML Integration

```bash
pip install mlpotion[tensorflow,zenml]
pip install mlpotion[pytorch,zenml]
```

### Everything

```bash
pip install mlpotion[all]
```

---

## 🚀 Quick Start

### TensorFlow Example

```python
import tensorflow as tf
from mlpotion.frameworks.tensorflow import (
    TFCSVDataLoader,
    TFDatasetOptimizer,
    TFModelTrainer,
    TensorFlowTrainingConfig,
)

# 1. Load data
loader = TFCSVDataLoader(
    file_pattern="data/train_*.csv",
    label_name="target",
)
dataset = loader.load()

# 2. Optimize dataset
optimizer = TFDatasetOptimizer(batch_size=32, cache=True)
dataset = optimizer.optimize(dataset)

# 3. Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. Train
trainer = TFModelTrainer()
config = TensorFlowTrainingConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
)
result = trainer.train(model, dataset, config)

# 5. Access results
print(f"Final loss: {result.metrics['loss']:.4f}")
print(f"Training time: {result.training_time:.2f}s")
```

### PyTorch Example

```python
import torch
import torch.nn as nn
from mlpotion.frameworks.pytorch import (
    PyTorchCSVDataset,
    PyTorchDataLoaderFactory,
    PyTorchModelTrainer,
    PyTorchTrainingConfig,
)

# 1. Create dataset
dataset = PyTorchCSVDataset(
    file_pattern="data/train_*.csv",
    label_name="target",
)

# 2. Create DataLoader
factory = PyTorchDataLoaderFactory(batch_size=32, shuffle=True)
dataloader = factory.create(dataset)

# 3. Create model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleModel()

# 4. Train
trainer = PyTorchModelTrainer()
config = PyTorchTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
result = trainer.train(model, dataloader, config)

# 5. Access results
print(f"Final loss: {result.metrics['loss']:.4f}")
print(f"Training time: {result.training_time:.2f}s")
```

---

## 📚 Core Concepts

### Protocol-Based Architecture

MLPotion uses Python protocols for maximum flexibility:

```python
from mlpotion.core.protocols import ModelTrainer, DataLoader
from mlpotion.core.config import TrainingConfig
from mlpotion.core.results import TrainingResult

# Any class implementing these protocols works!
class CustomTrainer:
    def train(
        self,
        model: Any,
        dataset: Any,
        config: TrainingConfig,
    ) -> TrainingResult:
        # Your custom training logic
        ...

# Type-safe usage
trainer: ModelTrainer = CustomTrainer()  # ✅ Type checks!
```

### Type-Safe Configuration

All configuration uses Pydantic for validation:

```python
from mlpotion.frameworks.tensorflow import TensorFlowTrainingConfig

# Type-safe configuration with validation
config = TensorFlowTrainingConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    optimizer="adam",
    loss="mse",
    metrics=["mae", "mse"],
)

# Pydantic validates all fields
config.epochs = -1  # ❌ ValidationError: ensure this value is greater than 0
```

### Structured Results

Results are typed dataclasses:

```python
result = trainer.train(model, dataset, config)

# Typed access
result.model              # Trained model
result.metrics            # dict[str, float]
result.history            # dict[str, list[float]]
result.training_time      # float | None
result.best_epoch         # int | None

# Helper methods
loss = result.get_metric("loss")
loss_history = result.get_history("loss")
```

---

## 🧩 Components

### TensorFlow Components

| Component | Purpose |
|-----------|---------|
| `TFCSVDataLoader` | Load CSV files → `tf.data.Dataset` |
| `TFDatasetOptimizer` | Optimize datasets (batching, prefetching, caching) |
| `TFModelTrainer` | Train Keras models |
| `TFModelEvaluator` | Evaluate Keras models |
| `TFModelExporter` | Export to SavedModel/H5/Keras format |
| `TFModelPersistence` | Save/load Keras models |

### PyTorch Components

| Component | Purpose |
|-----------|---------|
| `PyTorchCSVDataset` | CSV files → `torch.utils.data.Dataset` |
| `PyTorchDataLoaderFactory` | Create configured `DataLoader` |
| `PyTorchModelTrainer` | Train PyTorch models |
| `PyTorchModelEvaluator` | Evaluate PyTorch models |
| `PyTorchModelExporter` | Export to TorchScript/ONNX |
| `PyTorchModelPersistence` | Save/load PyTorch models |

---

## 🔌 Optional Integrations

### ZenML Integration

```python
from zenml import pipeline
from mlpotion.integrations.zenml.adapters import ZenMLAdapter
from mlpotion.frameworks.tensorflow import TFCSVDataLoader, TFModelTrainer

# Create components
loader = TFCSVDataLoader("data/*.csv")
trainer = TFModelTrainer()

# Adapt to ZenML steps
load_step = ZenMLAdapter.create_data_loader_step(loader)
train_step = ZenMLAdapter.create_training_step(trainer)

# Use in pipeline
@pipeline
def ml_pipeline():
    dataset = load_step()
    result = train_step(model, dataset, config)
    return result

# Run pipeline
ml_pipeline()
```

---

## 🧪 Testing

MLPotion is designed to be testable:

```python
import pytest
from mlpotion.core.protocols import ModelTrainer
from mlpotion.core.config import TrainingConfig
from mlpotion.core.results import TrainingResult

# Mock trainer for testing
class MockTrainer:
    def train(self, model, dataset, config):
        return TrainingResult(
            model=model,
            history={"loss": [0.5, 0.3]},
            metrics={"loss": 0.3},
            config=config,
        )

def test_training_pipeline():
    trainer: ModelTrainer = MockTrainer()
    config = TrainingConfig(epochs=1)
    
    result = trainer.train(model, dataset, config)
    
    assert result.metrics["loss"] == 0.3
```

---

## 📁 Project Structure

```
mlpotion/
├── core/                    # Framework-agnostic protocols & types
│   ├── protocols.py        # Protocol definitions
│   ├── config.py           # Configuration models (Pydantic)
│   ├── results.py          # Result models (dataclasses)
│   └── exceptions.py       # Custom exceptions
├── frameworks/              
│   ├── tensorflow/         # TensorFlow implementation
│   └── pytorch/            # PyTorch implementation
├── integrations/
│   └── zenml/              # Optional ZenML integration
└── utils/
    └── framework.py        # Framework detection utilities
```

---

## 🎯 Use Cases

### ✅ Standalone Scripts

Use MLPotion in plain Python scripts without any orchestration:

```python
from mlpotion.frameworks.tensorflow import TFModelTrainer

trainer = TFModelTrainer()
result = trainer.train(model, dataset, config)
```

### ✅ Jupyter Notebooks

Perfect for experimentation in notebooks:

```python
from mlpotion.frameworks.pytorch import PyTorchModelTrainer

result = PyTorchModelTrainer().train(model, dataloader, config)
```

### ✅ ZenML Pipelines

Integrate seamlessly with ZenML:

```python
from mlpotion.integrations.zenml import ZenMLAdapter

step = ZenMLAdapter.create_training_step(trainer)
```

### ✅ Custom Orchestration

Bring your own orchestrator:

```python
# Works with any orchestration framework
def my_training_step(model, dataset):
    trainer = TFModelTrainer()
    config = TensorFlowTrainingConfig(epochs=10)
    return trainer.train(model, dataset, config)
```

---

## 🛠️ Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mlpotion.git
cd mlpotion

# Install with dev dependencies
pip install -e ".[all,dev]"
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=mlpotion --cov-report=html

# Specific framework
pytest tests/unit/tensorflow/
pytest tests/unit/pytorch/
```

### Type Checking

```bash
mypy mlpotion/
```

### Code Formatting

```bash
# Format code
black mlpotion/ tests/

# Lint
ruff check mlpotion/ tests/
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Run type checking and tests
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by the need for reusable ML pipeline components
- Built with modern Python 3.10+ features
- Designed for production use

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/mlpotion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mlpotion/discussions)

---

## ⭐ Star History

If you find MLPotion useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the ML community**