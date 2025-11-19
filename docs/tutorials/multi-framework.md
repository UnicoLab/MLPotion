# Multi-Framework Tutorial 🔀

Learn how to use the same MLPotion components across TensorFlow, PyTorch, and Keras!

## The Power of Protocols 💪

MLPotion's protocol-based design means you can switch frameworks with minimal code changes:

```python
# Same pattern, different framework!

# TensorFlow
from mlpotion.frameworks.tensorflow import TFCSVDataLoader
loader = TFCSVDataLoader("data.csv", label_name="target")
dataset = loader.load()  # Returns tf.data.Dataset

# PyTorch
from mlpotion.frameworks.pytorch import PyTorchCSVDataset
dataset = PyTorchCSVDataset("data.csv", label_name="target")

# Keras
from mlpotion.frameworks.keras import KerasCSVDataLoader
loader = KerasCSVDataLoader("data.csv", label_name="target")
dataset = loader.load()  # Returns keras dataset
```

## Framework-Agnostic Code 🎯

Write code that works with any framework:

```python
from typing import Protocol
from mlpotion.core.protocols import DataLoader, ModelTrainer

def train_model(
    loader: DataLoader,
    trainer: ModelTrainer,
    config,
):
    """Works with TensorFlow, PyTorch, or Keras!"""
    dataset = loader.load()
    result = trainer.train(model, dataset, config)
    return result
```

## Switching Frameworks Mid-Project 🔄

```python
# Start with TensorFlow for prototyping
from mlpotion.frameworks.tensorflow import *

# Later: Switch to PyTorch for custom research
from mlpotion.frameworks.pytorch import *

# Finally: Deploy with Keras for production
from mlpotion.frameworks.keras import *
```

The same MLPotion patterns (to some possible extents) work everywhere!

---

<p align="center">
  <strong>One API, Multiple Frameworks!</strong> 🎨
</p>
