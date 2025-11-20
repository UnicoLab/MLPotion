# MLPotion 🧪

<p align="center">
  <img src="docs/logo.png" width="350" alt="MLPotion Logo"/>
  <br>
  <em>Modular Components for Machine Learning Pipelines</em>
  <br>
  <br>
  <p align="center"><strong>Provided and maintained by <a href="https://unicolab.ai">🦄 UnicoLab</a></strong></p>
</p>

---

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="http://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="Type checked: mypy"></a>
  <br>
  <a href="https://zenml.io"><img src="https://img.shields.io/badge/built%20with-ZenML-blue.svg" alt="ZenML"></a>
  <a href="https://keras.io"><img src="https://img.shields.io/badge/keras-3.0+-red.svg" alt="Keras 3"></a>
  <a href="https://tensorflow.org"><img src="https://img.shields.io/badge/TensorFlow-2.15+-orange.svg" alt="TensorFlow"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
</p>

---

## 🧪 The Formula for Better Pipelines

**MLPotion** is a library of composable, framework-agnostic building blocks for machine learning.

Instead of writing the same boilerplate training loops, data loaders, and evaluation scripts for every project, MLPotion provides standardized "atoms" that you can mix and match to build robust pipelines. It bridges the gap between experimental code and production systems.

### 🌟 Key Ingredients

*   **🧩 Modular Architecture**: Components are designed to be used independently or together. Swap out a data loader or a trainer without rewriting your entire codebase.
*   **♻️ Reusability**: Built on fully tested, production-ready components. Stop reinventing the wheel and trust in standardized atoms for your pipelines.
*   **🔌 Integrations**: Designed to be extensible to any MLOps tool. Currently, **ZenML** is fully implemented with reusable steps. Contributions for other integrations are welcome!
*   **🧠 ML Frameworks**: Components available for **Keras**, **PyTorch**, and **TensorFlow**. The architecture is extensible to any other framework—contributions are welcome!

---

## 📦 Installation

Install only what you need to keep your environment clean.

### Core (Base Protocols)
Perfect for defining custom implementations or lightweight usage.
```bash
poetry add mlpotion
```

### For Framework Users
```bash
# TensorFlow / Keras
poetry add mlpotion -E tensorflow

# PyTorch
poetry add mlpotion -E pytorch
```

### For MLOps (ZenML)
Combine your framework with ZenML capabilities.
```bash
poetry add mlpotion -E tensorflow -E zenml
# OR
poetry add mlpotion -E pytorch -E zenml
```

### The Full Lab
```bash
poetry add mlpotion -E all
```

---

## ⚡ Quickstart

Here is a simple example of defining a training configuration and running a trainer using Keras components.

```python
from mlpotion.frameworks.keras.training import ModelTrainer
from mlpotion.frameworks.keras.config import ModelTrainingConfig

# 1. Define your configuration 📝
# Strongly typed configs ensure you never miss a parameter
config = ModelTrainingConfig(
    epochs=10,
    batch_size=32,
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

# 2. Initialize the Trainer 🧪
trainer = ModelTrainer(config=config)

# 3. Run the pipeline 🚀
# (Assuming 'my_model' and datasets are ready)
history = trainer.train(
    model=my_model,
    train_data=train_ds,
    val_data=val_ds
)

print(f"✅ Training complete! Final accuracy: {history.history['accuracy'][-1]:.4f}")
```

> **ZenML User?**
> Drop this logic directly into a pipeline step:
> ```python
> from mlpotion.integrations.zenml.tensorflow.steps import train_model
> ```

---

## 🤝 Contributing

We welcome fellow chemists to the lab! 🥼
If you have a new component, a bug fix, or an idea for an improvement, please open an issue or submit a pull request.

---

<p align="center">
  <strong>Built with ❤️ by <a href="https://unicolab.ai">UnicoLab.ai</a></strong>
</p>
