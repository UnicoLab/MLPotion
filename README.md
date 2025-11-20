# MLPotion: Brew Your ML Magic! 🧪✨

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

Welcome, fellow alchemist! 🧙‍♂️ Ready to brew some machine learning magic without getting locked in a cauldron?

**MLPotion** is your chest of modular, mix-and-match ML building blocks that work across **Keras, TensorFlow, and PyTorch**. Think of it as LEGO® for ML pipelines, but with fewer foot injuries and more flexibility!

## Why MLPotion? 🤔

Ever felt trapped by a framework that forces you to do things "their way"? We've been there. That's why we created MLPotion:

*   **🎯 Framework Agnostic**: Write once, run anywhere (well, on Keras, TensorFlow, or PyTorch).
*   **🧱 Modular by Design**: Pick the pieces you need, leave the rest in the box.
*   **🔬 Type-Safe**: Python 3.10+ typing that actually helps you (mypy approved!).
*   **🚀 Production Ready**: Built for the real world, not just notebooks.
*   **🎨 Orchestration Flexible**: Works standalone OR with ZenML, Prefect, Airflow - your choice!
*   **📦 Install What You Need**: Core package works without any ML frameworks (you only install what you need)!
*   **🤝 Community-Driven**: Missing something? Contribute it back - we love community additions!

## What's in the Potion? 🧪

### ⚗️ Core Ingredients
*   Type-safe protocols for all components
*   Framework-agnostic result types
*   Consistent error handling
*   Zero-dependency core package

### 🔧 Framework Support
*   **Keras 3.0+** - The friendly one
*   **TensorFlow 2.15+** - The production workhorse
*   **PyTorch 2.0+** - The researcher's favorite

### 📊 Data Processing
*   CSV loaders for all frameworks
*   Dataset optimization utilities
*   Data transformers
*   Preprocessing pipelines

### 🎓 Training & Evaluation
*   Unified training interface
*   Comprehensive evaluation tools
*   Rich result objects
*   Training history tracking

### 💾 Model Management
*   Save/load model checkpoints
*   Export to production formats
*   Model inspection utilities
*   Multiple export formats

### 🔄 Orchestration Integration
*   ZenML integration built-in
*   Extensible to Prefect, Airflow, etc.
*   Works standalone (no orchestration needed!)

## The MLPotion Philosophy 🎭

> "A good potion doesn't force you to drink it a certain way. It just... works."
>
> — Ancient ML Alchemist Proverb (we just made that up)

We believe in:

1.  **Flexibility > Convention**: Your project, your rules
2.  **Simplicity > Complexity**: If it's hard to use, we failed
3.  **Type Safety > Runtime Surprises**: Catch errors before they bite
4.  **Modularity > Monoliths**: Use what you need, ignore the rest
5.  **Consistency > Chaos**: Same patterns across all frameworks
6.  **Community > Corporate**: Built by the community, for the community

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

## Feature Comparison 📊

| Feature | MLPotion | Framework-Only | All-in-One Solutions |
|---------|----------|----------------|---------------------|
| Multi-framework | ✅ Yes | ❌ No | ⚠️ Limited |
| Type Safety | ✅ Full | ⚠️ Partial | ⚠️ Partial |
| Modular Install | ✅ Yes | ❌ No | ❌ No |
| ZenML Native | ✅ Yes | ❌ Manual | ⚠️ Adapters |
| Learning Curve | 📈 Gentle | 📈 Framework-specific | 📈 Steep |
| Production Ready | ✅ Yes | ⚠️ DIY | ✅ Yes |
| Flexibility | 🌟🌟🌟🌟🌟 | 🌟🌟🌟🌟🌟 | 🌟🌟 |

## Who's This For? 🎯

**You'll love MLPotion if you:**
*   Switch between frameworks and hate rewriting everything
*   Value heavily tested code that you can reuse
*   Value type safety and IDE autocomplete (who doesn't?)
*   Want production-ready code without enterprise bloat
*   Believe ML pipelines should be composable and testable

**You might want something else if you:**
*   Do not like modularity
*   Do not like reusability
*   Are too lazy to contribute something that you can't already find here

## Community & Support 🤝

*   [**GitHub**](https://github.com/UnicoLab/MLPotion): Star, fork, contribute!
*   [**Issues**](https://github.com/UnicoLab/MLPotion/issues): Report bugs, request features.
*   [**UnicoLab**](https://unicolab.ai): Enterprise AI solutions.

---

<p align="center">
  <strong>Ready to brew some ML magic? Let's get started! 🧪✨</strong><br>
  <em>Built with ❤️ for the ML community by <a href="https://unicolab.ai">🦄 UnicoLab</a></em>
</p>
