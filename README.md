# MLPotion 🧪

<p align="center">
  <img src="docs/logo.png" width="350"/>
  <p align="center"><strong>Provided and maintained by <a href="https://unicolab.ai">🦄 UnicoLab</a></strong></p>
</p>

**ML-Potion** helps you brew your own machine-learning magic ✨—exactly the way you want it. Instead of forcing you into a rigid framework, it gives you a chest of mix-and-match building blocks (atoms, steps, components) you can snap together into fully custom training or inference pipelines.

Whether you're a fan of **Keras, TensorFlow, PyTorch**, or you’re bold enough to bring your own framework, everything is designed to be modular, composable, and delightfully flexible. Build pipelines by hand like a wizard mixing ingredients… or drop them straight into **ZenML** to get production-ready steps with clean, tested foundations.

If you prefer a “just-give-me-the-spell” workflow, ML-Potion also includes ready-made steps for common use cases—simple, predictable, and still fully customizable.
And when you need that special custom twist, we want you to contribute it back. 🧪 Your creation might become someone else’s favorite spell.

*Craft. Combine. Conjure.*

With ML-Potion, your ML pipeline becomes a potion worth sharing. 🚀


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
[![ZenML](https://img.shields.io/badge/built%20with-ZenML-blue.svg)](https://zenml.io)
[![Keras 3](https://img.shields.io/badge/keras-3.0+-red.svg)](https://keras.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-red.svg)](https://keras.io)
[![PyToch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://keras.io)
[![🦄 UnicoLab](https://img.shields.io/badge/UnicoLab-Enterprise%20AI-blue.svg)](https://unicolab.ai)
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

<p align="center">
  <strong>Built with ❤️ for the ML community by 🦄 UnicoLab.ai</strong>
</p>