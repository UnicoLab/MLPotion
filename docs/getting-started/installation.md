# Installation Guide 📥

Getting MLPotion installed is easier than pronouncing "scikit-learn" correctly! Let's get you set up.

## TL;DR - Just Tell Me What to Run! 🏃‍♂️

```bash
# Most common setup (using Poetry)
poetry add mlpotion -E tensorflow

# ... or using pip
pip install "mlpotion[tensorflow]"

# Or for PyTorch lovers
poetry add mlpotion -E pytorch

# Or if you can't decide (we don't judge)
poetry add mlpotion -E all
```

Done! Jump to [Quick Start](quickstart.md) to start brewing.

## Requirements 📋

Before you start, make sure you have:

- **Python >3.10** (we live on the edge, but not too close to it)
- **pip** (you probably have this already)
- **A sense of adventure** (optional, but recommended)

!!! warning "Python Version"
    MLPotion requires Python 3.10 or 3.11. Why? Because we use modern type hints that make your IDE actually helpful! Python 3.12 support is coming soon.

## Installation Options 🎯

MLPotion follows a "bring your own framework" philosophy. You only install what you need!

### Option 1: Core Only (Framework Agnostic) 🌟

Install just the core without any ML frameworks:

```bash
poetry add mlpotion
```

**Use this when:**

- You want to explore the package structure
- You're installing on systems without ML frameworks
- You're writing framework-agnostic code
- You're a minimalist at heart

**What you get:**

- Core protocols and interfaces
- Result types and configurations
- Utility functions
- Type stubs for IDE support

**What you don't get:**

- Actual ML framework implementations (obviously!)
- Data loaders (they need frameworks)
- Training components (ditto)

### Option 2: With TensorFlow 🔶

The production workhorse setup:

```bash
poetry add mlpotion -E tensorflow
```

**What's included:**

- MLPotion core
- TensorFlow 2.15+
- Keras 3.0+ (automatically included with TensorFlow)
- All TensorFlow-specific components

**Perfect for:**

- Production deployments
- TensorFlow ecosystem users
- Google Cloud Platform projects
- When you need tf.data.Dataset optimization

### Option 3: With PyTorch 🔥

The researcher's choice:

```bash
poetry add mlpotion -E pytorch
```

**What's included:**

- MLPotion core
- PyTorch 2.0+
- TorchVision (for image processing)
- All PyTorch-specific components

**Perfect for:**

- Research projects
- Academic work
- When you love `nn.Module`
- Dynamic computation graphs

### Option 4: With Keras 🎨

The friendly, backend-agnostic option:

```bash
poetry add mlpotion -E keras
```

**What's included:**

- MLPotion core
- Keras 3.0+ (standalone)
- Keras-specific components

**Perfect for:**

- Quick prototyping
- When you want to switch backends later
- Teaching and learning
- Keras fans (obviously!)

### Option 5: Everything! 🎉

When you can't choose or need it all:

```bash
poetry add mlpotion -E all
```

**What's included:**

- MLPotion core
- TensorFlow 2.15+
- PyTorch 2.0+ with TorchVision
- Keras 3.0+
- All framework-specific components

**Warning:**

This will install **a lot** of dependencies (~3GB). Your disk space might need therapy afterward.

### Option 6: With ZenML Integration 🔄

For the MLOps enthusiasts:

```bash
# TensorFlow + ZenML
poetry add mlpotion -E tensorflow -E zenml

# PyTorch + ZenML
poetry add mlpotion -E pytorch -E zenml

# Everything + ZenML (bold choice!)
poetry add mlpotion -E all -E zenml
```

**What you get extra:**

- ZenML integration components
- Pre-built pipeline steps (♻️ REUSABLE)
- Custom materializers
- ZenML-specific utilities

## Installing from Source 🛠️

Want the bleeding edge or contributing? Clone and install:

```bash
# Clone the repository
git clone https://github.com/UnicoLab/MLPotion.git
cd MLPotion

# Install in development mode
pip install -e .

# Or with extras
pip install -e ".[tensorflow]"
pip install -e ".[all]"
```

## Using Poetry 📦

We use Poetry for dependency management. If you prefer Poetry:

```bash
# Clone the repo
git clone https://github.com/UnicoLab/MLPotion.git
cd MLPotion

# Install with Poetry
poetry install

# Or with extras
poetry install -E tensorflow
poetry install -E pytorch
poetry install -E all
```

## Virtual Environments (Highly Recommended!) 🧊

Don't pollute your global Python! Use virtual environments:

### Using venv

```bash
# Create virtual environment
python -m venv mlpotion-env

# Activate (macOS/Linux)
source mlpotion-env/bin/activate

# Activate (Windows)
mlpotion-env\Scripts\activate

# Install MLPotion
pip install mlpotion[tensorflow]
```

### Using conda

```bash
# Create conda environment
conda create -n mlpotion python=3.10

# Activate
conda activate mlpotion

# Install MLPotion
pip install mlpotion[tensorflow]
```

## Verifying Installation ✅

Let's make sure everything works:

```python
# Test core installation
import mlpotion
print(f"MLPotion version: {mlpotion.__version__}")

# Check available frameworks
from mlpotion.utils import get_available_frameworks
print(f"Available frameworks: {get_available_frameworks()}")
```

Expected output:

```
MLPotion version: 0.1.0
Available frameworks: ['tensorflow', 'torch']  # Depends on what you installed
```

### Framework-Specific Tests

#### TensorFlow

```python
from mlpotion.frameworks.tensorflow import TFCSVDataLoader
print("✅ TensorFlow support is working!")
```

#### PyTorch

```python
from mlpotion.frameworks.pytorch import PyTorchCSVDataset
print("✅ PyTorch support is working!")
```

#### Keras

```python
from mlpotion.frameworks.keras import KerasCSVDataLoader
print("✅ Keras support is working!")
```

## Common Installation Issues 🔧

### Issue: TensorFlow not installing on M1/M2 Macs

**Problem:** Apple Silicon can be picky about TensorFlow.

**Solution:**

```bash
# Use conda for M1/M2 Macs
conda create -n mlpotion python=3.10
conda activate mlpotion
conda install -c apple tensorflow-deps
pip install mlpotion[tensorflow]
```

### Issue: PyTorch CUDA version mismatch

**Problem:** PyTorch CUDA version doesn't match your GPU.

**Solution:** Install PyTorch first with the correct CUDA version:

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install MLPotion core only
pip install mlpotion
```

### Issue: Conflicting dependencies

**Problem:** Package version conflicts.

**Solution:** Use a fresh virtual environment:

```bash
# Remove old environment
rm -rf venv

# Create fresh one
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install mlpotion[tensorflow]
```

### Issue: Import errors after installation

**Problem:** Python can't find the package.

**Solution:**

```python
# Check where Python looks for packages
import sys
print(sys.path)

# Verify installation location
pip show mlpotion
```

If they don't match, you might have multiple Python installations. Use `python -m pip` instead of `pip`.

## Upgrading MLPotion 🔄

Keep your potion fresh:

```bash
# Upgrade to latest version
pip install --upgrade mlpotion[tensorflow]

# Or force reinstall everything
pip install --force-reinstall mlpotion[tensorflow]
```

## Uninstalling 🗑️

Sad to see you go, but here's how:

```bash
# Uninstall MLPotion
pip uninstall mlpotion

# Remove the frameworks too (if you want)
pip uninstall tensorflow torch keras
```

## Docker Setup 🐳

Prefer containers? We got you:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install MLPotion with TensorFlow
RUN pip install --no-cache-dir mlpotion[tensorflow]

# Verify installation
RUN python -c "import mlpotion; print(mlpotion.__version__)"

WORKDIR /app
CMD ["python"]
```

Build and run:

```bash
docker build -t mlpotion:latest .
docker run -it mlpotion:latest python
```

## Next Steps 🚀

Installation complete! Now what?

1. **[Quick Start →](quickstart.md)** - Build your first pipeline in 5 minutes
2. **[Core Concepts →](concepts.md)** - Understand the architecture
3. **[Framework Guides →](../frameworks/tensorflow.md)** - Deep dive into your framework

## Need Help? 🆘

- Check the [FAQ](../faq.md) for common questions
- Open an issue on [GitHub](https://github.com/UnicoLab/MLPotion/issues)
- Join our [community discussions](https://github.com/UnicoLab/MLPotion/discussions)

---

<p align="center">
  <strong>Installation successful? Time to brew some magic!</strong> 🧪✨
</p>
