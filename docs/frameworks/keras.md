# Keras Guide 🎨

Complete guide to using MLPotion with Keras 3 - the user-friendly, backend-agnostic ML framework!

## Why Keras + MLPotion? 🤔

- **User-Friendly**: Simple, consistent API
- **Backend-Agnostic**: Switch between TensorFlow, PyTorch, and JAX
- **Fast Prototyping**: Build models quickly
- **Production-Ready**: Keras 3 is production-grade
- **MLPotion Benefits**: Type-safe, modular components

## Installation 📥

```bash
# Keras with TensorFlow backend (default)
poetry add mlpotion -E tensorflow

# Keras with PyTorch backend
poetry add mlpotion -E keras-pytorch

# Keras with JAX backend
poetry add mlpotion -E keras-jax
```

## Quick Example 🚀

```python
from mlpotion.frameworks.keras import (
    CSVDataLoader,
    ModelTrainer,
    ModelTrainingConfig,
)
import keras

# Load data
loader = CSVDataLoader("data.csv", label_name="target", batch_size=32)
dataset = loader.load()

# Create model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Train
trainer = ModelTrainer()
config = ModelTrainingConfig(epochs=10, learning_rate=0.001)
result = trainer.train(model, dataset, config)

print(f"Final loss: {result.metrics['loss']:.4f}")
```

For complete Keras documentation, see the [TensorFlow Guide](tensorflow.md) as Keras components use the same patterns!

---

<p align="center">
  <strong>Keras + MLPotion = Simplicity + Power!</strong> 🎨
</p>
