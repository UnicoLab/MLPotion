# MLPotion: Perfect User-Facing API Design

## The Golden Rule ✨

**Users only import from their framework module. Period.**

```python
# ✅ CORRECT - Everything from one place
from mlpotion.frameworks.tensorflow import (
    TFCSVDataLoader,
    TFModelTrainer,
    TensorFlowTrainingConfig,
    inspect_model,
)

# ❌ WRONG - Never make users do this!
from mlpotion.frameworks.tensorflow import TFCSVDataLoader
from mlpotion.frameworks.keras import KerasModelTrainer  # NO!
```

---

## Directory Structure

```
mlpotion/frameworks/
│
├── keras/                          # INTERNAL: Implementation
│   ├── __init__.py                # Exports: KerasModelTrainer, etc.
│   ├── config.py                  
│   ├── models/
│   │   └── inspection.py         
│   ├── training/
│   │   └── trainers.py           # KerasModelTrainer (actual code)
│   ├── evaluation/
│   │   └── evaluators.py         
│   └── deployment/
│
├── tensorflow/                     # PUBLIC: Complete TF API
│   ├── __init__.py                # RE-EXPORTS EVERYTHING
│   ├── config.py                  
│   ├── data/                      
│   │   ├── loaders.py            
│   │   └── optimizers.py         
│   ├── training/                  
│   │   └── trainers.py           # TFModelTrainer = alias
│   ├── evaluation/                
│   │   └── evaluators.py         
│   ├── deployment/                
│   │   ├── exporters.py          
│   │   └── persistence.py        
│   ├── models/                    
│   │   └── inspection.py         # Re-export from keras
│   └── utils/
│
└── pytorch/                        # PUBLIC: Complete PyTorch API
    ├── __init__.py                # Exports everything PyTorch
    ├── config.py
    ├── models/
    ├── training/
    ├── evaluation/
    └── deployment/
```

---

## Implementation: Complete Re-Exports

### 1. TensorFlow __init__.py (Complete API)

**`mlpotion/frameworks/tensorflow/__init__.py`:**

```python
"""TensorFlow framework implementation.

Complete TensorFlow API - users import everything from here.

Example:
    from mlpotion.frameworks.tensorflow import (
        TFCSVDataLoader,
        TFModelTrainer,
        TensorFlowTrainingConfig,
        inspect_model,
    )
    
    # All TensorFlow needs in one import!
"""

# ============================================================================
# Configuration
# ============================================================================
from mlpotion.frameworks.tensorflow.config import (
    TensorFlowTrainingConfig,
    TensorFlowEvaluationConfig,
    TensorFlowExportConfig,
)

# ============================================================================
# Data Loading & Processing (TensorFlow-specific)
# ============================================================================
from mlpotion.frameworks.tensorflow.data.loaders import TFCSVDataLoader
from mlpotion.frameworks.tensorflow.data.optimizers import TFDatasetOptimizer

# ============================================================================
# Training (from Keras, aliased)
# ============================================================================
from mlpotion.frameworks.tensorflow.training.trainers import TFModelTrainer

# ============================================================================
# Evaluation (from Keras, aliased)
# ============================================================================
from mlpotion.frameworks.tensorflow.evaluation.evaluators import TFModelEvaluator

# ============================================================================
# Deployment (from Keras, aliased)
# ============================================================================
from mlpotion.frameworks.tensorflow.deployment.exporters import TFModelExporter
from mlpotion.frameworks.tensorflow.deployment.persistence import (
    TFModelPersistence,
    save_model,
    load_model,
)

# ============================================================================
# Model Inspection (from Keras, re-exported)
# ============================================================================
from mlpotion.frameworks.tensorflow.models.inspection import (
    ModelInspector,
    inspect_model,
    get_io_shapes,
    validate_input_shape,
)

# ============================================================================
# Public API - Everything users need
# ============================================================================
__all__ = [
    # Configuration
    "TensorFlowTrainingConfig",
    "TensorFlowEvaluationConfig",
    "TensorFlowExportConfig",
    
    # Data
    "TFCSVDataLoader",
    "TFDatasetOptimizer",
    
    # Training
    "TFModelTrainer",
    
    # Evaluation
    "TFModelEvaluator",
    
    # Deployment
    "TFModelExporter",
    "TFModelPersistence",
    "save_model",
    "load_model",
    
    # Inspection
    "ModelInspector",
    "inspect_model",
    "get_io_shapes",
    "validate_input_shape",
]
```

### 2. TensorFlow Training (Alias to Keras)

**`mlpotion/frameworks/tensorflow/training/__init__.py`:**

```python
"""TensorFlow training components."""

from mlpotion.frameworks.tensorflow.training.trainers import TFModelTrainer

__all__ = ["TFModelTrainer"]
```

**`mlpotion/frameworks/tensorflow/training/trainers.py`:**

```python
"""TensorFlow model trainers.

Provides TensorFlow-friendly API that wraps Keras implementation.
"""

from mlpotion.frameworks.keras.training.trainers import KerasModelTrainer

# Create TensorFlow-named alias
# Users see "TFModelTrainer", internally uses KerasModelTrainer
TFModelTrainer = KerasModelTrainer

__all__ = ["TFModelTrainer"]
```

### 3. TensorFlow Models (Re-export from Keras)

**`mlpotion/frameworks/tensorflow/models/__init__.py`:**

```python
"""TensorFlow model utilities."""

from mlpotion.frameworks.tensorflow.models.inspection import (
    ModelInspector,
    inspect_model,
    get_io_shapes,
    validate_input_shape,
)

__all__ = [
    "ModelInspector",
    "inspect_model",
    "get_io_shapes",
    "validate_input_shape",
]
```

**`mlpotion/frameworks/tensorflow/models/inspection.py`:**

```python
"""TensorFlow model inspection.

Re-exports Keras model inspection with TensorFlow-friendly names.
"""

# Import from Keras (the actual implementation)
from mlpotion.frameworks.keras.models.inspection import (
    ModelInspector,
    inspect_model as keras_inspect_model,
    get_io_shapes as keras_get_io_shapes,
    validate_input_shape as keras_validate_input_shape,
)

# Re-export with same names (users don't see difference)
inspect_model = keras_inspect_model
get_io_shapes = keras_get_io_shapes
validate_input_shape = keras_validate_input_shape

__all__ = [
    "ModelInspector",
    "inspect_model",
    "get_io_shapes",
    "validate_input_shape",
]
```

### 4. TensorFlow Evaluation (Alias to Keras)

**`mlpotion/frameworks/tensorflow/evaluation/evaluators.py`:**

```python
"""TensorFlow model evaluators."""

from mlpotion.frameworks.keras.evaluation.evaluators import KerasModelEvaluator

# Create alias
TFModelEvaluator = KerasModelEvaluator

__all__ = ["TFModelEvaluator"]
```

### 5. TensorFlow Deployment (Alias to Keras)

**`mlpotion/frameworks/tensorflow/deployment/exporters.py`:**

```python
"""TensorFlow model exporters."""

from mlpotion.frameworks.keras.deployment.exporters import KerasModelExporter

# Create alias
TFModelExporter = KerasModelExporter

__all__ = ["TFModelExporter"]
```

**`mlpotion/frameworks/tensorflow/deployment/persistence.py`:**

```python
"""TensorFlow model persistence."""

from mlpotion.frameworks.keras.deployment.persistence import (
    KerasModelPersistence,
    save_model as keras_save_model,
    load_model as keras_load_model,
)

# Create aliases
TFModelPersistence = KerasModelPersistence
save_model = keras_save_model
load_model = keras_load_model

__all__ = [
    "TFModelPersistence",
    "save_model",
    "load_model",
]
```

### 6. TensorFlow Config (Alias to Keras)

**`mlpotion/frameworks/tensorflow/config.py`:**

```python
"""TensorFlow configuration.

Provides TensorFlow-friendly config names that wrap Keras configs.
"""

from mlpotion.frameworks.keras.config import (
    KerasTrainingConfig,
    KerasEvaluationConfig,
    KerasExportConfig,
)

# Create TensorFlow-named aliases
TensorFlowTrainingConfig = KerasTrainingConfig
TensorFlowEvaluationConfig = KerasEvaluationConfig
TensorFlowExportConfig = KerasExportConfig

__all__ = [
    "TensorFlowTrainingConfig",
    "TensorFlowEvaluationConfig",
    "TensorFlowExportConfig",
]
```

### 7. Keras __init__.py (For direct Keras users)

**`mlpotion/frameworks/keras/__init__.py`:**

```python
"""Keras framework implementation.

Backend-agnostic Keras components.

Example:
    import os
    os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"
    
    from mlpotion.frameworks.keras import (
        KerasModelTrainer,
        KerasTrainingConfig,
        inspect_model,
    )
"""

from mlpotion.frameworks.keras.config import (
    KerasTrainingConfig,
    KerasEvaluationConfig,
    KerasExportConfig,
)
from mlpotion.frameworks.keras.training.trainers import KerasModelTrainer
from mlpotion.frameworks.keras.evaluation.evaluators import KerasModelEvaluator
from mlpotion.frameworks.keras.deployment.exporters import KerasModelExporter
from mlpotion.frameworks.keras.deployment.persistence import (
    KerasModelPersistence,
    save_model,
    load_model,
)
from mlpotion.frameworks.keras.models.inspection import (
    ModelInspector,
    inspect_model,
    get_io_shapes,
    validate_input_shape,
)

__all__ = [
    # Config
    "KerasTrainingConfig",
    "KerasEvaluationConfig",
    "KerasExportConfig",
    
    # Training
    "KerasModelTrainer",
    
    # Evaluation
    "KerasModelEvaluator",
    
    # Deployment
    "KerasModelExporter",
    "KerasModelPersistence",
    "save_model",
    "load_model",
    
    # Inspection
    "ModelInspector",
    "inspect_model",
    "get_io_shapes",
    "validate_input_shape",
]
```

### 8. PyTorch __init__.py (Independent)

**`mlpotion/frameworks/pytorch/__init__.py`:**

```python
"""PyTorch framework implementation.

Complete PyTorch API - users import everything from here.

Example:
    from mlpotion.frameworks.pytorch import (
        PyTorchCSVDataset,
        PyTorchModelTrainer,
        PyTorchTrainingConfig,
        inspect_model,
    )
"""

from mlpotion.frameworks.pytorch.config import (
    PyTorchTrainingConfig,
    PyTorchEvaluationConfig,
    PyTorchExportConfig,
)
from mlpotion.frameworks.pytorch.data.datasets import PyTorchCSVDataset
from mlpotion.frameworks.pytorch.data.loaders import PyTorchDataLoaderFactory
from mlpotion.frameworks.pytorch.training.trainers import PyTorchModelTrainer
from mlpotion.frameworks.pytorch.evaluation.evaluators import PyTorchModelEvaluator
from mlpotion.frameworks.pytorch.deployment.exporters import PyTorchModelExporter
from mlpotion.frameworks.pytorch.deployment.persistence import (
    PyTorchModelPersistence,
    save_model,
    load_model,
)
from mlpotion.frameworks.pytorch.models.inspection import (
    ModelInspector,
    inspect_model,
    get_layer_info,
)

__all__ = [
    # Config
    "PyTorchTrainingConfig",
    "PyTorchEvaluationConfig",
    "PyTorchExportConfig",
    
    # Data
    "PyTorchCSVDataset",
    "PyTorchDataLoaderFactory",
    
    # Training
    "PyTorchModelTrainer",
    
    # Evaluation
    "PyTorchModelEvaluator",
    
    # Deployment
    "PyTorchModelExporter",
    "PyTorchModelPersistence",
    "save_model",
    "load_model",
    
    # Inspection
    "ModelInspector",
    "inspect_model",
    "get_layer_info",
]
```

---

## User Experience Examples

### Example 1: TensorFlow User (Everything in One Import)

```python
# Single import has EVERYTHING
from mlpotion.frameworks.tensorflow import (
    # Data
    TFCSVDataLoader,
    TFDatasetOptimizer,
    
    # Training
    TFModelTrainer,
    TensorFlowTrainingConfig,
    
    # Evaluation
    TFModelEvaluator,
    
    # Inspection
    inspect_model,
    get_io_shapes,
    
    # Deployment
    save_model,
    load_model,
)

# Use everything!
loader = TFCSVDataLoader("data.csv")
dataset = loader.load()

optimizer = TFDatasetOptimizer(batch_size=32)
dataset = optimizer.optimize(dataset)

trainer = TFModelTrainer()
config = TensorFlowTrainingConfig(epochs=10)
result = trainer.train(model, dataset, config)

evaluator = TFModelEvaluator()
eval_result = evaluator.evaluate(model, test_dataset)

shapes = get_io_shapes(model)
inspector = inspect_model(model)

save_model(model, "model.keras")
loaded_model = load_model("model.keras")
```

**User never knows or cares that some components come from Keras!**

### Example 2: PyTorch User (Everything in One Import)

```python
# Single import has EVERYTHING
from mlpotion.frameworks.pytorch import (
    # Data
    PyTorchCSVDataset,
    PyTorchDataLoaderFactory,
    
    # Training
    PyTorchModelTrainer,
    PyTorchTrainingConfig,
    
    # Evaluation
    PyTorchModelEvaluator,
    
    # Inspection
    inspect_model,
    
    # Deployment
    save_model,
    load_model,
)

# Use everything!
dataset = PyTorchCSVDataset("data.csv")
factory = PyTorchDataLoaderFactory(batch_size=32)
dataloader = factory.create(dataset)

trainer = PyTorchModelTrainer()
config = PyTorchTrainingConfig(epochs=10)
result = trainer.train(model, dataloader, config)

inspector = inspect_model(model)

save_model(model, "model.pth")
loaded_model = load_model("model.pth")
```

### Example 3: Keras-Only User (Lightweight)

```python
# Single import has EVERYTHING for Keras
import os
os.environ["KERAS_BACKEND"] = "jax"  # Choose backend

from mlpotion.frameworks.keras import (
    # Training
    KerasModelTrainer,
    KerasTrainingConfig,
    
    # Evaluation
    KerasModelEvaluator,
    
    # Inspection
    inspect_model,
    
    # Deployment
    save_model,
    load_model,
)

# Use Keras with JAX backend!
trainer = KerasModelTrainer()
config = KerasTrainingConfig(epochs=10)
result = trainer.train(model, dataset, config)
```

---

## Benefits

✅ **Simple User API**
- One import per framework
- Users never see internal structure
- Framework-focused, not scattered

✅ **Internal Modularity**
- Keras code is reusable
- Clear separation of concerns
- Easy to maintain

✅ **No Confusion**
- TensorFlow users: import from `tensorflow`
- PyTorch users: import from `pytorch`
- Keras users: import from `keras`

✅ **Flexibility**
- Keras users get lightweight install
- TensorFlow users get complete API
- PyTorch users stay independent

✅ **Future-Proof**
- Easy to add new frameworks
- Easy to expose Keras separately
- Modular and extensible

---

## Installation Behavior

```bash
# Keras only (~100 MB)
pip install mlpotion[keras]
# Can import from: mlpotion.frameworks.keras

# TensorFlow (~600 MB, includes Keras)
pip install mlpotion[tensorflow]
# Can import from: mlpotion.frameworks.tensorflow
# Can ALSO import from: mlpotion.frameworks.keras (for advanced users)

# PyTorch (~2 GB)
pip install mlpotion[pytorch]
# Can import from: mlpotion.frameworks.pytorch
```

---

## Key Insight

**Aliasing + Re-exporting = Perfect User Experience**

```python
# Internal structure (users don't see this)
mlpotion/frameworks/keras/training/trainers.py  # KerasModelTrainer

# TensorFlow re-exports (users see this)
mlpotion/frameworks/tensorflow/training/trainers.py  # TFModelTrainer = KerasModelTrainer
mlpotion/frameworks/tensorflow/__init__.py  # exports TFModelTrainer

# User experience (clean and simple!)
from mlpotion.frameworks.tensorflow import TFModelTrainer  # ✅ Perfect!
```

---

## Summary

**Perfect structure:**
1. ✅ `keras/` has all Keras implementation
2. ✅ `tensorflow/` re-exports from `keras/` + adds TF-specific
3. ✅ `pytorch/` is independent
4. ✅ Users import everything from their framework module
5. ✅ No scattered imports, ever!

**This is the perfect design!** 🎯