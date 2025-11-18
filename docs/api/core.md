# Core API Reference 📖

Complete API reference for MLPotion's framework-agnostic core components.

!!! info "Auto-Generated Documentation"
    This page will be automatically populated with API documentation from the source code once the package is fully documented. For now, refer to the source code and inline docstrings.

## Protocols

MLPotion uses Python protocols for type-safe, framework-agnostic interfaces:

### DataLoader

**Module**: `mlpotion.core.protocols`

Protocol for data loading components that produce framework-specific datasets.

```python
from typing import Protocol, TypeVar

DatasetT = TypeVar("DatasetT")

class DataLoader(Protocol[DatasetT]):
    def load(self) -> DatasetT:
        """Load data and return framework-specific dataset."""
        ...
```

### ModelTrainer

**Module**: `mlpotion.core.protocols`

Protocol for model training components.

```python
class ModelTrainer(Protocol[ModelT, DatasetT]):
    def train(
        self,
        model: ModelT,
        dataset: DatasetT,
        config: TrainingConfig,
        validation_dataset: DatasetT | None = None,
    ) -> TrainingResult[ModelT]:
        """Train a model and return results."""
        ...
```

### ModelEvaluator

**Module**: `mlpotion.core.protocols`

Protocol for model evaluation components.

### ModelPersistence

**Module**: `mlpotion.core.protocols`

Protocol for saving and loading models.

### ModelExporter

**Module**: `mlpotion.core.protocols`

Protocol for exporting models to production formats.

## Result Types

### TrainingResult

**Module**: `mlpotion.core.results`

Dataclass containing training results.

**Attributes**:
- `model: ModelT` - Trained model
- `history: dict[str, list[float]]` - Training history
- `metrics: dict[str, float]` - Final metrics
- `config: TrainingConfig` - Configuration used
- `training_time: float | None` - Training duration
- `best_epoch: int | None` - Best epoch number

### EvaluationResult

**Module**: `mlpotion.core.results`

Dataclass containing evaluation results.

**Attributes**:
- `metrics: dict[str, float]` - Evaluation metrics
- `config: EvaluationConfig` - Configuration used
- `evaluation_time: float | None` - Evaluation duration

### ExportResult

**Module**: `mlpotion.core.results`

Dataclass containing export results.

**Attributes**:
- `export_path: str` - Export location
- `format: str` - Export format
- `config: ExportConfig` - Configuration used
- `metadata: dict[str, Any]` - Additional metadata

## Configurations

### TrainingConfig

**Module**: `mlpotion.core.config`

Base configuration for training. Framework-specific configs inherit from this.

### EvaluationConfig

**Module**: `mlpotion.core.config`

Base configuration for evaluation.

### ExportConfig

**Module**: `mlpotion.core.config`

Base configuration for model export.

## Exceptions

### MLPotionError

**Module**: `mlpotion.core.exceptions`

Base exception for all MLPotion errors.

### DataLoadingError

**Module**: `mlpotion.core.exceptions`

Raised when data loading fails.

### TrainingError

**Module**: `mlpotion.core.exceptions`

Raised when training fails.

### EvaluationError

**Module**: `mlpotion.core.exceptions`

Raised when evaluation fails.

### ExportError

**Module**: `mlpotion.core.exceptions`

Raised when export fails.

## Utilities

### Framework Detection

**Module**: `mlpotion.utils.framework`

```python
def is_framework_available(framework: str) -> bool:
    """Check if a framework is available.

    Args:
        framework: Framework name ('tensorflow', 'torch', 'keras')

    Returns:
        True if framework is installed and importable
    """

def get_available_frameworks() -> list[str]:
    """Get list of available frameworks.

    Returns:
        List of installed framework names
    """
```

### Decorators

**Module**: `mlpotion.utils.decorators`

```python
def trycatch(
    default_return: Any = None,
    catch: tuple[type[Exception], ...] = (Exception,),
    reraise: bool = False,
):
    """Decorator for exception handling with configurable behavior."""
```

---

<p align="center">
  <em>For framework-specific APIs, see the respective framework documentation</em>
</p>
