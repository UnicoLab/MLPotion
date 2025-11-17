# ClassGuidelines

Design and testing guidelines for **framework-agnostic, protocol-based utility classes** such as:

- `KerasModelInspector`
- `KerasModelEvaluator`
- `KerasModelExporter`
- Future framework-specific helpers (e.g. PyTorch, XGBoost, etc.)

The goal: **small, predictable, side-effect-light classes** with:

- Clean public APIs (matching a protocol)
- Well-isolated internal helpers
- Clear logging and error handling
- Tests that are **fast**, **deterministic**, and **backend-agnostic** where possible


---

## 1. General Design Principles

### 1.1. Use Protocol-Oriented Design

Each concrete helper should implement a protocol from `mlpotion.core.protocols`, e.g.:

- `ModelInspectorProtocol[ModelType]`
- `ModelEvaluatorProtocol[ModelType]`
- `ModelExporterProtocol[ModelType]`
- `ModelPersistence[ModelType]` (if applicable)

**Guideline:**

- The **public methods** (like `.inspect(...)`, `.evaluate(...)`, `.export(...)`) must exactly match the protocol signature.
- All extra flexibility should be added via `**kwargs` (and documented).

Example:

```python
from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import Model
from mlpotion.core.protocols import ModelEvaluatorProtocol


@dataclass(slots=True)
class KerasModelEvaluator(ModelEvaluatorProtocol[Model]):
    def evaluate(self, model: Model, data: Any, **kwargs: Any) -> dict[str, float]:
        ...
1.2. Use @dataclass(slots=True) for Simple Config Objects
For classes that are essentially configuration + behavior, use:

@dataclass(slots=True)

Small, immutable-ish configuration fields

No heavy logic in __init__ beyond storing parameters

Example:

python
Copy code
@dataclass(slots=True)
class KerasModelInspector(ModelInspectorProtocol[Model]):
    include_layers: bool = True
    include_signatures: bool = True
Benefits:

Simpler construction

Automatic __repr__

Better memory usage with slots=True

1.3. Keep Public Methods Thin, Delegate to Internal Helpers
Pattern:

Public method:

Validate inputs

Extract & normalize kwargs

Logging

Call internal helpers for real work

Normalize return values to a consistent shape

Internal helpers:

Private methods: _validate_model, _ensure_compiled, _get_inputs, etc.

Focus on one concern each.

Example (KerasModelEvaluator):

python
Copy code
def evaluate(self, model: Model, data: Any, **kwargs: Any) -> dict[str, float]:
    self._validate_model(model)

    compile_params = kwargs.pop("compile_params", None)
    eval_params = kwargs.pop("eval_params", None)

    if kwargs:
        logger.warning(f"Unused evaluation kwargs in {self.__class__.__name__}: {list(kwargs.keys())}")

    self._ensure_compiled(model=model, compile_params=compile_params)

    eval_kwargs = dict(eval_params or {})
    eval_kwargs["return_dict"] = True

    result = self._call_evaluate(model=model, data=data, eval_kwargs=eval_kwargs)

    if not isinstance(result, dict):
        logger.warning("Wrapping non-dict result into {'metric_0': ...}")
        result = {"metric_0": float(result)}

    return {str(k): float(v) for k, v in result.items()}
1.4. Strong Validation Up Front
Always have a _validate_model (or equivalent) and call it at the start of public operations.

Example:

python
Copy code
def _validate_model(self, model: Model) -> None:
    if not isinstance(model, keras.Model):
        raise TypeError(
            f"{self.__class__.__name__} expects a keras.Model, got {type(model)!r}"
        )
Similarly for config mappings:

python
Copy code
def _validate_config(self, config: Mapping[str, Any] | None) -> None:
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("`config` must be a mapping (dict-like) or None.")
1.5. Use Duck-Typing for Advanced / Optional Features
For things like signatures, structured_input_signature, or other backend-specific attributes:

Use getattr(...) with defaults

Check with hasattr(...) and guard with try/except

Return safe fallback values

Example (_get_signatures):

python
Copy code
def _get_signatures(self, model: keras.Model) -> dict[str, Any]:
    signatures = getattr(model, "signatures", None)
    if not signatures or not isinstance(signatures, Mapping):
        return {}

    result: dict[str, Any] = {}
    for name, fn in signatures.items():
        inputs = {}
        outputs = {}

        if hasattr(fn, "structured_input_signature"):
            try:
                _, input_spec = fn.structured_input_signature
                inputs = str(input_spec)
            except Exception as exc:
                logger.warning(f"Failed to read structured_input_signature for '{name}': {exc}")

        if hasattr(fn, "structured_outputs"):
            try:
                outputs = str(fn.structured_outputs)
            except Exception as exc:
                logger.warning(f"Failed to read structured_outputs for '{name}': {exc}")

        result[name] = {"inputs": inputs, "outputs": outputs}

    return result
1.6. Make Return Types Boring and Predictable
Examples:

KerasModelInspector.inspect(...) → dict[str, Any] with stable top-level keys

KerasModelEvaluator.evaluate(...) → dict[str, float]

KerasModelExporter.export(...) → None (but logs clearly)

If something upstream returns multiple shapes (e.g. scalar vs dict), normalize it:

python
Copy code
if not isinstance(result, dict):
    logger.warning("Wrapping non-dict result into {'metric_0': ...}")
    result = {"metric_0": float(result)}
1.7. Log at Appropriate Levels
logger.info for high-level, user-visible actions:

“Inspecting Keras model…”

“Evaluating Keras model…”

“Compiling model with provided compile_params.”

“Exporting model to path X…”

logger.debug for detailed inspection:

Data types

Eval kwargs

Layer shapes

logger.warning for:

Suspicious conditions

Ignored kwargs

Non-dict evaluate result

Fallbacks in duck-typed areas

2. Specific Class Patterns
2.1. Inspector Classes (KerasModelInspector)
Key responsibilities:

Read-only operations.

No side effects other than logging.

Extract:

Model name, backend, trainable flag

Inputs / outputs (names, shapes, dtypes)

Parameter counts

Optional per-layer summary

Optional signatures

Structure:

python
Copy code
@dataclass(slots=True)
class KerasModelInspector(ModelInspectorProtocol[ModelLike]):
    include_layers: bool = True
    include_signatures: bool = True

    def inspect(self, model: ModelLike) -> dict[str, Any]:
        self._validate_model(model)
        backend_name = self._get_backend_name()

        info: dict[str, Any] = {
            "name": model.name,
            "backend": backend_name,
            "trainable": model.trainable,
            "inputs": self._get_inputs(model),
            "outputs": self._get_outputs(model),
            "parameters": self._get_param_counts(model),
        }
        info["input_names"] = [i["name"] for i in info["inputs"]]
        info["output_names"] = [o["name"] for o in info["outputs"]]

        if self.include_signatures:
            info["signatures"] = self._get_signatures(model)

        if self.include_layers:
            info["layers"] = self._get_layers_summary(model)

        return info
Design rules:

Don’t mutate the model.

Don’t require specific backends.

Be robust to missing attributes (subclassed models, etc.).

2.2. Evaluator Classes (KerasModelEvaluator)
Key responsibilities:

Wrap model.evaluate in a stable interface.

Normalize different data shapes:

(x, y) / (x, y, sample_weight)

{"x": x, "y": y, ...}

dataset / generator / x only

Ensure the result is dict[str, float].

Optionally handle compilation.

Structure:

python
Copy code
@dataclass(slots=True)
class KerasModelEvaluator(ModelEvaluatorProtocol[Model]):
    def evaluate(self, model: Model, data: Any, **kwargs: Any) -> dict[str, float]:
        self._validate_model(model)

        compile_params = kwargs.pop("compile_params", None)
        eval_params = kwargs.pop("eval_params", None)

        if kwargs:
            logger.warning(
                f"Unused evaluation kwargs in {self.__class__.__name__}: {list(kwargs.keys())}"
            )

        self._ensure_compiled(model=model, compile_params=compile_params)

        eval_kwargs = dict(eval_params or {})
        eval_kwargs["return_dict"] = True

        result = self._call_evaluate(model=model, data=data, eval_kwargs=eval_kwargs)

        if not isinstance(result, dict):
            logger.warning("Wrapping non-dict result into {'metric_0': ...}")
            result = {"metric_0": float(result)}

        return {str(k): float(v) for k, v in result.items()}
Design rules:

Don’t silently run uncompiled models: either compile or raise a clear error.

_is_compiled should be a best-effort heuristic, not rely on Keras internals too heavily.

_call_evaluate should isolate all “data shape” branching.

3. Testing Guidelines
3.1. General Testing Strategy
Use unittest + your existing TestBase for:

temp directory setup/teardown

shared fixtures if needed

Keep unit tests:

Fast

Deterministic

Backend-friendly (no heavy TF export if possible)

Use mocks to isolate:

model.evaluate

model.export

ExportArchive and other heavy I/O

Reserve integration tests for real end-to-end behavior (e.g. saving/loading real models).

3.2. Inspector Tests
For KerasModelInspector:

What to test:

Happy path, simple Sequential model

Name, backend, trainable flag.

Inputs / outputs shapes and dtypes.

Parameter counts (you can calculate expected manually).

Layers summary length and basic keys.

include_layers=False / include_signatures=False

layers and/or signatures keys are absent or empty.

Signatures handling with a dummy model

Create a fake object with .signatures mapping.

Each value exposing structured_input_signature and structured_outputs.

Ensure _get_signatures produces the expected dict.

Type validation

Passing a non-keras.Model raises TypeError.

Example test skeleton:

python
Copy code
import unittest
import keras
import numpy as np
from loguru import logger

from mlpotion.frameworks.keras.models.inspection import KerasModelInspector
from tests.core import TestBase


class TestKerasModelInspector(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(4,)),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        self.inspector = KerasModelInspector()

    def test_basic_inspection(self) -> None:
        info = self.inspector.inspect(self.model)

        self.assertEqual(info["name"], self.model.name)
        self.assertIn("backend", info)
        self.assertIn("trainable", info)
        self.assertIn("inputs", info)
        self.assertIn("outputs", info)
        self.assertIn("parameters", info)

        self.assertIsInstance(info["inputs"], list)
        self.assertIsInstance(info["outputs"], list)
        self.assertIsInstance(info["parameters"]["total"], int)

    def test_include_layers_false(self) -> None:
        inspector = KerasModelInspector(include_layers=False)
        info = inspector.inspect(self.model)
        self.assertNotIn("layers", info)

    def test_validate_model_raises_on_non_keras_model(self) -> None:
        with self.assertRaises(TypeError):
            self.inspector.inspect(object())  # type: ignore[arg-type]
3.3. Evaluator Tests
For KerasModelEvaluator:

Core ideas:

Use real Keras models only when:

It’s cheap (small model, small data).

You don’t need to tightly control internal compiled state.

Use mocks + patching when:

You need to control “compiled vs uncompiled” behavior.

You just want to assert on evaluate call arguments.

You want to simulate evaluate returning a scalar instead of a dict.

What to test:

Evaluate with (x, y) and compile_params

Patch _is_compiled to False to force compilation.

Patch model.compile to assert it’s called with given params.

Assert result is a dict of floats.

Error when uncompiled and no compile_params

Patch _is_compiled to False.

Expect RuntimeError.

Precompiled model: ignore compile_params

Patch _is_compiled to True.

Patch model.compile and assert not called.

Dispatch in _call_evaluate

For tuple (x, y)

For dict {"x": x, "y": y}

For “other” (e.g. a fake dataset)

Wrapping scalar result

evaluate returns 0.5 → final dict has {"metric_0": 0.5}.

Validation & _is_compiled heuristic

_validate_model rejects non-Keras models.

_is_compiled behavior tested with dummy objects, not real Keras internals.

Example test skeleton (simplified):

python
Copy code
import unittest
from unittest.mock import MagicMock, patch

import keras
import numpy as np
from loguru import logger

from mlpotion.frameworks.keras.evaluation import KerasModelEvaluator
from tests.core import TestBase


class TestKerasModelEvaluator(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.num_samples = 16
        self.num_features = 4

        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(self.num_samples, self.num_features)).astype("float32")
        self.y = rng.integers(0, 2, size=(self.num_samples, 1)).astype("float32")

        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.num_features,)),
                keras.layers.Dense(4, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.evaluator = KerasModelEvaluator()

    def test_evaluate_compiles_and_evaluates_tuple_data(self) -> None:
        compile_params = {"optimizer": "adam", "loss": "binary_crossentropy"}
        eval_params = {"verbose": 0}

        with patch.object(self.evaluator, "_is_compiled", return_value=False), patch.object(
            self.model,
            "compile",
            wraps=self.model.compile,
        ) as mock_compile:
            metrics = self.evaluator.evaluate(
                model=self.model,
                data=(self.x, self.y),
                compile_params=compile_params,
                eval_params=eval_params,
            )

        mock_compile.assert_called_once()
        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)

    def test_evaluate_raises_if_uncompiled_and_no_compile_params(self) -> None:
        with patch.object(self.evaluator, "_is_compiled", return_value=False):
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate(model=self.model, data=(self.x, self.y))

    def test_evaluate_wraps_scalar_result(self) -> None:
        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.compiled_loss = object()
                self.optimizer = object()
                self.evaluate = MagicMock(return_value=0.5)

        dummy = DummyModel()
        with patch.object(self.evaluator, "_is_compiled", return_value=True):
            metrics = self.evaluator.evaluate(dummy, data=(self.x, self.y))

        self.assertEqual(metrics["metric_0"], 0.5)
3.4. When to Use Real Models vs Mocks
Use real models when:

You want to ensure your code works with actual Keras API surface.

The operation is cheap:

Simple model.evaluate on small random data.

Simple .save(".keras")/.save(".h5").

Use mocks when:

Testing branching logic, not Keras internals.

Avoiding heavy side effects:

Exporting to SavedModel / onnx / ExportArchive.

Large data or long-running ops.

You need to control/override internal state like “compiled vs uncompiled”.

4. Summary Checklist
When designing a new class similar to KerasModelInspector or KerasModelEvaluator:

 Implements a protocol from mlpotion.core.protocols.

 Uses @dataclass(slots=True) if it has simple configuration.

 Public methods:

 Validate inputs.

 Normalize kwargs (compile_params, eval_params, config, …).

 Delegate logic to internal helpers.

 Return predictable types (e.g. dict[str, float]).

 Internal helpers:

 Small and single-responsibility (_ensure_compiled, _get_inputs, …).

 Duck-typed where necessary (signatures, backends).

 Logging:

 info for high-level actions.

 debug for detailed inspection.

 warning for suspicious/edge conditions.

 Tests:

 Unit tests are fast and use mocks where appropriate.

 Real Keras usage kept small and simple.

 Edge cases and validation behavior covered.

 Use TestBase for filesystem / temp dirs if needed.

Following these guidelines should keep your framework helpers stable, easy to maintain, and pleasant to test.