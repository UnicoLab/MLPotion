import unittest
from unittest.mock import patch

import keras
from loguru import logger

from mlpotion.frameworks.keras.models.inspection import ModelInspector
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class TestModelInspector(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up test model for {self.__class__.__name__}")

        self.num_features = 4

        # Simple Sequential model for inspection tests
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.num_features,)),
                keras.layers.Dense(8, activation="relu", name="dense_1"),
                keras.layers.Dense(1, activation="linear", name="dense_2"),
            ]
        )

        self.inspector = ModelInspector()

    # ------------------------------------------------------------------ #
    # Basic inspection
    # ------------------------------------------------------------------ #
    def test_basic_inspection_contains_expected_keys(self) -> None:
        """inspect() should return a dict with core metadata and structures."""
        logger.info("Testing basic inspection of Keras model")

        info = self.inspector.inspect(self.model)
        logger.info(f"Inspection result: {info}")

        # Top-level keys
        self.assertIn("name", info)
        self.assertIn("backend", info)
        self.assertIn("trainable", info)
        self.assertIn("inputs", info)
        self.assertIn("input_names", info)
        self.assertIn("outputs", info)
        self.assertIn("output_names", info)
        self.assertIn("parameters", info)

        # Optional keys enabled by default
        self.assertIn("layers", info)
        self.assertIn("signatures", info)

        # Types
        self.assertIsInstance(info["name"], str)
        self.assertIsInstance(info["backend"], str)
        self.assertIsInstance(info["trainable"], bool)
        self.assertIsInstance(info["inputs"], list)
        self.assertIsInstance(info["outputs"], list)
        self.assertIsInstance(info["input_names"], list)
        self.assertIsInstance(info["output_names"], list)
        self.assertIsInstance(info["parameters"], dict)
        self.assertIsInstance(info["layers"], list)
        self.assertIsInstance(info["signatures"], dict)

        # Inputs / outputs: at least one each
        self.assertGreaterEqual(len(info["inputs"]), 1)
        self.assertGreaterEqual(len(info["outputs"]), 1)

    # ------------------------------------------------------------------ #
    # Inputs / Outputs content
    # ------------------------------------------------------------------ #
    def test_inputs_and_outputs_have_shape_and_dtype(self) -> None:
        """Inputs/outputs entries should contain index, name, shape, dtype."""
        logger.info("Testing input/output specs structure")

        info = self.inspector.inspect(self.model)

        input_spec = info["inputs"][0]
        self.assertIn("index", input_spec)
        self.assertIn("name", input_spec)
        self.assertIn("shape", input_spec)
        self.assertIn("dtype", input_spec)

        output_spec = info["outputs"][0]
        self.assertIn("index", output_spec)
        self.assertIn("name", output_spec)
        self.assertIn("shape", output_spec)
        self.assertIn("dtype", output_spec)

        self.assertIsInstance(input_spec["shape"], tuple)
        self.assertIsInstance(output_spec["shape"], tuple)
        self.assertIsInstance(input_spec["dtype"], str)
        self.assertIsInstance(output_spec["dtype"], str)

    # ------------------------------------------------------------------ #
    # Parameter counts
    # ------------------------------------------------------------------ #
    def test_param_counts_are_correct(self) -> None:
        """_get_param_counts should compute correct trainable/total parameters."""
        logger.info("Testing parameter counting")

        params = self.inspector._get_param_counts(self.model)
        logger.info(f"Parameter counts: {params}")

        # Model architecture:
        # Input: (4,)
        # Dense(8): weights = 4*8, bias = 8 → 4*8 + 8 = 40
        # Dense(1): weights = 8*1, bias = 1 → 8 + 1 = 9
        # Total trainable = 40 + 9 = 49
        expected_trainable = 49
        expected_non_trainable = 0
        expected_total = expected_trainable + expected_non_trainable

        self.assertEqual(params["trainable"], expected_trainable)
        self.assertEqual(params["non_trainable"], expected_non_trainable)
        self.assertEqual(params["total"], expected_total)

    # ------------------------------------------------------------------ #
    # Layers summary
    # ------------------------------------------------------------------ #
    def test_layers_summary_matches_model_layers(self) -> None:
        """_get_layers_summary should produce one entry per model layer."""
        logger.info("Testing per-layer summary")

        layers_summary = self.inspector._get_layers_summary(self.model)
        logger.info(f"Layers summary: {layers_summary}")

        self.assertEqual(len(layers_summary), len(self.model.layers))

        for entry in layers_summary:
            self.assertIn("name", entry)
            self.assertIn("class_name", entry)
            self.assertIn("trainable", entry)
            self.assertIsInstance(entry["name"], str)
            self.assertIsInstance(entry["class_name"], str)
            self.assertIsInstance(entry["trainable"], bool)

            # input_shape / output_shape are optional
            # but if present, just ensure they are not None
            if "input_shape" in entry:
                self.assertIsNotNone(entry["input_shape"])
            if "output_shape" in entry:
                self.assertIsNotNone(entry["output_shape"])

    def test_include_layers_false_excludes_layers_key(self) -> None:
        """When include_layers=False, 'layers' should not be present in output."""
        logger.info("Testing include_layers=False behavior")

        inspector = ModelInspector(include_layers=False, include_signatures=True)
        info = inspector.inspect(self.model)
        self.assertNotIn("layers", info)
        self.assertIn("signatures", info)

    # ------------------------------------------------------------------ #
    # Signatures
    # ------------------------------------------------------------------ #
    def test_include_signatures_false_excludes_signatures_key(self) -> None:
        """When include_signatures=False, 'signatures' should not be present."""
        logger.info("Testing include_signatures=False behavior")

        inspector = ModelInspector(include_layers=True, include_signatures=False)
        info = inspector.inspect(self.model)
        self.assertIn("layers", info)
        self.assertNotIn("signatures", info)

    def test_get_signatures_from_dummy_model(self) -> None:
        """_get_signatures should inspect a duck-typed 'signatures' mapping."""
        logger.info("Testing _get_signatures with a dummy signatures object")

        class DummySignatureFn:
            def __init__(self) -> None:
                # mimic tf.function structured_* attributes
                self.structured_input_signature = (None, {"x": "input_spec"})
                self.structured_outputs = {"y": "output_spec"}

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.signatures = {
                    "serving_default": DummySignatureFn(),
                }

        dummy_model = DummyModel()
        sigs = self.inspector._get_signatures(dummy_model)
        logger.info(f"Signatures introspection result: {sigs}")

        self.assertIsInstance(sigs, dict)
        self.assertIn("serving_default", sigs)
        entry = sigs["serving_default"]
        self.assertIn("inputs", entry)
        self.assertIn("outputs", entry)
        # They are stringified in the implementation
        self.assertIsInstance(entry["inputs"], str)
        self.assertIsInstance(entry["outputs"], str)

    # ------------------------------------------------------------------ #
    # Backend detection
    # ------------------------------------------------------------------ #
    def test_get_backend_name_uses_keras_config_backend(self) -> None:
        """_get_backend_name should call keras.config.backend()."""
        logger.info("Testing _get_backend_name uses keras.config.backend")

        with patch("keras.config.backend", return_value="mock_backend") as mock_backend:
            backend_name = self.inspector._get_backend_name()

        mock_backend.assert_called_once()
        self.assertEqual(backend_name, "mock_backend")

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def test_inspect_raises_for_non_keras_model(self) -> None:
        """inspect() should raise TypeError for non-keras.Model instances."""
        logger.info("Testing inspect() with invalid model type")

        with self.assertRaises(TypeError):
            self.inspector.inspect(model=object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
