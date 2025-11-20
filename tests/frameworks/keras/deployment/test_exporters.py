import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import keras
import numpy as np
from loguru import logger

from mlpotion.frameworks.keras.deployment.exporters import ModelExporter
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class TestModelExporter(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Creating temp directory for {self.__class__.__name__}")

        self.keras_path = self.temp_dir / "test_model.keras"
        self.h5_path = self.temp_dir / "test_model.h5"
        self.export_dir = self.temp_dir / "saved_model_dir"
        logger.info(f"Keras path: {self.keras_path}")
        logger.info(f"H5 path: {self.h5_path}")
        logger.info(f"Export dir: {self.export_dir}")

        logger.info("Building test Keras model")
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(4,)),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        logger.info("Test Keras model created")

        self.exporter = ModelExporter()

    # ------------------------------------------------------------------ #
    # Native Keras serialization tests
    # ------------------------------------------------------------------ #
    def test_export_native_keras_format_inferred_from_extension(self) -> None:
        """Export with .keras extension should use native save and create a file."""
        logger.info("Testing export with .keras extension (format inferred)")

        self.exporter.export(self.model, str(self.keras_path))

        logger.info("Checking that .keras file exists")
        self.assertTrue(self.keras_path.exists())
        self.assertTrue(self.keras_path.is_file())

    def test_export_native_h5_format_explicit(self) -> None:
        """Export with explicit 'h5' format should use model.save and create a file."""
        logger.info("Testing export with explicit export_format='h5'")

        self.exporter.export(
            self.model,
            str(self.h5_path),
            export_format="h5",
        )

        logger.info("Checking that .h5 file exists")
        self.assertTrue(self.h5_path.exists())
        self.assertTrue(self.h5_path.is_file())

    # ------------------------------------------------------------------ #
    # Inference export via model.export
    # ------------------------------------------------------------------ #
    def test_export_with_model_export_uses_model_export_method(self) -> None:
        """Export with non-native format should call model.export(...) with config."""
        logger.info("Testing export via model.export(...)")

        # Attach a mock export method to the model
        self.model.export = MagicMock(name="export")

        config = {"foo": "bar"}
        export_format = "tf_saved_model"

        self.exporter.export(
            self.model,
            str(self.export_dir),
            export_format=export_format,
            config=config,
        )

        logger.info("Asserting model.export was called once with expected args")
        self.model.export.assert_called_once()
        args, kwargs = self.model.export.call_args

        self.assertEqual(args[0], str(self.export_dir))
        self.assertEqual(kwargs["format"], export_format)
        self.assertIn("foo", kwargs)
        self.assertEqual(kwargs["foo"], "bar")

        # NOTE: we do NOT assert on self.export_dir.exists() here because
        # model.export is mocked and does not actually create any files.

    def test_export_with_model_export_raises_for_missing_export_method(self) -> None:
        """_export_with_model_export should raise if model has no export method."""
        logger.info("Testing _export_with_model_export with a non-exporting model")

        dummy_model = object()  # not a keras.Model; we test the internal helper only

        with self.assertRaises(RuntimeError):
            self.exporter._export_with_model_export(
                model=dummy_model,  # type: ignore[arg-type]
                path=self.export_dir,
                export_format="tf_saved_model",
                config=None,
            )

    # ------------------------------------------------------------------ #
    # Export via ExportArchive (custom endpoint)
    # ------------------------------------------------------------------ #
    def test_export_with_export_archive(self) -> None:
        """Export with endpoint_name or input_specs should use ExportArchive."""
        logger.info("Testing export via ExportArchive with custom endpoint")

        input_spec = keras.InputSpec(shape=(None, 4), dtype="float32")
        endpoint_name = "serve"

        with patch("keras.export.ExportArchive") as MockExportArchive:
            archive_instance = MockExportArchive.return_value

            self.exporter.export(
                self.model,
                str(self.export_dir),
                export_format="tf_saved_model",
                endpoint_name=endpoint_name,
                input_specs=[input_spec],
            )

            logger.info("Asserting ExportArchive was constructed")
            MockExportArchive.assert_called_once()

            logger.info("Asserting archive.track(model) was called")
            archive_instance.track.assert_called_once_with(self.model)

            logger.info("Asserting archive.add_endpoint(...) was called")
            archive_instance.add_endpoint.assert_called_once()
            args, kwargs = archive_instance.add_endpoint.call_args

            # Endpoint name
            self.assertEqual(kwargs["name"], endpoint_name)

            # fn should be `self.model.call` – but compare based on binding,
            # not object identity of the bound method
            fn = kwargs["fn"]
            self.assertEqual(fn.__name__, "call")
            self.assertIs(fn.__self__, self.model)
            self.assertIs(fn.__func__, self.model.call.__func__)

            # input_signature should be a list (possibly of InputSpec)
            self.assertIsInstance(kwargs.get("input_signature"), list)
            self.assertEqual(len(kwargs["input_signature"]), 1)

            logger.info("Asserting archive.write_out(...) was called with export path")
            archive_instance.write_out.assert_called_once_with(str(self.export_dir))

    # ------------------------------------------------------------------ #
    # Warm-up behavior
    # ------------------------------------------------------------------ #
    def test_warmup_runs_only_one_batch(self) -> None:
        """Warm-up dataset should call the model exactly once."""
        logger.info("Testing warm-up behavior with a custom model")

        class WarmupModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            def call(self, inputs, *_, **__):  # type: ignore[override]
                self.call_count += 1
                return inputs

        warmup_model = WarmupModel()
        dataset = [
            np.zeros((2, 4), dtype="float32"),
            np.zeros((2, 4), dtype="float32"),
        ]

        # Avoid actually saving anything: patch _save_native_keras
        with patch.object(ModelExporter, "_save_native_keras") as mock_save:
            logger.info("Calling exporter.export with warm-up dataset")
            self.exporter.export(
                warmup_model,
                str(self.keras_path),
                dataset=dataset,
                export_format="keras",
            )

            logger.info("Asserting model.call was invoked exactly once")
            self.assertEqual(warmup_model.call_count, 1)

            logger.info("Asserting _save_native_keras was called once")
            mock_save.assert_called_once()

    # ------------------------------------------------------------------ #
    # Validation and format helpers
    # ------------------------------------------------------------------ #
    def test_validate_model_raises_for_non_keras_model(self) -> None:
        """_validate_model should reject non-keras.Model instances."""
        logger.info("Testing _validate_model with invalid model")

        with self.assertRaises(TypeError):
            self.exporter._validate_model(model=object())  # type: ignore[arg-type]

    def test_validate_config_accepts_mapping_or_none(self) -> None:
        """_validate_config should accept mapping or None and reject others."""
        logger.info("Testing _validate_config with valid configs")
        self.exporter._validate_config(config=None)
        self.exporter._validate_config(config={"a": 1})

        logger.info("Testing _validate_config with invalid config (non-mapping)")
        with self.assertRaises(TypeError):
            self.exporter._validate_config(config="not a mapping")  # type: ignore[arg-type]

    def test_infer_export_format_from_path(self) -> None:
        """_infer_export_format_from_path should map suffixes to formats."""
        logger.info("Testing _infer_export_format_from_path")

        self.assertEqual(
            self.exporter._infer_export_format_from_path(Path("model.keras")),
            "keras",
        )
        self.assertEqual(
            self.exporter._infer_export_format_from_path(Path("model.h5")),
            "h5",
        )
        # No known extension → default to 'tf_saved_model'
        self.assertEqual(
            self.exporter._infer_export_format_from_path(Path("model_dir")),
            "tf_saved_model",
        )

    def test_is_native_keras_format(self) -> None:
        """_is_native_keras_format should recognize 'keras' and 'h5'."""
        logger.info("Testing _is_native_keras_format")

        self.assertTrue(ModelExporter._is_native_keras_format("keras"))
        self.assertTrue(ModelExporter._is_native_keras_format("h5"))
        self.assertFalse(ModelExporter._is_native_keras_format("tf_saved_model"))
        self.assertFalse(ModelExporter._is_native_keras_format("onnx"))


if __name__ == "__main__":
    unittest.main()
