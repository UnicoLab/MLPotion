import unittest
from unittest.mock import MagicMock

import keras
import numpy as np
import pandas as pd
from loguru import logger

from mlpotion.frameworks.keras.config import KerasCSVTransformationConfig
from mlpotion.core.exceptions import DataTransformationError
from mlpotion.core.results import TransformationResult
from mlpotion.frameworks.keras.data.loaders import CSVSequence, CSVDataLoader
from mlpotion.frameworks.keras.data.transformers import CSVDataTransformer
from mlpotion.frameworks.keras.deployment.persistence import KerasModelPersistence
from tests.core import TestBase  # provides temp_dir, setUp/tearDown



class TestCSVDataTransformer(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up CSVDataTransformer tests in {self.temp_dir}")

        # Simple synthetic data: 10 samples, 2 features
        self.n_samples = 10
        self.n_features = 2
        rng = np.random.default_rng(123)
        self.features = rng.normal(size=(self.n_samples, self.n_features)).astype(
            "float32"
        )
        self.labels = rng.integers(0, 2, size=(self.n_samples,)).astype("float32")

        # Simple Keras model
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.n_features,)),
                keras.layers.Dense(4, activation="relu"),
                keras.layers.Dense(1, activation="linear"),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")

    def _make_config(
        self,
        data_output_path: str,
        per_batch: bool = False,
    ) -> KerasCSVTransformationConfig:
        """Create a minimal KerasCSVTransformationConfig for tests."""
        return KerasCSVTransformationConfig(
            data_output_path=data_output_path,
            data_output_per_batch=per_batch,
            # leave optional fields (file_pattern, model_path, batch_size, config) as defaults
        )

    # ------------------------------------------------------------------ #
    # End-to-end: Sequence + single output CSV
    # ------------------------------------------------------------------ #
    def test_transform_with_sequence_and_single_output_csv(self) -> None:
        """Transform a CSVSequence and save a single CSV file."""
        logger.info("Testing CSVDataTransformer with CSVSequence and single CSV output")

        seq = CSVSequence(
            features=self.features,
            labels=self.labels,
            batch_size=4,
            shuffle=False,
        )

        out_path = self.temp_dir / "predictions.csv"

        transformer = CSVDataTransformer(
            dataset=seq,
            model=self.model,
            data_output_path=str(out_path),
            data_output_per_batch=False,
        )
        # Avoid automatic input_columns from inspection to keep all columns
        transformer.input_columns = []  # falsy → use all columns

        config = KerasCSVTransformationConfig(data_output_path=str(out_path))

        result = transformer.transform(
            dataset=None,
            model=None,
            config=config,
        )
        self.assertIsInstance(result, TransformationResult)
        self.assertEqual(result.data_output_path, str(out_path))

        self.assertTrue(out_path.exists(), f"Output file {out_path} not found")
        df = pd.read_csv(out_path)
        logger.info(f"Transformed CSV shape: {df.shape}, columns: {df.columns.tolist()}")

        # Sequence yields only features as input (2 columns) + prediction
        self.assertGreaterEqual(df.shape[0], self.n_samples)
        self.assertIn("prediction", df.columns)
        self.assertEqual(df.shape[1], self.n_features + 1)

    # ------------------------------------------------------------------ #
    # End-to-end: DataFrame + per-batch CSV
    # ------------------------------------------------------------------ #
    def test_transform_with_dataframe_and_per_batch_output(self) -> None:
        """Transform a DataFrame and save one CSV per batch."""
        logger.info(
            "Testing CSVDataTransformer with DataFrame and per-batch CSV output"
        )

        df = pd.DataFrame(
            {
                "feature_0": self.features[:, 0],
                "feature_1": self.features[:, 1],
                "extra": np.arange(self.n_samples, dtype="float32"),
            }
        )

        out_dir = self.temp_dir / "batches"

        transformer = CSVDataTransformer(
            dataset=df,
            model=self.model,
            data_output_path=str(out_dir),
            data_output_per_batch=True,
            batch_size=4,
        )
        # IMPORTANT: only pass the real model inputs
        transformer.input_columns = ["feature_0", "feature_1"]

        config = self._make_config(str(out_dir), per_batch=True)

        result = transformer.transform(
            dataset=None,
            model=None,
            config=config,
        )
        self.assertEqual(result.data_output_path, str(out_dir))

        # Check batch files
        expected_files = [
            out_dir / "batch_00000.csv",
            out_dir / "batch_00001.csv",
            out_dir / "batch_00002.csv",
        ]
        for f in expected_files:
            self.assertTrue(f.exists(), f"Expected per-batch file missing: {f}")

        total_rows = 0
        for f in expected_files:
            batch_df = pd.read_csv(f)
            self.assertIn("prediction", batch_df.columns)
            total_rows += len(batch_df)

        self.assertEqual(total_rows, self.n_samples)


    # ------------------------------------------------------------------ #
    # _iter_batches: numpy with batch_size
    # ------------------------------------------------------------------ #
    def test_iter_batches_numpy_respects_batch_size(self) -> None:
        """_iter_batches should split NumPy arrays according to batch_size."""
        logger.info("Testing _iter_batches with NumPy array")

        x = np.arange(20, dtype="float32").reshape(10, 2)

        transformer = CSVDataTransformer(batch_size=3)
        batches = list(transformer._iter_batches(x))

        self.assertEqual(len(batches), 4)  # 3+3+3+1 rows
        self.assertEqual(batches[0].shape, (3, 2))
        self.assertEqual(batches[1].shape, (3, 2))
        self.assertEqual(batches[2].shape, (3, 2))
        self.assertEqual(batches[3].shape, (1, 2))

    # ------------------------------------------------------------------ #
    # _restrict_batch_columns behavior
    # ------------------------------------------------------------------ #
    def test_restrict_batch_columns_filters_and_preserves_order(self) -> None:
        """_restrict_batch_columns should filter to requested columns."""
        logger.info("Testing _restrict_batch_columns filtering")

        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            }
        )

        transformer = CSVDataTransformer(input_columns=["b", "x"])
        filtered = transformer._restrict_batch_columns(df)

        self.assertEqual(list(filtered.columns), ["b"])
        self.assertTrue(np.array_equal(filtered["b"].to_numpy(), [3, 4]))

    def test_restrict_batch_columns_raises_if_no_columns_present(self) -> None:
        """_restrict_batch_columns should raise when no requested columns exist."""
        logger.info("Testing _restrict_batch_columns error when no column matches")

        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
            }
        )

        transformer = CSVDataTransformer(input_columns=["x", "y"])

        with self.assertRaises(DataTransformationError):
            _ = transformer._restrict_batch_columns(df)

    # ------------------------------------------------------------------ #
    # _derive_input_columns_from_inspection
    # ------------------------------------------------------------------ #
    def test_derive_input_columns_from_inspection_normalizes_names(self) -> None:
        """_derive_input_columns_from_inspection should strip ':0' suffix."""
        logger.info("Testing _derive_input_columns_from_inspection normalization")

        transformer = CSVDataTransformer()
        transformer._model_inspection = {"input_names": ["feat_0:0", "feat_1"]}

        cols = transformer._derive_input_columns_from_inspection()
        self.assertEqual(cols, ["feat_0", "feat_1"])

    def test_derive_input_columns_from_inspection_no_inspection_returns_none(
        self,
    ) -> None:
        """If no inspection metadata, derived input columns should be None."""
        logger.info("Testing _derive_input_columns_from_inspection with no metadata")

        transformer = CSVDataTransformer()
        transformer._model_inspection = None

        cols = transformer._derive_input_columns_from_inspection()
        self.assertIsNone(cols)

    # ------------------------------------------------------------------ #
    # _load_data: data_loader vs dataset vs fallback
    # ------------------------------------------------------------------ #
    def test_load_data_uses_data_loader_when_provided(self) -> None:
        """_load_data should call data_loader.load when data_loader is set."""
        logger.info("Testing _load_data with data_loader")

        mock_loader = MagicMock(spec=CSVDataLoader)
        mock_loader.load.return_value = "DATASET"

        transformer = CSVDataTransformer(data_loader=mock_loader)
        dataset = transformer._load_data()

        self.assertEqual(dataset, "DATASET")
        self.assertEqual(transformer.dataset, "DATASET")
        mock_loader.load.assert_called_once()

    def test_load_data_uses_dataset_attribute_when_set(self) -> None:
        """_load_data should use self.dataset when data_loader is not set."""
        logger.info("Testing _load_data with existing dataset attribute")

        transformer = CSVDataTransformer(dataset="MY_DATA")
        dataset = transformer._load_data()
        self.assertEqual(dataset, "MY_DATA")

    def test_load_data_uses_fallback_argument(self) -> None:
        """_load_data should use fallback_dataset if no loader/attribute."""
        logger.info("Testing _load_data with fallback dataset")

        transformer = CSVDataTransformer()
        dataset = transformer._load_data(fallback_dataset="FALLBACK")
        self.assertEqual(dataset, "FALLBACK")
        self.assertEqual(transformer.dataset, "FALLBACK")

    def test_load_data_raises_when_no_dataset(self) -> None:
        """_load_data should raise DataTransformationError if no dataset."""
        logger.info("Testing _load_data error when no dataset provided")

        transformer = CSVDataTransformer()
        with self.assertRaises(DataTransformationError):
            _ = transformer._load_data()

    # ------------------------------------------------------------------ #
    # _load_model_and_inspection: various resolution paths
    # ------------------------------------------------------------------ #
    def test_load_model_and_inspection_uses_attached_model(self) -> None:
        """_load_model_and_inspection should inspect and reuse attached model."""
        logger.info("Testing _load_model_and_inspection with attached model")

        transformer = CSVDataTransformer(model=self.model)
        model = transformer._load_model_and_inspection(fallback_model=None)

        self.assertIs(model, self.model)
        self.assertIsNotNone(transformer._model_inspection)
        self.assertIn("input_names", transformer._model_inspection)

    def test_load_model_and_inspection_uses_model_persistence(self) -> None:
        """_load_model_and_inspection should use KerasModelPersistence when provided."""
        logger.info("Testing _load_model_and_inspection with model_persistence")

        mock_model = MagicMock(spec=keras.Model)
        inspection = {"input_names": ["x"]}

        mock_persistence = MagicMock(spec=KerasModelPersistence)
        mock_persistence.load.return_value = (mock_model, inspection)

        transformer = CSVDataTransformer(model_persistence=mock_persistence)
        model = transformer._load_model_and_inspection(fallback_model=None)

        self.assertIs(model, mock_model)
        self.assertIs(transformer.model, mock_model)
        self.assertEqual(transformer._model_inspection, inspection)
        mock_persistence.load.assert_called_once_with(inspect=True)

    def test_load_model_and_inspection_raises_when_no_model(self) -> None:
        """_load_model_and_inspection should raise when no model source is set."""
        logger.info("Testing _load_model_and_inspection error with no model")

        transformer = CSVDataTransformer()
        with self.assertRaises(DataTransformationError):
            _ = transformer._load_model_and_inspection(fallback_model=None)

    # ------------------------------------------------------------------ #
    # _batch_to_dataframe
    # ------------------------------------------------------------------ #
    def test_batch_to_dataframe_uses_feature_names(self) -> None:
        """_batch_to_dataframe should use custom feature_names when provided."""
        logger.info("Testing _batch_to_dataframe with custom feature_names")

        x = np.zeros((5, 2), dtype="float32")
        transformer = CSVDataTransformer(feature_names=["a", "b"])
        df = transformer._batch_to_dataframe(x)

        self.assertEqual(list(df.columns), ["a", "b"])

    def test_batch_to_dataframe_raises_on_feature_name_mismatch(self) -> None:
        """_batch_to_dataframe should raise if feature_names length mismatches."""
        logger.info("Testing _batch_to_dataframe error on feature_names mismatch")

        x = np.zeros((5, 2), dtype="float32")
        transformer = CSVDataTransformer(feature_names=["a"])

        with self.assertRaises(DataTransformationError):
            _ = transformer._batch_to_dataframe(x)

    # ------------------------------------------------------------------ #
    # _save_full_data_from_batches & _save_data_per_batch
    # ------------------------------------------------------------------ #
    def test_save_full_data_from_batches_requires_output_path(self) -> None:
        """_save_full_data_from_batches should raise if data_output_path is None."""
        logger.info("Testing _save_full_data_from_batches without data_output_path")

        transformer = CSVDataTransformer(data_output_path=None)
        with self.assertRaises(DataTransformationError):
            transformer._save_full_data_from_batches([])

    def test_save_data_per_batch_requires_output_path(self) -> None:
        """_save_data_per_batch should raise if data_output_path is None."""
        logger.info("Testing _save_data_per_batch without data_output_path")

        transformer = CSVDataTransformer(data_output_path=None)
        with self.assertRaises(DataTransformationError):
            transformer._save_data_per_batch([])

    # ------------------------------------------------------------------ #
    # _resolve_output_path
    # ------------------------------------------------------------------ #
    def test_resolve_output_path_prefers_attribute_over_config(self) -> None:
        """_resolve_output_path should use instance attribute if set."""
        logger.info("Testing _resolve_output_path with attribute and config")

        attr_path = self.temp_dir / "attr.csv"
        cfg_path = self.temp_dir / "cfg.csv"

        transformer = CSVDataTransformer(data_output_path=str(attr_path))
        config = KerasCSVTransformationConfig(data_output_path=str(cfg_path))

        resolved = transformer._resolve_output_path(config=config)
        self.assertEqual(resolved, str(attr_path))

    def test_resolve_output_path_uses_config_when_attribute_missing(self) -> None:
        """_resolve_output_path should use config.data_output_path when attribute is None."""
        logger.info("Testing _resolve_output_path with only config")

        cfg_path = self.temp_dir / "cfg.csv"

        transformer = CSVDataTransformer(data_output_path=None)
        config = KerasCSVTransformationConfig(data_output_path=str(cfg_path))

        resolved = transformer._resolve_output_path(config=config)
        self.assertEqual(resolved, str(cfg_path))
        self.assertEqual(transformer.data_output_path, str(cfg_path))


if __name__ == "__main__":
    unittest.main()
