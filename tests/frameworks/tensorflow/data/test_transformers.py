import unittest
from typing import Mapping, Any
from unittest.mock import MagicMock, patch

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

from mlpotion.core.exceptions import DataTransformationError
from mlpotion.core.results import TransformationResult
from mlpotion.frameworks.tensorflow.data.transformers import TFDataToCSVTransformer
from mlpotion.frameworks.keras.utils.formatter import KerasPredictionFormatter
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class TestTFDataToCSVTransformer(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up TFDataToCSVTransformer tests in {self.temp_dir}")

        # simple numeric data
        self.n_samples = 10
        self.n_features = 2
        rng = np.random.default_rng(123)
        self.features = rng.normal(
            size=(self.n_samples, self.n_features)
        ).astype("float32")

        # tf.data.Dataset of dict features
        self.dataset_features_only = tf.data.Dataset.from_tensor_slices(
            {
                "x0": self.features[:, 0],
                "x1": self.features[:, 1],
            }
        )

        # dataset with (features, labels)
        labels = rng.integers(0, 2, size=(self.n_samples,)).astype("float32")
        self.dataset_with_labels = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "x0": self.features[:, 0],
                    "x1": self.features[:, 1],
                },
                labels,
            )
        )

        # simple Keras model
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.n_features,)),
                keras.layers.Dense(4, activation="relu"),
                keras.layers.Dense(1, activation="linear"),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")

    # ------------------------------------------------------------------ #
    # __post_init__ validation
    # ------------------------------------------------------------------ #
    def test_init_raises_when_no_data_sources(self) -> None:
        """__post_init__ should raise if no data_loading_config and no dataset."""
        logger.info("Testing __post_init__ with missing data sources")

        with self.assertRaises(DataTransformationError) as ctx:
            TFDataToCSVTransformer(
                data_loading_config=None,
                dataset=None,
                model=self.model,
            )
        self.assertIn("Either DataLoadingConfig", str(ctx.exception))

    def test_init_raises_when_both_data_sources(self) -> None:
        """__post_init__ should raise if both data_loading_config and dataset set."""
        logger.info("Testing __post_init__ with both data sources")

        dummy_cfg = MagicMock()
        ds = self.dataset_features_only

        with self.assertRaises(DataTransformationError) as ctx:
            TFDataToCSVTransformer(
                data_loading_config=dummy_cfg,
                dataset=ds,
                model=self.model,
            )
        self.assertIn("Only one of DataLoadingConfig or dataset", str(ctx.exception))

    def test_init_raises_when_no_model_sources(self) -> None:
        """__post_init__ should raise if no model and no model_loading_config."""
        logger.info("Testing __post_init__ with missing model sources")

        with self.assertRaises(DataTransformationError) as ctx:
            TFDataToCSVTransformer(
                data_loading_config=MagicMock(),
                dataset=None,
                model=None,
                model_loading_config=None,
            )
        self.assertIn(
            "Either a Keras model or a ModelLoadingConfig must be provided",
            str(ctx.exception),
        )

    def test_init_raises_when_both_model_sources(self) -> None:
        """__post_init__ should raise if both model and model_loading_config set."""
        logger.info("Testing __post_init__ with both model sources")

        with self.assertRaises(DataTransformationError) as ctx:
            TFDataToCSVTransformer(
                data_loading_config=MagicMock(),
                dataset=None,
                model=self.model,
                model_loading_config=MagicMock(),
            )
        self.assertIn(
            "Only one of model or ModelLoadingConfig may be provided",
            str(ctx.exception),
        )

    # ------------------------------------------------------------------ #
    # _load_data
    # ------------------------------------------------------------------ #
    @patch(
        "mlpotion.frameworks.tensorflow.data.transformers.TFCSVDataLoader",
        autospec=True,
    )
    def test_load_data_uses_data_loading_config_and_tf_csv_loader(
        self, loader_cls_mock: MagicMock
    ) -> None:
        """_load_data should construct TFCSVDataLoader with config and call load()."""
        logger.info("Testing _load_data with DataLoadingConfig")

        # Dummy config object that returns a simple dict
        class DummyDataLoadingConfig:
            def model_dump(self) -> dict[str, Any]:
                return {
                    "file_pattern": "data/*.csv",
                    "batch_size": 16,
                    "column_names": ["x0", "x1"],
                    "label_name": "y",
                    "map_fn": None,
                    "config": {"num_epochs": 1, "shuffle": True},
                }

        cfg = DummyDataLoadingConfig()

        transformer = TFDataToCSVTransformer(
            data_loading_config=cfg,
            dataset=None,
            model=self.model,
        )

        loader_instance = loader_cls_mock.return_value
        loader_instance.load.return_value = "DS"

        ds = transformer._load_data()
        self.assertEqual(ds, "DS")
        self.assertEqual(transformer.dataset, "DS")

        loader_cls_mock.assert_called_once_with(
            file_pattern="data/*.csv",
            batch_size=16,
            column_names=["x0", "x1"],
            label_name="y",
            map_fn=None,
            config={"num_epochs": 1, "shuffle": True},
        )
        loader_instance.load.assert_called_once()

    def test_load_data_uses_existing_dataset_attribute(self) -> None:
        """_load_data should return self.dataset when set."""
        logger.info("Testing _load_data using existing dataset attribute")

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )

        ds = transformer._load_data()
        self.assertIs(ds, self.dataset_features_only)

    def test_load_data_uses_fallback_dataset_when_no_config_or_dataset(self) -> None:
        """_load_data should use fallback_dataset if no loader or attribute."""
        logger.info("Testing _load_data with fallback dataset")

        # To bypass __post_init__ data validation, give a dummy data_loading_config,
        # then override after construction.
        transformer = TFDataToCSVTransformer(
            data_loading_config=MagicMock(),
            dataset=None,
            model=self.model,
        )
        transformer.data_loading_config = None  # force no loader

        fallback = self.dataset_with_labels
        ds = transformer._load_data(fallback_dataset=fallback)
        self.assertIs(ds, fallback)
        self.assertIs(transformer.dataset, fallback)

    # ------------------------------------------------------------------ #
    # _load_model_and_inspection
    # ------------------------------------------------------------------ #
    def test_load_model_and_inspection_uses_attached_model(self) -> None:
        """When model is attached, it should be reused and inspected."""
        logger.info("Testing _load_model_and_inspection with attached model")

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )

        model = transformer._load_model_and_inspection(fallback_model=None)
        self.assertIs(model, self.model)
        self.assertIsNotNone(transformer._model_inspection)
        self.assertIn("input_names", transformer._model_inspection)

    @patch(
        "mlpotion.frameworks.tensorflow.data.transformers.KerasModelPersistence",
        autospec=True,
    )
    def test_load_model_and_inspection_uses_model_loading_config(
        self, persistence_cls_mock: MagicMock
    ) -> None:
        """_load_model_and_inspection should use KerasModelPersistence if config given."""
        logger.info("Testing _load_model_and_inspection with ModelLoadingConfig")

        # Dummy config with model_path attribute
        class DummyModelLoadingConfig:
            def __init__(self, model_path: str) -> None:
                self.model_path = model_path

        mlc = DummyModelLoadingConfig(model_path="models/my_model.keras")

        mock_model = MagicMock(spec=keras.Model)
        inspection = {"input_names": ["x0", "x1"]}

        persistence_instance = persistence_cls_mock.return_value
        persistence_instance.load.return_value = (mock_model, inspection)

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=None,
            model_loading_config=mlc,
        )

        model = transformer._load_model_and_inspection(fallback_model=None)

        self.assertIs(model, mock_model)
        self.assertIs(transformer.model, mock_model)
        self.assertEqual(transformer._model_inspection, inspection)
        persistence_cls_mock.assert_called_once_with(path="models/my_model.keras")
        persistence_instance.load.assert_called_once_with(inspect=True)

    def test_load_model_and_inspection_uses_fallback_model_when_no_other_source(
        self,
    ) -> None:
        """_load_model_and_inspection should inspect fallback model when no config."""
        logger.info("Testing _load_model_and_inspection with fallback model")

        # Construct with a dummy model source so __post_init__ passes,
        # then remove it to force fallback on argument.
        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            model_loading_config=None,
        )
        transformer.model = None  # remove attached model

        model = transformer._load_model_and_inspection(fallback_model=self.model)

        self.assertIs(model, self.model)
        self.assertIs(transformer.model, self.model)
        self.assertIsNotNone(transformer._model_inspection)
        self.assertIn("input_names", transformer._model_inspection)

    def test_load_model_and_inspection_raises_when_no_model_available(self) -> None:
        """_load_model_and_inspection should raise when no model is available."""
        logger.info("Testing _load_model_and_inspection error with no model")

        # Bypass __post_init__ model validation by constructing with a dummy
        # model_loading_config, then remove it.
        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            model_loading_config=None,
        )
        transformer.model = None
        transformer.model_loading_config = None

        with self.assertRaises(DataTransformationError):
            _ = transformer._load_model_and_inspection(fallback_model=None)

    # ------------------------------------------------------------------ #
    # _get_model_input_keys
    # ------------------------------------------------------------------ #
    def test_get_model_input_keys_from_signature(self) -> None:
        """If model_input_signature is set, its keys should be returned."""
        logger.info("Testing _get_model_input_keys from model_input_signature")

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        transformer.model_input_signature = {
            "x0": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "x1": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }

        keys = transformer._get_model_input_keys()
        self.assertEqual(keys, ["x0", "x1"])

    def test_get_model_input_keys_from_inspection_normalizes_names(self) -> None:
        """_get_model_input_keys should strip ':0' suffix from input_names."""
        logger.info("Testing _get_model_input_keys name normalization from inspection")

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        transformer.model_input_signature = None
        transformer._model_inspection = {"input_names": ["feat_0:0", "feat_1"]}

        keys = transformer._get_model_input_keys()
        self.assertEqual(keys, ["feat_0", "feat_1"])

    def test_get_model_input_keys_returns_none_when_no_metadata(self) -> None:
        """If no signature and no inspection, None should be returned."""
        logger.info("Testing _get_model_input_keys with no metadata")

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        transformer.model_input_signature = None
        transformer._model_inspection = None

        keys = transformer._get_model_input_keys()
        self.assertIsNone(keys)

    # ------------------------------------------------------------------ #
    # _batch_mapping_to_dataframe
    # ------------------------------------------------------------------ #
    def test_batch_mapping_to_dataframe_simple_1d(self) -> None:
        """_batch_mapping_to_dataframe should convert 1D tensors to scalar columns."""
        logger.info("Testing _batch_mapping_to_dataframe with simple 1D tensors")

        features: Mapping[str, tf.Tensor] = {
            "a": tf.constant([1.0, 2.0], dtype=tf.float32),
            "b": tf.constant([3.0, 4.0], dtype=tf.float32),
        }

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )

        df = transformer._batch_mapping_to_dataframe(features)
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertTrue(np.array_equal(df["a"].to_numpy(), [1.0, 2.0]))
        self.assertTrue(np.array_equal(df["b"].to_numpy(), [3.0, 4.0]))

    def test_batch_mapping_to_dataframe_multi_dim_feature(self) -> None:
        """Multi-dimensional features should become multiple columns name_0, name_1,..."""
        logger.info("Testing _batch_mapping_to_dataframe with multi-dimensional tensors")

        features: Mapping[str, tf.Tensor] = {
            "vec": tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32),
        }

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )

        df = transformer._batch_mapping_to_dataframe(features)
        self.assertEqual(sorted(df.columns), ["vec_0", "vec_1"])
        self.assertTrue(np.array_equal(df["vec_0"].to_numpy(), [1.0, 3.0]))
        self.assertTrue(np.array_equal(df["vec_1"].to_numpy(), [2.0, 4.0]))

    # ------------------------------------------------------------------ #
    # _restrict_batch_columns
    # ------------------------------------------------------------------ #
    def test_restrict_batch_columns_uses_all_columns_when_no_keys(self) -> None:
        """If no model input keys, all columns should be kept."""
        logger.info("Testing _restrict_batch_columns with no input keys")

        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
            }
        )

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        transformer.model_input_signature = None
        transformer._model_inspection = None

        filtered = transformer._restrict_batch_columns(df)
        self.assertTrue(filtered.equals(df))

    def test_restrict_batch_columns_filters_and_preserves_order(self) -> None:
        """_restrict_batch_columns should keep only expected columns in order."""
        logger.info("Testing _restrict_batch_columns filtering behavior")

        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            }
        )

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        transformer.model_input_signature = {
            "b": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "x": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }

        filtered = transformer._restrict_batch_columns(df)
        self.assertEqual(list(filtered.columns), ["b"])
        self.assertTrue(np.array_equal(filtered["b"].to_numpy(), [3, 4]))

    def test_restrict_batch_columns_raises_if_no_columns_present(self) -> None:
        """_restrict_batch_columns should raise when expected columns are absent."""
        logger.info("Testing _restrict_batch_columns error when no columns match")

        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
            }
        )

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        transformer.model_input_signature = {
            "x": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "y": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }

        with self.assertRaises(DataTransformationError):
            _ = transformer._restrict_batch_columns(df)

    # ------------------------------------------------------------------ #
    # _iter_batches
    # ------------------------------------------------------------------ #
    def test_iter_batches_features_only_dataset_respects_batch_size(self) -> None:
        """_iter_batches should yield dict batches honoring batch_size."""
        logger.info("Testing _iter_batches with features-only dataset")

        ds = self.dataset_features_only
        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=ds,
            model=self.model,
            batch_size=4,
        )

        batches = list(transformer._iter_batches(ds))
        self.assertEqual(len(batches), 3)  # 4 + 4 + 2

        self.assertIsInstance(batches[0], Mapping)
        self.assertEqual(
            set(batches[0].keys()),
            {"x0", "x1"},
        )

        first_batch_x0 = batches[0]["x0"].numpy()
        self.assertEqual(first_batch_x0.shape[0], 4)

    def test_iter_batches_dataset_with_labels_extracts_features(self) -> None:
        """_iter_batches should take features from (features, labels) dataset."""
        logger.info("Testing _iter_batches with (features, labels) dataset")

        ds = self.dataset_with_labels
        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=ds,
            model=self.model,
            batch_size=5,
        )

        batches = list(transformer._iter_batches(ds))
        # With batch_size=5, 10 examples → 2 batches
        self.assertEqual(len(batches), 2)
        for b in batches:
            self.assertIsInstance(b, Mapping)
            self.assertEqual(set(b.keys()), {"x0", "x1"})

    def test_iter_batches_raises_for_non_mapping_features(self) -> None:
        """_iter_batches should raise if dataset yields non-mapping features."""
        logger.info("Testing _iter_batches error with non-mapping elements")

        ds = tf.data.Dataset.from_tensor_slices(tf.range(5, dtype=tf.float32))

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=ds,
            model=self.model,
            batch_size=2,
        )

        with self.assertRaises(DataTransformationError):
            _ = list(transformer._iter_batches(ds))

    # ------------------------------------------------------------------ #
    # _transform_batch + KerasPredictionFormatter integration
    # ------------------------------------------------------------------ #
    @patch.object(KerasPredictionFormatter, "format", autospec=True)
    def test_transform_batch_uses_prediction_formatter(
        self, format_mock: MagicMock
    ) -> None:
        """_transform_batch should delegate prediction formatting to KerasPredictionFormatter."""
        logger.info("Testing _transform_batch uses KerasPredictionFormatter")

        features = {
            "x0": tf.constant([1.0, 2.0], dtype=tf.float32),
            "x1": tf.constant([3.0, 4.0], dtype=tf.float32),
        }

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        # ensure we do not fail on restriction
        transformer.model_input_signature = {
            "x0": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "x1": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }

        # patch formatter instance on transformer
        formatter_instance = transformer._prediction_formatter
        format_mock.return_value = pd.DataFrame({"dummy": [1, 2]})

        df_out = transformer._transform_batch(features=features, model=self.model)

        # Ensure formatter was called correctly
        self.assertIs(df_out, format_mock.return_value)
        format_mock.assert_called_once()
        # first arg is the formatter instance, then df & predictions
        _, args, _ = format_mock.mock_calls[0]
        self.assertIs(args[0], formatter_instance)
        self.assertIsInstance(args[1], pd.DataFrame)  # df_inputs
        # args[2] is predictions np.ndarray

    def test_transform_batch_end_to_end(self) -> None:
        """_transform_batch should produce a DataFrame with prediction columns."""
        logger.info("Testing _transform_batch end-to-end")

        features = {
            "x0": tf.constant(self.features[:, 0]),
            "x1": tf.constant(self.features[:, 1]),
        }

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
        )
        transformer.model_input_signature = {
            "x0": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "x1": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }

        df_out = transformer._transform_batch(features=features, model=self.model)
        self.assertEqual(df_out.shape[0], self.n_samples)
        # two inputs + prediction column (KerasPredictionFormatter default name)
        self.assertIn("prediction", df_out.columns)
        self.assertIn("x0", df_out.columns)
        self.assertIn("x1", df_out.columns)

    # ------------------------------------------------------------------ #
    # Saving helpers
    # ------------------------------------------------------------------ #
    def test_save_full_data_from_batches_requires_output_path(self) -> None:
        """_save_full_data_from_batches should raise if data_output_path is None."""
        logger.info("Testing _save_full_data_from_batches without data_output_path")

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            data_output_path=None,
        )

        with self.assertRaises(DataTransformationError):
            transformer._save_full_data_from_batches([])

    def test_save_data_per_batch_requires_output_path(self) -> None:
        """_save_data_per_batch should raise if data_output_path is None."""
        logger.info("Testing _save_data_per_batch without data_output_path")

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            data_output_path=None,
        )

        with self.assertRaises(DataTransformationError):
            transformer._save_data_per_batch([])

    def test_save_full_data_from_batches_writes_single_csv(self) -> None:
        """_save_full_data_from_batches should concatenate and write a single CSV."""
        logger.info("Testing _save_full_data_from_batches file output")

        df1 = pd.DataFrame({"x0": [1, 2], "prediction": [0.1, 0.2]})
        df2 = pd.DataFrame({"x0": [3, 4], "prediction": [0.3, 0.4]})
        out_path = self.temp_dir / "full_preds.csv"

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            data_output_path=str(out_path),
        )

        full_df = transformer._save_full_data_from_batches([df1, df2])

        self.assertTrue(out_path.exists())
        self.assertEqual(len(full_df), 4)
        loaded = pd.read_csv(out_path)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(list(loaded.columns), ["x0", "prediction"])

    def test_save_data_per_batch_writes_multiple_csvs_with_global_index(self) -> None:
        """_save_data_per_batch should write batches with global index ranges."""
        logger.info("Testing _save_data_per_batch per-batch file output")

        df1 = pd.DataFrame({"x0": [1, 2], "prediction": [0.1, 0.2]})
        df2 = pd.DataFrame({"x0": [3, 4, 5], "prediction": [0.3, 0.4, 0.5]})

        out_dir = self.temp_dir / "batches"

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            data_output_path=str(out_dir),
        )

        transformer._save_data_per_batch([df1, df2])

        expected_files = [
            out_dir / "batch_00000.csv",
            out_dir / "batch_00001.csv",
        ]
        for f in expected_files:
            self.assertTrue(f.exists(), f"Expected batch file missing: {f}")

        loaded1 = pd.read_csv(expected_files[0])
        loaded2 = pd.read_csv(expected_files[1])

        self.assertEqual(len(loaded1), 2)
        self.assertEqual(len(loaded2), 3)
        self.assertEqual(list(loaded1.columns), ["x0", "prediction"])

    # ------------------------------------------------------------------ #
    # _resolve_output_path
    # ------------------------------------------------------------------ #
    def test_resolve_output_path_prefers_attribute_over_config(self) -> None:
        """_resolve_output_path should prefer instance attribute over config."""
        logger.info("Testing _resolve_output_path with attribute and config")

        attr_path = self.temp_dir / "attr.csv"
        cfg_path = self.temp_dir / "cfg.csv"

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            data_output_path=str(attr_path),
        )

        config = MagicMock()
        config.data_output_path = str(cfg_path)

        resolved = transformer._resolve_output_path(config=config)
        self.assertEqual(resolved, str(attr_path))

    def test_resolve_output_path_uses_config_when_attribute_missing(self) -> None:
        """_resolve_output_path should use config.data_output_path when attribute is None."""
        logger.info("Testing _resolve_output_path with only config path")

        cfg_path = self.temp_dir / "cfg.csv"

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=self.dataset_features_only,
            model=self.model,
            data_output_path=None,
        )

        config = MagicMock()
        config.data_output_path = str(cfg_path)

        resolved = transformer._resolve_output_path(config=config)
        self.assertEqual(resolved, str(cfg_path))
        self.assertEqual(transformer.data_output_path, str(cfg_path))

    # ------------------------------------------------------------------ #
    # End-to-end transform
    # ------------------------------------------------------------------ #
    def test_transform_end_to_end_single_output_csv(self) -> None:
        """Full transform() call: dataset + model → single CSV with predictions."""
        logger.info("Testing transform() end-to-end with single CSV output")

        ds = self.dataset_features_only

        out_path = self.temp_dir / "predictions.csv"

        transformer = TFDataToCSVTransformer(
            data_loading_config=None,
            dataset=ds,
            model=self.model,
            batch_size=4,
            data_output_path=str(out_path),
            data_output_per_batch=False,
        )
        # ensure we restrict to correct features
        transformer.model_input_signature = {
            "x0": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "x1": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }

        # Config can be a simple object with data_output_path attribute
        config = MagicMock()
        config.data_output_path = str(out_path)

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

        self.assertEqual(df.shape[0], self.n_samples)
        self.assertIn("prediction", df.columns)
        # we expect 2 feature columns + prediction
        self.assertGreaterEqual(df.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
