import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

from mlpotion.core.exceptions import DataLoadingError
from mlpotion.frameworks.tensorflow.data.loaders import (
    CSVDataLoader,
    RecordDataLoader,
)
from tests.core import (
    TestBase,
)  # provides temp_dir, setUpClass/tearDownClass, setUp/tearDown


class TestCSVDataLoader(TestBase):
    def setUp(self) -> None:
        super().setUp()
        # Work with *relative* patterns to avoid "Non-relative patterns are unsupported"
        self._old_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        logger.info(f"Changed CWD to {self.temp_dir} for CSVDataLoader tests")

        # Create a small CSV file in the temp directory
        self.file_path = self.temp_dir / "test_data.csv"
        df = pd.DataFrame(
            {
                "feature_0": np.arange(4, dtype=np.float32),
                "feature_1": np.arange(4, dtype=np.float32) * 10.0,
                "target": [0, 1, 0, 1],
            }
        )
        df.to_csv(self.file_path, index=False)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        super().tearDown()

    # ------------------------------------------------------------------ #
    # _extract_and_validate_num_epochs
    # ------------------------------------------------------------------ #
    def test_extract_and_validate_num_epochs_valid(self) -> None:
        """Valid num_epochs should be extracted and removed from config."""
        logger.info("Testing _extract_and_validate_num_epochs with valid config")

        loader = CSVDataLoader(
            file_pattern="*.csv",
            batch_size=2,
            label_name="target",
            config={"num_epochs": 3, "shuffle": False},
        )

        self.assertEqual(loader.num_epochs, 3)
        self.assertNotIn("num_epochs", loader.config)
        self.assertEqual(loader.config.get("shuffle"), False)

    def test_extract_and_validate_num_epochs_non_integer_raises(self) -> None:
        """Non-integer num_epochs should raise DataLoadingError."""
        logger.info("Testing _extract_and_validate_num_epochs with non-integer")

        with self.assertRaises(DataLoadingError):
            _ = CSVDataLoader(
                file_pattern="*.csv",
                config={"num_epochs": "not-an-int"},
            )

    def test_extract_and_validate_num_epochs_non_positive_raises(self) -> None:
        """num_epochs <= 0 should raise DataLoadingError."""
        logger.info("Testing _extract_and_validate_num_epochs with num_epochs <= 0")

        with self.assertRaises(DataLoadingError):
            _ = CSVDataLoader(
                file_pattern="*.csv",
                config={"num_epochs": 0},
            )

    # ------------------------------------------------------------------ #
    # _validate_files_exist / _validate_finite_dataset
    # ------------------------------------------------------------------ #
    def test_validate_files_exist_raises_if_no_files(self) -> None:
        """Constructor should fail if no CSV files match the pattern."""
        logger.info("Testing _validate_files_exist with missing files")

        with self.assertRaises(DataLoadingError):
            _ = CSVDataLoader(
                file_pattern="missing_*.csv",
                config={"num_epochs": 1},
            )

    def test_validate_finite_dataset_raises_if_num_epochs_non_positive(self) -> None:
        """_validate_finite_dataset should guard against num_epochs <= 0."""
        logger.info("Testing _validate_finite_dataset explicit guard")

        loader = CSVDataLoader(
            file_pattern="*.csv",
            config={"num_epochs": 1},
        )
        # Force an invalid state and ensure guard fires
        loader.num_epochs = 0
        with self.assertRaises(DataLoadingError):
            loader._validate_finite_dataset()

    # ------------------------------------------------------------------ #
    # load(): happy path
    # ------------------------------------------------------------------ #
    def test_load_returns_dataset_with_features_and_labels(self) -> None:
        """load() should return a tf.data.Dataset with (features, labels)."""
        logger.info("Testing CSVDataLoader.load() basic behaviour")

        loader = CSVDataLoader(
            file_pattern="*.csv",
            batch_size=2,
            column_names=["feature_0", "feature_1", "target"],
            label_name="target",
            config={"num_epochs": 1, "shuffle": False},
        )
        ds = loader.load()

        self.assertIsInstance(ds, tf.data.Dataset)

        # Take a single batch and inspect structure
        features, labels = next(iter(ds))
        self.assertIsInstance(features, dict)
        self.assertIsInstance(labels, tf.Tensor)

        self.assertSetEqual(set(features.keys()), {"feature_0", "feature_1"})
        # Batch dimension should be 2 for our settings
        self.assertEqual(int(features["feature_0"].shape[0]), 2)
        self.assertEqual(int(labels.shape[0]), 2)

    def test_load_applies_map_fn_when_label_name_none(self) -> None:
        """map_fn should be applied when label_name is None (features-only dataset)."""
        logger.info("Testing CSVDataLoader.load() with map_fn")

        # Use all CSV columns and treat them as features-only (no label),
        # and let map_fn just modify feature_0.
        def map_fn(batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
            batch["feature_0"] = batch["feature_0"] + 42.0
            return batch

        loader = CSVDataLoader(
            file_pattern="*.csv",
            batch_size=4,
            # IMPORTANT: must match the actual CSV file (3 columns)
            column_names=["feature_0", "feature_1", "target"],
            label_name=None,
            map_fn=map_fn,
            config={"num_epochs": 1, "shuffle": False},
        )

        ds = loader.load()
        self.assertIsInstance(ds, tf.data.Dataset)

        batch = next(iter(ds))
        self.assertIsInstance(batch, dict)
        self.assertIn("feature_0", batch)

        feature_vals = batch["feature_0"].numpy()
        # original feature_0 is [0,1,2,3]
        expected = np.arange(4, dtype=np.float32) + 42.0
        np.testing.assert_allclose(np.sort(feature_vals), np.sort(expected))


class TestRecordDataLoader(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self._old_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        logger.info(f"Changed CWD to {self.temp_dir} for RecordDataLoader tests")

        # Create a small TFRecord file with simple features
        self.tfrecord_path = self.temp_dir / "data.tfrecord"
        with tf.io.TFRecordWriter(str(self.tfrecord_path)) as writer:
            for i in range(3):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "feature_0": tf.train.Feature(
                                float_list=tf.train.FloatList(value=[float(i)])
                            ),
                            "feature_1": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[i * 10])
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[i % 2])
                            ),
                        }
                    )
                )
                writer.write(example.SerializeToString())

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        super().tearDown()

    # ------------------------------------------------------------------ #
    # _validate_files_exist
    # ------------------------------------------------------------------ #
    def test_validate_files_exist_raises_if_no_tfrecord_files(self) -> None:
        """RecordDataLoader should fail if no TFRecord files match the pattern."""
        logger.info("Testing RecordDataLoader._validate_files_exist with missing files")

        with self.assertRaises(DataLoadingError):
            _ = RecordDataLoader(
                file_pattern="missing_*.tfrecord",
                batch_size=2,
            )

    def test_get_files_matching_pattern_returns_dataset_of_filenames(self) -> None:
        """_get_files_matching_pattern should return a tf.data.Dataset of filenames."""
        logger.info("Testing RecordDataLoader._get_files_matching_pattern")

        loader = RecordDataLoader(
            file_pattern="*.tfrecord",
            batch_size=2,
        )

        filenames_ds = loader._get_files_matching_pattern()
        self.assertIsInstance(filenames_ds, tf.data.Dataset)

        # Materialize one filename and ensure it ends with .tfrecord
        filenames_list = list(filenames_ds.as_numpy_iterator())
        self.assertGreaterEqual(len(filenames_list), 1)
        any_name = filenames_list[0].decode("utf-8")
        self.assertTrue(any_name.endswith(".tfrecord"))

    # ------------------------------------------------------------------ #
    # _apply_column_label_selection
    # ------------------------------------------------------------------ #
    def test_apply_column_label_selection_no_label_no_column_filter(self) -> None:
        """If column_names and label_name are None, example should be returned unchanged."""
        logger.info("Testing _apply_column_label_selection with no label/columns")

        loader = RecordDataLoader(
            file_pattern="*.tfrecord",
            batch_size=2,
            column_names=None,
            label_name=None,
        )

        example = {
            "feature_0": tf.constant([1.0]),
            "feature_1": tf.constant([2.0]),
        }

        result = loader._apply_column_label_selection(example)
        self.assertIs(result, example)

    def test_apply_column_label_selection_with_label_and_columns(self) -> None:
        """_apply_column_label_selection should split features and label."""
        logger.info("Testing _apply_column_label_selection with label and columns")

        loader = RecordDataLoader(
            file_pattern="*.tfrecord",
            batch_size=2,
            column_names=["feature_0"],
            label_name="label",
        )

        example = {
            "feature_0": tf.constant([1.0]),
            "feature_1": tf.constant([2.0]),
            "label": tf.constant([1]),
        }

        features, label = loader._apply_column_label_selection(example)
        self.assertIsInstance(features, dict)
        self.assertIn("feature_0", features)
        self.assertNotIn("feature_1", features)
        self.assertTrue(tf.reduce_all(tf.equal(label, example["label"])))

    def test_apply_column_label_selection_raises_if_label_missing(self) -> None:
        """Missing label_name in example should raise DataLoadingError."""
        logger.info("Testing _apply_column_label_selection with missing label")

        loader = RecordDataLoader(
            file_pattern="*.tfrecord",
            batch_size=2,
            column_names=["feature_0"],
            label_name="label",
        )

        example = {
            "feature_0": tf.constant([1.0]),
            "feature_1": tf.constant([2.0]),
        }

        with self.assertRaises(DataLoadingError):
            _ = loader._apply_column_label_selection(example)


if __name__ == "__main__":
    unittest.main()
