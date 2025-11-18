import unittest

import numpy as np
import pandas as pd
from loguru import logger

from mlpotion.frameworks.keras.utils.formatter import (
    PredictionFormatter,
)
from tests.core import TestBase


class TestPredictionFormatter(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up PredictionFormatter tests in {self.temp_dir}")
        self.formatter = PredictionFormatter()

    # ------------------------------------------------------------------ #
    # Basic behaviour
    # ------------------------------------------------------------------ #
    def test_format_with_empty_dataframe_returns_unchanged(self) -> None:
        """Empty DataFrame should be returned unchanged and not counted."""
        logger.info("Testing format() with empty DataFrame")

        df = pd.DataFrame()
        result = self.formatter.format(df, predictions=np.array([1, 2, 3]))

        self.assertTrue(result.empty)
        self.assertIs(result, df)  # in-place modification, same object
        self.assertEqual(self.formatter.predicted_rows_count, 0)

    def test_format_adds_single_prediction_column_from_1d_array(self) -> None:
        """1D numpy array should become a 'prediction' column."""
        logger.info("Testing format() with 1D numpy array predictions")

        df = pd.DataFrame({"x": np.arange(5)})
        preds = np.arange(5).astype("float32")

        result = self.formatter.format(df, preds)

        self.assertIs(result, df)
        self.assertIn("prediction", df.columns)
        np.testing.assert_array_equal(df["prediction"].to_numpy(), preds)
        self.assertEqual(self.formatter.predicted_rows_count, len(df))

    def test_format_broadcasts_scalar_prediction(self) -> None:
        """Scalar prediction should be broadcast to all rows."""
        logger.info("Testing scalar prediction broadcasting")

        df = pd.DataFrame({"x": [10, 20, 30]})
        preds = 0.5

        result = self.formatter.format(df, preds)

        self.assertIn("prediction", result.columns)
        np.testing.assert_array_equal(result["prediction"].to_numpy(), np.full(3, 0.5))
        self.assertEqual(self.formatter.predicted_rows_count, 3)

    # ------------------------------------------------------------------ #
    # Dict / list / iterable handling
    # ------------------------------------------------------------------ #
    def test_format_dict_predictions_adds_multiple_columns(self) -> None:
        """Dict of predictions should produce one column per key."""
        logger.info("Testing dict predictions → multiple columns")

        df = pd.DataFrame({"id": [1, 2, 3]})
        preds = {
            "score": np.array([0.1, 0.2, 0.3]),
            "flag": 1.0,
        }

        result = self.formatter.format(df, preds)

        self.assertIn("score", result.columns)
        self.assertIn("flag", result.columns)
        np.testing.assert_array_equal(result["score"].to_numpy(), [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(result["flag"].to_numpy(), np.full(3, 1.0))
        self.assertEqual(self.formatter.predicted_rows_count, 3)

    def test_format_list_predictions_creates_output_columns(self) -> None:
        """List/tuple should produce output_0, output_1, ... columns."""
        logger.info("Testing list predictions → output_i columns")

        df = pd.DataFrame({"x": [0, 1, 2]})
        preds = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
        ]

        result = self.formatter.format(df, preds)

        self.assertIn("output_0", result.columns)
        self.assertIn("output_1", result.columns)
        np.testing.assert_array_equal(result["output_0"].to_numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result["output_1"].to_numpy(), [10.0, 20.0, 30.0])
        self.assertEqual(self.formatter.predicted_rows_count, 3)

    def test_format_iterable_single_element_prediction(self) -> None:
        """Non-list iterable with single element should become 'prediction'."""
        logger.info("Testing iterable-of-one predictions")

        df = pd.DataFrame({"x": [0, 1, 2]})
        preds_gen = (p for p in [np.array([5.0, 6.0, 7.0])])

        result = self.formatter.format(df, preds_gen)

        self.assertIn("prediction", result.columns)
        np.testing.assert_array_equal(result["prediction"].to_numpy(), [5.0, 6.0, 7.0])
        self.assertEqual(self.formatter.predicted_rows_count, 3)

    # ------------------------------------------------------------------ #
    # Multi-output / 2D arrays
    # ------------------------------------------------------------------ #
    def test_multi_output_2d_array_creates_indexed_columns(self) -> None:
        """2D predictions (N, C) should create name_0, name_1, ... columns."""
        logger.info("Testing 2D (N, C) predictions → name_i columns")

        df = pd.DataFrame({"x": [0, 1, 2]})
        preds = np.arange(6).reshape(3, 2).astype("float32")

        result = self.formatter.format(df, preds)

        # For 2D ndarray, column base name is "prediction"
        self.assertIn("prediction_0", result.columns)
        self.assertIn("prediction_1", result.columns)
        np.testing.assert_array_equal(
            result["prediction_0"].to_numpy(), [0.0, 2.0, 4.0]
        )
        np.testing.assert_array_equal(
            result["prediction_1"].to_numpy(), [1.0, 3.0, 5.0]
        )
        self.assertEqual(self.formatter.predicted_rows_count, 3)

    def test_2d_single_column_array_becomes_single_prediction_column(self) -> None:
        """2D (N, 1) should be squeezed into a single 'prediction' column."""
        logger.info("Testing 2D (N,1) predictions → 'prediction' column")

        df = pd.DataFrame({"x": [0, 1, 2]})
        preds = np.array([[1.0], [2.0], [3.0]], dtype="float32")

        result = self.formatter.format(df, preds)

        self.assertIn("prediction", result.columns)
        np.testing.assert_array_equal(result["prediction"].to_numpy(), [1.0, 2.0, 3.0])
        self.assertEqual(self.formatter.predicted_rows_count, 3)

    # ------------------------------------------------------------------ #
    # Length mismatch / broadcasting
    # ------------------------------------------------------------------ #
    def test_length_mismatch_broadcasts_first_value(self) -> None:
        """Length mismatch should broadcast the first value to all rows."""
        logger.info("Testing length mismatch → broadcast first value")

        df = pd.DataFrame({"x": [0, 1, 2, 3]})
        preds = np.array([9.0, 8.0], dtype="float32")

        result = self.formatter.format(df, preds)

        self.assertIn("prediction", result.columns)
        # All rows should get the first value (9.0)
        np.testing.assert_array_equal(result["prediction"].to_numpy(), np.full(4, 9.0))
        self.assertEqual(self.formatter.predicted_rows_count, 4)

    # ------------------------------------------------------------------ #
    # Unsupported types
    # ------------------------------------------------------------------ #
    def test_unsupported_predictions_type_adds_no_columns(self) -> None:
        """Unsupported prediction types should leave DataFrame unchanged."""
        logger.info("Testing unsupported predictions type")

        df = pd.DataFrame({"x": [1, 2, 3]})
        original_cols = list(df.columns)

        result = self.formatter.format(df, predictions="not-a-valid-prediction-type")

        self.assertEqual(list(result.columns), original_cols)
        self.assertEqual(self.formatter.predicted_rows_count, 0)

    # ------------------------------------------------------------------ #
    # Multiple calls accumulate predicted_rows_count
    # ------------------------------------------------------------------ #
    def test_predicted_rows_count_accumulates_across_calls(self) -> None:
        """predicted_rows_count should accumulate across successful format calls."""
        logger.info("Testing predicted_rows_count accumulation")

        df1 = pd.DataFrame({"x": [0, 1, 2]})
        df2 = pd.DataFrame({"x": [10, 20]})

        self.formatter.format(df1, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(self.formatter.predicted_rows_count, 3)

        self.formatter.format(df2, np.array([5.0, 6.0]))
        self.assertEqual(self.formatter.predicted_rows_count, 5)


if __name__ == "__main__":
    unittest.main()
