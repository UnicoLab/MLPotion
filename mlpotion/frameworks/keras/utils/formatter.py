from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(slots=True)
class KerasPredictionFormatter:
    """Format Keras model predictions into a pandas DataFrame (no TensorFlow).

    This class is backend-agnostic as long as predictions are convertible
    to NumPy arrays or scalars (typical Keras 3 behaviour on any backend).

    Attributes:
        predicted_rows_count: Counter of how many rows were augmented with
            predictions across calls.

    Example:
        ```python
        import keras
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({"feature": np.arange(5)})
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(1, activation="linear"),
            ]
        )

        preds = model.predict(df[["feature"]].to_numpy())

        formatter = KerasPredictionFormatter()
        df_with_preds = formatter.format(df, preds)
        print(df_with_preds.head())
        ```
    """

    predicted_rows_count: int = field(default=0, init=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def format(self, df: pd.DataFrame, predictions: Any) -> pd.DataFrame:
        """Format predictions and add them as columns to a DataFrame.

        Args:
            df: Input DataFrame to augment with prediction columns.
            predictions: Model predictions. Can be:
                - np.ndarray
                - scalar (float, int, np.number)
                - dict[str, (array | scalar)]
                - list/tuple of arrays or scalars (multi-output)

        Returns:
            DataFrame with added prediction columns.
        """
        if df.empty:
            logger.warning("Received empty DataFrame; nothing to format.")
            return df

        pred_map = self._normalize_predictions(predictions)
        if not pred_map:
            logger.warning(
                f"Unsupported predictions type: {type(predictions)!r}. "
                "No prediction columns added."
            )
            return df

        prediction_columns_added = 0
        for out_name, data in pred_map.items():
            try:
                added = self._add_column(df, out_name, data)
                prediction_columns_added += int(added)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error adding prediction column '{out_name}': {exc}")

        if prediction_columns_added > 0:
            self.predicted_rows_count += len(df)
        else:
            logger.warning("No prediction columns were added. Check model output format.")

        logger.debug(
            f"Added predictions for {len(df)} rows, "
            f"{prediction_columns_added} prediction column(s)."
        )
        return df

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _normalize_predictions(self, predictions: Any) -> dict[str, Any]:
        """Normalize various prediction shapes into a name -> value mapping."""
        # dict[str, ...]
        if isinstance(predictions, dict):
            return dict(predictions)

        # list/tuple → multi-output unnamed; name them output_0, output_1, ...
        if isinstance(predictions, (list, tuple)):
            return {f"output_{i}": p for i, p in enumerate(predictions)}

        # Single array-like or scalar → single column called "prediction"
        if isinstance(predictions, (np.ndarray, np.number, float, int)):
            return {"prediction": predictions}

        # Anything iterable but not list/tuple/dict → try to treat as first output
        if isinstance(predictions, Iterable) and not isinstance(predictions, (str, bytes)):
            preds_list = list(predictions)
            if len(preds_list) == 1:
                return {"prediction": preds_list[0]}

        # Unsupported type
        return {}

    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert prediction data to a NumPy array if possible."""
        if isinstance(data, np.ndarray):
            return data

        if isinstance(data, (np.number, float, int)):
            # Scalar
            return np.asarray([data])

        try:
            return np.asarray(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not convert predictions to numpy array: {exc}")
            return np.asarray([])

    def _add_column(self, df: pd.DataFrame, name: str, data: Any) -> bool:
        """Add a single prediction column to the DataFrame if possible.

        Returns:
            True if a column was added, False otherwise.
        """
        arr = self._to_numpy(data)
        if arr.size == 0:
            logger.warning(
                f"Prediction data for '{name}' is empty after conversion; "
                "column not added."
            )
            return False

        # Flatten if multi-dimensional but keep 2D for multi-class outputs
        if arr.ndim == 1:
            values = arr
        elif arr.ndim == 2 and arr.shape[1] == 1:
            values = arr[:, 0]
        else:
            # For shape (N, C) we can either create one column per class
            # or keep as-is. Here we keep as 2D only if shape matches.
            if arr.shape[0] == len(df):
                # Create one column per class: name_0, name_1, ...
                for idx in range(arr.shape[1]):
                    col_name = f"{name}_{idx}"
                    df[col_name] = arr[:, idx]
                    logger.debug(
                        f"Added multi-output predictions column '{col_name}' "
                        f"with shape {arr[:, idx].shape}"
                    )
                return True
            # Fallback to flatten
            values = arr.reshape(-1)

        # Now we have 1D values
        if len(values) == len(df):
            df[name] = values
            logger.debug(
                f"Added prediction column '{name}' with {len(values)} values."
            )
            return True

        if len(values) == 1:
            df[name] = np.full(len(df), values[0])
            logger.debug(
                f"Broadcasted scalar prediction for '{name}' to {len(df)} rows."
            )
            return True

        logger.warning(
            f"Length mismatch for '{name}': expected {len(df)}, got {len(values)}. "
            "Using the first value and broadcasting."
        )
        df[name] = np.full(len(df), values[0])
        return True
