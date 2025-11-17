from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

from mlpotion.core.config import (
    DataTransformationConfig,
    DataLoadingConfig,
    ModelLoadingConfig,
)
from mlpotion.core.exceptions import DataTransformationError
from mlpotion.core.protocols import DataTransformer
from mlpotion.core.results import TransformationResult
from mlpotion.frameworks.keras.deployment.persistence import KerasModelPersistence
from mlpotion.frameworks.keras.models.inspection import KerasModelInspector
from mlpotion.frameworks.keras.utils.formatter import KerasPredictionFormatter
from mlpotion.frameworks.tensorflow.data.loaders import TFCSVDataLoader
from mlpotion.utils import trycatch


@dataclass(slots=True)
class TFDataToCSVTransformer(DataTransformer):
    """Transform TensorFlow data to CSV using a Keras model.

    This is the TensorFlow analogue of the Keras CSV transformer:

    - Resolves data from:
        * a `DataLoadingConfig` → `TFCSVDataLoader`,
        * an explicit `dataset` (tf.data.Dataset),
        * the `dataset` argument to `transform(...)` (fallback).
    - Resolves model from:
        * an attached `keras.Model`,
        * a `ModelLoadingConfig` / path via `KerasModelPersistence`,
        * the `model` argument to `transform(...)` (fallback).
    - Optionally uses `KerasModelInspector` to derive input names from the model,
      and restricts batch features to those inputs.
    - Uses `KerasPredictionFormatter` to attach predictions to each batch
      as columns in a pandas DataFrame.
    - Saves either:
        * one CSV per batch (`data_output_per_batch=True`), or
        * one concatenated CSV (`data_output_per_batch=False`).

    Expected dataset element shapes:
        - `(features_dict, labels)` or `features_dict` where `features_dict`
          is `dict[str, tf.Tensor]`.

    Args:
        data_loading_config: Optional DataLoadingConfig for TFCSVDataLoader.
        model_loading_config: Optional ModelLoadingConfig used to resolve
            model path for KerasModelPersistence.
        model: Optional compiled Keras model.
        dataset: Optional tf.data.Dataset.
        model_input_signature: Optional mapping of input name → TensorSpec.
            If provided, it is used to restrict batch columns.
        batch_size: Batch size for transformation; if <= 0, dataset is assumed
            to be already batched.
        data_output_path: Path to save transformed CSV(s).
        data_output_per_batch: If True, write per-batch CSVs; otherwise
            write a single concatenated CSV.
    """

    # Data sources
    data_loading_config: DataLoadingConfig | None = None
    dataset: tf.data.Dataset | None = None

    # Model sources
    model_loading_config: ModelLoadingConfig | None = None
    model: keras.Model | None = None

    # Optional explicit input signature (dict[name, TensorSpec])
    model_input_signature: dict[str, tf.TensorSpec] | None = None

    # Transformation / IO configuration
    batch_size: int = 10_000
    data_output_path: str | None = None
    data_output_per_batch: bool = False

    # Internal / derived state
    _model_inspection: dict[str, Any] | None = field(default=None, init=False)
    _prediction_formatter: KerasPredictionFormatter = field(
        default_factory=KerasPredictionFormatter,
        init=False,
    )

    def __post_init__(self) -> None:
        self._validate_data_sources()
        self._validate_model_sources()

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _validate_data_sources(self) -> None:
        """Validate that we have either a loader config or a dataset."""
        if self.data_loading_config is None and self.dataset is None:
            raise DataTransformationError(
                "Either DataLoadingConfig (file_pattern, etc.) or a Dataset "
                "must be provided."
            )
        if self.data_loading_config is not None and self.dataset is not None:
            raise DataTransformationError(
                "Only one of DataLoadingConfig or dataset may be provided."
            )

    def _validate_model_sources(self) -> None:
        """Validate that we have either a model or a model loading config."""
        if self.model is None and self.model_loading_config is None:
            raise DataTransformationError(
                "Either a Keras model or a ModelLoadingConfig must be provided."
            )
        if self.model is not None and self.model_loading_config is not None:
            raise DataTransformationError(
                "Only one of model or ModelLoadingConfig may be provided."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @trycatch(
        error=DataTransformationError,
        success_msg="✅ Data transformed and saved to {config.data_output_path}",
    )
    def transform(
        self,
        dataset: tf.data.Dataset | None,
        model: keras.Model | None,
        config: DataTransformationConfig,
        **kwargs: Any,
    ) -> TransformationResult:
        """Transform dataset using a Keras model and save predictions as CSV.

        The `dataset` and `model` arguments are mainly for protocol compatibility.

        Resolution order:

        - Dataset:
            1. If `data_loading_config` is set → use `TFCSVDataLoader`.
            2. Else, if `self.dataset` is set → use it.
            3. Else, use `dataset` argument.

        - Model:
            1. If `self.model` is set → use it, and inspect if needed.
            2. Else, if `model_loading_config` is set → use `KerasModelPersistence`.
            3. Else, use `model` argument and inspect it.

        `config.data_output_path` is used as a fallback output path when the
        instance attribute is None.
        """
        resolved_dataset = self._load_data(fallback_dataset=dataset)
        resolved_model = self._load_model_and_inspection(fallback_model=model)
        output_path = self._resolve_output_path(config=config)

        logger.info(
            "Starting TF data transformation: "
            f"output_path={output_path}, per_batch={self.data_output_per_batch}"
        )

        batches = self._transform_data(
            dataset=resolved_dataset,
            model=resolved_model,
        )

        if self.data_output_per_batch:
            self._save_data_per_batch(batches)
        else:
            self._save_full_data_from_batches(batches)

        return TransformationResult(data_output_path=output_path)

    # ------------------------------------------------------------------ #
    # Data / model loading
    # ------------------------------------------------------------------ #
    def _load_data(
        self,
        fallback_dataset: tf.data.Dataset | None = None,
    ) -> tf.data.Dataset:
        """Resolve dataset from DataLoadingConfig, instance attribute, or argument."""
        if self.data_loading_config is not None:
            logger.info("Loading dataset via TFCSVDataLoader and DataLoadingConfig")
            cfg = (
                self.data_loading_config.model_dump()
                if hasattr(self.data_loading_config, "model_dump")
                else self.data_loading_config.dict()
            )
            loader = TFCSVDataLoader(
                file_pattern=cfg["file_pattern"],
                batch_size=cfg.get("batch_size", 32),
                column_names=cfg.get("column_names"),
                label_name=cfg.get("label_name"),
                map_fn=cfg.get("map_fn"),
                config=cfg.get("config"),
            )
            ds = loader.load()
            self.dataset = ds
            return ds

        if self.dataset is not None:
            logger.info("Using dataset stored on transformer instance")
            return self.dataset

        if fallback_dataset is not None:
            logger.info("Using dataset passed to transform()")
            self.dataset = fallback_dataset
            return fallback_dataset

        raise DataTransformationError(
            "No dataset provided. Set `data_loading_config`, `dataset`, or pass "
            "`dataset` to transform()."
        )

    def _load_model_and_inspection(
        self,
        fallback_model: keras.Model | None = None,
    ) -> keras.Model:
        """Resolve model and inspection metadata using KerasModelPersistence/Inspector."""
        # 1) Attached model
        if self.model is not None:
            logger.info("Using model stored on transformer instance")
            if self._model_inspection is None:
                logger.info("Inspecting attached model with KerasModelInspector")
                inspector = KerasModelInspector()
                self._model_inspection = inspector.inspect(self.model)
            return self.model

        # 2) ModelLoadingConfig → persistence
        if self.model_loading_config is not None:
            model_path = getattr(self.model_loading_config, "model_path", None)
            if model_path is None:
                raise DataTransformationError(
                    "ModelLoadingConfig must define `model_path`."
                )
            logger.info(
                "Loading model via KerasModelPersistence (inspect=True) "
                f"from: {model_path}"
            )
            persistence = KerasModelPersistence(path=model_path)
            model, inspection = persistence.load(inspect=True)
            self.model = model
            self._model_inspection = inspection
            return model

        # 3) Fallback model argument
        if fallback_model is not None:
            logger.info("Using model passed to transform()")
            self.model = fallback_model
            logger.info("Inspecting fallback model with KerasModelInspector")
            inspector = KerasModelInspector()
            self._model_inspection = inspector.inspect(fallback_model)
            return fallback_model

        raise DataTransformationError(
            "No model provided. Set `model`, `model_loading_config`, or pass "
            "`model` to transform()."
        )

    def _resolve_output_path(self, config: DataTransformationConfig) -> str:
        """Resolve output path from instance attribute or config."""
        if self.data_output_path is not None:
            return self.data_output_path

        path = getattr(config, "data_output_path", None)
        if path is None:
            raise DataTransformationError(
                "data_output_path must be set either on transformer or in config."
            )

        self.data_output_path = str(path)
        return self.data_output_path

    # ------------------------------------------------------------------ #
    # Model inspection → input keys
    # ------------------------------------------------------------------ #
    def _get_model_input_keys(self) -> list[str] | None:
        """Derive model input keys from signature or inspection metadata."""
        # Explicit signature wins
        if self.model_input_signature:
            return list(self.model_input_signature.keys())

        if not self._model_inspection:
            logger.info(
                "No inspection metadata available; using all batch keys as inputs."
            )
            return None

        raw_names = self._model_inspection.get("input_names") or []
        if not raw_names:
            logger.info(
                "Inspection metadata has no 'input_names'; using all batch keys."
            )
            return None

        def _normalize(name: str) -> str:
            if ":" in name:
                return name.split(":", maxsplit=1)[0]
            return name

        keys = [_normalize(str(n)) for n in raw_names]
        logger.info("Derived model input keys from inspection: {keys}", keys=keys)
        return keys or None

    # ------------------------------------------------------------------ #
    # Batch iteration and transformation
    # ------------------------------------------------------------------ #
    def _iter_batches(
        self,
        dataset: tf.data.Dataset,
    ) -> Iterator[Mapping[str, tf.Tensor]]:
        """Yield feature batches from a tf.data.Dataset.

        Handles both:
            - dataset yielding `dict[str, tf.Tensor]`
            - dataset yielding `(features_dict, labels)`
        """
        if self.batch_size and self.batch_size > 0:
            logger.info(
                "Transforming dataset in batches of size {batch_size}",
                batch_size=self.batch_size,
            )
            ds = dataset.batch(self.batch_size)
        else:
            logger.info(
                "No positive batch_size set; treating dataset as already batched."
            )
            ds = dataset

        for elem in ds:
            if isinstance(elem, tuple) and len(elem) >= 1:
                features = elem[0]
            else:
                features = elem

            if not isinstance(features, Mapping):
                raise DataTransformationError(
                    "TFDataToCSVTransformer expects dataset elements where the "
                    "features part is a Mapping[str, tf.Tensor]. "
                    f"Got type: {type(features)!r}"
                )
            yield features  # type: ignore[return-value]

    def _transform_data(
        self,
        dataset: tf.data.Dataset,
        model: keras.Model,
    ) -> Iterator[pd.DataFrame]:
        """Transform all batches in the dataset and yield DataFrames."""
        for idx, features in enumerate(self._iter_batches(dataset)):
            logger.debug(f"Transforming batch {idx}")
            yield self._transform_batch(features=features, model=model)

    def _transform_batch(
        self,
        features: Mapping[str, tf.Tensor],
        model: keras.Model,
    ) -> pd.DataFrame:
        """Transform a single feature batch into a DataFrame with predictions."""
        df_features = self._batch_mapping_to_dataframe(features)
        df_inputs = self._restrict_batch_columns(df_features)

        logger.debug(
            "Running model.predict on batch of shape {shape}",
            shape=df_inputs.shape,
        )

        # Keras usually accepts DataFrame directly, but we can still use NumPy
        x_np = df_inputs.to_numpy()
        preds = model.predict(x_np, verbose=0)

        # Use prediction formatter to attach predictions
        df_with_preds = self._prediction_formatter.format(df_inputs, preds)
        return df_with_preds

    def _batch_mapping_to_dataframe(
        self,
        features: Mapping[str, tf.Tensor],
    ) -> pd.DataFrame:
        """Convert a mapping of tf.Tensor to a pandas DataFrame on CPU."""
        columns: dict[str, np.ndarray] = {}
        for name, tensor in features.items():
            # Ensure tensor is on CPU and 1D/2D as needed
            arr = tensor.numpy()
            if arr.ndim > 1:
                arr = arr.reshape(arr.shape[0], -1)
                if arr.shape[1] != 1:
                    # For multi-dimensional features, we encode as separate columns
                    for i in range(arr.shape[1]):
                        columns[f"{name}_{i}"] = arr[:, i]
                    continue
                arr = arr[:, 0]
            columns[name] = arr
        return pd.DataFrame(columns)

    def _restrict_batch_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict DataFrame to model input columns if input keys are known.

        If no input keys are known, all columns are kept as inputs.

        Raises:
            DataTransformationError: If none of the expected keys are present.
        """
        input_keys = self._get_model_input_keys()
        if not input_keys:
            logger.debug(
                "No explicit model input keys; using all DataFrame columns as inputs."
            )
            return df

        expected = list(input_keys)
        present = [c for c in expected if c in df.columns]

        if not present:
            logger.error(
                "None of the expected model input keys are present in the batch DataFrame."
            )
            raise DataTransformationError(
                f"Expected any of {sorted(expected)}, "
                f"but got DataFrame columns {sorted(df.columns)}"
            )

        logger.info("Restricting batch to model input columns: {cols}", cols=present)
        return df[present]

    # ------------------------------------------------------------------ #
    # Saving helpers
    # ------------------------------------------------------------------ #
    def _save_data_per_batch(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> None:
        """Save transformed data batch-by-batch with a global unique index."""
        if self.data_output_path is None:
            raise DataTransformationError("data_output_path is not set.")

        base_path = Path(self.data_output_path)
        global_row_idx = 0

        for batch_idx, batch_df in enumerate(batches):
            if batch_df.empty:
                logger.warning(f"Skipping empty batch {batch_idx}")
                continue

            batch_df = batch_df.reset_index(drop=True)
            batch_len = len(batch_df)
            batch_df.index = range(global_row_idx, global_row_idx + batch_len)
            global_row_idx += batch_len

            logger.info(
                f"Saving batch {batch_idx} with shape {batch_df.shape}, "
                f"index range [{batch_df.index[0]}, {batch_df.index[-1]}]"
            )
            self._save_single_batch(
                batch_df=batch_df,
                batch_idx=batch_idx,
                base_path=base_path,
            )

    def _save_single_batch(
        self,
        batch_df: pd.DataFrame,
        batch_idx: int,
        base_path: Path,
    ) -> None:
        """Persist a single batch DataFrame as `batch_00000.csv`, etc."""
        if base_path.suffix:  # user gave a file → use its parent as directory
            out_dir = base_path.parent
        else:
            out_dir = base_path

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"batch_{batch_idx:05d}.csv"

        batch_df.to_csv(out_file, index=False)
        logger.info("Saved batch {idx} to {path}", idx=batch_idx, path=out_file)

    def _save_full_data_from_batches(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> pd.DataFrame:
        """Concatenate all batches and save as a single CSV file."""
        if self.data_output_path is None:
            raise DataTransformationError("data_output_path is not set.")

        all_batches = list(batches)
        if not all_batches:
            logger.warning("No batches to save; returning empty DataFrame.")
            full_df = pd.DataFrame()
        else:
            full_df = pd.concat(all_batches, ignore_index=True)

        out_path = Path(self.data_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving full dataset with shape {shape}", shape=full_df.shape)
        full_df.to_csv(out_path, index=False)

        return full_df
