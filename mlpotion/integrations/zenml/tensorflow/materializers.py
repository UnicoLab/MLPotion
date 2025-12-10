"""Custom materializers for TensorFlow types."""

import json
from pathlib import Path
from typing import Any, Type

import tensorflow as tf
from zenml.enums import ArtifactType
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.materializers.materializer_registry import materializer_registry

logger = get_logger(__name__)

# Check if TensorFlow types are actually classes (not Mocks during testing)
_is_tensor_real_class = isinstance(tf.Tensor, type) and not hasattr(
    tf.Tensor, "_mock_name"
)
_is_tensorspec_real_class = isinstance(tf.TensorSpec, type) and not hasattr(
    tf.TensorSpec, "_mock_name"
)
_is_dataset_real_class = isinstance(tf.data.Dataset, type) and not hasattr(
    tf.data.Dataset, "_mock_name"
)

# Use object as fallback for ASSOCIATED_TYPES when types are mocked
_TensorForTypes = tf.Tensor if _is_tensor_real_class else object
_TensorSpecForTypes = tf.TensorSpec if _is_tensorspec_real_class else object
_DatasetForTypes = tf.data.Dataset if _is_dataset_real_class else object


class TensorMaterializer(BaseMaterializer):
    """Materializer for TensorFlow Tensor objects.

    This materializer handles the serialization and deserialization of `tf.Tensor` objects.
    It saves tensors as binary protobuf files (`tensor.pb`) using `tf.io.serialize_tensor`.
    """

    ASSOCIATED_TYPES = (_TensorForTypes,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type[Any]) -> tf.Tensor:  # noqa: ARG002
        """Load a TensorFlow Tensor from the artifact store.

        Args:
            data_type: The type of the data to load (should be `tf.Tensor`).

        Returns:
            tf.Tensor: The loaded tensor.
        """
        logger.info("Loading TensorFlow tensor...")
        try:
            tensor_path = Path(self.uri) / "tensor.pb"
            return tf.io.parse_tensor(
                tf.io.read_file(str(tensor_path)), out_type=tf.float32
            )
        except Exception as e:
            logger.error(f"Failed to load tensor: {e}")
            raise

    def save(self, data: tf.Tensor) -> None:
        """Save a TensorFlow Tensor to the artifact store.

        Args:
            data: The tensor to save.
        """
        logger.info("Saving TensorFlow tensor...")
        try:
            Path(self.uri).mkdir(parents=True, exist_ok=True)
            tensor_path = Path(self.uri) / "tensor.pb"
            tf.io.write_file(str(tensor_path), tf.io.serialize_tensor(data))
            logger.info("✅ Successfully saved TensorFlow tensor")
        except Exception as e:
            logger.error(f"Failed to save tensor: {e}")
            raise


# Register the TensorMaterializer (only if real class)
if _is_tensor_real_class:
    try:
        materializer_registry.register_and_overwrite_type(
            key=tf.Tensor,
            type_=TensorMaterializer,
        )
    except Exception:
        pass  # Registration may fail in test environments


class TensorSpecMaterializer(BaseMaterializer):
    """Materializer for TensorFlow TensorSpec objects.

    This materializer handles the serialization and deserialization of `tf.TensorSpec` objects.
    It saves the spec as a JSON file (`spec.json`) containing shape, dtype, and other metadata.
    """

    ASSOCIATED_TYPES = (_TensorSpecForTypes,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type[Any]) -> tf.TensorSpec:  # noqa: ARG002
        """Load a TensorFlow TensorSpec from the artifact store.

        Args:
            data_type: The type of the data to load (should be `tf.TensorSpec`).

        Returns:
            tf.TensorSpec: The loaded tensor spec.
        """
        logger.info("Loading TensorFlow TensorSpec...")
        try:
            spec_path = Path(self.uri) / "spec.json"
            with open(spec_path) as f:
                spec_dict = json.load(f)
            # Reconstruct TensorSpec from dict representation
            return tf.TensorSpec.from_spec(spec_dict)
        except Exception as e:
            logger.error(f"Failed to load TensorSpec: {e}")
            raise

    def save(self, data: tf.TensorSpec) -> None:
        """Save a TensorFlow TensorSpec to the artifact store.

        Args:
            data: The tensor spec to save.
        """
        logger.info("Saving TensorFlow TensorSpec...")
        try:
            Path(self.uri).mkdir(parents=True, exist_ok=True)
            spec_path = Path(self.uri) / "spec.json"
            # Convert TensorSpec to serializable dict format
            spec_dict = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            }
            with open(spec_path, "w") as f:
                json.dump(spec_dict, f, indent=2)
            logger.info("✅ Successfully saved TensorFlow TensorSpec")
        except Exception as e:
            logger.error(f"Failed to save TensorSpec: {e}")
            raise


# Register the TensorSpecMaterializer (only if real class)
if _is_tensorspec_real_class:
    try:
        materializer_registry.register_and_overwrite_type(
            key=tf.TensorSpec,
            type_=TensorSpecMaterializer,
        )
    except Exception:
        pass  # Registration may fail in test environments


class TFRecordDatasetMaterializer(BaseMaterializer):
    """Generic TFRecord materializer for `tf.data.Dataset`.

    This materializer is designed to be robust and round-trip safe for
    datasets produced by `tf.data.experimental.make_csv_dataset`, and in
    general for any dataset whose `element_spec` is a nested structure of:

        - dict / tuple / list containers
        - `tf.TensorSpec` leaves

    It works as follows:

    * Save:
        - Reads `dataset.element_spec` and serializes it to JSON.
        - For each batch (dataset element), recursively flattens it to a
          list of tensors in a deterministic order implied by the spec.
        - Writes a single `tf.train.Example` per batch, with features
          named "f0", "f1", ... corresponding to each leaf tensor.

    * Load:
        - Deserializes `element_spec` from JSON.
        - Builds a `feature_description` for `tf.io.parse_single_example`
          using the leaf specs.
        - Parses each example into a list of tensors.
        - Recursively unflattens the list back into the same nested
          structure as `element_spec`.

    This supports all typical `make_csv_dataset` shapes:

        1. label_name=None:
           element: dict[str, Tensor]

        2. label_name="target":
           element: (dict[str, Tensor], Tensor)

        3. label_name=["t1", "t2"]:
           element: (dict[str, Tensor], dict[str, Tensor])

    and also more complex nesting as long as it's composed of
    dict / tuple / list and TensorSpec leaves.
    """

    ASSOCIATED_TYPES = (tf.data.Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, data: tf.data.Dataset) -> None:
        """Serialize a `tf.data.Dataset` to TFRecord + metadata JSON."""
        dataset_dir = Path(self.uri)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        tfrecord_path = str(dataset_dir / "data.tfrecord")
        metadata_path = dataset_dir / "metadata.json"

        element_spec = data.element_spec

        logger.info("Saving dataset to TFRecord: %s", tfrecord_path)
        logger.info("Dataset element_spec: %s", element_spec)

        # Handle cardinality
        cardinality = data.cardinality().numpy()
        logger.info("Dataset cardinality: %s", cardinality)

        if cardinality == tf.data.INFINITE_CARDINALITY:
            logger.warning("Infinite dataset detected. Taking first 100000 batches.")
            data = data.take(100_000)
        elif cardinality == tf.data.UNKNOWN_CARDINALITY:
            logger.warning("Unknown dataset cardinality. Taking first 100000 batches.")
            data = data.take(100_000)
        else:
            logger.info("Finite dataset with %s batches.", cardinality)

        # Serialize element_spec so we can restore structure and leaf specs
        serialized_spec = self._serialize_element_spec(element_spec)
        flat_spec_leaves = self._flatten_element_spec(element_spec)
        num_leaves = len(flat_spec_leaves)

        # Get concrete shapes from the first batch element (if available)
        # We store shapes WITHOUT the batch dimension to handle variable batch sizes
        concrete_shapes = None
        try:
            first_batch = next(iter(data.take(1)))
            flat_tensors_sample: list[tf.Tensor] = []
            self._flatten_data_with_spec(first_batch, element_spec, flat_tensors_sample)
            # Store the shape WITHOUT the first (batch) dimension
            # This allows the materializer to work with variable batch sizes
            concrete_shapes = []
            for t in flat_tensors_sample:
                shape_list = list(t.shape.as_list())
                # Remove the first (batch) dimension, keep the rest
                if len(shape_list) > 1:
                    shape_without_batch = [None] + shape_list[1:]  # None for batch dim
                else:
                    shape_without_batch = [None]  # Just batch dimension
                concrete_shapes.append(shape_without_batch)
        except Exception:
            # If we can't get a sample, proceed without concrete shapes
            pass

        metadata = {
            "format_version": "3.1",  # Increment version for new feature
            "element_spec": serialized_spec,
            "num_leaves": num_leaves,
            "concrete_shapes": concrete_shapes,  # Store actual shapes if available
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Write TFRecord
        writer = tf.io.TFRecordWriter(tfrecord_path)
        batch_count = 0

        for batch in data:
            flat_tensors: list[tf.Tensor] = []
            self._flatten_data_with_spec(batch, element_spec, flat_tensors)

            if len(flat_tensors) != num_leaves:
                raise ValueError(
                    f"Flattened batch has {len(flat_tensors)} leaves but "
                    f"element_spec indicates {num_leaves}."
                )

            example = self._flat_tensors_to_example(flat_tensors)
            writer.write(example.SerializeToString())
            batch_count += 1

            if batch_count % 100 == 0:
                logger.info("Written %d batches...", batch_count)

        writer.close()
        logger.info("Successfully saved %d batches to TFRecord.", batch_count)

    def load(self, data_type: Type[Any]) -> tf.data.Dataset:
        """Deserialize a `tf.data.Dataset` from TFRecord + metadata JSON."""
        dataset_dir = Path(self.uri)
        tfrecord_path = str(dataset_dir / "data.tfrecord")
        metadata_path = dataset_dir / "metadata.json"

        logger.info("Loading dataset from TFRecord: %s", tfrecord_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        element_spec = self._deserialize_element_spec(metadata["element_spec"])
        num_leaves = metadata["num_leaves"]
        concrete_shapes = metadata.get(
            "concrete_shapes", None
        )  # May be None for older versions

        logger.info("Loaded element_spec: %s", element_spec)
        logger.info("Expected number of leaves: %s", num_leaves)
        if concrete_shapes:
            logger.info("Concrete shapes available: %s", concrete_shapes)

        flat_spec_leaves = self._flatten_element_spec(element_spec)
        if len(flat_spec_leaves) != num_leaves:
            raise ValueError(
                f"Metadata num_leaves={num_leaves} but element_spec "
                f"has {len(flat_spec_leaves)} leaves."
            )

        # Build feature description for parsing
        feature_description = self._build_feature_description(flat_spec_leaves)

        def parse_fn(serialized_example: tf.Tensor) -> Any:
            parsed = tf.io.parse_single_example(serialized_example, feature_description)

            flat_tensors: list[tf.Tensor] = []
            for i, (_, leaf_spec) in enumerate(flat_spec_leaves):
                key = f"f{i}"

                if leaf_spec.dtype in (tf.float32, tf.float64, tf.int32, tf.int64):
                    # Numeric: stored as VarLenFeature, results in 1D tensor
                    dense = tf.sparse.to_dense(parsed[key])
                    tensor = tf.cast(dense, leaf_spec.dtype)

                    # Use concrete shape if available, otherwise fall back to spec-based logic
                    if concrete_shapes and i < len(concrete_shapes):
                        # We have the actual shape from when the data was saved
                        concrete_shape = concrete_shapes[i]
                        # Replace None with -1 for reshape
                        target_shape = [
                            d if d is not None else -1 for d in concrete_shape
                        ]
                        tensor = tf.reshape(tensor, target_shape)
                        # Set the shape with proper None values
                        tensor.set_shape(concrete_shape)
                    else:
                        # Fallback to spec-based reshaping (legacy behavior)
                        if leaf_spec.shape.rank is not None:
                            if leaf_spec.shape.rank == 1:
                                # Original was 1D, VarLen already gives us 1D - just set shape
                                tensor.set_shape(leaf_spec.shape)
                            elif leaf_spec.shape.rank > 1:
                                # Original was multi-dimensional - need to reshape from 1D
                                shape_list = leaf_spec.shape.as_list()
                                none_indices = [
                                    i for i, d in enumerate(shape_list) if d is None
                                ]

                                if len(none_indices) <= 1:
                                    # Safe to reshape with at most one -1
                                    target_shape = [
                                        d if d is not None else -1 for d in shape_list
                                    ]
                                    tensor = tf.reshape(tensor, target_shape)
                                    tensor.set_shape(leaf_spec.shape)
                        else:
                            # Unknown rank - just set shape
                            tensor.set_shape(leaf_spec.shape)
                else:
                    # Other dtypes: stored as serialized bytes
                    serialized = parsed[key]
                    tensor = tf.io.parse_tensor(serialized, out_type=leaf_spec.dtype)
                    tensor.set_shape(leaf_spec.shape)

                flat_tensors.append(tensor)

            # Rebuild nested structure
            flat_iter = iter(flat_tensors)
            return self._unflatten_data_with_spec(element_spec, flat_iter)

        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

        logger.info("Successfully loaded dataset from TFRecord.")
        logger.info("Dataset cardinality: %s", dataset.cardinality().numpy())

        return dataset

    # ------------------------------------------------------------------
    # Flatten / unflatten with spec
    # ------------------------------------------------------------------

    def _flatten_element_spec(
        self, spec: Any, path: str = ""
    ) -> list[tuple[str, tf.TensorSpec]]:
        """Return a list of (path, TensorSpec) leaves in deterministic order."""
        leaves: list[tuple[str, tf.TensorSpec]] = []

        if isinstance(spec, tf.TensorSpec):
            leaves.append((path or "root", spec))
        elif isinstance(spec, dict):
            for key in sorted(spec.keys()):
                sub_path = f"{path}.dict[{key!r}]" if path else f"dict[{key!r}]"
                leaves.extend(self._flatten_element_spec(spec[key], sub_path))
        elif isinstance(spec, tuple):
            for idx, sub_spec in enumerate(spec):
                sub_path = f"{path}.tuple[{idx}]" if path else f"tuple[{idx}]"
                leaves.extend(self._flatten_element_spec(sub_spec, sub_path))
        elif isinstance(spec, list):
            for idx, sub_spec in enumerate(spec):
                sub_path = f"{path}.list[{idx}]" if path else f"list[{idx}]"
                leaves.extend(self._flatten_element_spec(sub_spec, sub_path))
        else:
            raise ValueError(f"Unsupported element_spec leaf type: {type(spec)}")

        return leaves

    def _flatten_data_with_spec(
        self, data: Any, spec: Any, out: list[tf.Tensor]
    ) -> None:
        """Flatten `data` into `out` according to the structure in `spec`."""
        if isinstance(spec, tf.TensorSpec):
            # Leaf: convert to tensor and append
            tensor = tf.convert_to_tensor(data)
            out.append(tensor)
        elif isinstance(spec, dict):
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict data matching spec, got: {type(data)}")
            for key in sorted(spec.keys()):
                self._flatten_data_with_spec(data[key], spec[key], out)
        elif isinstance(spec, tuple):
            if not isinstance(data, tuple):
                raise TypeError(f"Expected tuple data matching spec, got: {type(data)}")
            if len(data) != len(spec):
                raise ValueError(
                    f"Tuple length mismatch: data len={len(data)}, spec len={len(spec)}"
                )
            for idx, sub_spec in enumerate(spec):
                self._flatten_data_with_spec(data[idx], sub_spec, out)
        elif isinstance(spec, list):
            if not isinstance(data, list):
                raise TypeError(f"Expected list data matching spec, got: {type(data)}")
            if len(data) != len(spec):
                raise ValueError(
                    f"List length mismatch: data len={len(data)}, spec len={len(spec)}"
                )
            for idx, sub_spec in enumerate(spec):
                self._flatten_data_with_spec(data[idx], sub_spec, out)
        else:
            raise ValueError(f"Unsupported spec type in flatten: {type(spec)}")

    def _unflatten_data_with_spec(self, spec: Any, flat_iter: Any) -> Any:
        """Rebuild nested structure from flat iterator according to `spec`."""
        if isinstance(spec, tf.TensorSpec):
            return next(flat_iter)
        elif isinstance(spec, dict):
            return {
                key: self._unflatten_data_with_spec(sub_spec, flat_iter)
                for key, sub_spec in sorted(spec.items())
            }
        elif isinstance(spec, tuple):
            return tuple(
                self._unflatten_data_with_spec(sub_spec, flat_iter) for sub_spec in spec
            )
        elif isinstance(spec, list):
            return [
                self._unflatten_data_with_spec(sub_spec, flat_iter) for sub_spec in spec
            ]
        else:
            raise ValueError(f"Unsupported spec type in unflatten: {type(spec)}")

    # ------------------------------------------------------------------
    # TFRecord feature helpers
    # ------------------------------------------------------------------

    def _flat_tensors_to_example(
        self, flat_tensors: list[tf.Tensor]
    ) -> tf.train.Example:
        """Convert flattened tensors into a `tf.train.Example`."""
        features: dict[str, tf.train.Feature] = {}

        for i, tensor in enumerate(flat_tensors):
            key = f"f{i}"
            feature = self._tensor_to_feature(tensor)
            features[key] = feature

        return tf.train.Example(features=tf.train.Features(feature=features))

    def _tensor_to_feature(self, tensor: tf.Tensor) -> tf.train.Feature:
        """Convert tensor to tf.train.Feature.

        - Numeric tensors are flattened and stored as VarLen-like lists.
        - Other dtypes are serialized to bytes.
        """
        tensor = tf.convert_to_tensor(tensor)
        flat_tensor = tf.reshape(tensor, [-1])

        if tensor.dtype in (tf.float32, tf.float64):
            return tf.train.Feature(
                float_list=tf.train.FloatList(value=flat_tensor.numpy())
            )

        if tensor.dtype in (tf.int32, tf.int64):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=flat_tensor.numpy())
            )

        # Fallback: serialize any other dtype as bytes
        serialized = tf.io.serialize_tensor(tensor)
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized.numpy()])
        )

    def _build_feature_description(
        self, flat_spec_leaves: list[tuple[str, tf.TensorSpec]]
    ) -> dict[str, tf.io.FixedLenFeature | tf.io.VarLenFeature]:
        """Build feature_description for `tf.io.parse_single_example`."""
        feature_description: dict[str, tf.io.FixedLenFeature | tf.io.VarLenFeature] = {}

        for i, (_, leaf_spec) in enumerate(flat_spec_leaves):
            key = f"f{i}"

            if leaf_spec.dtype in (tf.float32, tf.float64):
                feature_description[key] = tf.io.VarLenFeature(tf.float32)
            elif leaf_spec.dtype in (tf.int32, tf.int64):
                feature_description[key] = tf.io.VarLenFeature(tf.int64)
            else:
                # Store other dtypes as a single serialized string
                feature_description[key] = tf.io.FixedLenFeature([], tf.string)

        return feature_description

    # ------------------------------------------------------------------
    # element_spec (de)serialization
    # ------------------------------------------------------------------

    def _serialize_element_spec(self, spec: Any) -> dict:
        """Serialize element_spec to a JSON-friendly dict."""
        if isinstance(spec, tf.TensorSpec):
            # Store both the full shape and the rank
            shape_list = [int(d) if d is not None else -1 for d in spec.shape]
            return {
                "type": "TensorSpec",
                "shape": shape_list,
                "rank": spec.shape.rank,  # Store rank explicitly
                "dtype": spec.dtype.name,
            }
        if isinstance(spec, dict):
            return {
                "type": "dict",
                "value": {k: self._serialize_element_spec(v) for k, v in spec.items()},
            }
        if isinstance(spec, tuple):
            return {
                "type": "tuple",
                "value": [self._serialize_element_spec(s) for s in spec],
            }
        if isinstance(spec, list):
            return {
                "type": "list",
                "value": [self._serialize_element_spec(s) for s in spec],
            }
        raise ValueError(f"Unsupported element_spec type: {type(spec)}")

    def _deserialize_element_spec(self, spec_dict: dict) -> Any:
        """Deserialize element_spec from a JSON-friendly dict."""
        spec_type = spec_dict["type"]

        if spec_type == "TensorSpec":
            shape = [None if d == -1 else d for d in spec_dict["shape"]]
            return tf.TensorSpec(shape=shape, dtype=getattr(tf, spec_dict["dtype"]))

        if spec_type == "dict":
            return {
                k: self._deserialize_element_spec(v)
                for k, v in spec_dict["value"].items()
            }

        if spec_type == "tuple":
            return tuple(self._deserialize_element_spec(s) for s in spec_dict["value"])

        if spec_type == "list":
            return [self._deserialize_element_spec(s) for s in spec_dict["value"]]

        raise ValueError(f"Unsupported spec_type: {spec_type}")


# Register the TensorflowDatasetMaterializer with the actual Dataset class (only if real class)
# if _is_dataset_real_class:
#     try:
#         materializer_registry.register_and_overwrite_type(
#             key=tf.data.Dataset,
#             type_=TFRecordDatasetMaterializer,
#             artifact_type=ArtifactType.DATA,
#         )
#     except Exception:
#         pass  # Registration may fail in test environments


class TFConfigDatasetMaterializer(BaseMaterializer):
    """Materializer for tf.data.Dataset created from CSV files.

    Instead of serializing the entire dataset to TFRecords, this materializer
    stores only the configuration needed to recreate the dataset using
    `tf.data.experimental.make_csv_dataset`. This is much more efficient and
    avoids shape-related issues during serialization/deserialization.

    This materializer works specifically with datasets created via:
    - `tf.data.experimental.make_csv_dataset`
    - MLPotion's `TFCSVDataLoader`

    Advantages:
    - Lightweight: Only stores config, not data
    - Fast: No TFRecord serialization overhead
    - Reliable: Recreates dataset with exact same parameters
    - Flexible: Works with any subsequent transformations (batching, shuffling, etc.)
    """

    ASSOCIATED_TYPES = (tf.data.Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> tf.data.Dataset:
        """Load dataset by recreating it from stored configuration.

        Args:
            data_type: The type of the data to load.

        Returns:
            Recreated tf.data.Dataset with the same configuration.
        """
        config_path = Path(self.uri) / "config.json"

        logger.info("Loading CSV dataset config from: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info("Recreating dataset with config: %s", config)

        # Use CSVDataLoader to recreate the dataset
        # This ensures we handle empty lines correctly (unlike make_csv_dataset)
        from mlpotion.frameworks.tensorflow.data.loaders import CSVDataLoader

        # Extract parameters for CSVDataLoader
        loader_config = {
            "file_pattern": config["file_pattern"],
            "batch_size": config["batch_size"],
            "label_name": config.get("label_name"),
            "column_names": config.get("column_names"),
        }

        # Handle num_epochs and other config
        extra_params = config.get("extra_params", {})
        if "num_epochs" in config:
            extra_params["num_epochs"] = config["num_epochs"]
        elif "num_epochs" not in extra_params:
            extra_params["num_epochs"] = 1

        if extra_params:
            loader_config["config"] = extra_params

        # Create loader and load dataset
        loader = CSVDataLoader(**loader_config)
        dataset = loader.load()

        # Apply any transformations that were recorded
        transformations = config.get("transformations", [])
        for transform in transformations:
            transform_type = transform["type"]
            params = transform["params"]

            if transform_type == "batch":
                dataset = dataset.batch(params["batch_size"])
            elif transform_type == "shuffle":
                dataset = dataset.shuffle(params["buffer_size"])
            elif transform_type == "prefetch":
                buffer_size = params["buffer_size"]
                if buffer_size == "AUTOTUNE":
                    buffer_size = tf.data.AUTOTUNE
                dataset = dataset.prefetch(buffer_size)
            elif transform_type == "unbatch":
                dataset = dataset.unbatch()
            elif transform_type == "repeat":
                count = params.get("count")
                dataset = dataset.repeat(count)
            # Add more transformation types as needed

        logger.info("✅ Successfully recreated CSV dataset")
        return dataset

    def save(self, data: tf.data.Dataset) -> None:
        """Save dataset configuration instead of actual data.

        This method attempts to extract the original CSV loading configuration
        from the dataset. If the dataset doesn't have this metadata, it falls
        back to the TFRecord materializer.

        Args:
            data: The dataset to save configuration for.
        """
        config_path = Path(self.uri) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("🔵 TFConfigDatasetMaterializer.save() called")
        logger.info("Saving CSV dataset config to: %s", config_path)
        logger.debug("Dataset type: %s", type(data))
        logger.debug("URI: %s", self.uri)

        # Try to extract configuration from the dataset
        # This requires the dataset to have been created with our loader
        # or to have metadata attached
        config = self._extract_config_from_dataset(data)

        if config is None:
            logger.warning(
                "❌ Could not extract CSV config from dataset. "
                "This materializer only works with datasets created from CSV files. "
                "Falling back to TFRecord materializer."
            )
            logger.debug(
                "Dataset attributes: %s",
                [attr for attr in dir(data) if not attr.startswith("__")],
            )
            # Fall back to TFRecord materializer
            from mlpotion.integrations.zenml.tensorflow.materializers import (
                TFRecordDatasetMaterializer,
            )

            logger.info("🔄 Falling back to TFRecordDatasetMaterializer")
            try:
                tfrecord_materializer = TFRecordDatasetMaterializer(self.uri)
                tfrecord_materializer.save(data)
                logger.info("✅ Successfully saved dataset as TFRecord")
            except Exception as e:
                logger.error(f"Failed to save as TFRecord: {e}")
                raise
            return

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("✅ Successfully saved CSV dataset config to: %s", config_path)

    def _extract_config_from_dataset(self, dataset: tf.data.Dataset) -> dict | None:
        """Extract CSV loading configuration from dataset.

        This attempts to find the original make_csv_dataset parameters by:
        1. Checking for attached metadata (if dataset was created by our loader)
        2. Inspecting the dataset's internal structure
        3. Using reasonable defaults if parameters can't be extracted

        Args:
            dataset: The dataset to extract config from.

        Returns:
            Configuration dict or None if this isn't a CSV dataset.
        """
        # Try to get metadata that was attached by CSVDataLoader
        # Check multiple ways in case the attribute is stored differently
        logger.debug("Checking for _csv_config attribute on dataset")

        # Method 1: hasattr check
        if hasattr(dataset, "_csv_config"):
            config = dataset._csv_config
            logger.info("✅ Found _csv_config via hasattr: %s", config)
            return config

        # Method 2: getattr with default
        try:
            config = getattr(dataset, "_csv_config", None)
            if config is not None:
                logger.info("✅ Found _csv_config via getattr: %s", config)
                return config
        except Exception as e:
            logger.debug("getattr failed: %s", e)

        # Method 3: Check __dict__ directly (in case attribute is set but hasattr fails)
        try:
            if hasattr(dataset, "__dict__") and "_csv_config" in dataset.__dict__:
                config = dataset.__dict__["_csv_config"]
                logger.info("✅ Found _csv_config in __dict__: %s", config)
                return config
        except Exception as e:
            logger.debug("__dict__ check failed: %s", e)

        logger.debug("❌ _csv_config not found after all checks")

        # Try to infer from dataset structure
        # This is a heuristic approach for datasets without explicit metadata
        # NOTE: We only infer if we're confident it's a CSV dataset.
        # For datasets created from tensor_slices or other sources, we return None
        # to allow fallback to TFRecord.
        try:
            # Check if this looks like a CSV dataset
            _ = dataset.element_spec

            # CSV datasets typically have (OrderedDict, label) or just OrderedDict structure
            # But we should be more conservative - only infer if we're very confident
            # For now, we'll only return config if _csv_config exists, not infer
            # This ensures proper fallback behavior
            pass

        except Exception as e:
            logger.debug("Failed to infer CSV config: %s", e)

        return None

    def _create_default_config(self, element_spec: Any) -> dict:
        """Create a default configuration when exact params aren't available.

        This creates a best-effort configuration that may not perfectly
        recreate the original dataset but will produce a compatible one.

        Args:
            element_spec: The element spec of the dataset.

        Returns:
            Default configuration dict.
        """
        logger.warning(
            "Creating default CSV config - original parameters not available. "
            "The recreated dataset may not exactly match the original."
        )

        return {
            "file_pattern": "**/*.csv",  # Placeholder - user must update
            "batch_size": 32,
            "label_name": None,
            "column_names": None,
            "num_epochs": 1,
            "extra_params": {"ignore_errors": True},
            "transformations": [],
            "_is_default": True,
            "_note": "This config was auto-generated. Update file_pattern and other params as needed.",
        }


if _is_dataset_real_class:
    try:
        materializer_registry.register_and_overwrite_type(
            key=tf.data.Dataset,
            type_=TFConfigDatasetMaterializer,
        )
    except Exception:
        pass  # Registration may fail in test environments
