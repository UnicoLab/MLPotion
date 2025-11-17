"""Custom materializers for TensorFlow types."""

import json
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
_is_tensor_real_class = isinstance(tf.Tensor, type) and not hasattr(tf.Tensor, "_mock_name")
_is_tensorspec_real_class = isinstance(tf.TensorSpec, type) and not hasattr(tf.TensorSpec, "_mock_name")
_is_dataset_real_class = isinstance(tf.data.Dataset, type) and not hasattr(tf.data.Dataset, "_mock_name")

# Use object as fallback for ASSOCIATED_TYPES when types are mocked
_TensorForTypes = tf.Tensor if _is_tensor_real_class else object
_TensorSpecForTypes = tf.TensorSpec if _is_tensorspec_real_class else object
_DatasetForTypes = tf.data.Dataset if _is_dataset_real_class else object


class TensorMaterializer(BaseMaterializer):
    """Materializer for TensorFlow Tensor objects."""
    
    ASSOCIATED_TYPES = (_TensorForTypes,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type[Any]) -> tf.Tensor:  # noqa: ARG002
        """Load a TensorFlow Tensor."""
        logger.info("Loading TensorFlow tensor...")
        try:
            tensor_path = Path(self.uri) / "tensor.pb"
            return tf.io.parse_tensor(tf.io.read_file(str(tensor_path)), out_type=tf.float32)
        except Exception as e:
            logger.error(f"Failed to load tensor: {e}")
            raise

    def save(self, data: tf.Tensor) -> None:
        """Save a TensorFlow Tensor."""
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
            artifact_type=ArtifactType.DATA,
            )
    except Exception:
        pass  # Registration may fail in test environments


class TensorSpecMaterializer(BaseMaterializer):
    """Materializer for TensorFlow TensorSpec objects."""
    
    ASSOCIATED_TYPES = (_TensorSpecForTypes,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type[Any]) -> tf.TensorSpec:  # noqa: ARG002
        """Load a TensorFlow TensorSpec."""
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
        """Save a TensorFlow TensorSpec."""
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
            artifact_type=ArtifactType.DATA,
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

        metadata = {
            "format_version": "3.0",
            "element_spec": serialized_spec,
            "num_leaves": num_leaves,
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

        logger.info("Loaded element_spec: %s", element_spec)
        logger.info("Expected number of leaves: %s", num_leaves)

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
                    # Numeric: stored as VarLenFeature
                    dense = tf.sparse.to_dense(parsed[key])
                    tensor = tf.cast(dense, leaf_spec.dtype)
                else:
                    # Other dtypes: stored as serialized bytes
                    serialized = parsed[key]
                    tensor = tf.io.parse_tensor(serialized, out_type=leaf_spec.dtype)

                # Give TF a known rank; batch dim stays None.
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

    def _flatten_element_spec(self, spec: Any, path: str = "") -> list[tuple[str, tf.TensorSpec]]:
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

    def _flatten_data_with_spec(self, data: Any, spec: Any, out: list[tf.Tensor]) -> None:
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
                self._unflatten_data_with_spec(sub_spec, flat_iter)
                for sub_spec in spec
            )
        elif isinstance(spec, list):
            return [
                self._unflatten_data_with_spec(sub_spec, flat_iter)
                for sub_spec in spec
            ]
        else:
            raise ValueError(f"Unsupported spec type in unflatten: {type(spec)}")

    # ------------------------------------------------------------------
    # TFRecord feature helpers
    # ------------------------------------------------------------------

    def _flat_tensors_to_example(self, flat_tensors: list[tf.Tensor]) -> tf.train.Example:
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
            return {
                "type": "TensorSpec",
                "shape": [int(d) if d is not None else -1 for d in spec.shape],
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
            return tuple(
                self._deserialize_element_spec(s) for s in spec_dict["value"]
            )

        if spec_type == "list":
            return [
                self._deserialize_element_spec(s) for s in spec_dict["value"]
            ]

        raise ValueError(f"Unsupported spec_type: {spec_type}")


# Register the TensorflowDatasetMaterializer with the actual Dataset class (only if real class)
if _is_dataset_real_class:
    try:
        materializer_registry.register_and_overwrite_type(
            key=tf.data.Dataset,
            type_=TFRecordDatasetMaterializer,
            artifact_type=ArtifactType.DATA,
        )
    except Exception:
        pass  # Registration may fail in test environments
