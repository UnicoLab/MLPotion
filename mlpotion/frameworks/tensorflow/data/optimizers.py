"""TensorFlow dataset optimization."""

import tensorflow as tf
from mlpotion.frameworks.tensorflow.config import DataOptimizationConfig
from mlpotion.core.protocols import DatasetOptimizer as DatasetOptimizerProtocol
from loguru import logger


class DatasetOptimizer(DatasetOptimizerProtocol[tf.data.Dataset]):
    """Optimize TensorFlow datasets for training performance.

    This class applies a standard set of performance optimizations to a `tf.data.Dataset`:
    caching, shuffling, batching, and prefetching. These are critical for preventing
    data loading bottlenecks during training.

    Attributes:
        batch_size (int): The number of samples per batch.
        shuffle_buffer_size (int | None): Size of the shuffle buffer. If None, shuffling is disabled.
        prefetch (bool): Whether to prefetch data (uses `tf.data.AUTOTUNE`).
        cache (bool): Whether to cache the dataset in memory.

    Example:
        ```python
        from mlpotion.frameworks.tensorflow import DatasetOptimizer

        # Create optimizer
        optimizer = DatasetOptimizer(
            batch_size=32,
            shuffle_buffer_size=1000,
            cache=True,
            prefetch=True
        )

        # Apply to a raw dataset
        optimized_dataset = optimizer.optimize(raw_dataset)
        ```
    """

    def __init__(
        self,
        batch_size: int = 32,
        shuffle_buffer_size: int | None = None,
        prefetch: bool = True,
        cache: bool = False,
    ) -> None:
        """Initialize dataset optimizer.

        Args:
            batch_size: Batch size
            shuffle_buffer_size: Buffer size for shuffling (None = no shuffle)
            prefetch: Whether to prefetch batches
            cache: Whether to cache dataset in memory
        """
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch = prefetch
        self.cache = cache

    def optimize(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Optimize dataset for training.

        Applies optimizations in the following order:
        1. **Cache**: Caches data in memory (if enabled).
        2. **Shuffle**: Randomizes data order (if `shuffle_buffer_size` is set).
        3. **Batch**: Groups data into batches.
        4. **Prefetch**: Prepares the next batch while the current one is being processed.

        Args:
            dataset: The input `tf.data.Dataset`.

        Returns:
            tf.data.Dataset: The optimized dataset pipeline.
        """
        logger.info("Applying dataset optimizations...")

        # Track transformations for CSV materializer
        transformations = []
        if hasattr(dataset, "_csv_config"):
            transformations = dataset._csv_config.get("transformations", [])

        # Cache first (before shuffling/batching)
        if self.cache:
            logger.info("Caching dataset in memory")
            dataset = dataset.cache()
            # Note: cache() doesn't need to be recorded as it's a performance optimization

        # Shuffle before batching
        if self.shuffle_buffer_size:
            logger.info(f"Shuffling with buffer size {self.shuffle_buffer_size}")
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                reshuffle_each_iteration=True,
            )
            transformations.append(
                {
                    "type": "shuffle",
                    "params": {"buffer_size": self.shuffle_buffer_size},
                }
            )

        # Batch
        logger.info(f"Batching with size {self.batch_size}")
        dataset = dataset.batch(self.batch_size)
        transformations.append(
            {
                "type": "batch",
                "params": {"batch_size": self.batch_size},
            }
        )

        # Prefetch last for best performance
        if self.prefetch:
            logger.info("Prefetching with AUTOTUNE")
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            transformations.append(
                {
                    "type": "prefetch",
                    "params": {"buffer_size": "AUTOTUNE"},
                }
            )

        # Preserve CSV config if it exists
        if hasattr(dataset, "_csv_config"):
            dataset._csv_config["transformations"] = transformations

        return dataset

    @classmethod
    def from_config(cls, config: DataOptimizationConfig) -> "DatasetOptimizer":
        """Create optimizer from configuration.

        Args:
            config: Optimization configuration

        Returns:
            Configured optimizer instance
        """
        return cls(
            batch_size=config.batch_size,
            shuffle_buffer_size=config.shuffle_buffer_size,
            prefetch=config.prefetch,
            cache=config.cache,
        )
