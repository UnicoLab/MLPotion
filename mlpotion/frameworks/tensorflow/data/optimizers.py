"""TensorFlow dataset optimization."""

import tensorflow as tf
from mlpotion.frameworks.tensorflow.config import DataOptimizationConfig
from mlpotion.core.protocols import DatasetOptimizer
from loguru import logger


class TFDatasetOptimizer(DatasetOptimizer[tf.data.Dataset]):
    """Optimize TensorFlow datasets for training performance.

    Example:
        optimizer = TFDatasetOptimizer(batch_size=32, cache=True)
        optimized_dataset = optimizer.optimize(raw_dataset)
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

        Applies optimizations in order:
        1. Cache (if enabled)
        2. Shuffle (if buffer size provided)
        3. Batch
        4. Prefetch (if enabled)

        Args:
            dataset: Input dataset

        Returns:
            Optimized dataset
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
            transformations.append({
                "type": "shuffle",
                "params": {"buffer_size": self.shuffle_buffer_size},
            })

        # Batch
        logger.info(f"Batching with size {self.batch_size}")
        dataset = dataset.batch(self.batch_size)
        transformations.append({
            "type": "batch",
            "params": {"batch_size": self.batch_size},
        })

        # Prefetch last for best performance
        if self.prefetch:
            logger.info("Prefetching with AUTOTUNE")
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            transformations.append({
                "type": "prefetch",
                "params": {"buffer_size": "AUTOTUNE"},
            })

        # Preserve CSV config if it exists
        if hasattr(dataset, "_csv_config"):
            dataset._csv_config["transformations"] = transformations

        return dataset

    @classmethod
    def from_config(cls, config: DataOptimizationConfig) -> "TFDatasetOptimizer":
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