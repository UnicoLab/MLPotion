import unittest

import tensorflow as tf
from loguru import logger

from mlpotion.frameworks.tensorflow.config import DataOptimizationConfig
from mlpotion.frameworks.tensorflow.data.optimizers import DatasetOptimizer
from tests.core import (
    TestBase,
)  # provides temp_dir, setUpClass/tearDownClass, setUp/tearDown


class TestDatasetOptimizer(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info("Setting up DatasetOptimizer tests")

        # Simple 1D dataset: 10 elements [0..9]
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            tf.range(10, dtype=tf.int32)
        )

    # ------------------------------------------------------------------ #
    # optimize(): basic batching + prefetch (defaults)
    # ------------------------------------------------------------------ #
    def test_optimize_batches_and_prefetches_by_default(self) -> None:
        """optimize() should batch and prefetch with default settings."""
        logger.info("Testing DatasetOptimizer default behavior")

        optimizer = DatasetOptimizer()  # batch_size=32, prefetch=True, no shuffle/cache
        optimized = optimizer.optimize(self.raw_dataset)

        self.assertIsInstance(optimized, tf.data.Dataset)

        # Materialize batches
        batches = list(optimized)
        # With batch_size=32 and only 10 elements, we get a single batch of size 10
        self.assertEqual(len(batches), 1)
        self.assertEqual(int(batches[0].shape[0]), 10)

        # Check that all elements are preserved
        all_vals = tf.concat(batches, axis=0).numpy().tolist()
        self.assertEqual(all_vals, list(range(10)))

    # ------------------------------------------------------------------ #
    # optimize(): batching with custom batch_size, no prefetch
    # ------------------------------------------------------------------ #
    def test_optimize_respects_custom_batch_size(self) -> None:
        """optimize() should respect the configured batch_size."""
        logger.info("Testing DatasetOptimizer with custom batch size")

        optimizer = DatasetOptimizer(batch_size=4, prefetch=False)
        optimized = optimizer.optimize(self.raw_dataset)

        batches = list(optimized)
        # 10 elements with batch size 4 -> 3 batches: 4,4,2
        self.assertEqual(len(batches), 3)
        self.assertEqual(int(batches[0].shape[0]), 4)
        self.assertEqual(int(batches[1].shape[0]), 4)
        self.assertEqual(int(batches[2].shape[0]), 2)

        concatenated = tf.concat(batches, axis=0).numpy().tolist()
        self.assertEqual(concatenated, list(range(10)))

    # ------------------------------------------------------------------ #
    # optimize(): shuffle + cache should not change set of elements
    # ------------------------------------------------------------------ #
    def test_optimize_with_shuffle_and_cache_preserves_elements(self) -> None:
        """Using shuffle and cache should not change the multiset of elements."""
        logger.info("Testing DatasetOptimizer with shuffle and cache")

        optimizer = DatasetOptimizer(
            batch_size=3,
            shuffle_buffer_size=10,
            prefetch=False,
            cache=True,
        )
        optimized = optimizer.optimize(self.raw_dataset)

        # Unbatch to inspect all elements after transformations
        unbatched = optimized.unbatch()
        vals = list(unbatched.as_numpy_iterator())

        # Same number of elements and same multiset, possibly different order
        self.assertEqual(len(vals), 10)
        self.assertCountEqual(vals, list(range(10)))  # order-insensitive

    # ------------------------------------------------------------------ #
    # from_config(): construction from DataOptimizationConfig
    # ------------------------------------------------------------------ #
    def test_from_config_creates_optimizer_with_matching_fields(self) -> None:
        """from_config() should transfer config fields onto the optimizer."""
        logger.info("Testing DatasetOptimizer.from_config")

        config = DataOptimizationConfig(
            batch_size=16,
            shuffle_buffer_size=128,
            prefetch=False,
            cache=True,
        )

        optimizer = DatasetOptimizer.from_config(config)

        self.assertEqual(optimizer.batch_size, 16)
        self.assertEqual(optimizer.shuffle_buffer_size, 128)
        self.assertFalse(optimizer.prefetch)
        self.assertTrue(optimizer.cache)


if __name__ == "__main__":
    unittest.main()
