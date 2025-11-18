import logging
from typing import Annotated
import tensorflow as tf
from typing import Any
from zenml import log_step_metadata, step
from loguru import logger

# loading
from mlpotion.frameworks.tensorflow.config import DataLoadingConfig
from mlpotion.frameworks.tensorflow.data.loaders import TFCSVDataLoader

# optimizing
from mlpotion.frameworks.tensorflow.config import DataOptimizationConfig
from mlpotion.frameworks.tensorflow.data.optimizers import TFDatasetOptimizer

logger = logging.getLogger(__name__)

@step
def load_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str = "target",
    metadata: dict[str, Any] | None = None,
) -> Annotated[tf.data.Dataset, "TF DataSet"]:
    """Load data from local CSV files using TensorFlow's efficient loading."""
    logger.info(f"Loading data from: {file_path}")

    # defining configuration
    config = DataLoadingConfig(
        file_pattern=file_path,
        batch_size=batch_size,
        label_name=label_name,
    )

    # initializing data loader
    loader = TFCSVDataLoader(
        **config.dict()
    )
    # loading data
    dataset = loader.load()

    # adding metadata
    if metadata:
        log_step_metadata(metadata=metadata)

    return dataset


@step
def optimize_data(
    dataset: tf.data.Dataset,
    batch_size: int = 32,
    shuffle_buffer_size: int | None = None,
    prefetch: bool = True,
    cache: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Annotated[tf.data.Dataset, "TF DataSet"]:
    """Optimize TensorFlow dataset for training performance."""
    logger.info("Optimizing dataset for training performance")

    config = DataOptimizationConfig(
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch=prefetch,
        cache=cache,
    )

    optimizer = TFDatasetOptimizer(
        **config.dict()
    )
    dataset = optimizer.optimize(dataset)

    # adding metadata
    if metadata:
        log_step_metadata(metadata=metadata)
        
    return dataset