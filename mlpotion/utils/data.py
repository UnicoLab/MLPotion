#!/usr/bin/env python3
"""Generate sample CSV data for examples and tests."""

import numpy as np
import pandas as pd
from loguru import logger


def generate_regression_data(
    file_path: str | None = None, n_samples: int = 1000, n_features: int = 10
) -> pd.DataFrame:
    """Generate sample regression data.

    Args:
        file_path: Path to save the data
        n_samples: Number of samples to generate
        n_features: Number of features

    Returns:
        DataFrame with features and target
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate target as linear combination + noise
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.1

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # saving data
    if file_path:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved data to {file_path}")

    return df


def generate_classification_data(
    file_path: str | None = None, n_samples: int = 1000, n_features: int = 10
) -> pd.DataFrame:
    """Generate sample classification data.

    Args:
        file_path: Path to save the data
        n_samples: Number of samples to generate
        n_features: Number of features

    Returns:
        DataFrame with features and binary target
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate binary target
    weights = np.random.randn(n_features)
    logits = X @ weights
    y = (logits > 0).astype(int)

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # saving data
    if file_path:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved data to {file_path}")

    return df
