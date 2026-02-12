"""Pytest configuration for FlowyML integration tests.

Unlike ZenML, FlowyML does not require special environment variables
for running steps outside a full pipeline context. Steps can be called
directly as regular Python functions.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip FlowyML tests if flowyml is not installed."""
    try:
        import flowyml  # noqa: F401
    except ImportError:
        skip_flowyml = pytest.mark.skip(reason="flowyml not installed")
        for item in items:
            item.add_marker(skip_flowyml)
