"""Pytest configuration for ZenML integration tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_zenml_test_environment():
    """Set up a temporary ZenML configuration for testing.

    This enables running ZenML steps without requiring a full stack initialization,
    which is ideal for unit testing individual steps.
    """
    # Store original environment variables
    original_run_without_stack = os.environ.get("ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK")
    original_analytics = os.environ.get("ZENML_ANALYTICS_OPT_IN")
    original_logging = os.environ.get("ZENML_LOGGING_VERBOSITY")

    # Set environment variables to run steps without stack
    os.environ["ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK"] = "true"
    os.environ["ZENML_ANALYTICS_OPT_IN"] = "false"
    os.environ["ZENML_LOGGING_VERBOSITY"] = "ERROR"

    yield

    # Restore original environment
    if original_run_without_stack is not None:
        os.environ["ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK"] = original_run_without_stack
    else:
        os.environ.pop("ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK", None)

    if original_analytics is not None:
        os.environ["ZENML_ANALYTICS_OPT_IN"] = original_analytics
    else:
        os.environ.pop("ZENML_ANALYTICS_OPT_IN", None)

    if original_logging is not None:
        os.environ["ZENML_LOGGING_VERBOSITY"] = original_logging
    else:
        os.environ.pop("ZENML_LOGGING_VERBOSITY", None)
