"""Framework detection and validation utilities."""

import sys
from typing import Literal

from mlpotion.core.exceptions import FrameworkNotInstalledError

FrameworkName = Literal["tensorflow", "torch"]


def is_framework_available(framework: FrameworkName) -> bool:
    """Check if a framework is available.

    Args:
        framework: Framework name ("tensorflow" or "torch")

    Returns:
        True if framework is installed and importable
    """
    try:
        if framework == "tensorflow":
            import tensorflow

            return True
        elif framework == "torch":
            import torch

            return True
        return False
    except ImportError:
        return False


def require_framework(framework: FrameworkName, install_command: str) -> None:
    """Require a framework to be installed.

    Args:
        framework: Framework name
        install_command: Installation command to show in error

    Raises:
        FrameworkNotInstalledError: If framework is not installed
    """
    if not is_framework_available(framework):
        raise FrameworkNotInstalledError(
            f"{framework} is not installed. "
            f"Install it with: pip install {install_command}"
        )


def get_available_frameworks() -> list[FrameworkName]:
    """Get list of available frameworks.

    Returns:
        List of framework names that are installed
    """
    frameworks: list[FrameworkName] = ["tensorflow", "torch"]
    return [f for f in frameworks if is_framework_available(f)]