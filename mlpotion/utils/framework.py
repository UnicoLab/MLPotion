"""Framework detection and validation utilities."""
from typing import Literal, Mapping
from loguru import logger
from importlib import import_module

from mlpotion.core.exceptions import FrameworkNotInstalledError

FrameworkName = Literal["tensorflow", "torch", "keras", "jax", "zenml"]


class FrameworkChecker:
    """Utility class to check availability of ML frameworks."""

    _FRAMEWORK_IMPORTS: Mapping[FrameworkName, str] = {
        "tensorflow": "tensorflow",
        "torch": "torch",
        "keras": "keras",
        "jax": "jax",
        "zenml": "zenml",
    }

    @classmethod
    def is_available(cls, framework: FrameworkName) -> bool:
        """Check whether a framework is installed and importable.

        Args:
            framework: Framework identifier supported by this checker.

        Returns:
            True if the framework can be imported, otherwise False.

        Raises:
            ValueError: If the provided framework is not known.

        Example:
            ```python
            if FrameworkChecker.is_available("torch"):
                print("PyTorch is installed!")
            else:
                print("PyTorch is missing.")
            ```
        """
        if framework not in cls._FRAMEWORK_IMPORTS:
            msg = f"Unsupported framework: {framework}"
            logger.error(msg)
            raise ValueError(msg)

        module_name = cls._FRAMEWORK_IMPORTS[framework]

        try:
            import_module(module_name)
            return True
        except ImportError:
            return False

# Convenience alias for backwards compatibility
def is_framework_available(framework: FrameworkName) -> bool:
    return FrameworkChecker.is_available(framework)


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
            f"Install it with: poetry add {install_command}"
        )


def get_available_frameworks() -> list[FrameworkName]:
    """Get list of available frameworks.

    Returns:
        List of framework names that are installed
    """
    frameworks: list[FrameworkName] = list(FrameworkChecker._FRAMEWORK_IMPORTS.keys())
    return [f for f in frameworks if is_framework_available(f)]