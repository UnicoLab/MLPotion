"""FlowyML integration for MLPotion.

Provides reusable FlowyML pipeline steps that wrap MLPotion's
framework-agnostic ML components with full FlowyML features:
assets, context injection, caching, retry, and resource specs.

Install: pip install mlpotion[flowyml]
"""

try:
    from flowyml.core.step import step, Step  # noqa: F401

    FLOWYML_AVAILABLE = True
except ImportError:
    FLOWYML_AVAILABLE = False

if FLOWYML_AVAILABLE:
    from mlpotion.integrations.flowyml.adapters import FlowyMLAdapter  # noqa: F401

__all__ = ["FlowyMLAdapter"]
