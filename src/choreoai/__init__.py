"""ChoreoAI package metadata and lightweight exports."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("choreoai")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
