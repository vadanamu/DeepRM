"""
DeepRM package.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # resolve the installed package version at runtime
    # An unpacked sdist run with src on the path finds its incomplete egg-info: no version (None) on Python 3.12,
    # a KeyError later.
    __version__ = version("deeprm") or "0+unknown"
except (PackageNotFoundError, KeyError):  # a source tree that is not installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
