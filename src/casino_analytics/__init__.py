"""casino_analytics package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("casino-analytics")
except PackageNotFoundError:
    __version__ = "0.0.0"
