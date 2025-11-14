from .vault import Vault
from .utils import download_url, wrap

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # type: ignore

try:
    __version__ = _pkg_version("vector_vault")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"