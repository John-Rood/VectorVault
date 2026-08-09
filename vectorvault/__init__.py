from .vault import Vault
from .utils import download_url, wrap
from .local_storage import LocalStorageManager, LocalVaultStorageManager
from .model_catalog import (
    ModelCapabilityError,
    default_thinking_level,
    enrich_model_metadata,
    get_allowed_thinking_levels,
    get_default_thinking_level,
    get_model_capability,
    get_model_thinking_catalog,
    get_provider_thinking_kwargs,
    list_thinking_levels,
    load_model_catalog,
    resolve_model_alias,
    serialize_model_catalog,
    translate_thinking_level,
    validate_thinking_level,
)

# Compatibility-friendly public names for the thinking capability API.
ThinkingLevelError = ModelCapabilityError
get_thinking_capability = get_model_capability
resolve_thinking_model = resolve_model_alias

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore

try:
    __version__ = _pkg_version("vector-vault")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "7.4.9.14"
