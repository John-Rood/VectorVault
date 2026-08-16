"""Canonical packaged model and thinking-capability catalog.

The JSON resource is the package's sole model/capability authority.  API and UI
layers should consume these helpers instead of maintaining parallel tables.
"""
from __future__ import annotations

import copy
import json
from importlib import resources
from typing import Any, Dict, Iterable, List, Optional

_RESOURCE = "model_catalog.json"
_CATALOG_CACHE: Optional[Dict[str, Any]] = None


class ModelCapabilityError(ValueError):
    """Raised when a model or thinking-level combination is invalid."""


def _validate_catalog(catalog: Dict[str, Any]) -> None:
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("models"), list):
        raise ModelCapabilityError("Invalid packaged model catalog schema")
    index: Dict[str, Dict[str, Any]] = {}
    for entry in catalog["models"]:
        model = entry.get("id")
        if not model or model in index:
            raise ModelCapabilityError(f"Invalid or duplicate model id {model!r}")
        provider = entry.get("provider")
        if provider not in {"openai", "anthropic", "gemini", "grok"}:
            raise ModelCapabilityError(f"Invalid provider for model {model!r}")
        if not isinstance(entry.get("context_window"), int) or entry["context_window"] <= 0:
            raise ModelCapabilityError(f"Invalid context window for model {model!r}")
        if not isinstance(entry.get("frontend"), bool):
            raise ModelCapabilityError(f"Invalid frontend flag for model {model!r}")
        thinking = entry.get("thinking") or {}
        levels = thinking.get("levels")
        if not isinstance(levels, list) or len(levels) != len(set(levels)):
            raise ModelCapabilityError(f"Invalid thinking levels for model {model!r}")
        supported = thinking.get("supported")
        if supported is not bool(levels):
            raise ModelCapabilityError(f"Thinking support mismatch for model {model!r}")
        default = thinking.get("default")
        if default is not None and default not in levels:
            raise ModelCapabilityError(f"Invalid thinking default for model {model!r}")
        translation = thinking.get("translation")
        if supported and not isinstance(translation, dict):
            raise ModelCapabilityError(f"Missing thinking translation for model {model!r}")
        if not supported and translation is not None:
            raise ModelCapabilityError(f"Unexpected thinking translation for model {model!r}")
        if not isinstance(thinking.get("omit_parameters"), list):
            raise ModelCapabilityError(f"Invalid omitted parameters for model {model!r}")
        index[model] = entry
    for provider, model in catalog.get("defaults", {}).items():
        if model not in index or index[model]["provider"] != provider:
            raise ModelCapabilityError(f"Invalid {provider!r} default model {model!r}")
    for model, entry in index.items():
        alias = entry.get("alias_for")
        if alias and alias not in index:
            raise ModelCapabilityError(f"Unknown alias target {alias!r} for model {model!r}")
        if alias and index[alias]["provider"] != entry["provider"]:
            raise ModelCapabilityError(f"Cross-provider alias target {alias!r} for model {model!r}")
        seen = set()
        while alias:
            if alias in seen:
                raise ModelCapabilityError(f"Model alias cycle detected at {alias!r}")
            seen.add(alias)
            alias = index[alias].get("alias_for")
    lists = catalog.get("lists")
    if not isinstance(lists, dict):
        raise ModelCapabilityError("Invalid packaged model catalog lists")
    for name, models in lists.items():
        if not isinstance(models, list) or len(models) != len(set(models)):
            raise ModelCapabilityError(f"Invalid model list {name!r}")
        unknown = set(models) - set(index)
        if unknown:
            raise ModelCapabilityError(f"Unknown models in list {name!r}: {sorted(unknown)!r}")


def load_model_catalog() -> Dict[str, Any]:
    """Load and return a defensive copy of the packaged canonical catalog."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        package = resources.files(__package__)
        with package.joinpath(_RESOURCE).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        _validate_catalog(loaded)
        _CATALOG_CACHE = loaded
    return copy.deepcopy(_CATALOG_CACHE)


def get_model_thinking_catalog() -> Dict[str, Any]:
    """Compatibility name returning a defensive copy of the canonical catalog."""
    return load_model_catalog()


def _index() -> Dict[str, Dict[str, Any]]:
    return {entry["id"]: entry for entry in load_model_catalog()["models"]}


def resolve_model_alias(model: Optional[str], *, allow_unknown: bool = False) -> str:
    """Resolve a catalog alias (and ``default``) to its concrete model ID.

    ``allow_unknown`` exists for legacy request paths that accept private or
    fine-tuned provider model IDs. Catalog/UI callers remain strict by default.
    """
    catalog = load_model_catalog()
    current = model or "default"
    if current == "Default":
        current = "default"
    if current == "default":
        current = catalog["defaults"]["openai"]
    index = {entry["id"]: entry for entry in catalog["models"]}
    if current not in index:
        if allow_unknown:
            return current
        raise ModelCapabilityError(f"Unknown model {current!r}")
    seen = set()
    while index[current].get("alias_for"):
        if current in seen:
            raise ModelCapabilityError(f"Model alias cycle detected at {current!r}")
        seen.add(current)
        current = index[current]["alias_for"]
    return current


def get_model_capability(model: Optional[str]) -> Dict[str, Any]:
    """Return copied capability metadata for a model or alias."""
    requested = model or "default"
    if requested == "Default":
        requested = "default"
    resolved = resolve_model_alias(requested)
    source = _index()[resolved]
    result = copy.deepcopy(source)
    result["requested_model"] = requested
    result["resolved_model"] = resolved
    return result


def list_thinking_levels(model: Optional[str]) -> List[str]:
    """Return the ordered thinking levels supported by ``model``."""
    return list(get_model_capability(model)["thinking"]["levels"])


def get_allowed_thinking_levels(model: Optional[str]) -> List[str]:
    """Compatibility name for :func:`list_thinking_levels`."""
    return list_thinking_levels(model)


def default_thinking_level(model: Optional[str]) -> Optional[str]:
    """Return the documented provider default, or ``None`` when unsupported."""
    return get_model_capability(model)["thinking"]["default"]


def get_default_thinking_level(model: Optional[str]) -> Optional[str]:
    """Compatibility name for :func:`default_thinking_level`."""
    return default_thinking_level(model)


def supports_thinking(model: Optional[str]) -> bool:
    return bool(get_model_capability(model)["thinking"]["supported"])


def validate_thinking_level(model: Optional[str], thinking_level: Optional[str]) -> Optional[str]:
    """Validate and normalize a model/thinking selection.

    Omission is deliberately preserved as ``None`` so existing callers continue
    to use each provider's pre-existing default behavior.
    """
    # Omission is a true no-op, including for user-supplied fine-tuned model IDs.
    # This preserves the package's historical support for models not in its catalog.
    if thinking_level is None or thinking_level == "":
        return None
    resolve_model_alias(model)
    normalized = str(thinking_level).strip().lower()
    allowed = list_thinking_levels(model)
    if normalized not in allowed:
        rendered = ", ".join(allowed) if allowed else "none"
        raise ModelCapabilityError(
            f"Thinking level {thinking_level!r} is not supported by model {model!r}; "
            f"allowed values: {rendered}"
        )
    return normalized


def translate_thinking_level(model: Optional[str], thinking_level: Optional[str]) -> Dict[str, Any]:
    """Translate a validated generic level to provider-safe SDK kwargs."""
    level = validate_thinking_level(model, thinking_level)
    if level is None:
        return {}
    thinking = get_model_capability(model)["thinking"]
    kind = thinking["translation"]["kind"]
    if kind in {"openai_reasoning_effort", "xai_reasoning_effort"}:
        return {"reasoning_effort": level}
    if kind == "anthropic_effort":
        translated: Dict[str, Any] = {"output_config": {"effort": level}}
        if thinking.get("adaptive_thinking"):
            translated["thinking"] = {"type": "adaptive"}
        return translated
    if kind == "gemini_thinking_level":
        return {"thinking_config": {"thinking_level": level.upper()}}
    raise ModelCapabilityError(
        f"No provider translation is defined for thinking-capable model {model!r}"
    )


def get_provider_thinking_kwargs(model: Optional[str], thinking_level: Optional[str]) -> Dict[str, Any]:
    """Compatibility name for :func:`translate_thinking_level`."""
    return translate_thinking_level(model, thinking_level)


def should_omit_parameter(model: Optional[str], thinking_level: Optional[str], parameter: str) -> bool:
    """Return whether an explicit level requires omission of a sampling field."""
    level = validate_thinking_level(model, thinking_level)
    if level is None:
        return False
    return parameter in get_model_capability(model)["thinking"].get("omit_parameters", [])


def provider_for_model(model: Optional[str]) -> str:
    return get_model_capability(model)["provider"]


def model_maps(frontend: bool = False) -> Dict[str, Dict[str, Any]]:
    """Build SDK-compatible provider token maps from the JSON resource."""
    catalog = load_model_catalog()
    result: Dict[str, Dict[str, Any]] = {
        provider: {} for provider in ("openai", "grok", "anthropic", "gemini")
    }
    for entry in catalog["models"]:
        if not frontend or entry.get("frontend"):
            result[entry["provider"]][entry["id"]] = entry["context_window"]
    for provider, default in catalog["defaults"].items():
        result[provider]["default"] = default
    return result


def aliases() -> Dict[str, str]:
    return {
        entry["id"]: entry["alias_for"]
        for entry in load_model_catalog()["models"]
        if entry.get("alias_for")
    }


def serialize_model_catalog(frontend: bool = True) -> Dict[str, Any]:
    """Serialize stable model metadata for API/UI consumers."""
    catalog = load_model_catalog()
    rows = []
    for entry in catalog["models"]:
        if frontend and not entry.get("frontend"):
            continue
        capability = get_model_capability(entry["id"])
        thinking = capability["thinking"]
        rows.append({
            "model": entry["id"],
            "resolved_model": capability["resolved_model"],
            "token_limit": entry["context_window"],
            "provider": entry["provider"],
            "thinking_levels": list(thinking["levels"]),
            "default_thinking_level": thinking["default"],
            "supports_thinking": thinking["supported"],
            "alias_for": entry.get("alias_for"),
        })
    if frontend:
        default = catalog["defaults"]["openai"]
        default_capability = get_model_capability(default)
        rows.insert(0, {
            "model": "default",
            "resolved_model": default,
            "token_limit": default_capability["context_window"],
            "provider": "openai",
            "thinking_levels": list_thinking_levels(default),
            "default_thinking_level": default_thinking_level(default),
            "supports_thinking": supports_thinking(default),
            "alias_for": default,
        })
    return {"schema_version": catalog["schema_version"], "models": rows}


def enrich_model_metadata(models: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy API model rows and add the canonical nested ``thinking`` shape."""
    enriched = []
    for row in models:
        copied = copy.deepcopy(row)
        model = copied.get("model") or copied.get("id")
        capability = get_model_capability(model)
        thinking = capability["thinking"]
        copied["thinking"] = {
            "levels": list(thinking["levels"]),
            "default": thinking["default"],
            "supported": thinking["supported"],
        }
        copied["resolved_model"] = capability["resolved_model"]
        enriched.append(copied)
    return enriched


# Backward/phase-2 compatibility aliases.
ThinkingLevelError = ModelCapabilityError
resolve_thinking_model = resolve_model_alias
get_thinking_capability = get_model_capability

_MAPS = model_maps(False)
_FRONT_MAPS = model_maps(True)
OPENAI_MODELS = _MAPS["openai"]
OPENAI_FRONT_MODELS = _FRONT_MAPS["openai"]
GROK_MODELS = _MAPS["grok"]
GROK_FRONT_MODELS = _FRONT_MAPS["grok"]
ANTHROPIC_MODELS = _MAPS["anthropic"]
ANTHROPIC_FRONT_MODELS = _FRONT_MAPS["anthropic"]
GEMINI_MODELS = _MAPS["gemini"]
GEMINI_FRONT_MODELS = _FRONT_MAPS["gemini"]
LATEST_MODELS_MAP = aliases()
_lists = load_model_catalog()["lists"]
OPENAI_IMG_CAPABLE = _lists["OPENAI_IMG_CAPABLE"]
OPENAI_NO_STREAM_LIST = _lists["OPENAI_NO_STREAM_LIST"]
OPENAI_NO_TEMPERATURE_LIST = _lists["OPENAI_NO_TEMPERATURE_LIST"]
ANTHROPIC_NO_TEMPERATURE_LIST = _lists["ANTHROPIC_NO_TEMPERATURE_LIST"]
GEMINI_MULTIMODAL_MODELS = _lists["GEMINI_MULTIMODAL_MODELS"]
GEMINI_THINKING_MODELS = _lists["GEMINI_THINKING_MODELS"]
MODEL_METADATA = {
    entry["id"]: {
        **copy.deepcopy(entry.get("metadata", {})),
        "context_window": entry["context_window"],
    }
    for entry in load_model_catalog()["models"]
}
