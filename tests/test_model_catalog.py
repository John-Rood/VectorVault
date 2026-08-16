import json
from importlib import resources

import pytest

import vectorvault
from vectorvault.model_catalog import (
    ModelCapabilityError,
    default_thinking_level,
    enrich_model_metadata,
    get_allowed_thinking_levels,
    get_default_thinking_level,
    get_model_capability,
    get_model_thinking_catalog,
    list_thinking_levels,
    load_model_catalog,
    resolve_model_alias,
    serialize_model_catalog,
    translate_thinking_level,
    validate_thinking_level,
)


def test_packaged_resource_is_readable_complete_and_defensive():
    resource = resources.files("vectorvault").joinpath("model_catalog.json")
    assert resource.is_file()
    raw = json.loads(resource.read_text(encoding="utf-8"))
    assert raw == load_model_catalog()
    assert raw["schema_version"] == 1
    assert len(raw["models"]) == 97
    copied = get_model_thinking_catalog()
    copied["models"].clear()
    assert len(load_model_catalog()["models"]) == 97


def test_every_catalog_entry_has_explicit_coherent_thinking_contract():
    catalog = load_model_catalog()
    ids = {entry["id"] for entry in catalog["models"]}
    assert len(ids) == len(catalog["models"])
    for entry in catalog["models"]:
        thinking = entry["thinking"]
        assert set((
            "supported", "levels", "default", "omitted_behavior",
            "translation", "adaptive_thinking", "omit_parameters",
        )).issubset(thinking)
        levels = thinking["levels"]
        assert thinking["supported"] is bool(levels)
        assert len(levels) == len(set(levels))
        assert thinking["default"] is None or thinking["default"] in levels
        if entry.get("alias_for"):
            assert entry["alias_for"] in ids
        for level in levels:
            assert validate_thinking_level(entry["id"], level.upper()) == level
            assert translate_thinking_level(entry["id"], level)
        assert validate_thinking_level(entry["id"], None) is None
        assert translate_thinking_level(entry["id"], None) == {}


def test_first_party_verified_provider_capabilities_and_aliases():
    assert list_thinking_levels("gpt-5.6") == ["none", "low", "medium", "high", "xhigh", "max"]
    assert default_thinking_level("gpt-5.6") == "medium"
    assert list_thinking_levels("gpt-5.5") == ["none", "low", "medium", "high", "xhigh"]
    assert list_thinking_levels("claude-opus-5") == ["low", "medium", "high", "xhigh", "max"]
    assert default_thinking_level("claude-opus-5") == "high"
    assert list_thinking_levels("grok-4.6") == ["low", "medium", "high", "xhigh"]
    assert default_thinking_level("grok-4.6") == "high"
    assert list_thinking_levels("grok-4.5") == ["low", "medium", "high"]
    assert default_thinking_level("grok-4.5") == "high"
    assert list_thinking_levels("gemini-3.7-flash") == ["low", "medium", "high"]
    assert default_thinking_level("gemini-3.7-flash") == "high"
    assert list_thinking_levels("gemini-3.6-flash") == ["minimal", "low", "medium", "high"]
    assert default_thinking_level("gemini-3.6-flash") == "medium"
    assert default_thinking_level("gemini-2.5-pro") is None
    assert resolve_model_alias("grok-latest") == "grok-4.3"
    assert list_thinking_levels("grok-4.5-latest") == list_thinking_levels("grok-4.5")
    assert resolve_model_alias("gemini-latest") == "gemini-3.7-flash"


def test_provider_translations_are_sdk_safe():
    assert translate_thinking_level("gpt-5.6", "max") == {"reasoning_effort": "max"}
    assert translate_thinking_level("grok-4.6", "xhigh") == {"reasoning_effort": "xhigh"}
    assert translate_thinking_level("grok-4.5", "medium") == {"reasoning_effort": "medium"}
    assert translate_thinking_level("claude-opus-5", "xhigh") == {
        "output_config": {"effort": "xhigh"},
        "thinking": {"type": "adaptive"},
    }
    assert translate_thinking_level("claude-opus-4-5", "low") == {
        "output_config": {"effort": "low"},
    }
    assert translate_thinking_level("gemini-3.7-flash", "high") == {
        "thinking_config": {"thinking_level": "HIGH"},
    }
    assert translate_thinking_level("gemini-3.6-flash", "minimal") == {
        "thinking_config": {"thinking_level": "MINIMAL"},
    }
    assert list_thinking_levels("gemini-2.5-pro") == []
    with pytest.raises(ModelCapabilityError, match="allowed values: none"):
        translate_thinking_level("gemini-2.5-pro", "low")


def test_invalid_combinations_fail_and_omission_remains_backward_compatible():
    assert list_thinking_levels("gpt-4o") == []
    with pytest.raises(ModelCapabilityError, match="allowed values: none"):
        validate_thinking_level("gpt-4o", "high")
    with pytest.raises(ModelCapabilityError, match="Unknown model"):
        validate_thinking_level("fine-tuned-private", "high")
    assert validate_thinking_level("fine-tuned-private", None) is None
    assert translate_thinking_level("fine-tuned-private", None) == {}


def test_serialization_and_enrichment_expose_stable_api_ui_shapes():
    payload = serialize_model_catalog(frontend=True)
    expected = sum(1 for entry in load_model_catalog()["models"] if entry["frontend"])
    assert payload["schema_version"] == 1
    assert len(payload["models"]) == expected + 1
    default_row = payload["models"][0]
    assert default_row["model"] == "default"
    default_capability = get_model_capability(default_row["resolved_model"])
    assert isinstance(default_row["token_limit"], int)
    assert default_row["token_limit"] == default_capability["context_window"] == 1_050_000
    row = next(row for row in payload["models"] if row["model"] == "gpt-5.6")
    assert row["thinking_levels"] == ["none", "low", "medium", "high", "xhigh", "max"]
    enriched = enrich_model_metadata([{"model": "gpt-5.6", "label": "GPT"}])
    assert enriched == [{
        "model": "gpt-5.6",
        "label": "GPT",
        "thinking": {
            "levels": ["none", "low", "medium", "high", "xhigh", "max"],
            "default": "medium",
            "supported": True,
        },
        "resolved_model": "gpt-5.6",
    }]


def test_required_helpers_are_public_package_exports():
    for name in (
        "get_allowed_thinking_levels", "get_default_thinking_level",
        "validate_thinking_level", "enrich_model_metadata",
        "get_model_thinking_catalog", "translate_thinking_level",
    ):
        assert callable(getattr(vectorvault, name))
    assert get_allowed_thinking_levels("gpt-5.6") == list_thinking_levels("gpt-5.6")
    assert get_default_thinking_level("gpt-5.6") == "medium"
