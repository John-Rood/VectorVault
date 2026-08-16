from vectorvault.model_catalog import (
    MODEL_METADATA,
    default_thinking_level,
    list_thinking_levels,
    load_model_catalog,
    model_maps,
    resolve_model_alias,
    translate_thinking_level,
)


def test_august_2026_models_are_complete_and_frontend_visible():
    catalog = load_model_catalog()
    rows = {row["id"]: row for row in catalog["models"]}
    assert catalog["defaults"]["grok"] == "grok-4.6"
    assert catalog["defaults"]["gemini"] == "gemini-3.7-flash"
    assert rows["grok-4.6"]["frontend"] is True
    assert rows["grok-4.6"]["context_window"] == 500_000
    assert rows["gemini-3.7-flash"]["frontend"] is True
    assert rows["gemini-3.7-flash"]["context_window"] == 1_048_576
    assert model_maps(True)["grok"]["grok-4.6"] == 500_000
    assert model_maps(True)["gemini"]["gemini-3.7-flash"] == 1_048_576


def test_august_2026_thinking_and_alias_contracts():
    assert list_thinking_levels("grok-4.6") == ["low", "medium", "high", "xhigh"]
    assert default_thinking_level("grok-4.6") == "high"
    assert translate_thinking_level("grok-4.6", "xhigh") == {"reasoning_effort": "xhigh"}
    assert resolve_model_alias("grok-4.6-latest") == "grok-4.6"
    assert list_thinking_levels("gemini-3.7-flash") == ["low", "medium", "high"]
    assert default_thinking_level("gemini-3.7-flash") == "high"
    assert translate_thinking_level("gemini-3.7-flash", "low") == {
        "thinking_config": {"thinking_level": "LOW"}
    }
    assert resolve_model_alias("gemini-latest") == "gemini-3.7-flash"


def test_august_2026_official_metadata():
    grok = MODEL_METADATA["grok-4.6"]
    assert grok["context_window"] == 500_000
    assert grok["max_output_tokens"] is None
    assert grok["input_modalities"] == ["text", "image"]
    assert "function_calling" in grok["tools"]
    gemini = MODEL_METADATA["gemini-3.7-flash"]
    assert gemini["context_window"] == 1_048_576
    assert gemini["max_output_tokens"] == 65_536
    assert "video" in gemini["input_modalities"]
    assert "computer_use" in gemini["tools"]
