import pytest

from vectorvault.ai import ANTHROPIC_NO_TEMPERATURE_LIST, OPENAI_NO_TEMPERATURE_LIST, get_front_models
from vectorvault.model_catalog import (
    ModelCapabilityError, default_thinking_level, get_model_capability,
    list_thinking_levels, load_model_catalog, resolve_model_alias,
    translate_thinking_level, validate_thinking_level,
)


def test_october_stable_models_limits_and_current_selector_membership():
    current = get_front_models()
    for model, context, maximum_input in (
        ('gpt-6.1-sol', 1_050_000, 922_000),
        ('claude-sonnet-5-5', 1_000_000, 1_000_000),
    ):
        capability = get_model_capability(model)
        assert current[model] == context
        assert capability['metadata']['release_stage'] == 'stable'
        assert capability['metadata']['max_input_tokens'] == maximum_input
        assert capability['metadata']['max_output_tokens'] == 128_000
        assert resolve_model_alias(model) == model
        assert validate_thinking_level(model, None) is None
        assert translate_thinking_level(model, None) == {}
    assert 'gpt-6.1-sol' in OPENAI_NO_TEMPERATURE_LIST
    assert 'claude-sonnet-5-5' in ANTHROPIC_NO_TEMPERATURE_LIST


def test_gpt_61_sol_wire_verified_chat_completions_efforts():
    assert list_thinking_levels('gpt-6.1-sol') == ['low', 'medium', 'high', 'xhigh']
    assert default_thinking_level('gpt-6.1-sol') == 'medium'
    for level in list_thinking_levels('gpt-6.1-sol'):
        assert translate_thinking_level('gpt-6.1-sol', level) == {'reasoning_effort': level}
    for level in ('none', 'minimal', 'max'):
        with pytest.raises(ModelCapabilityError):
            validate_thinking_level('gpt-6.1-sol', level)
    assert 'Responses' in get_model_capability('gpt-6.1-sol')['metadata']['chat_completions_tool_note']


def test_sonnet_55_adaptive_efforts_and_separate_between_tools_mode():
    assert list_thinking_levels('claude-sonnet-5-5') == ['low', 'medium', 'high', 'xhigh', 'max']
    assert default_thinking_level('claude-sonnet-5-5') == 'high'
    for level in list_thinking_levels('claude-sonnet-5-5'):
        assert translate_thinking_level('claude-sonnet-5-5', level) == {
            'output_config': {'effort': level}, 'thinking': {'type': 'adaptive'},
        }
    for level in ('none', 'between_tools'):
        with pytest.raises(ModelCapabilityError):
            validate_thinking_level('claude-sonnet-5-5', level)


def test_october_release_preserves_prior_defaults_and_saved_model_aliases():
    assert load_model_catalog()['defaults'] == {
        'openai': 'gpt-6-astra', 'anthropic': 'claude-opus-5-5',
        'grok': 'grok-4.7', 'gemini': 'gemini-3.8-flash',
    }
    assert resolve_model_alias('claude-latest') == 'claude-opus-5-5'
    assert resolve_model_alias('grok-latest') == 'grok-4.3'
    assert list_thinking_levels('gpt-6-sol')[0] == 'none'
    assert resolve_model_alias('gemini-2.0-flash') == 'gemini-3.6-flash'


def test_newly_deprecated_sonnet_45_remains_backend_only_until_retirement():
    from vectorvault.ai import get_all_models
    assert 'claude-sonnet-4-5' not in get_front_models()
    assert 'claude-sonnet-4-5' in get_all_models()
    metadata = get_model_capability('claude-sonnet-4-5')['metadata']
    assert metadata['release_stage'] == 'deprecated'
    assert metadata['deprecated'] == '2026-09-30'
    assert metadata['retirement'] == '2026-11-30'
    assert translate_thinking_level('claude-sonnet-4-5', None) == {}
