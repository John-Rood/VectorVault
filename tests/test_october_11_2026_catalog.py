import pytest

from vectorvault.ai import (
    ANTHROPIC_NO_TEMPERATURE_LIST, get_all_models, get_front_models,
)
from vectorvault.model_catalog import (
    ModelCapabilityError, default_thinking_level, get_model_capability,
    list_thinking_levels, load_model_catalog, provider_for_model,
    resolve_model_alias, translate_thinking_level, validate_thinking_level,
)
from test_provider_thinking import _anthropic_platform


def test_claude_haiku_55_is_a_stable_current_adaptive_effort_model():
    capability = get_model_capability('claude-haiku-5-5')
    metadata = capability['metadata']
    assert provider_for_model('claude-haiku-5-5') == 'anthropic'
    assert get_front_models()['claude-haiku-5-5'] == 1_000_000
    assert resolve_model_alias('claude-haiku-5-5') == 'claude-haiku-5-5'
    assert metadata['max_output_tokens'] == 128_000
    assert metadata['max_input_tokens'] == 1_000_000
    assert metadata['release_stage'] == 'stable'
    assert metadata['released'] == '2026-10-07'
    assert metadata['input_modalities'] == ['text', 'image']
    assert {'browser_use', 'computer_use', 'web_search', 'web_fetch', 'code_execution'} <= set(metadata['tools'])
    assert list_thinking_levels('claude-haiku-5-5') == ['low', 'medium', 'high', 'xhigh', 'max']
    # Anthropic documents medium as the Haiku 5.5 default (high on Sonnet 5.5).
    assert default_thinking_level('claude-haiku-5-5') == 'medium'
    assert default_thinking_level('claude-sonnet-5-5') == 'high'
    assert 'claude-haiku-5-5' in ANTHROPIC_NO_TEMPERATURE_LIST
    assert metadata['unsupported_parameters'] == ['temperature', 'top_p', 'top_k']


def test_claude_haiku_55_translation_and_omission_compatibility():
    assert translate_thinking_level('claude-haiku-5-5', None) == {}
    for level in list_thinking_levels('claude-haiku-5-5'):
        assert translate_thinking_level('claude-haiku-5-5', level) == {
            'output_config': {'effort': level}, 'thinking': {'type': 'adaptive'},
        }
    for level in ('none', 'minimal', 'between_tools', 'disabled'):
        with pytest.raises(ModelCapabilityError):
            validate_thinking_level('claude-haiku-5-5', level)
    platform, recorder = _anthropic_platform()
    platform.no_temperature_list = ANTHROPIC_NO_TEMPERATURE_LIST
    assert platform.make_call([], 'claude-haiku-5-5', temperature=0.8, thinking_level='xhigh') == 'complete'
    assert list(platform.stream_call([], 'claude-haiku-5-5', temperature=0.8, thinking_level='low')) == ['streamed']
    assert platform.make_call([], 'claude-haiku-5-5', temperature=0.8) == 'complete'
    assert recorder.calls == [
        {'model': 'claude-haiku-5-5', 'messages': [], 'max_tokens': 8192,
         'output_config': {'effort': 'xhigh'}, 'thinking': {'type': 'adaptive'}},
        {'model': 'claude-haiku-5-5', 'messages': [], 'max_tokens': 8192, 'stream': True,
         'output_config': {'effort': 'low'}, 'thinking': {'type': 'adaptive'}},
        {'model': 'claude-haiku-5-5', 'messages': [], 'max_tokens': 8192},
    ]
    with pytest.raises(ModelCapabilityError):
        platform.make_call([], 'claude-haiku-5-5', thinking_level='none')
    assert len(recorder.calls) == 3


def test_current_anthropic_computer_use_toolset_id_is_the_wire_verified_one():
    for model in ('claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5'):
        assert get_model_capability(model)['metadata']['computer_use_tool_version'] == 'computer_toolset_20260801'


def test_grok_43_supports_xhigh_with_low_provider_default():
    assert list_thinking_levels('grok-4.3') == ['none', 'low', 'medium', 'high', 'xhigh']
    assert default_thinking_level('grok-4.3') == 'low'
    assert translate_thinking_level('grok-4.3', 'xhigh') == {'reasoning_effort': 'xhigh'}
    assert translate_thinking_level('grok-4.3', None) == {}
    with pytest.raises(ModelCapabilityError):
        validate_thinking_level('grok-4.3', 'max')
    # Newer Grok defaults and levels are unchanged.
    assert default_thinking_level('grok-4.7') == 'high'
    assert list_thinking_levels('grok-4.5') == ['low', 'medium', 'high']


@pytest.mark.parametrize('model,replacement', [
    ('gemini-3.7-flash', 'gemini-3.8-flash'),
    ('gemini-3.5-flash', 'gemini-3.6-flash'),
])
def test_deprecated_gemini_models_leave_selectors_but_keep_saved_flow_compatibility(model, replacement):
    assert model not in get_front_models()
    assert model in get_all_models()
    metadata = get_model_capability(model)['metadata']
    assert metadata['release_stage'] == 'deprecated'
    assert metadata['deprecated'] == '2026-10-08'
    assert metadata['routed_to'] == replacement
    assert resolve_model_alias(model) == model
    assert translate_thinking_level(model, None) == {}
    assert translate_thinking_level(model, 'low') == {'thinking_config': {'thinking_level': 'LOW'}}


def test_gpt_54_nano_is_deprecated_backend_only_with_unchanged_effort_contract():
    assert 'gpt-5.4-nano' not in get_front_models()
    assert 'gpt-5.4-nano' in get_all_models()
    metadata = get_model_capability('gpt-5.4-nano')['metadata']
    assert metadata['release_stage'] == 'deprecated'
    assert metadata['retirement'] == '2027-04-01'
    assert metadata['replacement'] == 'gpt-6-luna'
    assert list_thinking_levels('gpt-5.4-nano') == ['none', 'low', 'medium', 'high', 'xhigh']


def test_defaults_aliases_and_wire_verified_openai_contracts_are_unchanged():
    assert load_model_catalog()['defaults'] == {
        'openai': 'gpt-6-astra', 'anthropic': 'claude-opus-5-5',
        'grok': 'grok-4.7', 'gemini': 'gemini-3.8-flash',
    }
    assert resolve_model_alias('claude-latest') == 'claude-opus-5-5'
    assert resolve_model_alias('gemini-latest') == 'gemini-3.8-flash'
    assert resolve_model_alias('grok-latest') == 'grok-4.3'
    assert list_thinking_levels('gpt-6-astra') == ['low', 'medium', 'high', 'xhigh']
    assert list_thinking_levels('gpt-6.1-sol') == ['low', 'medium', 'high', 'xhigh']
    assert list_thinking_levels('gemini-3.8-flash') == ['low', 'medium', 'high']
