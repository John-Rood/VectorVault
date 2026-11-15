"""Public November models preserve package-owned effort and saved-flow omission."""
import pytest
from vectorvault.ai import ANTHROPIC_NO_TEMPERATURE_LIST, get_front_models
from vectorvault.model_catalog import (
    ModelCapabilityError, get_model_capability, list_thinking_levels,
    default_thinking_level, load_model_catalog, translate_thinking_level,
)
from test_provider_thinking import _anthropic_platform, _gemini_platform, _dump_gemini_call

MODELS = {
    'claude-fable-5-2': (['low', 'medium', 'high', 'xhigh', 'max'], 'high'),
    'gemini-4-carbon': (['low', 'medium', 'high'], 'medium'),
    'gemini-4-hydrogen': (['low', 'medium', 'high', 'xhigh'], 'high'),
}

@pytest.mark.parametrize('model', MODELS)
def test_new_public_models_limits_defaults_and_omission(model):
    levels, default = MODELS[model]
    assert model in get_front_models()
    assert load_model_catalog()['catalog_date'] == load_model_catalog()['catalog_version'] == '2026-11-15'
    capability = get_model_capability(model)
    assert capability['context_window'] == (1_000_000 if model.startswith('claude') else 1_048_576)
    assert capability['metadata']['max_output_tokens'] == (128_000 if model.startswith('claude') else 1_048_576)
    assert list_thinking_levels(model) == levels
    assert default_thinking_level(model) == default
    assert translate_thinking_level(model, None) == {}
    assert load_model_catalog()['defaults'] == {
        'openai': 'gpt-6-astra', 'anthropic': 'claude-opus-5-5',
        'grok': 'grok-4.7', 'gemini': 'gemini-3.8-flash',
    }

@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('level', [None, 'low', 'medium', 'high', 'xhigh', 'max'])
def test_fable_52_provider_translation_and_sampling_omission(stream, level):
    platform, recorder = _anthropic_platform()
    platform.no_temperature_list = ANTHROPIC_NO_TEMPERATURE_LIST
    result = (list(platform.stream_call([], 'claude-fable-5-2', temperature=.7, thinking_level=level))
              if stream else platform.make_call([], 'claude-fable-5-2', temperature=.7, thinking_level=level))
    assert result == (['streamed'] if stream else 'complete')
    expected = {'model': 'claude-fable-5-2', 'messages': [], 'max_tokens': 8192}
    if stream:
        expected['stream'] = True
    if level:
        expected.update(output_config={'effort': level}, thinking={'type': 'adaptive'})
    assert recorder.calls == [expected]
    metadata = get_model_capability('claude-fable-5-2')['metadata']
    assert 'zero data retention' in metadata['zero_data_retention_note']
    assert 'account bound' in metadata['preserved_thinking_note']

@pytest.mark.parametrize('model', ['gemini-4-carbon', 'gemini-4-hydrogen'])
@pytest.mark.parametrize('stream', [False, True])
def test_gemini_4_exact_provider_controls_and_ignored_parameter_omission(model, stream):
    for level in [None, *MODELS[model][0]]:
        platform, recorder = _gemini_platform()
        kwargs = dict(temperature=.8, top_p=.7, top_k=20, candidate_count=2,
                      thinking_budget=128, cached_content='unsupported',
                      max_output_tokens=262_144, thinking_level=level)
        result = (list(platform.stream_call([], model, **kwargs)) if stream
                  else platform.make_call([], model, **kwargs))
        assert result == (['streamed'] if stream else 'complete')
        config = {'max_output_tokens': 262_144}
        if level:
            config['thinking_config'] = {'thinking_level': level.upper()}
        assert _dump_gemini_call(recorder.calls[0]) == {
            'model': model, 'contents': [], 'config': config,
        }
    metadata = get_model_capability(model)['metadata']
    assert metadata['context_accounting'] == 'unified_input_thinking_output'
    assert metadata['per_request_output_tokens']['rest_nonstream'] == (65_536 if 'hydrogen' in model else 32_768)
    assert metadata['caching'] == 'implicit_only'

@pytest.mark.parametrize('model,invalid', [
    ('claude-fable-5-2', 'none'), ('gemini-4-carbon', 'xhigh'),
    ('gemini-4-carbon', 'minimal'), ('gemini-4-hydrogen', 'minimal'),
    ('gemini-4-hydrogen', 'max'),
])
def test_invalid_efforts_fail_before_sdk(model, invalid):
    platform, recorder = (_anthropic_platform() if model.startswith('claude') else _gemini_platform())
    with pytest.raises(ModelCapabilityError):
        platform.make_call([], model, thinking_level=invalid)
    with pytest.raises(ModelCapabilityError):
        platform.stream_call([], model, thinking_level=invalid)
    assert not recorder.calls
