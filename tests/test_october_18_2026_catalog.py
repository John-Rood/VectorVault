import pytest

from vectorvault.ai import GEMINI_MULTIMODAL_MODELS, GEMINI_THINKING_MODELS, get_all_models, get_front_models
from vectorvault.model_catalog import (
    ModelCapabilityError, default_thinking_level, get_model_capability,
    list_thinking_levels, load_model_catalog, resolve_model_alias,
    translate_thinking_level, validate_thinking_level,
)
from test_provider_thinking import _dump_gemini_call, _gemini_platform


def test_argon_stable_unified_context_long_decode_metadata():
    model = 'gemini-4-argon'
    capability = get_model_capability(model)
    metadata = capability['metadata']
    assert get_front_models()[model] == 1_048_576
    assert model in GEMINI_MULTIMODAL_MODELS
    assert model in GEMINI_THINKING_MODELS
    assert metadata['released'] == '2026-10-13'
    assert metadata['release_stage'] == 'ga'
    assert metadata['context_accounting'] == 'unified_input_thinking_output'
    assert metadata['max_output_tokens'] == 1_048_576
    assert metadata['default_max_output_tokens'] == 65_536
    assert metadata['per_request_output_tokens'] == {'rest_nonstream': 32_768, 'stream': 65_536}
    assert metadata['caching'] == 'implicit_only'
    # Computer Use went GA on 2026-10-30 (see test_november_01_2026_catalog).
    assert 'computer_use' in metadata['tools']
    assert 'preview_tools' not in metadata
    assert metadata['modalities']['input'] == ['text', 'image', 'video', 'audio', 'pdf']
    assert metadata['modalities']['output'] == ['text']
    assert list_thinking_levels(model) == ['low', 'medium', 'high']
    assert default_thinking_level(model) == 'medium'


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('level', [None, 'low', 'medium', 'high'])
def test_argon_exact_provider_translation_sampling_and_cache_omission(stream, level):
    platform, recorder = _gemini_platform()
    kwargs = dict(temperature=.8, top_p=.7, top_k=20, candidate_count=2,
                  thinking_budget=1000, cached_content='unsupported-cache',
                  max_output_tokens=262_144, thinking_level=level)
    result = (list(platform.stream_call([], 'gemini-4-argon', **kwargs)) if stream
              else platform.make_call([], 'gemini-4-argon', **kwargs))
    assert result == (['streamed'] if stream else 'complete')
    config = {'max_output_tokens': 262_144}
    if level:
        config['thinking_config'] = {'thinking_level': level.upper()}
    assert _dump_gemini_call(recorder.calls[0]) == {
        'model': 'gemini-4-argon', 'contents': [], 'config': config,
    }
    assert translate_thinking_level('gemini-4-argon', None) == {}


@pytest.mark.parametrize('level', ['none', 'minimal', 'xhigh', 'max', 'disabled'])
def test_argon_rejects_invalid_thinking_before_sdk(level):
    platform, recorder = _gemini_platform()
    with pytest.raises(ModelCapabilityError):
        platform.make_call([], 'gemini-4-argon', thinking_level=level)
    with pytest.raises(ModelCapabilityError):
        platform.stream_call([], 'gemini-4-argon', thinking_level=level)
    assert recorder.calls == []


@pytest.mark.parametrize('model', ['gpt-4o', 'gpt-4o-mini'])
def test_deprecated_openai_models_remain_backend_compatible(model):
    assert model not in get_front_models()
    assert model in get_all_models()
    assert resolve_model_alias(model) == model
    metadata = get_model_capability(model)['metadata']
    assert metadata['deprecated'] == '2026-10-14'
    assert metadata['retirement'] == '2027-01-14'
    assert metadata['replacement'] == 'gpt-6-luna'
    assert validate_thinking_level(model, None) is None


def test_omitted_defaults_and_alias_targets_are_intentionally_preserved():
    assert load_model_catalog()['defaults']['gemini'] == 'gemini-3.8-flash'
    assert resolve_model_alias('gemini-latest') == 'gemini-3.8-flash'
    assert resolve_model_alias('default') == 'gpt-6-astra'
