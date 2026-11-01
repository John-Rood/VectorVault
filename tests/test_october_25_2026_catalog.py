"""October 25, 2026 catalog: OpenAI shutdowns and Anthropic SDK 2 sampling."""
import pytest

from vectorvault.ai import (
    ANTHROPIC_MODELS,
    MODEL_METADATA,
    ANTHROPIC_SAMPLING_API_VERSION,
    LATEST_MODELS_MAP,
    get_all_models,
    get_front_models,
)
from vectorvault.model_catalog import (
    ModelCapabilityError,
    get_model_capability,
    list_thinking_levels,
    load_model_catalog,
    resolve_model_alias,
    serialize_model_catalog,
    translate_thinking_level,
    validate_thinking_level,
)
from test_provider_thinking import _anthropic_platform, _openai_platform

RETIRED_OPENAI = {
    'o4-mini': ('gpt-5.6-terra', '2026-10-23'),
    'o1': ('gpt-5.6-sol', '2026-10-23'),
    'o3-mini': ('gpt-5.6-sol', '2026-10-23'),
    'gpt-4': ('gpt-5.6-sol', '2026-10-23'),
    'gpt-3.5-turbo': ('gpt-5.6-terra', '2026-10-23'),
    'o1-mini': ('gpt-5.6-terra', '2025-10-27'),
    'chatgpt-4o-latest': ('gpt-5.6-sol', '2026-02-17'),
    'gpt-5-chat-latest': ('gpt-5.6-sol', '2026-07-23'),
    'gpt-5.1-chat-latest': ('gpt-5.6-sol', '2026-07-23'),
    'gpt-5.2-chat-latest': ('gpt-5.6-sol', '2026-08-10'),
    'gpt-5.3-chat-latest': ('gpt-5.6-sol', '2026-08-10'),
}


@pytest.mark.parametrize('model,expected', sorted(RETIRED_OPENAI.items()))
def test_retired_openai_ids_stay_loadable_but_route_to_substitutes(model, expected):
    target, shutdown = expected
    assert model not in get_front_models()
    assert model in get_all_models()
    assert LATEST_MODELS_MAP[model] == target
    assert resolve_model_alias(model) == target
    # Alias rows keep their own retirement record; capability lookups resolve to the target.
    metadata = MODEL_METADATA[model]
    assert metadata['release_stage'] == 'retired'
    assert metadata['shutdown'] == shutdown
    assert metadata['local_substitute'] == target
    # Omitted thinking stays a true no-op for saved flows.
    assert validate_thinking_level(model, None) is None
    assert translate_thinking_level(model, None) == {}
    assert list_thinking_levels(model) == list_thinking_levels(target)


def test_retired_ids_never_appear_in_current_selector_payload():
    rows = {row['model'] for row in serialize_model_catalog(frontend=True)['models']}
    assert rows.isdisjoint(RETIRED_OPENAI)
    assert 'gpt-4o-audio-preview' not in rows
    all_rows = {row['model']: row for row in serialize_model_catalog(frontend=False)['models']}
    for model, (target, _) in RETIRED_OPENAI.items():
        assert all_rows[model]['resolved_model'] == target


@pytest.mark.parametrize('level', ['low', 'medium', 'high', 'xhigh'])
def test_saved_o4_mini_flow_keeps_its_reasoning_level_on_substitute(level):
    platform, recorder = _openai_platform()
    assert platform.make_call([], 'o4-mini', thinking_level=level) == 'complete'
    assert list(platform.stream_call([], 'o4-mini', thinking_level=level)) == ['streamed']
    assert recorder.calls == [
        {'model': 'gpt-5.6-terra', 'messages': [], 'reasoning_effort': level},
        {'model': 'gpt-5.6-terra', 'messages': [], 'stream': True, 'reasoning_effort': level},
    ]


@pytest.mark.parametrize('model', ['o1', 'o1-mini'])
def test_formerly_non_streaming_ids_now_stream_through_substitute(model):
    platform, recorder = _openai_platform()
    assert list(platform.stream_call([], model)) == ['streamed']
    assert recorder.calls[0]['model'] == RETIRED_OPENAI[model][0]
    assert recorder.calls[0]['stream'] is True


def test_audio_preview_is_not_aliased_to_audio_only_successor():
    assert 'gpt-4o-audio-preview' not in LATEST_MODELS_MAP
    assert resolve_model_alias('gpt-4o-audio-preview') == 'gpt-4o-audio-preview'
    metadata = get_model_capability('gpt-4o-audio-preview')['metadata']
    assert metadata['release_stage'] == 'retired'
    assert metadata['replacement'] == 'gpt-audio-1.5'


def test_o4_mini_rejects_levels_its_substitute_rejects():
    with pytest.raises(ModelCapabilityError):
        validate_thinking_level('o4-mini', 'max')


def test_claude_sonnet_4_5_context_matches_models_api():
    assert ANTHROPIC_MODELS['claude-sonnet-4-5'] == 200_000
    capability = get_model_capability('claude-sonnet-4-5')
    assert capability['context_window'] == 200_000
    assert capability['metadata']['context_window'] == 200_000
    assert capability['metadata']['max_output_tokens'] == 64_000


def test_defaults_and_catalog_version():
    catalog = load_model_catalog()
    assert catalog['defaults'] == {
        'openai': 'gpt-6-astra', 'anthropic': 'claude-opus-5-5',
        'grok': 'grok-4.7', 'gemini': 'gemini-3.8-flash',
    }
    assert catalog['catalog_version'] >= '2026-10-25'


def _sdk2_create(recorder):
    original_create = recorder.create

    # SDK 2.x: no temperature keyword, documented extra_body/extra_headers.
    def create(*, model, messages, max_tokens, stream=False, output_config=None,
               thinking=None, extra_body=None, extra_headers=None):
        params = {'model': model, 'messages': messages, 'max_tokens': max_tokens}
        if stream:
            params['stream'] = True
        if output_config is not None:
            params['output_config'] = output_config
        if thinking is not None:
            params['thinking'] = thinking
        if extra_body is not None:
            params['extra_body'] = extra_body
        if extra_headers is not None:
            params['extra_headers'] = extra_headers
        return original_create(**params)
    return create


@pytest.mark.parametrize('model,level', [
    ('claude-opus-4-5', None), ('claude-opus-4-5', 'high'),
    ('claude-sonnet-4-6', None), ('claude-haiku-4-5', None),
])
def test_anthropic_sdk2_sampling_requests_pin_compatible_api_version(model, level):
    platform, recorder = _anthropic_platform()
    platform.client.messages.create = _sdk2_create(recorder)
    assert platform.make_call([], model, temperature=.3, thinking_level=level) == 'complete'
    assert list(platform.stream_call([], model, temperature=.3, thinking_level=level)) == ['streamed']
    assert len(recorder.calls) == 2
    for call in recorder.calls:
        assert 'temperature' not in call
        assert call['extra_body'] == {'temperature': .3}
        assert call['extra_headers'] == {'anthropic-version': ANTHROPIC_SAMPLING_API_VERSION}
    assert ANTHROPIC_SAMPLING_API_VERSION == '2023-06-01'


def test_anthropic_sdk2_explicit_effort_on_sonnet_4_6_omits_sampling_and_header():
    # Sonnet 4.6 omits temperature once an explicit effort is selected, so the
    # request stays on the SDK's default API version.
    platform, recorder = _anthropic_platform()
    platform.client.messages.create = _sdk2_create(recorder)
    assert platform.make_call([], 'claude-sonnet-4-6', temperature=.3, thinking_level='low') == 'complete'
    call = recorder.calls[0]
    assert call['output_config'] == {'effort': 'low'}
    assert 'extra_body' not in call and 'extra_headers' not in call


@pytest.mark.parametrize('model', ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'])
def test_anthropic_sdk2_no_sampling_models_use_sdk_default_api_version(model):
    from vectorvault.ai import ANTHROPIC_NO_TEMPERATURE_LIST
    platform, recorder = _anthropic_platform()
    platform.no_temperature_list = ANTHROPIC_NO_TEMPERATURE_LIST
    platform.client.messages.create = _sdk2_create(recorder)
    assert platform.make_call([], model, temperature=.3, thinking_level='low') == 'complete'
    assert list(platform.stream_call([], model, temperature=.3)) == ['streamed']
    for call in recorder.calls:
        assert 'extra_body' not in call
        assert 'extra_headers' not in call


def test_installed_anthropic_sdk_bridge_is_valid_for_real_client():
    """Exercise the bridge against the actually installed SDK signature."""
    import inspect

    import anthropic
    from vectorvault.ai import AnthropicPlatform

    platform = object.__new__(AnthropicPlatform)
    platform.client = anthropic.Anthropic(api_key='test-key-not-used')
    params = platform._message_create_params({'model': 'claude-sonnet-4-6', 'temperature': 0})
    signature = inspect.signature(platform.client.messages.create).parameters
    if 'temperature' in signature:
        assert params == {'model': 'claude-sonnet-4-6', 'temperature': 0}
    else:
        assert params['extra_body'] == {'temperature': 0}
        assert params['extra_headers']['anthropic-version'] == ANTHROPIC_SAMPLING_API_VERSION
        inspect.signature(platform.client.messages.create).bind(
            max_tokens=1, messages=[], **params)
