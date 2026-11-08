"""November 1, 2026 catalog: GPT-6.1 Terra, OpenAI Dec 11 shutdowns, retired Claude IDs."""
import pytest

from vectorvault.ai import (
    LATEST_MODELS_MAP,
    MODEL_METADATA,
    OPENAI_IMG_CAPABLE,
    OPENAI_NO_TEMPERATURE_LIST,
    get_all_models,
    get_front_models,
)
from vectorvault.model_catalog import (
    ModelCapabilityError,
    default_thinking_level,
    get_model_capability,
    list_thinking_levels,
    load_model_catalog,
    resolve_model_alias,
    serialize_model_catalog,
    translate_thinking_level,
    validate_thinking_level,
)
from test_provider_thinking import _anthropic_platform, _openai_platform

TERRA = 'gpt-6.1-terra'


def test_gpt_6_1_terra_is_a_current_stable_option_with_documented_limits():
    assert get_front_models()[TERRA] == 1_050_000
    capability = get_model_capability(TERRA)
    metadata = capability['metadata']
    assert capability['provider'] == 'openai'
    assert capability['resolved_model'] == TERRA
    assert metadata['max_input_tokens'] == 922_000
    assert metadata['max_output_tokens'] == 128_000
    assert metadata['modalities'] == {'input': ['text', 'image'], 'output': ['text']}
    assert metadata['release_stage'] == 'stable'
    assert metadata['released'] == '2026-10-26'
    assert set(metadata['endpoints']) == {'chat_completions', 'responses'}
    assert 'Responses' in metadata['chat_completions_tool_note']
    assert TERRA in OPENAI_IMG_CAPABLE
    assert TERRA in OPENAI_NO_TEMPERATURE_LIST


def test_gpt_6_1_terra_thinking_contract_matches_active_route():
    assert list_thinking_levels(TERRA) == ['none', 'low', 'medium', 'high', 'xhigh']
    assert default_thinking_level(TERRA) == 'medium'
    assert validate_thinking_level(TERRA, None) is None
    assert translate_thinking_level(TERRA, None) == {}
    for level in list_thinking_levels(TERRA):
        assert translate_thinking_level(TERRA, level) == {'reasoning_effort': level}
    for rejected in ('minimal', 'max', 'bogus'):
        with pytest.raises(ModelCapabilityError):
            validate_thinking_level(TERRA, rejected)


def test_gpt_6_1_terra_payloads_omit_sampling_and_keep_omission():
    platform, recorder = _openai_platform()
    platform.no_temperature_list = OPENAI_NO_TEMPERATURE_LIST
    assert platform.make_call([], TERRA, temperature=.7, thinking_level='none') == 'complete'
    assert list(platform.stream_call([], TERRA, temperature=.7, thinking_level='xhigh')) == ['streamed']
    assert platform.make_call([], TERRA, temperature=.7) == 'complete'
    assert recorder.calls == [
        {'model': TERRA, 'messages': [], 'reasoning_effort': 'none'},
        {'model': TERRA, 'messages': [], 'stream': True, 'reasoning_effort': 'xhigh'},
        {'model': TERRA, 'messages': []},
    ]
    with pytest.raises(ModelCapabilityError):
        platform.make_call([], TERRA, thinking_level='max')
    assert len(recorder.calls) == 3


DEC_11_SHUTDOWNS = {
    'o3': 'gpt-5.6-sol',
    'gpt-5-mini': 'gpt-5.6-terra',
    'gpt-5-nano': 'gpt-5.6-luna',
    'gpt-5': 'gpt-5.6-sol',
    'o3-pro': 'gpt-5.6-sol',
}


@pytest.mark.parametrize('model,replacement', sorted(DEC_11_SHUTDOWNS.items()))
def test_dec_11_shutdown_ids_leave_selectors_but_still_call_themselves(model, replacement):
    assert model not in get_front_models()
    assert model in get_all_models()
    # Still callable upstream until 2026-12-11, so not aliased yet.
    assert model not in LATEST_MODELS_MAP
    assert resolve_model_alias(model) == model
    metadata = MODEL_METADATA[model]
    assert metadata['release_stage'] == 'deprecated'
    assert metadata['retirement'] == '2026-12-11'
    assert metadata['replacement'] == replacement
    assert validate_thinking_level(model, None) is None


@pytest.mark.parametrize('model,levels', [
    ('o3', ['low', 'medium', 'high', 'xhigh']),
    ('gpt-5-mini', ['minimal', 'low', 'medium', 'high']),
    ('gpt-5-nano', ['minimal', 'low', 'medium', 'high']),
])
def test_saved_flows_on_hidden_dec_11_models_keep_exact_levels(model, levels):
    assert list_thinking_levels(model) == levels
    platform, recorder = _openai_platform()
    for level in levels:
        assert platform.make_call([], model, thinking_level=level) == 'complete'
        assert recorder.calls[-1] == {'model': model, 'messages': [], 'reasoning_effort': level}


RETIRED_CLAUDE = {
    'claude-opus-4-1': ('claude-opus-4-8', '2026-08-05'),
    'claude-sonnet-4-0': ('claude-sonnet-5-5', '2026-06-15'),
    'claude-sonnet-4-20250514': ('claude-sonnet-5-5', '2026-06-15'),
    'claude-3-7-sonnet-latest': ('claude-sonnet-5-5', '2026-02-19'),
    'claude-3-5-sonnet-latest': ('claude-sonnet-5-5', '2025-10-28'),
    'claude-3-5-haiku-latest': ('claude-haiku-4-5', '2026-02-19'),
}


@pytest.mark.parametrize('model,expected', sorted(RETIRED_CLAUDE.items()))
def test_retired_claude_ids_route_locally_to_documented_replacements(model, expected):
    target, retired = expected
    assert model not in get_front_models()
    assert model in get_all_models()
    assert LATEST_MODELS_MAP[model] == target
    assert resolve_model_alias(model) == target
    metadata = MODEL_METADATA[model]
    assert metadata['release_stage'] == 'retired'
    assert metadata['retirement'] == retired
    assert metadata['local_substitute'] == target
    assert list_thinking_levels(model) == list_thinking_levels(target)
    assert validate_thinking_level(model, None) is None
    assert translate_thinking_level(model, None) == {}


def test_retired_claude_ids_never_reach_the_provider():
    platform, recorder = _anthropic_platform()
    from vectorvault.ai import ANTHROPIC_NO_TEMPERATURE_LIST
    platform.no_temperature_list = ANTHROPIC_NO_TEMPERATURE_LIST
    for model, (target, _) in RETIRED_CLAUDE.items():
        assert platform.make_call([], model, temperature=.4) == 'complete'
        assert recorder.calls[-1]['model'] == target
    # Sonnet 5.5 rejects sampling, Haiku 4.5 keeps it.
    sent = {call['model']: call for call in recorder.calls}
    assert 'temperature' not in sent['claude-sonnet-5-5']
    assert 'temperature' not in sent['claude-opus-4-8']
    assert sent['claude-haiku-4-5']['temperature'] == .4


def test_selector_payload_and_compatibility_rows():
    current = {row['model'] for row in serialize_model_catalog(frontend=True)['models']}
    assert TERRA in current
    assert current.isdisjoint(DEC_11_SHUTDOWNS)
    assert current.isdisjoint(RETIRED_CLAUDE)
    rows = {row['model']: row for row in serialize_model_catalog(frontend=False)['models']}
    for model, (target, _) in RETIRED_CLAUDE.items():
        assert rows[model]['resolved_model'] == target


def test_argon_computer_use_is_ga():
    metadata = get_model_capability('gemini-4-argon')['metadata']
    assert 'computer_use' in metadata['tools']
    assert 'preview_tools' not in metadata


def test_defaults_unchanged_and_catalog_version():
    catalog = load_model_catalog()
    assert catalog['defaults'] == {
        'openai': 'gpt-6-astra', 'anthropic': 'claude-opus-5-5',
        'grok': 'grok-4.7', 'gemini': 'gemini-3.8-flash',
    }
    assert catalog['catalog_version'] >= '2026-11-01'
