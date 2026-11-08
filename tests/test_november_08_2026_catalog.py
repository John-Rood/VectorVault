"""Google's November 2 deprecations must not change saved-flow reasoning semantics."""
import pytest
from vectorvault.ai import get_all_models, get_front_models, LATEST_MODELS_MAP
from vectorvault.model_catalog import (
    ModelCapabilityError, default_thinking_level, get_model_capability,
    list_thinking_levels, load_model_catalog, resolve_model_alias,
    serialize_model_catalog, translate_thinking_level, validate_thinking_level,
)
from test_provider_thinking import _gemini_platform, _dump_gemini_call

DEPRECATED = {'gemini-2.5-pro': 'gemini-3.8-flash',
              'gemini-2.5-flash': 'gemini-3.5-flash-lite'}

@pytest.mark.parametrize('model,replacement', DEPRECATED.items())
def test_google_deprecations_are_hidden_without_premature_local_routing(model, replacement):
    assert model not in get_front_models()
    assert model in get_all_models()
    assert model not in LATEST_MODELS_MAP
    assert resolve_model_alias(model) == model
    metadata = get_model_capability(model)['metadata']
    assert metadata['release_stage'] == 'deprecated'
    assert metadata['deprecated'] == '2026-11-02'
    assert metadata['provider_redirect_date'] == '2026-12-01'
    assert metadata['replacement'] == replacement
    assert 'before that date' in metadata['deprecation_note']

@pytest.mark.parametrize('model', DEPRECATED)
def test_legacy_budget_only_models_do_not_acquire_named_thinking_levels(model):
    assert list_thinking_levels(model) == []
    assert default_thinking_level(model) is None
    assert validate_thinking_level(model, None) is None
    assert validate_thinking_level(model, '') is None
    assert translate_thinking_level(model, None) == {}
    for level in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
        with pytest.raises(ModelCapabilityError):
            validate_thinking_level(model, level)

@pytest.mark.parametrize('model', DEPRECATED)
def test_hidden_model_still_emits_exact_legacy_id_and_omits_thinking(model):
    platform, recorder = _gemini_platform()
    assert platform.make_call([], model) == 'complete'
    assert list(platform.stream_call([], model)) == ['streamed']
    for call in recorder.calls:
        sent = _dump_gemini_call(call)
        assert sent['model'] == model
        assert 'thinking_config' not in sent.get('config', {})
    with pytest.raises(ModelCapabilityError):
        platform.make_call([], model, thinking_level='low')
    assert len(recorder.calls) == 2

def test_selector_payload_defaults_and_account_restricted_flash_lite():
    current = {r['model'] for r in serialize_model_catalog(frontend=True)['models']}
    all_rows = {r['model']: r for r in serialize_model_catalog(frontend=False)['models']}
    assert len(current) == 43
    assert current.isdisjoint(DEPRECATED)
    for model, replacement in DEPRECATED.items():
        assert all_rows[model]['metadata']['replacement'] == replacement
        assert all_rows[model]['thinking_levels'] == []
    assert load_model_catalog()['defaults']['gemini'] == 'gemini-3.8-flash'
    assert resolve_model_alias('gemini-latest') == 'gemini-3.8-flash'
    assert 'gemini-2.5-flash-lite' not in current
    # Google explicitly says Flash-Lite is not part of the Nov 2 deprecation.
    assert get_model_capability('gemini-2.5-flash-lite')['metadata'].get('release_stage') != 'deprecated'
    assert load_model_catalog()['catalog_version'] == '2026-11-08'
