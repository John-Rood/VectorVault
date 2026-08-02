import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_ai_module():
    """Import the catalog module without installing provider SDKs in this source-only repo."""
    for name in ('openai', 'tiktoken', 'anthropic', 'httpx'):
        sys.modules.setdefault(name, types.ModuleType(name))
    google = sys.modules.setdefault('google', types.ModuleType('google'))
    genai = sys.modules.setdefault('google.genai', types.ModuleType('google.genai'))
    genai.types = types.SimpleNamespace()
    google.genai = genai
    spec = importlib.util.spec_from_file_location('vectorvault_catalog_under_test', Path(__file__).parents[1] / 'vectorvault' / 'ai.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ai = load_ai_module()


class ModelCatalogTests(unittest.TestCase):
    def test_august_2026_context_limits(self):
        for model in ('gpt-5.6', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'):
            self.assertEqual(ai.OPENAI_MODELS[model], 1_050_000)
            self.assertEqual(ai.MODEL_METADATA[model]['max_input_tokens'], 922_000)
            self.assertEqual(ai.MODEL_METADATA[model]['max_output_tokens'], 128_000)
        for model in ('claude-fable-5', 'claude-opus-5', 'claude-sonnet-5'):
            self.assertEqual(ai.ANTHROPIC_MODELS[model], 1_000_000)
        for model in ('claude-fable-5', 'claude-opus-5'):
            self.assertEqual(ai.MODEL_METADATA[model]['max_output_tokens'], 128_000)
        self.assertEqual(ai.MODEL_METADATA['claude-sonnet-5']['max_output_tokens'], 64_000)
        self.assertEqual(ai.GROK_MODELS['grok-4.5'], 256_000)
        for model in ('grok-4.3', 'grok-4.20', 'grok-4.20-0309-reasoning', 'grok-4.20-0309-non-reasoning'):
            self.assertEqual(ai.GROK_MODELS[model], 1_000_000)
        self.assertEqual(ai.GROK_MODELS['grok-build-0.1'], 256_000)
        self.assertEqual(ai.GROK_MODELS['grok-latest'], 1_000_000)
        self.assertEqual(ai.OPENAI_MODELS['gpt-5.4'], 1_050_000)
        self.assertEqual(ai.OPENAI_MODELS['gpt-5.5'], 1_050_000)
        for model in ('gpt-5.4-mini', 'gpt-5.4-nano', 'chat-latest'):
            self.assertEqual(ai.OPENAI_MODELS[model], 400_000)
            self.assertEqual(ai.MODEL_METADATA[model]['max_output_tokens'], 128_000)
        self.assertEqual(ai.GEMINI_MODELS['gemini-3.6-flash'], 1_048_576)
        self.assertEqual(ai.GEMINI_MODELS['gemini-2.5-pro'], 1_048_576)
        self.assertEqual(ai.MODEL_METADATA['gemini-2.5-pro']['max_output_tokens'], 65_536)
        self.assertEqual(ai.GEMINI_MODELS['gemini-3.5-flash-lite'], 1_048_576)
        self.assertEqual(ai.GEMINI_MODELS['gemini-3.1-pro-preview'], 1_048_576)
        self.assertEqual(ai.GEMINI_MODELS['gemini-3-pro-image'], 131_072)
        self.assertEqual(ai.MODEL_METADATA['gemini-3-pro-image']['max_output_tokens'], 32_768)
        for model in ('claude-opus-4-8', 'claude-opus-4-7', 'claude-opus-4-6'):
            self.assertEqual(ai.ANTHROPIC_MODELS[model], 1_000_000)
        self.assertEqual(ai.OPENAI_MODELS['gpt-4'], 8_192)
        self.assertEqual(ai.OPENAI_MODELS['gpt-3.5-turbo'], 16_385)
        self.assertEqual(ai.OPENAI_MODELS['gpt-5.3-chat-latest'], 128_000)

    def test_latest_and_default_routes(self):
        expected = {
            'chatgpt-latest': 'chat-latest',
            'gpt-5.3': 'gpt-5.4',
            'gpt-5.4-chat-latest': 'gpt-5.4',
            'gpt-5.5-chat-latest': 'gpt-5.5',
            'claude-latest': 'claude-opus-5',
            'grok-latest': 'grok-4.3',
            'grok-4-3': 'grok-4.3',
            'grok-3': 'grok-4.3',
            'gemini-latest': 'gemini-3.6-flash',
            'gemini-3.1-pro': 'gemini-3.1-pro-preview',
            'gemini-3-pro-preview': 'gemini-3.1-pro-preview',
            'gemini-3-pro-image-preview': 'gemini-3-pro-image',
            'gemini-2.0-flash': 'gemini-3.6-flash',
        }
        for alias, target in expected.items():
            self.assertEqual(ai.LATEST_MODELS_MAP[alias], target)
        self.assertNotIn('gpt-5.3-chat-latest', ai.LATEST_MODELS_MAP)
        self.assertEqual(ai.MODEL_METADATA['gpt-5.3-chat-latest']['max_output_tokens'], 16_384)
        self.assertEqual(ai.OPENAI_MODELS['default'], 'gpt-5.6')
        self.assertEqual(ai.ANTHROPIC_MODELS['default'], 'claude-opus-5')
        self.assertEqual(ai.GROK_MODELS['default'], 'grok-4.5')
        self.assertEqual(ai.GEMINI_MODELS['default'], 'gemini-3.6-flash')

    def test_compatibility_aliases_are_resolved_before_openai_call(self):
        captured = []

        def create(**kwargs):
            captured.append(kwargs['model'])
            message = types.SimpleNamespace(content='ok')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        platform = ai.OpenAIPlatform.__new__(ai.OpenAIPlatform)
        platform.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )
        platform.no_temperature_list = []
        for alias, upstream in (
            ('chatgpt-latest', 'chat-latest'),
            ('gpt-5.3', 'gpt-5.4'),
            ('gpt-5.4-chat-latest', 'gpt-5.4'),
            ('gpt-5.5-chat-latest', 'gpt-5.5'),
        ):
            self.assertEqual(platform.make_call([], alias, timeout=1), 'ok')
            self.assertEqual(captured[-1], upstream)
        self.assertNotIn('chatgpt-latest', captured)

    def test_front_catalog_hides_legacy_but_backend_retains_it(self):
        backend = ai.get_all_models()
        frontend = ai.get_front_models()
        for legacy in ('gpt-5.3', 'gpt-5.4-chat-latest', 'chatgpt-latest', 'claude-sonnet-4-0', 'claude-sonnet-4-20250514', 'grok-3', 'grok-4', 'grok-4-3', 'gemini-3.1-pro'):
            self.assertIn(legacy, backend)
            self.assertNotIn(legacy, frontend)
        self.assertNotIn('claude-mythos-5', backend)
        self.assertNotIn('gemini-3-pro-preview', frontend)
        self.assertIn('gemini-3.1-pro-preview', backend)
        self.assertNotIn('gemini-3.1-pro-preview', frontend)

    def test_current_multimodal_and_sampling_capabilities(self):
        for model in ('gpt-5.6', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'):
            self.assertIn(model, ai.OPENAI_IMG_CAPABLE)
        for model in ('claude-fable-5', 'claude-opus-5', 'claude-sonnet-5'):
            self.assertIn(model, ai.ANTHROPIC_NO_TEMPERATURE_LIST)
        for model in ('gemini-3.6-flash', 'gemini-3.5-flash-lite'):
            self.assertIn(model, ai.GEMINI_MULTIMODAL_MODELS)
            self.assertIn(model, ai.GEMINI_THINKING_MODELS)


if __name__ == '__main__':
    unittest.main()
