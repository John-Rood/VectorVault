from types import SimpleNamespace

from vectorvault.ai import LLMClient
from vectorvault.vault import Vault


class FakePlatform:
    def __init__(self):
        self.default_model = "gpt-5.6"
        self.model_token_limits = {"gpt-5.6": 1_050_000}
        self.calls = []

    def get_tokens(self, text, encoding_name=None):
        return len(str(text))

    def model_check(self, token_count, model):
        return model

    def make_call(self, messages, model, temperature=None, timeout=None, thinking_level=None, **kwargs):
        self.calls.append(("make", model, thinking_level, kwargs))
        return "complete"

    def stream_call(self, messages, model, temperature=None, timeout=None, thinking_level=None, **kwargs):
        self.calls.append(("stream", model, thinking_level, kwargs))
        return iter(["streamed"])


def test_llmclient_public_entrypoints_propagate_named_thinking_level():
    platform = FakePlatform()
    client = LLMClient(platform)
    assert client.text_llm("hello", thinking_level="high") == "complete"
    assert client.llm("hello", thinking_level="low") == "complete"
    assert client.llm_sys("hello", thinking_level="max") == "complete"
    assert client.llm_instruct("hello", "answer", thinking_level="medium") == "complete"
    assert client.llm_w_context("hello", "context", thinking_level="high") == "complete"
    assert list(client.llm_stream("hello", thinking_level="low")) == ["streamed"]
    assert list(client.llm_w_context_stream("hello", "context", thinking_level="max")) == ["streamed"]
    assert client.summarize("hello", thinking_level="medium") == "complete"
    assert list(client.summarize_stream("hello", thinking_level="high")) == ["streamed"]
    assert [call[2] for call in platform.calls] == [
        "high", "low", "max", "medium", "high", "low", "max", "medium", "high"
    ]


class FakeRateLimiter:
    max_attempts = 1
    current_delay = 0

    def on_success(self):
        pass

    def on_failure(self):
        pass


class FakeVaultAI:
    def __init__(self):
        self.calls = []

    def llm(self, *args, thinking_level=None, **kwargs):
        self.calls.append(("llm", thinking_level))
        return "complete"

    def llm_stream(self, *args, thinking_level=None, **kwargs):
        self.calls.append(("llm_stream", thinking_level))
        return iter(["streamed"])


def _bare_vault():
    vault = object.__new__(Vault)
    vault._all_models = {"default": "gpt-5.6"}
    vault._ai = FakeVaultAI()
    vault.load_ai = lambda model=None: None
    vault.rate_limiter = FakeRateLimiter()
    vault.verbose = False
    return vault


def test_vault_get_chat_validates_resolves_and_propagates():
    vault = _bare_vault()
    assert vault.get_chat("hello", model="gpt-5.6", thinking_level="HIGH") == "complete"
    assert vault._ai.calls == [("llm", "high")]


def test_vault_get_chat_stream_validates_resolves_and_propagates():
    vault = _bare_vault()
    assert list(vault.get_chat_stream("hello", model="gpt-5.6", thinking_level="MAX")) == [
        "streamed", "!END"
    ]
    assert vault._ai.calls == [("llm_stream", "max")]
