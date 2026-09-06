from types import SimpleNamespace

import pytest

from vectorvault.ai import AnthropicPlatform, GeminiPlatform, GrokPlatform, OpenAIPlatform
from vectorvault.model_catalog import ModelCapabilityError


class OpenAICompletionsRecorder:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="streamed"))])]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="complete"))])


class AnthropicMessagesRecorder:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="thinking_delta", thinking="private"),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="streamed"),
                ),
                SimpleNamespace(type="message_stop"),
            ]
        return SimpleNamespace(content=[
            SimpleNamespace(type="thinking", thinking="private"),
            SimpleNamespace(type="text", text="complete"),
        ])


class GeminiModelsRecorder:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="complete")

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(text="streamed")]


def _openai_platform():
    recorder = OpenAICompletionsRecorder()
    platform = object.__new__(OpenAIPlatform)
    platform.client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    platform.no_temperature_list = []
    platform.no_stream_list = []
    return platform, recorder


def _grok_platform():
    recorder = OpenAICompletionsRecorder()
    platform = object.__new__(GrokPlatform)
    platform.client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    return platform, recorder


def _anthropic_platform():
    recorder = AnthropicMessagesRecorder()
    platform = object.__new__(AnthropicPlatform)
    platform.client = SimpleNamespace(messages=recorder)
    platform.no_temperature_list = []
    return platform, recorder


def _gemini_platform():
    recorder = GeminiModelsRecorder()
    platform = object.__new__(GeminiPlatform)
    platform._client_initialized = False
    platform.client = SimpleNamespace(models=recorder)
    platform.get_client = lambda model=None: platform.client
    return platform, recorder


def _dump_gemini_call(call):
    result = {"model": call["model"], "contents": call["contents"]}
    if "config" in call:
        result["config"] = call["config"].model_dump(exclude_none=True, mode="json")
    return result


def test_openai_nonstream_and_stream_emit_exact_reasoning_kwargs():
    platform, recorder = _openai_platform()
    assert platform.make_call([], "gpt-5.6", temperature=0.7, thinking_level="max") == "complete"
    assert list(platform.stream_call([], "gpt-5.6", temperature=0.7, thinking_level="low")) == ["streamed"]
    assert recorder.calls == [
        {"model": "gpt-5.6", "messages": [], "reasoning_effort": "max"},
        {"model": "gpt-5.6", "messages": [], "stream": True, "reasoning_effort": "low"},
    ]


def test_september_2026_new_models_emit_provider_safe_kwargs():
    openai_platform, openai_recorder = _openai_platform()
    assert openai_platform.make_call([], "gpt-6-astra", thinking_level="max") == "complete"
    assert openai_recorder.calls == [{
        "model": "gpt-6-astra", "messages": [], "reasoning_effort": "max"
    }]

    anthropic_platform, anthropic_recorder = _anthropic_platform()
    assert anthropic_platform.make_call([], "claude-fable-5-1", thinking_level="xhigh") == "complete"
    assert anthropic_recorder.calls == [{
        "model": "claude-fable-5-1",
        "messages": [],
        "max_tokens": 8192,
        "output_config": {"effort": "xhigh"},
        "thinking": {"type": "adaptive"},
    }]

    gemini_platform, gemini_recorder = _gemini_platform()
    assert gemini_platform.make_call([], "gemini-3.8-flash", thinking_level="minimal") == "complete"
    assert [_dump_gemini_call(call) for call in gemini_recorder.calls] == [{
        "model": "gemini-3.8-flash",
        "contents": [],
        "config": {
            "temperature": 0.0,
            "thinking_config": {"thinking_level": "MINIMAL"},
        },
    }]


def test_xai_nonstream_and_stream_emit_exact_reasoning_kwargs():
    platform, recorder = _grok_platform()
    assert platform.make_call([], "grok-4.5", thinking_level="medium") == "complete"
    assert list(platform.stream_call([], "grok-4.5", thinking_level="high")) == ["streamed"]
    assert recorder.calls == [
        {"model": "grok-4.5", "messages": [], "reasoning_effort": "medium"},
        {"model": "grok-4.5", "messages": [], "stream": True, "reasoning_effort": "high"},
    ]


def test_anthropic_nonstream_and_stream_emit_exact_effort_and_adaptive_kwargs():
    platform, recorder = _anthropic_platform()
    assert platform.make_call([], "claude-opus-5", temperature=0.5, thinking_level="xhigh") == "complete"
    assert list(platform.stream_call([], "claude-opus-5", temperature=0.5, thinking_level="low")) == ["streamed"]
    assert recorder.calls == [
        {
            "model": "claude-opus-5",
            "messages": [],
            "max_tokens": 8192,
            "output_config": {"effort": "xhigh"},
            "thinking": {"type": "adaptive"},
        },
        {
            "model": "claude-opus-5",
            "messages": [],
            "max_tokens": 8192,
            "stream": True,
            "output_config": {"effort": "low"},
            "thinking": {"type": "adaptive"},
        },
    ]


@pytest.mark.parametrize("level", ("minimal", "low", "medium", "high"))
def test_gemini_all_declared_levels_emit_exact_sdk_thinking_config(level):
    platform, recorder = _gemini_platform()
    assert platform.make_call([], "gemini-3.6-flash", thinking_level=level) == "complete"
    assert [_dump_gemini_call(call) for call in recorder.calls] == [
        {
            "model": "gemini-3.6-flash",
            "contents": [],
            "config": {
                "temperature": 0.0,
                "thinking_config": {"thinking_level": level.upper()},
            },
        }
    ]


def test_gemini_stream_emits_exact_sdk_thinking_config():
    platform, recorder = _gemini_platform()
    assert list(platform.stream_call([], "gemini-3.6-flash", thinking_level="high")) == ["streamed"]
    assert [_dump_gemini_call(call) for call in recorder.calls] == [
        {
            "model": "gemini-3.6-flash",
            "contents": [],
            "config": {"temperature": 0.0, "thinking_config": {"thinking_level": "HIGH"}},
        }
    ]


def test_omission_preserves_exact_legacy_kwargs_for_sync_and_stream():
    openai_platform, openai_recorder = _openai_platform()
    grok_platform, grok_recorder = _grok_platform()
    anthropic_platform, anthropic_recorder = _anthropic_platform()
    gemini_platform, gemini_recorder = _gemini_platform()

    openai_platform.make_call([], "gpt-5.6", temperature=0.4)
    list(openai_platform.stream_call([], "gpt-5.6", temperature=0.4))
    grok_platform.make_call([], "grok-4.5", temperature=0.4)
    list(grok_platform.stream_call([], "grok-4.5", temperature=0.4))
    anthropic_platform.make_call([], "claude-opus-5", temperature=0.4)
    list(anthropic_platform.stream_call([], "claude-opus-5", temperature=0.4))
    gemini_platform.make_call([], "gemini-3.6-flash", temperature=0.4)
    list(gemini_platform.stream_call([], "gemini-3.6-flash", temperature=0.4))

    assert openai_recorder.calls == [
        {"model": "gpt-5.6", "messages": [], "temperature": 0.4},
        {"model": "gpt-5.6", "messages": [], "stream": True, "temperature": 0.4},
    ]
    # Grok's historical streaming path intentionally does not send temperature.
    assert grok_recorder.calls == [
        {"model": "grok-4.5", "messages": [], "temperature": 0.4},
        {"model": "grok-4.5", "messages": [], "stream": True},
    ]
    assert anthropic_recorder.calls == [
        {"model": "claude-opus-5", "messages": [], "max_tokens": 8192, "temperature": 0.4},
        {"model": "claude-opus-5", "messages": [], "max_tokens": 8192, "stream": True, "temperature": 0.4},
    ]
    assert [_dump_gemini_call(call) for call in gemini_recorder.calls] == [
        {"model": "gemini-3.6-flash", "contents": [], "config": {"temperature": 0.4}},
        {"model": "gemini-3.6-flash", "contents": [], "config": {"temperature": 0.4}},
    ]


def test_non_thinking_and_unproven_generic_levels_fail_before_sdk_call():
    openai_platform, openai_recorder = _openai_platform()
    gemini_platform, gemini_recorder = _gemini_platform()
    for method in (openai_platform.make_call, openai_platform.stream_call):
        with pytest.raises(ModelCapabilityError):
            method([], "gpt-4o", thinking_level="high")
    for method in (gemini_platform.make_call, gemini_platform.stream_call):
        with pytest.raises(ModelCapabilityError):
            method([], "gemini-2.5-pro", thinking_level="low")
    assert openai_recorder.calls == []
    assert gemini_recorder.calls == []


def test_xai_grok_4_3_reasoning_none_translates_without_sampling_churn():
    platform, recorder = _grok_platform()
    assert platform.make_call([], "grok-4.3", temperature=0.4, thinking_level="none") == "complete"
    assert recorder.calls == [{
        "model": "grok-4.3",
        "messages": [],
        "temperature": 0.4,
        "reasoning_effort": "none",
    }]


def test_august_2026_provider_translation_reaches_exact_sdk_parameters():
    grok_platform, grok_recorder = _grok_platform()
    gemini_platform, gemini_recorder = _gemini_platform()
    assert grok_platform.make_call([], "grok-4.6", thinking_level="xhigh") == "complete"
    assert grok_recorder.calls == [
        {"model": "grok-4.6", "messages": [], "reasoning_effort": "xhigh"}
    ]
    assert gemini_platform.make_call([], "gemini-3.7-flash", thinking_level="low") == "complete"
    assert [_dump_gemini_call(call) for call in gemini_recorder.calls] == [
        {
            "model": "gemini-3.7-flash",
            "contents": [],
            "config": {
                "temperature": 0.0,
                "thinking_config": {"thinking_level": "LOW"},
            },
        }
    ]
