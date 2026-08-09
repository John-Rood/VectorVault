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


def test_openai_nonstream_and_stream_emit_reasoning_effort_and_omit_temperature():
    platform, recorder = _openai_platform()
    assert platform.make_call([], "gpt-5.6", temperature=0.7, thinking_level="max") == "complete"
    assert list(platform.stream_call([], "gpt-5.6", temperature=0.7, thinking_level="low")) == ["streamed"]
    assert recorder.calls[0]["reasoning_effort"] == "max"
    assert recorder.calls[1]["reasoning_effort"] == "low"
    assert "temperature" not in recorder.calls[0]
    assert "temperature" not in recorder.calls[1]


def test_xai_nonstream_and_stream_emit_reasoning_effort():
    platform, recorder = _grok_platform()
    assert platform.make_call([], "grok-4.5", thinking_level="medium") == "complete"
    assert list(platform.stream_call([], "grok-4.5", thinking_level="high")) == ["streamed"]
    assert recorder.calls[0]["reasoning_effort"] == "medium"
    assert recorder.calls[1]["reasoning_effort"] == "high"


def test_anthropic_nonstream_and_stream_emit_effort_adaptive_thinking_and_text_only():
    platform, recorder = _anthropic_platform()
    assert platform.make_call([], "claude-opus-5", temperature=0.5, thinking_level="xhigh") == "complete"
    assert list(platform.stream_call([], "claude-opus-5", temperature=0.5, thinking_level="low")) == ["streamed"]
    assert recorder.calls[0]["output_config"] == {"effort": "xhigh"}
    assert recorder.calls[0]["thinking"] == {"type": "adaptive"}
    assert recorder.calls[1]["output_config"] == {"effort": "low"}
    assert "temperature" not in recorder.calls[0]
    assert "temperature" not in recorder.calls[1]


def test_gemini_nonstream_and_stream_construct_sdk_thinking_config():
    platform, recorder = _gemini_platform()
    assert platform.make_call([], "gemini-3.6-flash", thinking_level="minimal") == "complete"
    assert list(platform.stream_call([], "gemini-3.6-flash", thinking_level="high")) == ["streamed"]
    assert platform.make_call([], "gemini-2.5-pro", thinking_level="low") == "complete"
    first = recorder.calls[0]["config"].model_dump(exclude_none=True)
    second = recorder.calls[1]["config"].model_dump(exclude_none=True)
    legacy = recorder.calls[2]["config"].model_dump(exclude_none=True)
    assert str(first["thinking_config"]["thinking_level"]).lower().endswith("minimal")
    assert str(second["thinking_config"]["thinking_level"]).lower().endswith("high")
    assert legacy["thinking_config"]["thinking_budget"] == 1024


def test_omission_emits_no_provider_thinking_fields():
    openai_platform, openai_recorder = _openai_platform()
    anthropic_platform, anthropic_recorder = _anthropic_platform()
    gemini_platform, gemini_recorder = _gemini_platform()
    openai_platform.make_call([], "gpt-5.6")
    anthropic_platform.make_call([], "claude-opus-5")
    gemini_platform.make_call([], "gemini-3.6-flash")
    assert "reasoning_effort" not in openai_recorder.calls[0]
    assert "output_config" not in anthropic_recorder.calls[0]
    assert "thinking" not in anthropic_recorder.calls[0]
    gemini_dump = gemini_recorder.calls[0]["config"].model_dump(exclude_none=True)
    assert "thinking_config" not in gemini_dump


def test_invalid_selection_fails_before_sdk_call_for_streaming_and_nonstreaming():
    platform, recorder = _openai_platform()
    with pytest.raises(ModelCapabilityError):
        platform.make_call([], "gpt-4o", thinking_level="high")
    with pytest.raises(ModelCapabilityError):
        platform.stream_call([], "gpt-4o", thinking_level="high")
    assert recorder.calls == []
