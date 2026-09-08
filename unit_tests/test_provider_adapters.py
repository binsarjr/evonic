"""Provider format regressions at the shared LLM boundary."""

import copy
import json

import pytest

from backend.provider.adapters import (
    AnthropicProvider, CODEX_CLIENT_VERSION, CodexProvider, OllamaProvider,
    OpenAIProvider, get_provider,
)


@pytest.mark.parametrize("api_format,cls", [
    ("openai", OpenAIProvider), ("ollama", OllamaProvider),
    ("anthropic", AnthropicProvider), ("codex", CodexProvider),
])
def test_provider_preserves_messages_and_tool_choice(api_format, cls):
    provider = get_provider({"api_format": api_format})
    assert isinstance(provider, cls)
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Find a record", "parameters": {"type": "object"},
    }}]
    original = copy.deepcopy((messages, tools))
    payload = provider.build_payload("model", messages, 256, 0.5, tools, "lookup")
    assert (messages, tools) == original
    assert payload["model"] == "model"
    if api_format == "anthropic":
        assert payload["system"] == "system"
        assert payload["messages"] == messages[1:]
        assert payload["tool_choice"] == {"type": "tool", "name": "lookup"}
        assert payload["tools"][0]["input_schema"] == {"type": "object"}
    elif api_format == "codex":
        assert payload["tool_choice"] == {"type": "function", "name": "lookup"}
        assert payload["input"][0]["role"] == "system"
        assert payload["store"] is False
    else:
        assert payload["messages"] == messages
        assert payload["tool_choice"]["function"]["name"] == "lookup"
        if api_format == "ollama":
            assert payload["options"] == {"num_predict": 256, "temperature": 0.5}
        else:
            assert payload["max_tokens"] == 256
            assert payload["temperature"] == 0.5


def test_anthropic_response_keeps_tool_calls_and_usage():
    result = AnthropicProvider({}).normalize_response({
        "content": [{"type": "text", "text": "Searching"},
                    {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": "x"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3},
    })
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Searching"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call-1"
    assert json.loads(call["function"]["arguments"]) == {"q": "x"}
    assert result["usage"]["total_tokens"] == 15
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 3


def test_ollama_response_and_codex_discovery_keep_existing_formats():
    result = OllamaProvider({}).normalize_response({
        "message": {"content": "answer", "reasoning_content": "summary"},
        "prompt_eval_count": 4, "eval_count": 2,
    })
    assert result["choices"][0]["message"]["reasoning_content"] == "summary"
    assert result["usage"]["total_tokens"] == 6
    codex = CodexProvider({})
    assert codex.discovery_params() == {"client_version": CODEX_CLIENT_VERSION}
    models = codex.parse_models({"models": ["old-model", {"slug": "new-model", "extra": 1}]})
    assert [m["id"] for m in models] == ["old-model", "new-model"]
    assert models[1]["extra"] == 1
