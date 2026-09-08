"""Provider format regressions at the shared LLM boundary."""

import copy
import json
import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.provider.anthropic_provider import AnthropicProvider
from backend.provider.codex_provider import CODEX_CLIENT_VERSION, CodexProvider
from backend.provider.factory import get_provider
from backend.provider.ollama_provider import OllamaProvider
from backend.provider.openai_provider import OpenAIProvider
from backend.llm_client import LLMClient


def test_provider_modules_define_at_most_one_class():
    package = Path(__file__).resolve().parents[1] / 'backend/provider'
    for path in package.glob('*.py'):
        classes = [node.name for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, ast.ClassDef)]
        assert len(classes) <= 1, f'{path.name} defines multiple classes: {classes}'


@pytest.mark.parametrize('reverse', [False, True])
def test_provider_modules_import_in_a_fresh_process(reverse):
    root = Path(__file__).resolve().parents[1]
    modules = sorted(('backend.provider.' + path.stem
                      for path in (root / 'backend/provider').glob('*.py')
                      if path.stem != '__init__'), reverse=reverse)
    subprocess.run([sys.executable, '-c',
                    'import importlib, sys; [importlib.import_module(name) for name in sys.argv[1:]]',
                    *modules], cwd=root, check=True, timeout=30)


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


@pytest.mark.parametrize('provider,url', [
    ('openrouter', 'https://openrouter.ai/api/v1'),
    ('togetherai', 'https://api.together.xyz/v1'),
    ('opencode_zen', 'https://opencode.ai/zen/v1'),
    ('opencode_go', 'https://opencode.ai/zen/go/v1'),
    ('cavoti', 'https://gateway.example/v1'),
    ('ollama', 'http://localhost:11434/v1'),
    ('llama.cpp', 'http://localhost:8080/v1'),
    ('custom', 'https://custom.example/v1'),
])
def test_compatible_providers_share_the_same_implementation(provider, url):
    adapter = get_provider({'provider': provider, 'base_url': url, 'api_format': 'openai'})
    assert type(adapter) is OpenAIProvider


def test_llm_client_delegates_without_knowing_the_provider_class():
    class ExampleProvider(OpenAIProvider):
        def chat_completion(self, messages, **kwargs):
            return {'success': True, 'messages': messages, 'options': kwargs}

        def test_connection(self):
            return {'success': True, 'message': 'example connection'}

        def get_actual_model_name(self, force_refresh=False):
            self._cached_model_name = 'actual-model'
            return self._cached_model_name

    client = LLMClient({'provider': 'custom', 'base_url': 'https://example.test/v1',
                        'model_name': 'configured', 'timeout': 5})
    client.timeout = 120
    client.max_retries = 0
    with patch('backend.llm_client.get_provider', side_effect=ExampleProvider) as factory:
        result = client.chat_completion([{'role': 'user', 'content': 'hi'}], max_tokens=64)
        assert result['success']
        assert result['options']['max_tokens'] == 64
        assert factory.call_args.args[0]['timeout'] == 120
        assert factory.call_args.args[0]['max_retries'] == 0
        assert client.test_connection()['message'] == 'example connection'
        assert client.get_actual_model_name() == 'actual-model'
        before = factory.call_count
        assert client.get_actual_model_name() == 'actual-model'
        assert factory.call_count == before
        client.get_actual_model_name(force_refresh=True)
        assert factory.call_count == before + 1


def test_provider_input_rules_preserve_caller_history():
    messages = [{'role': 'assistant', 'content': 'answer', 'reasoning_content': 'reason'},
                {'role': 'assistant', 'content': 'next'}]
    original = copy.deepcopy(messages)
    generic = get_provider({'base_url': 'https://opencode.ai/zen/go/v1'})
    assert generic.prepare_messages(messages, True)[1]['reasoning_content'] == ''
    cerebras = get_provider({'base_url': 'https://api.cerebras.ai/v1'})
    assert all('reasoning_content' not in m for m in cerebras.prepare_messages(messages, True))
    claude = get_provider({'api_format': 'anthropic', 'model_name': 'claude',
                          'base_url': 'https://api.anthropic.com/v1'})
    images = [{'role': 'user', 'content': [{'type': 'image_url',
               'image_url': {'url': 'data:image/png;base64,abc'}}]}]
    prepared = claude.prepare_messages(images, True)
    assert prepared[0]['content'][0]['source']['media_type'] == 'image/png'
    gemma = get_provider({'model_name': 'gemma4'})
    assert gemma.prepare_messages([{'role': 'user', 'content': 'hi'}], True)[0]['content'].startswith('<|think|>')
    assert messages == original
    assert images[0]['content'][0]['type'] == 'image_url'


@pytest.mark.parametrize('api_format,url,data,path', [
    ('openai', 'https://example.test/v1', {'data': [{'id': 'one'}]}, '/models'),
    ('ollama', 'http://localhost:11434/api', {'models': [{'name': 'one'}]}, '/tags'),
])
def test_connection_uses_the_provider_discovery_endpoint(api_format, url, data, path):
    client = LLMClient({'base_url': url, 'api_format': api_format, 'api_key': 'model-key'})
    response = MagicMock(status_code=200)
    response.json.return_value = data
    with patch('backend.provider.base.requests.get', return_value=response) as get:
        assert client.test_connection()['available_models'] == 1
    assert get.call_args.args[0] == url + path
    assert get.call_args.kwargs['headers']['Authorization'] == 'Bearer model-key'


@pytest.mark.parametrize('api_format,first,success', [
    ('openai', {'error': {'code': 503, 'message': 'unavailable'}},
     {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}], 'usage': {}}),
    ('anthropic', {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'busy'}},
     {'content': [{'type': 'text', 'text': 'ok'}], 'stop_reason': 'end_turn', 'usage': {}}),
])
def test_shared_retry_path_uses_provider_error_classification(api_format, first, success):
    client = LLMClient({'base_url': 'https://api.example.test/v1', 'api_format': api_format,
                        'api_key': 'key', 'max_tokens': 64, 'timeout': 5})
    client.max_retries = 1
    response = MagicMock(status_code=200)
    response.json.side_effect = [first, success]
    with patch('backend.provider.openai_provider.requests.post', return_value=response) as post, \
         patch('backend.provider.openai_provider.time.sleep') as sleep, \
         patch('backend.llm_usage_events.record_llm_usage') as usage:
        result = client.chat_completion([{'role': 'user', 'content': 'hi'}])
    assert result['success']
    assert post.call_count == 2
    sleep.assert_called_once_with(2)
    usage.assert_called_once()


@pytest.mark.parametrize('override', [None, 'sk-ant-api-model-key'])
def test_anthropic_credential_source_is_shared_by_completion_and_discovery(override):
    from models.db import db
    from backend.provider.claude_code import SYSTEM_PREFIX

    provider_id = db.create_provider({'id': 'native-claude', 'api_format': 'anthropic',
        'base_url': 'https://api.anthropic.com/v1', 'api_key': 'provider-key'})
    client = LLMClient({'provider': provider_id, 'model_name': 'claude',
                        'api_key': override, 'max_tokens': 64})
    response = MagicMock(status_code=200)
    response.json.return_value = {'content': [{'type': 'text', 'text': 'ok'}],
                                  'stop_reason': 'end_turn', 'usage': {}}
    catalog = MagicMock(status_code=200)
    catalog.json.return_value = {'data': [{'id': 'claude'}]}
    with patch('backend.provider.claude_code.resolve_credential',
               return_value=('sk-ant-oat-token', True)) as resolve, \
         patch('backend.provider.claude_code.claude_code_version', return_value='1.0.0'), \
         patch('backend.provider.openai_provider.requests.post', return_value=response) as post, \
         patch('backend.provider.base.requests.get', return_value=catalog) as get:
        assert client.chat_completion([{'role': 'user', 'content': 'hi'}])['success']
        assert client.test_connection()['available_models'] == 1
    assert post.call_args.kwargs['headers'] == get.call_args.kwargs['headers']
    if override:
        resolve.assert_not_called()
        assert post.call_args.kwargs['headers']['x-api-key'] == override
        assert 'system' not in post.call_args.kwargs['json']
    else:
        assert resolve.call_count == 2
        assert resolve.call_args.args[1] == provider_id
        assert post.call_args.kwargs['headers']['Authorization'] == 'Bearer sk-ant-oat-token'
        assert post.call_args.kwargs['json']['system'] == SYSTEM_PREFIX
