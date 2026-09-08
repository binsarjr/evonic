"""Per-model effort discovery, persistence and provider request boundaries."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import shutil
import subprocess

import pytest

from backend.llm_client import LLMClient
from backend.reasoning_capabilities import model_reasoning_capabilities, apply_reasoning_effort
from backend.reasoning_effort_error import ReasoningEffortError
from backend.provider.codex_client import CodexClient
from models.db import db


@pytest.fixture
def client():
    from app import app
    app.config['TESTING'] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session['authenticated'] = True
    return client


def _provider(api_format='openai', base_url='https://api.deepseek.com/v1'):
    pid = db.create_provider({'id': 'effort-test', 'api_format': api_format,
                              'base_url': base_url, 'api_key': 'test-key'})
    return db.get_provider(pid)


def _model(**extra):
    return {'provider': 'effort-test', 'model_name': 'deepseek-v4-pro',
            'name': 'Effort model', 'type': 'remote', **extra}


def test_metadata_is_per_model_and_endpoint_bound():
    provider = _provider('codex', 'https://chatgpt.com/backend-api/codex')
    db.save_provider_model_capabilities(provider, [{
        'id': 'one', 'supported_reasoning_levels': [{'effort': 'low'}, {'effort': 'ultra'}],
        'default_reasoning_level': 'low',
    }, {'id': 'two', 'supported_reasoning_levels': [{'effort': 'high'}]}])
    assert db.get_model_reasoning_capabilities(_model(model_name='one')) == {
        'efforts': ['low', 'ultra'], 'default_effort': 'low', 'manual': False,
    }
    assert db.get_model_reasoning_capabilities(_model(model_name='two'))['efforts'] == ['high']
    assert db.get_model_reasoning_capabilities(_model(model_name='unknown'))['efforts'] == []
    assert db.get_model_reasoning_capabilities(_model(model_name='one',
        base_url='https://gateway.example/v1'))['efforts'] == []
    db.update_provider(provider['id'], {'base_url': 'https://chatgpt.com/other'})
    assert db.get_model_reasoning_capabilities(_model(model_name='one'))['efforts'] == []
    # An old, in-flight discovery response cannot restore support at the new endpoint.
    db.save_provider_model_capabilities(provider, [{'id': 'one'}])
    assert db.get_provider(provider['id'])['model_capabilities'] == '{}'


def test_anthropic_uses_advertised_levels_only():
    provider = {'api_format': 'anthropic', 'base_url': 'https://api.anthropic.com/v1', 'model_name': 'claude'}
    metadata = {'capabilities': {'effort': {'supported': True, 'low': {'supported': True},
                'high': {'supported': True}, 'max': {'supported': False}}}}
    assert model_reasoning_capabilities(provider, metadata) == {
        'efforts': ['low', 'high'], 'default_effort': 'high', 'manual': False,
    }
    assert model_reasoning_capabilities(provider)['efforts'] == []
    payload = {'output_config': {'format': {'type': 'json_schema'}}}
    apply_reasoning_effort(payload, provider, 'high')
    assert payload['output_config']['effort'] == 'high'
    assert 'format' in payload['output_config']
    assert 'thinking' not in payload


@pytest.mark.parametrize('url,api_format', [
    ('https://openrouter.ai/api/v1', 'openai'),
    ('https://generativelanguage.googleapis.com/v1beta/openai', 'openai'),
    ('https://api.deepseek.com.evil.example/v1', 'openai'),
    ('http://api.deepseek.com/v1', 'openai'),
    ('https://api.deepseek.com/v1', 'anthropic'),
])
def test_gateway_and_other_formats_do_not_inherit_deepseek_support(url, api_format):
    provider = {'base_url': url, 'api_format': api_format, 'model_name': 'deepseek-v4-pro'}
    assert model_reasoning_capabilities(provider)['efforts'] == []


def test_model_routes_preserve_effort_and_reject_invalid_updates(client):
    _provider()
    response = client.post('/api/models', json=_model(reasoning_effort='max', is_default=1))
    assert response.status_code == 200
    mid = response.json['model_id']
    saved = client.get('/api/models/' + mid).json
    assert saved['reasoning_effort'] == 'max'
    assert saved['reasoning_capabilities']['efforts'] == ['low', 'high', 'max']
    assert db.get_default_model()['reasoning_effort'] == 'max'
    assert db.get_model_by_shortcode(saved['shortcode'])['reasoning_effort'] == 'max'
    assert db.get_models_by_provider('effort-test')[0]['reasoning_effort'] == 'max'
    clone = client.post('/api/models/' + mid + '/clone').json
    assert db.get_model_by_id(clone['model_id'])['reasoning_effort'] == 'max'
    response = client.put('/api/models/' + clone['model_id'],
                          json={'reasoning_effort': 'medium', 'is_default': 1})
    assert response.status_code == 400
    assert db.get_default_model()['id'] == mid
    assert client.put('/api/models/' + mid, json={'base_url': 'https://gateway.example/v1'}).status_code == 400
    rows = db.get_llm_models()
    db.save_llm_models(rows)
    assert db.get_model_by_id(mid)['reasoning_effort'] == 'max'
    with pytest.raises(ReasoningEffortError):
        db.save_llm_models([{**rows[0], 'reasoning_effort': 'invalid'}])
    assert len(db.get_llm_models()) == len(rows)
    assert client.put('/api/models/' + mid, json={'reasoning_effort': None}).status_code == 200
    assert db.get_model_by_id(mid)['reasoning_effort'] is None


def test_capability_endpoint_uses_effective_config(client):
    _provider()
    url = '/api/providers/effort-test/reasoning-capabilities'
    result = client.get(url, query_string={'model_name': 'deepseek-v4-pro'}).json
    assert result['can_refresh'] is True
    assert result['reasoning_capabilities']['efforts'] == ['low', 'high', 'max']
    result = client.get(url, query_string={'model_name': 'deepseek-v4-pro',
                                          'base_url': 'https://openrouter.ai/api/v1'}).json
    assert result['reasoning_supported'] is False
    assert result['reasoning_capabilities']['efforts'] == []


@pytest.mark.parametrize('model,last,default', [
    ('gpt-6-astra', 'ultra', 'medium'),
    ('gpt-5.6-sol', 'ultra', 'low'),
    ('gpt-5.6-terra', 'ultra', 'medium'),
    ('gpt-5.6-luna', 'max', 'medium'),
])
def test_codex_edit_has_defaults_before_discovery(client, model, last, default):
    provider = _provider('codex', 'https://chatgpt.com/backend-api/codex')
    url = '/api/providers/effort-test/reasoning-capabilities'
    query = {'model_name': model, 'api_format': 'codex'}
    result = client.get(url, query_string=query).json['reasoning_capabilities']
    assert result['efforts'][-1] == last
    assert result['default_effort'] == default
    assert db.validate_model_reasoning(_model(model_name=model, reasoning_effort=last)) == last
    # Empty snapshots saved before built-in support must not mask the defaults.
    snapshot = {'base_url': provider['base_url'], 'api_format': 'codex',
                'models': {model: {'efforts': [], 'default_effort': None}}}
    with db._connect() as conn:
        conn.execute('UPDATE providers SET model_capabilities = ? WHERE id = ?',
                     (json.dumps(snapshot), provider['id']))
        conn.commit()
    assert client.get(url, query_string=query).json['reasoning_capabilities'] == result
    db.save_provider_model_capabilities(provider, [{'id': model,
        'supported_reasoning_levels': [{'effort': 'high'}], 'default_reasoning_level': 'high'}])
    assert client.get(url, query_string=query).json['reasoning_capabilities']['efforts'] == ['high']
    assert client.get(url, query_string={**query, 'base_url': 'https://gateway.example/v1'}).json[
        'reasoning_capabilities']['efforts'] == []


def test_failed_or_empty_discovery_keeps_verified_support(client):
    provider = _provider('codex', 'https://chatgpt.com/backend-api/codex')
    db.save_provider_model_capabilities(provider, [{'id': 'model',
        'supported_reasoning_levels': [{'effort': 'high'}]}])
    before = db.get_provider(provider['id'])['model_capabilities']
    with patch('backend.provider.oauth_codex.get_valid_token', return_value='token'), \
         patch('httpx.get') as fetch:
        fetch.return_value = MagicMock(status_code=200)
        fetch.return_value.json.return_value = {'models': []}
        assert client.post('/api/providers/effort-test/fetch-models').json['success']
        fetch.return_value = MagicMock(status_code=503, text='unavailable')
        assert not client.post('/api/providers/effort-test/fetch-models').json['success']
    assert db.get_provider(provider['id'])['model_capabilities'] == before


@pytest.mark.parametrize('effort,enable,expected', [('max', True, 'max'), ('max', False, None), (None, True, None)])
def test_runtime_effort_is_independent_of_legacy_thinking(effort, enable, expected):
    _provider()
    client = LLMClient(_model(reasoning_effort=effort, thinking=False, timeout=10))
    response = MagicMock(status_code=200)
    response.json.return_value = {'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                                              'finish_reason': 'stop'}], 'usage': {}}
    with patch('backend.llm_client.requests.post', return_value=response) as post:
        result = client.chat_completion([{'role': 'user', 'content': 'hi'}], enable_thinking=enable)
    assert result['success'], result
    payload = post.call_args.kwargs['json']
    assert payload.get('reasoning_effort') == expected
    if expected is None:
        assert 'reasoning_effort' not in payload


def test_invalid_runtime_effort_fails_before_network():
    _provider()
    client = LLMClient(_model(reasoning_effort='ultra'))
    with patch('backend.llm_client.requests.post') as post:
        result = client.chat_completion([{'role': 'user', 'content': 'hi'}])
    assert result['error_type'] == 'configuration_error'
    post.assert_not_called()


def test_codex_effort_merges_summary_and_fast_mode():
    client = CodexClient('token', 'https://chatgpt.com/backend-api/codex')
    response = MagicMock(status_code=200)
    response.json.return_value = {'output': [], 'usage': {}}
    with patch('backend.provider.codex_client.httpx.post', return_value=response) as post:
        client.send_request('gpt-5.6-luna', [{'role': 'user', 'content': 'hi'}],
                            stream=False, reasoning=True, reasoning_effort='high', service_tier='priority')
    payload = post.call_args.kwargs['json']
    assert payload['reasoning'] == {'summary': 'auto', 'effort': 'high'}
    assert payload['service_tier'] == 'priority'


def test_anthropic_runtime_effort_without_native_thinking():
    provider = _provider('anthropic', 'https://api.anthropic.com/v1')
    db.save_provider_model_capabilities(provider, [{'id': 'claude', 'capabilities': {
        'effort': {'supported': True, 'high': {'supported': True}},
    }}])
    client = LLMClient(_model(model_name='claude', reasoning_effort='high', thinking=False))
    response = MagicMock(status_code=200)
    response.json.return_value = {'content': [{'type': 'text', 'text': 'ok'}],
                                  'stop_reason': 'end_turn', 'usage': {}}
    with patch('backend.llm_client.requests.post', return_value=response) as post:
        assert client.chat_completion([{'role': 'user', 'content': 'hi'}])['success']
    payload = post.call_args.kwargs['json']
    assert payload['output_config'] == {'effort': 'high'}
    assert 'thinking' not in payload


def test_codex_stream_effort_merges_summary():
    client = CodexClient('token', 'https://chatgpt.com/backend-api/codex')
    with patch('backend.provider.codex_client.httpx.Client') as http:
        stream = http.return_value.__enter__.return_value.stream
        response = stream.return_value.__enter__.return_value
        response.status_code = 200
        response.iter_lines.return_value = []
        list(client.stream_chunks('gpt-5.6-luna', [], reasoning=True, reasoning_effort='high'))
    assert stream.call_args.kwargs['json']['reasoning'] == {'summary': 'auto', 'effort': 'high'}


def test_reasoning_dropdown_behavior():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required for frontend behavior checks')
    subprocess.run([node, 'unit_tests/test_settings_models_reasoning.js'],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=15)


@pytest.mark.parametrize('api_format,url,field', [
    ('codex', 'https://chatgpt.com/backend-api/codex', 'reasoning'),
    ('anthropic', 'https://api.anthropic.com/v1', 'output_config'),
    ('openai', 'https://api.deepseek.com/v1', 'reasoning_effort'),
])
def test_unidentified_model_accepts_manual_effort(client, api_format, url, field):
    provider = _provider(api_format, url)
    model = _model(model_name='new-model', reasoning_effort='custom_level')
    caps = db.get_model_reasoning_capabilities(model)
    assert caps['manual'] is True and caps['efforts'] == []
    created = client.post('/api/models', json=model)
    assert created.status_code == 200
    mid = created.json['model_id']
    saved = client.get('/api/models/' + mid).json
    assert saved['reasoning_effort'] == 'custom_level'
    assert saved['reasoning_capabilities']['manual'] is True
    resolved = db.resolve_model_config(saved)
    payload = {}
    apply_reasoning_effort(payload, resolved, db.validate_model_reasoning(saved))
    assert payload[field] == ('custom_level' if field == 'reasoning_effort' else {'effort': 'custom_level'})
    for invalid in ('high!', 'a' * 33, 123):
        assert client.put('/api/models/' + mid, json={'reasoning_effort': invalid}).status_code == 400
    assert client.put('/api/models/' + mid, json={'reasoning_effort': ''}).status_code == 200
    assert db.get_model_by_id(mid)['reasoning_effort'] is None
    assert not db.get_model_reasoning_capabilities({**model, 'base_url': 'https://gateway.example'})['manual']


def test_explicitly_unsupported_metadata_disables_manual_effort():
    provider = _provider('anthropic', 'https://api.anthropic.com/v1')
    db.save_provider_model_capabilities(provider, [{'id': 'no-reasoning',
        'capabilities': {'effort': {'supported': False}}}])
    model = _model(model_name='no-reasoning', reasoning_effort='high')
    assert db.get_model_reasoning_capabilities(model)['manual'] is False
    with pytest.raises(ReasoningEffortError):
        db.validate_model_reasoning(model)
