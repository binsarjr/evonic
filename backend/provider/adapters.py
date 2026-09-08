"""Provider adapters for the API formats supported by Evonic."""

import json

import httpx

from backend.provider.base import BaseProvider

CODEX_CLIENT_VERSION = "0.153.4"


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible transport, also used by gateways and custom endpoints."""

    def parse_models(self, data):
        rows = data.get("data", data.get("models", []))
        if not isinstance(rows, list):
            return []
        return [{**m, "id": m.get("id", ""), "name": m.get("id", "")}
                for m in rows if isinstance(m, dict)]

    def normalize_response(self, data):
        return data

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = {
                    "type": "function", "function": {"name": tool_choice}}

        return payload


class OllamaProvider(OpenAIProvider):
    """Ollama's native chat and model-list formats."""

    completion_path = "/chat"
    models_path = "/tags"

    def parse_models(self, data):
        return [{**m, "id": m.get("name", ""), "name": m.get("name", "")}
                for m in data.get("models", []) if isinstance(m, dict)]

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = {
                    "type": "function", "function": {"name": tool_choice}}

        return payload

    def normalize_response(self, result):
        ollama_message = result.get("message", {})
        ollama_content = ollama_message.get("content", "")
        ollama_reasoning = ollama_message.get("reasoning_content", "")
        prompt_eval = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)
        transformed_message = {
            "role": "assistant",
            "content": ollama_content,
        }
        if ollama_reasoning:
            transformed_message["reasoning_content"] = ollama_reasoning
        result = {
            "choices": [
                {
                    "message": transformed_message,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_eval,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval + eval_count,
            },
        }

        return result


class AnthropicProvider(OpenAIProvider):
    """Anthropic Messages payloads and responses."""

    completion_path = "/messages"

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        # Anthropic API uses /messages endpoint with different payload structure.
        # Extract system messages into top-level "system" field.
        system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]
        payload = {
            "model": model,
            "messages": non_system_msgs,
            "max_tokens": max_tokens,
        }
        if system_msgs:
            payload["system"] = "\n\n".join(system_msgs) if len(system_msgs) > 1 else system_msgs[0]
        if oauth:
            from backend.provider.claude_code import SYSTEM_PREFIX
            payload["system"] = SYSTEM_PREFIX + (
                "\n\n" + payload["system"] if payload.get("system") else ""
            )
        if temperature is not None:
            payload["temperature"] = temperature
        # Transform OpenAI tools -> Anthropic tools format
        if tools:
            anthropic_tools = []
            for tool in tools:
                fn = tool.get("function", {})
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
            payload["tools"] = anthropic_tools
            payload["tool_choice"] = (
                {"type": "tool", "name": tool_choice}
                if tool_choice else {"type": "auto"})

        return payload

    def normalize_response(self, result):
        anthropic_content = result.get("content", [])
        anthropic_stop = result.get("stop_reason", "end_turn")
        anthropic_usage = result.get("usage", {})

        # Map stop_reason to finish_reason
        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        mapped_finish = stop_map.get(anthropic_stop, "stop")

        # Parse content blocks
        text_parts = []
        tool_calls_list = []
        for block in anthropic_content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls_list.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

        transformed_message = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if tool_calls_list:
            transformed_message["tool_calls"] = tool_calls_list

        # Map usage from Anthropic to OpenAI field names
        anthropic_input = anthropic_usage.get("input_tokens", 0)
        anthropic_output = anthropic_usage.get("output_tokens", 0)
        anthropic_cached = anthropic_usage.get("cache_read_input_tokens", 0) or 0
        result = {
            "choices": [
                {
                    "message": transformed_message,
                    "finish_reason": mapped_finish,
                }
            ],
            "usage": {
                "prompt_tokens": anthropic_input,
                "completion_tokens": anthropic_output,
                "total_tokens": anthropic_input + anthropic_output,
                "prompt_tokens_details": {
                    "cached_tokens": anthropic_cached,
                },
            },
        }

        return result


class CodexProvider(OpenAIProvider):
    """Codex Responses API; CodexClient owns its SSE transport."""

    completion_path = "/responses"
    http_client = httpx

    def discovery_params(self):
        return {"client_version": CODEX_CLIENT_VERSION}

    def discovery_headers(self, headers):
        from backend.provider.oauth_codex import extract_account_id
        result = dict(headers)
        result.update({"User-Agent": f"codex_cli_rs/{CODEX_CLIENT_VERSION}",
                       "originator": "codex_cli_rs"})
        account = extract_account_id(result.get("Authorization", "").removeprefix("Bearer "))
        if account:
            result["ChatGPT-Account-ID"] = account
        return result

    def parse_models(self, data):
        rows = data.get("models", data.get("data", []))
        if not isinstance(rows, list) or not rows:
            return [
                {"id": "gpt-5.6-sol", "name": "GPT-5.6 Sol"},
                {"id": "gpt-5.6-terra", "name": "GPT-5.6 Terra"},
                {"id": "gpt-5.6-luna", "name": "GPT-5.6 Luna"},
            ]
        models = []
        for item in rows:
            if isinstance(item, str):
                models.append({"id": item, "name": item})
            elif isinstance(item, dict):
                mid = item.get("id") or item.get("slug") or item.get("model") or item.get("name", "")
                models.append({**item, "id": mid, "name": mid})
        return models

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        from backend.provider.codex_client import CodexClient
        payload = {"model": model, "input": CodexClient._convert_messages(messages),
                   "stream": True, "store": False}
        if tools:
            payload["tools"] = CodexClient._convert_tools(tools)
            if tool_choice:
                payload["tool_choice"] = {"type": "function", "name": tool_choice}
        return payload


def get_provider(config):
    """Select the adapter using the effective API format and endpoint."""
    api_format = config.get("api_format", "openai")
    if api_format == "codex":
        cls = CodexProvider
    elif api_format == "anthropic":
        cls = AnthropicProvider
    elif api_format == "ollama" or "ollama.com" in (config.get("base_url") or ""):
        cls = OllamaProvider
    else:
        cls = OpenAIProvider
    return cls(config)
