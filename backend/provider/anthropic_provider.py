import json

from backend.provider.openai_provider import OpenAIProvider
from backend.provider.provider_auth_error import ProviderAuthError


class AnthropicProvider(OpenAIProvider):
    """Anthropic Messages payloads and responses."""

    completion_path = "/messages"

    def request_headers(self, discovery=False):
        from backend.provider.claude_code import auth_headers, is_oauth_token, resolve_credential
        if self.config.get("_model_api_key_override"):
            token = self.api_key
            oauth = is_oauth_token(token or "")
        else:
            from models.db import db
            token, oauth = resolve_credential(db, self.provider or "anthropic")
        if discovery and not token:
            raise ProviderAuthError("Not connected. Set up credentials first.")
        self.config["_oauth"] = oauth
        return auth_headers(token or "", oauth)

    def response_error(self, result):
        if result.get("type") != "error":
            return super().response_error(result)
        error = result.get("error", {})
        text = (f"{error.get('type', '')} {error.get('message', '')}"
                if isinstance(error, dict) else str(error)).lower()
        return error, any(word in text for word in (
            "rate_limit", "overloaded", "unavailable", "internal_error", "server_error",
        ))


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
