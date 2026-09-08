import time
from typing import Any, Dict, List, Optional

import httpx

from backend.provider.common import _format_llm_error, _normalize_system_messages
from backend.provider.openai_provider import OpenAIProvider
from backend.provider.provider_auth_error import ProviderAuthError
from backend.provider.reasoning_capabilities import reasoning_capabilities
from evaluator.api_logger import log_api_call

CODEX_CLIENT_VERSION = "0.153.4"


class CodexProvider(OpenAIProvider):
    """Codex Responses API; CodexClient owns its SSE transport."""

    completion_path = "/responses"
    http_client = httpx
    reasoning_host = "chatgpt.com"

    def request_headers(self, discovery=False):
        from models.db import db
        from backend.provider.oauth_codex import get_valid_token
        token = get_valid_token(db, self.provider or "codex")
        if not token:
            raise ProviderAuthError("Not connected. Click Connect first.")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        max_tokens: Optional[int] = None,
        log_file: Optional[str] = None,
        tool_choice: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delegate chat completion to CodexClient (Responses API)."""
        messages = _normalize_system_messages(messages)
        from models.db import db as _db
        from backend.provider.oauth_codex import get_valid_token
        from backend.provider.codex_client import CodexClient

        start_time = time.time()
        token = get_valid_token(_db, self.provider or "codex")
        if not token:
            return {
                "response": {"error": _format_llm_error("auth_error")},
                "duration_ms": 0,
                "success": False,
                "error_type": "auth_error",
                "error_detail": "Codex OAuth token missing or expired. Reconnect in Settings.",
            }

        client = CodexClient(token, self.base_url)
        if max_tokens is None:
            max_tokens = self.max_tokens
        result = client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or 4096,
            temperature=temperature if temperature is not None else self.temperature,
            tools=tools,
            reasoning=bool(self.thinking),
            reasoning_effort=reasoning_effort,
            timeout=self.timeout or 120,
            tool_choice=tool_choice,
            service_tier=getattr(self, "service_tier", None),
        )
        duration_ms = int((time.time() - start_time) * 1000)

        if not result.get("success"):
            from evaluator.api_logger import log_api_call as _log_call
            _log_call(messages, None, duration_ms, error=result.get("error", ""), log_file=log_file)
            return {
                "response": {"error": _format_llm_error(result.get("error_type", "unknown_error"))},
                "duration_ms": duration_ms,
                "success": False,
                "error_type": result.get("error_type", "unknown_error"),
                "error_detail": result.get("error", ""),
            }

        # Extract content for logging
        choices = result["response"].get("choices", [])
        response_text = ""
        thinking_text = ""
        if choices:
            msg = choices[0].get("message", {})
            response_text = msg.get("content", "")
            thinking_text = msg.get("reasoning_content", "")
        log_api_call(messages, response_text, duration_ms, log_file=log_file, thinking=thinking_text or None)

        # Usage is mapped by CodexClient from the Responses API's
        # input_tokens/output_tokens. Hardcoded zeros here previously froze
        # the context monitor (guarded on prompt_tokens > 0) at whatever the
        # last non-codex model reported.
        usage = result["response"].get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens", 0) or 0
        reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
        usage_details_available = bool(prompt_details or completion_details)

        # Emit the shared usage event for consumers such as the Token Monitor.
        from backend.llm_usage_events import record_llm_usage
        record_llm_usage(
            model=result["response"].get("model") or self.model,
            provider=self.provider or "codex",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            usage_details_available=usage_details_available,
            duration_ms=duration_ms,
            messages=messages,
            response_text=response_text,
        )

        return {
            "response": result["response"],
            "request_payload": {"model": self.model, "messages": messages,
                                "tools": tools},
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "success": True,
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Codex via OAuth token."""
        try:
            from models.db import db as _db
            from backend.provider.oauth_codex import get_valid_token
            from backend.provider.codex_client import CodexClient

            provider = _db.get_provider(self.provider or "codex")
            if not provider:
                return {"success": False, "error": "Codex provider not found"}
            token = get_valid_token(_db, provider["id"])
            if not token:
                return {"success": False, "error": "Not connected to Codex. Complete OAuth flow first."}
            client = CodexClient(token, self.base_url)
            return client.test_connection()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_reasoning_capabilities(self, model, metadata=None):
        if not self.supports_reasoning_effort:
            return reasoning_capabilities()
        metadata = metadata or {}
        levels = metadata.get("supported_reasoning_levels")
        if not isinstance(levels, list):
            return reasoning_capabilities()
        return reasoning_capabilities(
            [item.get("effort") for item in levels if isinstance(item, dict)],
            metadata.get("default_reasoning_level"),
        )

    def apply_reasoning_effort(self, payload, effort):
        if effort is not None:
            payload.setdefault("reasoning", {})["effort"] = effort

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
