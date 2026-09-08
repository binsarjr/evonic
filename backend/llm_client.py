"""LLM client module for OpenAI-compatible API interactions.

Provides the LLMClient class for chat completions, thinking tag handling,
and response parsing. Supports multiple model backends and formats
including standard XML thinking tags, Gemma 4, and Qwen formats.

"""

import json
import re
from typing import Any, Dict, List, Optional

from backend.provider.factory import get_provider
from backend.provider.reasoning_effort_error import ReasoningEffortError

from backend.provider.common import (
    _format_llm_error, _normalize_system_messages, _split_trailing_think_close,
    strip_thinking_tags,
)


class LLMClient:
    """Client for OpenAI-compatible LLM chat completion APIs.

    Resolves configuration and delegates I/O to the provider contract.
    Shared response extraction stays independent of the selected provider.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize LLMClient with optional model_config.

        Args:
            model_config: Dict with keys: base_url, api_key, model_name, timeout,
                         thinking (bool), thinking_budget (int), max_tokens, temperature,
                         and optional service_tier and reasoning_effort.
                         If None, uses the default model from DB or config.py defaults.
        """
        self.provider = None
        self.reasoning_effort = None
        self.service_tier = model_config.get("service_tier") if model_config else None
        self._model_api_key_override = False
        if model_config:
            self._model_api_key_override = bool(model_config.get("api_key"))
            try:
                from models.db import db
                model_config = db.resolve_model_config(model_config)
            except Exception:
                pass
            self.provider = model_config.get("provider")
            self.base_url = model_config.get("base_url")
            self.api_key = model_config.get("api_key")
            self.model = model_config.get("model_name")
            self.timeout = model_config.get("timeout")
            self.thinking = model_config.get("thinking", False)
            self.thinking_budget = model_config.get("thinking_budget", 0)
            self.reasoning_effort = model_config.get("reasoning_effort")
            self.max_tokens = model_config.get("max_tokens")
            self.temperature = model_config.get("temperature")
            self.api_format = model_config.get("api_format", "openai")
        else:
            try:
                from models.db import db

                dm = db.get_default_model()
                if dm:
                    self._model_api_key_override = bool(dm.get("api_key"))
                    dm = db.resolve_model_config(dm)
                    self.provider = dm.get("provider")
                    self.base_url = dm.get("base_url")
                    self.api_key = dm.get("api_key")
                    self.model = dm.get("model_name")
                    self.timeout = dm.get("timeout")
                    self.thinking = bool(dm.get("thinking", False))
                    self.thinking_budget = int(dm.get("thinking_budget", 0))
                    self.reasoning_effort = dm.get("reasoning_effort")
                    self.max_tokens = dm.get("max_tokens")
                    self.temperature = dm.get("temperature")
                    self.api_format = dm.get("api_format", "openai")
                else:
                    self.base_url = None
                    self.api_key = None
                    self.model = None
                    self.timeout = None
                    self.thinking = False
                    self.thinking_budget = 0
                    self.max_tokens = None
                    self.temperature = None
                    self.api_format = "openai"
            except Exception:
                self.base_url = None
                self.api_key = None
                self.model = None
                self.timeout = None
                self.thinking = False
                self.thinking_budget = 0
                self.max_tokens = None
                self.temperature = None
                self.api_format = "openai"
        self._cached_model_name = None
        # Cache for global LLM settings (avoids repeated DB reads in hot path).
        # TTL-based, simple dict — intentionally lock-free (worst case: 1 extra DB read).
        # Optional per-call retry override. When set (not None), it takes
        # precedence over the global llm_max_retries setting. Callers that
        # need bounded latency (e.g. photo validation fallback chains) set
        # this to 0 so one slow provider cannot stall the whole request.
        self.max_retries: Optional[int] = None
        self._settings_cache = {}

    def _provider_adapter(self):
        """Build the provider from current settings, including per-call overrides."""
        fields = (
            'provider', 'base_url', 'api_key', 'api_format', 'timeout',
            'thinking', 'thinking_budget', 'reasoning_effort', 'max_tokens',
            'temperature', 'service_tier', 'max_retries',
        )
        config = {key: getattr(self, key, None) for key in fields}
        config.update({
            'model_name': getattr(self, 'model', None),
            '_model_api_key_override': getattr(self, '_model_api_key_override', False),
            '_cached_model_name': getattr(self, '_cached_model_name', None),
            '_settings_cache': getattr(self, '_settings_cache', {}),
        })
        return get_provider(config)

    def get_actual_model_name(self, force_refresh: bool = False) -> str:
        """Ask the provider for the actual model name, caching a successful lookup."""
        if self._cached_model_name and not force_refresh:
            return self._cached_model_name
        provider = self._provider_adapter()
        model = provider.get_actual_model_name(force_refresh)
        self._cached_model_name = provider._cached_model_name
        return model

    def test_connection(self) -> Dict[str, Any]:
        """Test the model endpoint through its provider contract."""
        return self._provider_adapter().test_connection()

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        max_tokens: Optional[int] = None,
        log_file: Optional[str] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenAI-compatible endpoint.

        Processes messages (normalizes quotes, injects thinking prompts for
        Gemma 4, merges multiple system messages), applies token budget
        adjustments for thinking mode, and retries on transient errors
        with exponential backoff.

        Args:
            messages: List of message dicts with role and content keys.
            tools: Optional list of tool definitions for function calling.
            temperature: Optional override for model temperature.
            enable_thinking: If True and model supports thinking, enables
                reasoning mode and applies the configured effort. False skips the effort
                override without changing the provider default. Defaults to True.
            max_tokens: Optional override for max output tokens. If None,
                uses self.max_tokens (doubled when thinking is active).
            log_file: Optional path for API call logging.
            tool_choice: Optional function name that the provider must call.

        Returns:
            Dict with response, duration_ms, token counts, success flag,
            and error_type/error_detail on failure.

        Note:
            Retries on 5xx errors, timeouts, and connection errors with
            exponential backoff (max 60s between retries). Configurable
            retry count via llm_max_retries setting (DB default: 5).
        """
        effort = None
        if enable_thinking and getattr(self, 'reasoning_effort', None) is not None:
            from models.db import db
            try:
                effort = db.validate_model_reasoning({
                    'provider': self.provider, 'base_url': self.base_url,
                    'api_format': self.api_format, 'model_name': self.model,
                    'reasoning_effort': self.reasoning_effort,
                })
            except ReasoningEffortError as e:
                return {'success': False, 'duration_ms': 0,
                        'response': {'error': _format_llm_error('configuration_error')},
                        'error_type': 'configuration_error', 'error_detail': str(e)}
        return self._provider_adapter().chat_completion(
            messages, tools=tools, temperature=temperature,
            enable_thinking=enable_thinking, max_tokens=max_tokens, log_file=log_file,
            tool_choice=tool_choice, reasoning_effort=effort,
        )

    def extract_content(
        self, response: Dict[str, Any], strip_thinking: bool = True
    ) -> str:
        """Extract text content from LLM response."""
        if not response.get("success"):
            error_type = response.get("error_type", "unknown_error")
            user_msg = _format_llm_error(error_type)
            # Include error_detail only for internal/admin contexts — never raw API responses
            error_detail = response.get("error_detail", "")
            if error_detail:
                return f"{user_msg}\n\nDetails: {error_detail}"
            return user_msg

        choices = response["response"].get("choices", [])
        if not choices:
            return "No response generated"

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return json.dumps({"tool_calls": tool_calls}, indent=2)

        content = message.get("content", "")
        if not content:
            content = message.get("reasoning_content", "")
        if not content:
            return "No content generated"

        if strip_thinking:
            cleaned, _ = strip_thinking_tags(content)
            return cleaned
        return content

    def extract_content_with_thinking(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract both thinking and final content from LLM response.

        Handles:
        1. llama.cpp --reasoning mode: thinking in message.reasoning_content
        2. Tag-based thinking: <think>...</think> or Gemma4/Qwen XML formats
        """
        if not response.get("success"):
            return {
                "content": self.extract_content(response),
                "thinking": None,
                "raw": None,
            }

        choices = response["response"].get("choices", [])
        if not choices:
            return {"content": "No response generated", "thinking": None, "raw": None}

        message = choices[0].get("message", {})
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content") or message.get("reasoning")
        tool_calls = message.get("tool_calls")

        if tool_calls:
            tool_content = json.dumps({"tool_calls": tool_calls}, indent=2)
            return {
                "content": tool_content,
                "thinking": (reasoning_content or "").strip() or None,
                "raw": tool_content,
                "tool_calls": tool_calls,
            }

        if content and "<|tool_call>" in content:
            from evaluator.gemma4_parser import (
                extract_gemma4_tool_calls,
                gemma4_tool_calls_to_openai_format,
            )

            gemma4_calls = extract_gemma4_tool_calls(content)
            if gemma4_calls:
                openai_calls = gemma4_tool_calls_to_openai_format(gemma4_calls)
                tool_content = json.dumps({"tool_calls": openai_calls}, indent=2)
                return {
                    "content": tool_content,
                    "thinking": reasoning_content,
                    "raw": content,
                    "tool_calls": openai_calls,
                }

        if content and "<tool_call>" in content:
            from evaluator.qwen_parser import (
                extract_qwen_tool_calls,
                qwen_tool_calls_to_openai_format,
                strip_qwen_tool_calls,
            )

            qwen_calls = extract_qwen_tool_calls(content)
            if qwen_calls:
                openai_calls = qwen_tool_calls_to_openai_format(qwen_calls)
                visible_content = strip_qwen_tool_calls(content)
                return {
                    "content": visible_content,
                    "thinking": (reasoning_content or "").strip() or None,
                    "raw": content,
                    "tool_calls": openai_calls,
                }

        reasoning_text = (reasoning_content or "").strip()
        embedded_final = None
        if reasoning_text and ("</think>" in reasoning_text or "</thinking>" in reasoning_text):
            reasoning_text, embedded_final = _split_trailing_think_close(reasoning_text)
        if reasoning_text:
            cleaned = strip_thinking_tags(content)[0] if content else ""
            if not cleaned and embedded_final:
                cleaned = embedded_final
            # Check for Qwen-style XML tool calls that may appear in
            # Fallback: when cleaned is empty and no embedded_final, the model
            # put its entire response in reasoning_content (e.g. Qwen via llama.cpp).
            if not cleaned and not embedded_final and not tool_calls:
                cleaned = reasoning_text
            # reasoning_content instead of content (common with Qwen-based models).
            # Two forms: (a) trailing after </think> or </thinking> in embedded_final,
            # (b) directly in reasoning_text when content is empty.
            xml_source = None
            if embedded_final and "<tool_call>" in embedded_final:
                xml_source = embedded_final
            elif not cleaned and reasoning_text and "<tool_call>" in reasoning_text:
                xml_source = reasoning_text
            if xml_source:
                from evaluator.qwen_parser import (
                    extract_qwen_tool_calls,
                    qwen_tool_calls_to_openai_format,
                    strip_qwen_tool_calls,
                )
                qwen_calls = extract_qwen_tool_calls(xml_source)
                if qwen_calls:
                    openai_calls = qwen_tool_calls_to_openai_format(qwen_calls)
                    visible_content = strip_qwen_tool_calls(xml_source)
                    return {
                        "content": visible_content,
                        "thinking": reasoning_text or None,
                        "raw": content,
                        "tool_calls": openai_calls,
                    }
            return {"content": cleaned, "thinking": reasoning_text, "raw": content}

        if content:
            cleaned, thinking = strip_thinking_tags(content)
            return {"content": cleaned, "thinking": thinking, "raw": content}

        return {"content": "No content generated", "thinking": None, "raw": None}

    def get_error_info(self, response: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Get error information from a failed response."""
        if response.get("success"):
            return None
        return {
            "type": response.get("error_type", "unknown"),
            "message": response["response"].get("error", "Unknown error")
            if isinstance(response.get("response"), dict)
            else str(response.get("response")),
            "detail": response.get("error_detail", ""),
            "duration_ms": response.get("duration_ms", 0),
        }

    def extract_tool_calls(
        self, response: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from LLM response."""
        if not response.get("success"):
            return None
        choices = response["response"].get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        return message.get("tool_calls")


# Global LLM client instance (initialized once at startup, uses DB default model)
llm_client = LLMClient()


def get_llm_client() -> LLMClient:
    """Create a fresh LLMClient that reads the latest default model from DB.

    Use this in request handlers when you need the current default model
    without restarting the server.  Each call creates a new LLMClient
    instance that queries db.get_default_model() at init time.
    """
    return LLMClient()
