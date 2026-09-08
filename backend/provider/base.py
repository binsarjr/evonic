"""Contract for provider wire formats and model discovery."""

from abc import ABC, abstractmethod
import re
import time
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlsplit

import requests
import httpx

from backend.provider.common import _format_llm_error


class ReasoningCapabilities(TypedDict):
    efforts: List[str]
    default_effort: Optional[str]


class ProviderAuthError(ValueError):
    """A provider requires credentials before it can send the request."""


class ReasoningEffortError(ValueError):
    """The selected effort is not supported by the effective provider/model."""


def reasoning_capabilities(efforts=(), default=None) -> ReasoningCapabilities:
    """Normalize provider metadata without inventing a universal effort enum."""
    values = list(dict.fromkeys(
        value for value in efforts
        if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value)
    ))
    return {"efforts": values, "default_effort": default if default in values else None}


def validate_reasoning_effort(effort, capabilities):
    if effort is None or effort == "":
        return None
    if not isinstance(effort, str) or effort not in capabilities["efforts"]:
        raise ReasoningEffortError(
            "Reasoning effort is not supported by this provider/model. "
            "Fetch models to refresh support, or select Default (provider)."
        )
    return effort


class BaseProvider(ABC):
    """Adapt provider-specific payloads without changing the LLM client API."""

    completion_path = "/chat/completions"
    models_path = "/models"
    http_client = requests
    reasoning_host = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = (config.get("base_url") or "").rstrip("/")
        self.provider = config.get("provider") or config.get("id")
        self.api_key = config.get("api_key")
        self.model = config.get("model_name")
        self.timeout = config.get("timeout")
        self.thinking = config.get("thinking", False)
        self.thinking_budget = config.get("thinking_budget", 0)
        self.max_tokens = config.get("max_tokens")
        self.temperature = config.get("temperature")
        self.service_tier = config.get("service_tier")
        self.max_retries = config.get("max_retries")
        self._cached_model_name = config.get("_cached_model_name")
        self._settings_cache = config.get("_settings_cache", {})

    def _get_cached_setting(self, cache_key, db_func, *args):
        now = time.time()
        if now - self._settings_cache.get("_timestamp", 0) > 30:
            self._settings_cache.clear()
            self._settings_cache["_timestamp"] = now
        if cache_key not in self._settings_cache:
            self._settings_cache[cache_key] = db_func(*args)
        return self._settings_cache[cache_key]

    @abstractmethod
    def chat_completion(
        self, messages: List[Dict[str, Any]], tools=None, temperature=None,
        enable_thinking=True, max_tokens=None, log_file=None, tool_choice=None,
        reasoning_effort=None,
    ) -> Dict[str, Any]:
        """Return the existing completion envelope with response, usage and errors."""
        pass

    @abstractmethod
    def request_headers(self, discovery=False) -> Dict[str, str]:
        """Resolve credentials and construct headers for this provider."""
        pass

    def get_actual_model_name(self, force_refresh=False):
        return self.model

    def test_connection(self) -> Dict[str, Any]:
        try:
            response = self.fetch_models(timeout=10)
            if response.status_code == 200:
                return {"success": True, "message": f"Connected to {self.base_url}",
                        "available_models": len(self.parse_models(response.json()))}
            return {"success": False, "error": _format_llm_error("api_error")}
        except ProviderAuthError as e:
            return {"success": False, "error": str(e)}
        except (requests.exceptions.Timeout, httpx.TimeoutException):
            return {"success": False, "error": _format_llm_error("timeout_error")}
        except (requests.exceptions.ConnectionError, httpx.ConnectError):
            return {"success": False, "error": _format_llm_error("connection_error")}
        except Exception:
            return {"success": False, "error": _format_llm_error("unknown_error")}

    @property
    def supports_reasoning_effort(self) -> bool:
        endpoint = urlsplit(self.base_url)
        return bool(self.reasoning_host and endpoint.scheme == "https"
                    and endpoint.hostname == self.reasoning_host)

    def get_reasoning_capabilities(self, model: str, metadata=None) -> ReasoningCapabilities:
        return reasoning_capabilities()

    def apply_reasoning_effort(self, payload, effort):
        if effort is not None:
            raise ReasoningEffortError("Reasoning effort is not supported by this provider.")

    def discovery_params(self) -> Dict[str, Any]:
        return {}

    def discovery_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        return dict(headers)

    def fetch_models(self, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
        return self.http_client.get(
            self.base_url + self.models_path,
            headers=self.discovery_headers(self.request_headers(discovery=True) if headers is None else headers),
            params=self.discovery_params(),
            timeout=timeout,
        )

    @abstractmethod
    def parse_models(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return model entries with an id and a display name."""
        pass

    @abstractmethod
    def build_payload(
        self, model: str, messages: List[Dict[str, Any]], max_tokens: Optional[int],
        temperature: Optional[float] = None, tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None, oauth: bool = False,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return an OpenAI-chat-style response for the shared runtime."""
        pass
