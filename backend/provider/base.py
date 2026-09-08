"""Contract for provider wire formats and model discovery."""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlsplit

import requests


class ReasoningCapabilities(TypedDict):
    efforts: List[str]
    default_effort: Optional[str]


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

    def fetch_models(self, headers: Dict[str, str], timeout: int = 15):
        return self.http_client.get(
            self.base_url + self.models_path,
            headers=self.discovery_headers(headers),
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
