"""Contract for provider wire formats and model discovery."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests


class BaseProvider(ABC):
    """Adapt provider-specific payloads without changing the LLM client API."""

    completion_path = "/chat/completions"
    models_path = "/models"
    http_client = requests

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = (config.get("base_url") or "").rstrip("/")

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
