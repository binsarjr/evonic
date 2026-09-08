"""Select the provider implementation from the effective model configuration."""

from urllib.parse import urlsplit

from backend.provider.anthropic_provider import AnthropicProvider
from backend.provider.cerebras_provider import CerebrasProvider
from backend.provider.codex_provider import CodexProvider
from backend.provider.deepseek_provider import DeepSeekProvider
from backend.provider.ollama_provider import OllamaProvider
from backend.provider.openai_provider import OpenAIProvider


def get_provider(config):
    """Select the adapter using the effective API format and endpoint."""
    api_format = config.get("api_format", "openai")
    if api_format == "codex":
        cls = CodexProvider
    elif api_format == "ollama" or "ollama.com" in (config.get("base_url") or ""):
        cls = OllamaProvider
    elif api_format == "anthropic":
        cls = AnthropicProvider
    elif api_format == "openai" and urlsplit(config.get("base_url") or "").hostname == "api.deepseek.com":
        cls = DeepSeekProvider
    elif api_format == "openai" and "cerebras.ai" in (config.get("base_url") or ""):
        cls = CerebrasProvider
    else:
        cls = OpenAIProvider
    return cls(config)
