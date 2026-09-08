from backend.provider.openai_provider import OpenAIProvider
from backend.provider.reasoning_capabilities import reasoning_capabilities


class DeepSeekProvider(OpenAIProvider):
    """Direct DeepSeek API, reusing the OpenAI-compatible wire format."""

    reasoning_host = "api.deepseek.com"

    def get_reasoning_capabilities(self, model, metadata=None):
        # /models only returns IDs. Keep documented support here until it gains capabilities.
        # https://api-docs.deepseek.com/api/create-chat-completion/
        if self.supports_reasoning_effort and model in {
            "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp",
        }:
            return reasoning_capabilities(["low", "high", "max"], "high")
        return reasoning_capabilities()

    def apply_reasoning_effort(self, payload, effort):
        if effort is not None:
            payload["reasoning_effort"] = effort
