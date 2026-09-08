from backend.provider.openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """Direct DeepSeek API, reusing the OpenAI-compatible wire format."""
