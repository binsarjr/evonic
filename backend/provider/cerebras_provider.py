from backend.provider.openai_provider import OpenAIProvider


class CerebrasProvider(OpenAIProvider):
    """Cerebras rejects non-standard reasoning fields in input messages."""

    def prepare_messages(self, messages, enable_thinking):
        processed = super().prepare_messages(messages, enable_thinking)
        for message in processed:
            message.pop("reasoning_content", None)
        return processed
