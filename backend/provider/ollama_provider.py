from backend.provider.openai_provider import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Ollama's native chat and model-list formats."""

    completion_path = "/chat"
    models_path = "/tags"

    def parse_models(self, data):
        return [{**m, "id": m.get("name", ""), "name": m.get("name", "")}
                for m in data.get("models", []) if isinstance(m, dict)]

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = {
                    "type": "function", "function": {"name": tool_choice}}

        return payload

    def normalize_response(self, result):
        ollama_message = result.get("message", {})
        ollama_content = ollama_message.get("content", "")
        ollama_reasoning = ollama_message.get("reasoning_content", "")
        prompt_eval = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)
        transformed_message = {
            "role": "assistant",
            "content": ollama_content,
        }
        if ollama_reasoning:
            transformed_message["reasoning_content"] = ollama_reasoning
        result = {
            "choices": [
                {
                    "message": transformed_message,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_eval,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval + eval_count,
            },
        }

        return result
