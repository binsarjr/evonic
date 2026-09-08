"""Shared message normalization and response/error parsing."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from evaluator.gemma4_parser import is_gemma4_format, strip_gemma4_thinking

_LLM_ERROR_MESSAGES = {
    "configuration_error": "Invalid model reasoning effort. Refresh models or select Default in Settings.",
    "api_error": "The LLM API is temporarily unavailable.",
    "rate_limit_error": "API rate limit exceeded. Please wait and try again.",
    "auth_error": "API authentication failed. Check your API key.",
    "timeout_error": "Request timed out. The service may be slow or unavailable.",
    "request_timeout": "Request timed out. The service may be slow or unavailable.",
    "connection_error": "Cannot connect to the LLM server.",
    "context_length_error": "Conversation too long. Start a new session.",
    "generation_timeout": "The AI ran out of tokens before finishing. Try a shorter conversation.",
    "tool_call_json_error": "The AI generated an invalid tool call. Retrying with a correction.",
    "provider_error": "The LLM provider is experiencing issues. Please try again shortly.",
    "llm_error": "The LLM returned an error. Please try again.",
    "unknown_error": "An unexpected error occurred with the LLM service.",
    "parse_error": "The LLM provider returned an invalid response. Check the provider configuration.",
}


def _normalize_system_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a provider-safe message snapshot with system instructions first.

    Strict chat templates, including Qwen templates used by llama.cpp, reject
    any system-role message after the conversation begins. Consolidate every
    system message at index 0 while preserving the exact order and fields of
    all non-system messages. Caller-owned dictionaries and the input list are
    never mutated.
    """
    copied_messages = [message.copy() for message in messages]
    system_messages = [
        message for message in copied_messages
        if message.get("role") == "system"
    ]
    if not system_messages:
        return copied_messages

    non_system_messages = [
        message for message in copied_messages
        if message.get("role") != "system"
    ]
    merged_system = system_messages[0].copy()
    if len(system_messages) > 1:
        def _content_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if content is None:
                return ""
            return json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        merged_system["content"] = "\n\n".join(
            _content_text(message.get("content", ""))
            for message in system_messages
        )

    return [merged_system, *non_system_messages]


def _format_llm_error(error_type: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Format an LLM error type into a user-friendly message.

    Args:
        error_type: Internal error classification (e.g. 'api_error', 'timeout').
        context: Optional dict for appending non-sensitive context (e.g. session_id).

    Returns:
        User-friendly error string — never exposes API keys or raw API responses.
    """
    user_msg = _LLM_ERROR_MESSAGES.get(error_type, _LLM_ERROR_MESSAGES["unknown_error"])
    if context:
        if context.get("session_id"):
            user_msg += f" (Session: {context['session_id']})"
    return user_msg


def _parse_sse_error_frame(raw_text: str) -> Optional[Dict[str, str]]:
    """Extract the error type/message from an SSE ``event: error`` frame.

    Some OpenAI-compatible gateways (e.g. cavoti) return transient failures as
    Server-Sent-Events even for non-streaming requests (``"stream": false``)::

        event: error
        data: {"error": {"type": "service_unavailable",
                         "message": "Service temporarily unavailable, please retry later"}}

    Returns ``{"type": ..., "message": ...}`` when the body is an SSE error
    frame, otherwise ``None``.
    """
    if not raw_text or "event:" not in raw_text or "data:" not in raw_text:
        return None
    event_name = None
    data_chunks = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("event:"):
            event_name = stripped[len("event:"):].strip()
        elif stripped.startswith("data:"):
            data_chunks.append(stripped[len("data:"):].strip())
    if event_name != "error" or not data_chunks:
        return None
    try:
        payload = json.loads("\n".join(data_chunks))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error_obj = payload.get("error", payload)
    if isinstance(error_obj, dict):
        return {
            "type": str(error_obj.get("type", "")),
            "message": str(error_obj.get("message", "")),
        }
    return {"type": "", "message": str(error_obj)}


def _split_trailing_think_close(text: str) -> Tuple[str, Optional[str]]:
    """Split text on </think> or </thinking> marker — returns (actual_thinking, trailing_final_response).

    Some backends accidentally include </think> and the final response inside
    the reasoning_content field. This extracts the trailing response.
    Returns (original_text, None) if no close tag found or nothing follows it.
    """
    if not text:
        return text, None
    close_tag = None
    if "</thinking>" in text:
        close_tag = "</thinking>"
    elif "</think>" in text:
        close_tag = "</think>"
    else:
        return text, None
    parts = text.split(close_tag, 1)
    actual = parts[0].strip()
    trailing = parts[1].strip() if len(parts) > 1 else ""
    return actual or text, trailing or None


def strip_thinking_tags(content: str) -> Tuple[str, Optional[str]]:
    """
    Strip thinking tags from content with auto-format detection.

    Supports:
    - Standard: <think>...</think> or <thinking>...</thinking>
    - Gemma 4: <|channel>thought...<channel|>

    Returns:
        Tuple of (cleaned_content, thinking_content)
    """
    if not content:
        return content, None

    if is_gemma4_format(content):
        return strip_gemma4_thinking(content)

    thinking_pattern = r"<(?:think|thinking)>(.*?)</(?:think|thinking)>"
    thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)
    cleaned = re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()
    thinking_content = "\n".join(thinking_matches) if thinking_matches else None

    # Gemma4-12B tokenizer produces "** text**" (space after opening **)
    # for certain tokens, breaking markdown bold rendering.
    # Collapse the space: "** text**" → "**text**"
    _fix_bold = lambda s: re.sub(r'(^|\s)\*\* ', r'\1**', s) if s else s
    cleaned = _fix_bold(cleaned)

    # Edge case: model put the final response inside <think>/<thinking> tags, leaving cleaned empty.
    # Check if thinking_content itself has an embedded </think>/</thinking> that signals end-of-thinking.
    if not cleaned and thinking_content:
        actual_thinking, embedded_final = _split_trailing_think_close(thinking_content)
        if embedded_final:
            return _fix_bold(embedded_final), actual_thinking

    # Fallback: handle missing opening <think>/<thinking> tag (common with vLLM)
    if not thinking_content and ("</think>" in content or "</thinking>" in content):
        close_tag = "</thinking>" if "</thinking>" in content else "</think>"
        parts = content.split(close_tag, 1)
        thinking_text = parts[0].strip()
        cleaned_text = parts[1].strip() if len(parts) > 1 else ""
        if thinking_text:
            return _fix_bold(cleaned_text) or "", thinking_text
        return _fix_bold(cleaned_text) or content.replace(close_tag, "").strip(), None

    return cleaned, thinking_content


