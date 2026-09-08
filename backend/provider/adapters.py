"""Provider adapters for the API formats supported by Evonic."""

import json
import time
from typing import Any, Dict, List, Optional

import requests
from urllib.parse import urlsplit

import httpx

from backend.provider.base import BaseProvider, ProviderAuthError, reasoning_capabilities
from backend.provider.common import (
    _format_llm_error, _normalize_system_messages, _parse_sse_error_frame, strip_thinking_tags,
)
from backend.normalizer import normalize_llm_text
from evaluator.api_logger import log_api_call

CODEX_CLIENT_VERSION = "0.153.4"


def _convert_multimodal_to_claude(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style multimodal content blocks to Anthropic format.

    Handles image_url, input_audio, and video_url content types.
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_parts = []
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            ptype = part.get("type")
            if ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    try:
                        header, b64data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                    except (ValueError, IndexError):
                        media_type, b64data = "image/jpeg", url
                    new_parts.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64data},
                    })
                else:
                    new_parts.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            elif ptype == "input_audio":
                # Convert to Claude audio format if supported; otherwise pass through
                audio_info = part.get("input_audio") or {}
                b64data = audio_info.get("data", "")
                fmt = audio_info.get("format", "wav")
                # Map format to MIME type for Claude
                fmt_to_mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
                               "mpeg": "audio/mpeg", "webm": "audio/webm"}
                media_type = fmt_to_mime.get(fmt, f"audio/{fmt}")
                new_parts.append({
                    "type": "audio",
                    "source": {"type": "base64", "media_type": media_type, "data": b64data},
                })
            elif ptype == "video_url":
                # Convert video data URL to Claude format; pass through external URLs
                url = (part.get("video_url") or {}).get("url", "")
                if url.startswith("data:"):
                    try:
                        header, b64data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                    except (ValueError, IndexError):
                        media_type, b64data = "video/mp4", url
                    new_parts.append({
                        "type": "video",
                        "source": {"type": "base64", "media_type": media_type, "data": b64data},
                    })
                else:
                    new_parts.append({
                        "type": "video",
                        "source": {"type": "url", "url": url},
                    })
            else:
                new_parts.append(part)
        result.append({**msg, "content": new_parts})
    return result


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible transport, also used by gateways and custom endpoints."""

    def request_headers(self, discovery=False):
        headers = {"Content-Type": "application/json"}
        if not discovery:
            headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def response_error(self, result):
        if "error" not in result:
            return None
        error = result["error"]
        code = error.get("code") if isinstance(error, dict) else None
        message = (error.get("message", "") if isinstance(error, dict) else str(error)).lower()
        transient = ((isinstance(code, int) and code >= 500)
                     or str(code) in ("429", "503", "502", "529", "500")
                     or any(word in message for word in ("provider", "overloaded", "unavailable")))
        return error, transient

    def get_actual_model_name(self, force_refresh: bool = False) -> str:
        """Get the actual model name from the remote endpoint.

        For llama.cpp servers, fetches from /props endpoint.
        Falls back to configured model name if endpoint is unavailable.
        """
        if self._cached_model_name and not force_refresh:
            return self._cached_model_name

        try:
            props_url = f"{self.base_url.rstrip('/v1')}/props"
            response = requests.get(props_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "model_alias" in data:
                    self._cached_model_name = data["model_alias"]
                    return self._cached_model_name
        except Exception:
            pass

        # Only trust /v1/models when exactly one model is returned (local servers).
        try:
            response = self.fetch_models(timeout=5)
            if response.status_code == 200:
                models = self.parse_models(response.json())
                if len(models) == 1:
                    key = "id" if "id" in models[0] else "name"
                    self._cached_model_name = models[0].get(key, self.model)
                    return self._cached_model_name
        except Exception:
            pass

        return self.model

    def prepare_messages(self, messages, enable_thinking):
        # Normalize quote-like characters in all outgoing messages so the LLM is
        # less likely to reproduce them verbatim inside tool call argument strings,
        # which would cause llama.cpp --jinja to fail JSON parsing.
        processed_messages = []
        thinking_injected = False
        actual_model = self._cached_model_name or self.model or ""
        model_lower = actual_model.lower()
        is_gemma4 = (
            "gemma-4" in model_lower
            or "gemma4" in model_lower
            or "gemma-4-base" in model_lower
        )

        for msg in messages:
            new_msg = msg.copy()
            if isinstance(new_msg.get("content"), str):
                new_msg["content"] = normalize_llm_text(new_msg["content"])
            if not thinking_injected and is_gemma4 and enable_thinking:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "system") and content:
                    new_msg["content"] = "<|think|>\n" + new_msg["content"]
                    thinking_injected = True
            processed_messages.append(new_msg)

        # Handle reasoning_content field based on thinking mode.
        # Some models (e.g. DeepSeek-v4) produce reasoning_content automatically
        # even without explicit thinking mode. Detect this by checking if any
        # assistant message already carries reasoning_content — if so, preserve
        # it so the API receives it back on the next call.
        _has_reasoning = any(
            _msg.get("reasoning_content")
            for _msg in processed_messages
            if _msg.get("role") == "assistant"
        )
        if self.thinking or _has_reasoning:
            # Ensure every assistant message has the field (some APIs require it
            # even on turns where the model produced no reasoning).
            for _msg in processed_messages:
                if _msg.get("role") == "assistant" and "reasoning_content" not in _msg:
                    _msg["reasoning_content"] = ""
        else:
            # No thinking configured and no reasoning in history — strip the
            # field so APIs that reject unknown fields are not affected.
            for _msg in processed_messages:
                _msg.pop("reasoning_content", None)

        # Claude API uses {"type":"image","source":{...}} instead of OpenAI's image_url format.
        if (self.model or "").lower().startswith("claude") or "anthropic.com" in self.base_url:
            processed_messages = _convert_multimodal_to_claude(processed_messages)

        return processed_messages

    def chat_completion(
        self, messages, tools=None, temperature=None, enable_thinking=True,
        max_tokens=None, log_file=None, tool_choice=None, reasoning_effort=None,
    ):
        """Run the shared HTTP request/retry path using provider format hooks."""
        url = self.base_url + self.completion_path
        if max_tokens is None:
            max_tokens = self.max_tokens
        # When thinking is active, the model's internal chain-of-thought consumes tokens
        # from the same max_tokens budget before producing any output. Double the budget
        # so actual output isn't crowded out by heavy reasoning. Only applied when thinking
        # is enabled and no explicit thinking_budget cap is configured (thinking_budget > 0
        # means the caller already sized the budget intentionally).
        if self.thinking and enable_thinking and not self.thinking_budget:
            max_tokens = max_tokens * 2
        try:
            from models.db import db as _db

            _ctx_len = int(self._get_cached_setting("llm_context_length", _db.get_setting, "llm_context_length", 0) or 0)
            _prompt_buf = int(self._get_cached_setting("llm_prompt_buffer", _db.get_setting, "llm_prompt_buffer", 2048) or 2048)
            if _ctx_len > 0:
                max_tokens = min(max_tokens, _ctx_len - _prompt_buf)
        except Exception:
            pass

        # Caller > model setting > omit (let server decide)
        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )

        processed_messages = self.prepare_messages(_normalize_system_messages(messages), enable_thinking)
        headers = self.request_headers()
        payload = self.build_payload(
            self.model, processed_messages, max_tokens, effective_temperature,
            tools, tool_choice, oauth=self.config.get("_oauth", False),
        )
        self.apply_reasoning_effort(payload, reasoning_effort)

        try:
            from models.db import db as _db

            _val = self._get_cached_setting("llm_max_retries", _db.get_setting, "llm_max_retries", None)
            max_retries = self.max_retries if self.max_retries is not None else (int(_val) if _val is not None else 5)
        except Exception:
            max_retries = self.max_retries if self.max_retries is not None else 5
        last_error_result = None

        for attempt in range(1 + max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    url, json=payload, headers=headers, timeout=(10, self.timeout)
                )
                duration_ms = int((time.time() - start_time) * 1000)

                if response.status_code >= 500:
                    raw_text = response.text
                    # llama.cpp returns 500 when it cannot parse tool call arguments
                    # as JSON (e.g. unescaped quotes from Jinja templates). Retrying
                    # regenerates the identical broken call — return a distinct error
                    # type so the caller can inject a correction prompt instead.
                    if "Failed to parse tool call arguments" in raw_text:
                        error_msg = f"LLM API server error: {response.status_code} - {raw_text[:300]}"
                        log_api_call(
                            messages,
                            None,
                            duration_ms,
                            error=error_msg,
                            log_file=log_file,
                        )
                        user_msg = _format_llm_error("tool_call_json_error")
                        return {
                            "response": {"error": user_msg},
                            "duration_ms": duration_ms,
                            "success": False,
                            "error_type": "tool_call_json_error",
                            "error_detail": error_msg,
                        }
                    # Generic server error — retryable with exponential backoff
                    error_msg = f"LLM API server error: {response.status_code} - {raw_text[:200]}"
                    log_api_call(
                        messages,
                        None,
                        duration_ms,
                        error=f"[attempt {attempt + 1}/{1 + max_retries}] {error_msg}",
                        log_file=log_file,
                    )
                    last_error_result = {
                        "response": {"error": _format_llm_error("api_error")},
                        "duration_ms": duration_ms,
                        "success": False,
                        "error_type": "api_error",
                        "error_detail": error_msg,
                    }
                    if attempt < max_retries:
                        time.sleep(min(2 ** (attempt + 1), 60))
                        continue
                    return last_error_result

                if response.status_code != 200:
                    error_msg = (
                        f"LLM API error: {response.status_code} - {response.text[:200]}"
                    )
                    log_api_call(
                        messages, None, duration_ms, error=error_msg, log_file=log_file
                    )
                    return {
                        "response": {"error": _format_llm_error("api_error")},
                        "duration_ms": duration_ms,
                        "success": False,
                        "error_type": "api_error",
                        "error_detail": error_msg,
                    }

                result = response.json()

                failure = self.response_error(result)
                if failure is not None:
                    error_obj, is_transient = failure
                    error_detail = str(error_obj)
                    log_api_call(
                        messages,
                        None,
                        duration_ms,
                        error=f"[attempt {attempt + 1}/{1 + max_retries}] {error_detail}"
                        if is_transient
                        else error_detail,
                        log_file=log_file,
                    )
                    _et = "provider_error" if is_transient else "llm_error"
                    last_error_result = {
                        "response": result,
                        "duration_ms": duration_ms,
                        "success": False,
                        "error_type": _et,
                        "error_detail": error_detail,
                    }
                    if is_transient and attempt < max_retries:
                        time.sleep(min(2 ** (attempt + 1), 60))
                        continue
                    last_error_result["response"] = {"error": _format_llm_error(_et)}
                    return last_error_result

                result = self.normalize_response(result)
                choices = result.get("choices", [])
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    reasoning = (
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    )

                    if finish_reason == "length" and not content:
                        error_detail = f"Generation hit max_tokens limit ({payload['max_tokens']}) without producing final answer. Reasoning length: {len(reasoning)} chars."
                        log_api_call(
                            messages,
                            None,
                            duration_ms,
                            error=error_detail,
                            log_file=log_file,
                        )
                        user_msg = _format_llm_error("generation_timeout")
                        return {
                            "response": {"error": user_msg},
                            "duration_ms": duration_ms,
                            "success": False,
                            "error_type": "generation_timeout",
                            "error_detail": error_detail,
                        }

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get(
                    "total_tokens", prompt_tokens + completion_tokens
                )
                prompt_details = usage.get("prompt_tokens_details") or {}
                completion_details = usage.get("completion_tokens_details") or {}
                cached_tokens = prompt_details.get("cached_tokens", 0) or 0
                reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
                usage_details_available = bool(prompt_details or completion_details)

                response_text = ""
                thinking_text = ""
                if choices:
                    msg = choices[0].get("message", {})
                    raw_content = msg.get("content", "") or ""
                    thinking_text = (
                        msg.get("reasoning_content") or msg.get("reasoning") or ""
                    )
                    if thinking_text:
                        response_text = raw_content
                    else:
                        response_text, thinking_text = strip_thinking_tags(raw_content)
                        thinking_text = thinking_text or ""
                log_api_call(
                    messages,
                    response_text,
                    duration_ms,
                    log_file=log_file,
                    thinking=thinking_text or None,
                )

                # Generic usage telemetry — emits an 'llm_usage' event any plugin
                # can observe. Never disturbs the LLM path (record_llm_usage swallows).
                from backend.llm_usage_events import record_llm_usage
                record_llm_usage(
                    model=self._cached_model_name or self.model,
                    provider=self.provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    usage_details_available=usage_details_available,
                    duration_ms=duration_ms,
                    messages=messages,
                    response_text=response_text,
                )

                return {
                    "response": result,
                    "request_payload": payload,  # byte-exact body POSTed to provider
                    "duration_ms": duration_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "success": True,
                }

            except requests.exceptions.Timeout:
                elapsed_ms = int((time.time() - start_time) * 1000)
                log_api_call(
                    messages,
                    None,
                    elapsed_ms,
                    error=f"[attempt {attempt + 1}/{1 + max_retries}] Timeout after {self.timeout}s",
                    log_file=log_file,
                )
                last_error_result = {
                    "response": {"error": _format_llm_error("request_timeout")},
                    "duration_ms": elapsed_ms,
                    "success": False,
                    "error_type": "request_timeout",
                    "error_detail": f"HTTP request timed out after {self.timeout} seconds.",
                }
                if attempt < max_retries:
                    time.sleep(min(2 ** (attempt + 1), 60))
                    continue
                return last_error_result

            except requests.exceptions.ConnectionError as e:
                log_api_call(
                    messages,
                    None,
                    0,
                    error=f"[attempt {attempt + 1}/{1 + max_retries}] Connection error: {str(e)[:100]}",
                    log_file=log_file,
                )
                last_error_result = {
                    "response": {"error": _format_llm_error("connection_error")},
                    "duration_ms": 0,
                    "success": False,
                    "error_type": "connection_error",
                    "error_detail": f"Could not connect to LLM server at {self.base_url}.",
                }
                # Connection refused/unreachable means the server is down — the
                # full llm_max_retries exponential backoff (60s+) won't revive
                # it. Retry once with a short wait, then return so the caller
                # (llm_loop) can move to its own retry/fallback path quickly.
                if attempt < min(max_retries, 1):
                    time.sleep(2)
                    continue
                return last_error_result

            except json.JSONDecodeError:
                elapsed_ms = (
                    int((time.time() - start_time) * 1000)
                    if "start_time" in locals()
                    else 0
                )
                raw_text = getattr(response, "text", "(no response)")
                raw_snippet = raw_text[:500]

                # Some gateways return transient failures as SSE error frames
                # (`event: error` / `data: {...}`) even when the request is
                # non-streaming. Extract the embedded error and classify
                # transient upstream errors as provider_error so callers can
                # retry or fall back to the next configured model.
                sse_error = _parse_sse_error_frame(raw_text)
                if sse_error is not None:
                    err_type = (sse_error.get("type") or "").lower()
                    err_message = sse_error.get("message") or ""
                    err_blob = (err_type + ": " + err_message).strip(": ")
                    is_transient = (
                        "unavailable" in err_type
                        or "overloaded" in err_type
                        or "internal" in err_type
                        or "rate_limit" in err_type
                        or "timeout" in err_type
                        or "server_error" in err_type
                        or "unavailable" in err_message.lower()
                        or "overloaded" in err_message.lower()
                        or "retry later" in err_message.lower()
                    )
                    error_type = "provider_error" if is_transient else "parse_error"
                    error_detail = (
                        f"LLM provider returned SSE error event: "
                        f"{err_blob or '(unknown error)'}. "
                        f"Raw response: {raw_snippet}"
                    )
                    log_api_call(
                        messages,
                        None,
                        elapsed_ms,
                        error=(
                            f"[attempt {attempt + 1}/{1 + max_retries}] "
                            f"SSE error event: {err_blob or raw_snippet}"
                        ),
                        log_file=log_file,
                    )
                    last_error_result = {
                        "response": {"error": _format_llm_error(error_type)},
                        "duration_ms": elapsed_ms,
                        "success": False,
                        "error_type": error_type,
                        "error_detail": error_detail,
                    }
                    # Transient upstream error — retry once with a short wait,
                    # then return so the caller (llm_loop / describe_image) can
                    # move to its own retry/fallback path quickly.
                    if is_transient and attempt < min(max_retries, 1):
                        time.sleep(2)
                        continue
                    return last_error_result

                log_api_call(
                    messages,
                    None,
                    elapsed_ms,
                    error=f"JSON decode failed. Raw response: {raw_snippet}",
                    log_file=log_file,
                )
                return {
                    "response": {"error": _format_llm_error("parse_error")},
                    "duration_ms": elapsed_ms,
                    "success": False,
                    "error_type": "parse_error",
                    "error_detail": f"Received invalid (non-JSON) response from LLM server. Raw response: {raw_snippet}",
                }

            except Exception as e:
                elapsed_ms = (
                    int((time.time() - start_time) * 1000)
                    if "start_time" in locals()
                    else 0
                )
                log_api_call(
                    messages, None, elapsed_ms, error=str(e)[:200], log_file=log_file
                )
                return {
                    "response": {"error": _format_llm_error("unknown_error")},
                    "duration_ms": elapsed_ms,
                    "success": False,
                    "error_type": "unknown_error",
                    "error_detail": str(e),
                }

        return last_error_result

    def parse_models(self, data):
        rows = data.get("data", data.get("models", []))
        if not isinstance(rows, list):
            return []
        return [{**m, "id": m.get("id", ""), "name": m.get("id", "")}
                for m in rows if isinstance(m, dict)]

    def normalize_response(self, data):
        return data

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = {
                    "type": "function", "function": {"name": tool_choice}}

        return payload


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


class AnthropicProvider(OpenAIProvider):
    """Anthropic Messages payloads and responses."""

    completion_path = "/messages"
    reasoning_host = "api.anthropic.com"

    def request_headers(self, discovery=False):
        from backend.provider.claude_code import auth_headers, is_oauth_token, resolve_credential
        if self.config.get("_model_api_key_override"):
            token = self.api_key
            oauth = is_oauth_token(token or "")
        else:
            from models.db import db
            token, oauth = resolve_credential(db, self.provider or "anthropic")
        if discovery and not token:
            raise ProviderAuthError("Not connected. Set up credentials first.")
        self.config["_oauth"] = oauth
        return auth_headers(token or "", oauth)

    def response_error(self, result):
        if result.get("type") != "error":
            return super().response_error(result)
        error = result.get("error", {})
        text = (f"{error.get('type', '')} {error.get('message', '')}"
                if isinstance(error, dict) else str(error)).lower()
        return error, any(word in text for word in (
            "rate_limit", "overloaded", "unavailable", "internal_error", "server_error",
        ))

    def get_reasoning_capabilities(self, model, metadata=None):
        if not self.supports_reasoning_effort:
            return reasoning_capabilities()
        capabilities = (metadata or {}).get("capabilities")
        effort = capabilities.get("effort") if isinstance(capabilities, dict) else None
        if not isinstance(effort, dict) or not effort.get("supported"):
            return reasoning_capabilities()
        # The Models API advertises support per level, including model-specific max/xhigh.
        levels = [level for level in ("low", "medium", "high", "xhigh", "max")
                  if isinstance(effort.get(level), dict) and effort[level].get("supported")]
        return reasoning_capabilities(levels, "high")

    def apply_reasoning_effort(self, payload, effort):
        if effort is not None:
            payload.setdefault("output_config", {})["effort"] = effort

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        # Anthropic API uses /messages endpoint with different payload structure.
        # Extract system messages into top-level "system" field.
        system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]
        payload = {
            "model": model,
            "messages": non_system_msgs,
            "max_tokens": max_tokens,
        }
        if system_msgs:
            payload["system"] = "\n\n".join(system_msgs) if len(system_msgs) > 1 else system_msgs[0]
        if oauth:
            from backend.provider.claude_code import SYSTEM_PREFIX
            payload["system"] = SYSTEM_PREFIX + (
                "\n\n" + payload["system"] if payload.get("system") else ""
            )
        if temperature is not None:
            payload["temperature"] = temperature
        # Transform OpenAI tools -> Anthropic tools format
        if tools:
            anthropic_tools = []
            for tool in tools:
                fn = tool.get("function", {})
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
            payload["tools"] = anthropic_tools
            payload["tool_choice"] = (
                {"type": "tool", "name": tool_choice}
                if tool_choice else {"type": "auto"})

        return payload

    def normalize_response(self, result):
        anthropic_content = result.get("content", [])
        anthropic_stop = result.get("stop_reason", "end_turn")
        anthropic_usage = result.get("usage", {})

        # Map stop_reason to finish_reason
        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        mapped_finish = stop_map.get(anthropic_stop, "stop")

        # Parse content blocks
        text_parts = []
        tool_calls_list = []
        for block in anthropic_content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls_list.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

        transformed_message = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if tool_calls_list:
            transformed_message["tool_calls"] = tool_calls_list

        # Map usage from Anthropic to OpenAI field names
        anthropic_input = anthropic_usage.get("input_tokens", 0)
        anthropic_output = anthropic_usage.get("output_tokens", 0)
        anthropic_cached = anthropic_usage.get("cache_read_input_tokens", 0) or 0
        result = {
            "choices": [
                {
                    "message": transformed_message,
                    "finish_reason": mapped_finish,
                }
            ],
            "usage": {
                "prompt_tokens": anthropic_input,
                "completion_tokens": anthropic_output,
                "total_tokens": anthropic_input + anthropic_output,
                "prompt_tokens_details": {
                    "cached_tokens": anthropic_cached,
                },
            },
        }

        return result


class CodexProvider(OpenAIProvider):
    """Codex Responses API; CodexClient owns its SSE transport."""

    completion_path = "/responses"
    http_client = httpx
    reasoning_host = "chatgpt.com"

    def request_headers(self, discovery=False):
        from models.db import db
        from backend.provider.oauth_codex import get_valid_token
        token = get_valid_token(db, self.provider or "codex")
        if not token:
            raise ProviderAuthError("Not connected. Click Connect first.")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        max_tokens: Optional[int] = None,
        log_file: Optional[str] = None,
        tool_choice: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delegate chat completion to CodexClient (Responses API)."""
        messages = _normalize_system_messages(messages)
        from models.db import db as _db
        from backend.provider.oauth_codex import get_valid_token
        from backend.provider.codex_client import CodexClient

        start_time = time.time()
        token = get_valid_token(_db, self.provider or "codex")
        if not token:
            return {
                "response": {"error": _format_llm_error("auth_error")},
                "duration_ms": 0,
                "success": False,
                "error_type": "auth_error",
                "error_detail": "Codex OAuth token missing or expired. Reconnect in Settings.",
            }

        client = CodexClient(token, self.base_url)
        if max_tokens is None:
            max_tokens = self.max_tokens
        result = client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or 4096,
            temperature=temperature if temperature is not None else self.temperature,
            tools=tools,
            reasoning=bool(self.thinking),
            reasoning_effort=reasoning_effort,
            timeout=self.timeout or 120,
            tool_choice=tool_choice,
            service_tier=getattr(self, "service_tier", None),
        )
        duration_ms = int((time.time() - start_time) * 1000)

        if not result.get("success"):
            from evaluator.api_logger import log_api_call as _log_call
            _log_call(messages, None, duration_ms, error=result.get("error", ""), log_file=log_file)
            return {
                "response": {"error": _format_llm_error(result.get("error_type", "unknown_error"))},
                "duration_ms": duration_ms,
                "success": False,
                "error_type": result.get("error_type", "unknown_error"),
                "error_detail": result.get("error", ""),
            }

        # Extract content for logging
        choices = result["response"].get("choices", [])
        response_text = ""
        thinking_text = ""
        if choices:
            msg = choices[0].get("message", {})
            response_text = msg.get("content", "")
            thinking_text = msg.get("reasoning_content", "")
        log_api_call(messages, response_text, duration_ms, log_file=log_file, thinking=thinking_text or None)

        # Usage is mapped by CodexClient from the Responses API's
        # input_tokens/output_tokens. Hardcoded zeros here previously froze
        # the context monitor (guarded on prompt_tokens > 0) at whatever the
        # last non-codex model reported.
        usage = result["response"].get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens", 0) or 0
        reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
        usage_details_available = bool(prompt_details or completion_details)

        # Emit the shared usage event for consumers such as the Token Monitor.
        from backend.llm_usage_events import record_llm_usage
        record_llm_usage(
            model=result["response"].get("model") or self.model,
            provider=self.provider or "codex",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            usage_details_available=usage_details_available,
            duration_ms=duration_ms,
            messages=messages,
            response_text=response_text,
        )

        return {
            "response": result["response"],
            "request_payload": {"model": self.model, "messages": messages,
                                "tools": tools},
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "success": True,
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Codex via OAuth token."""
        try:
            from models.db import db as _db
            from backend.provider.oauth_codex import get_valid_token
            from backend.provider.codex_client import CodexClient

            provider = _db.get_provider(self.provider or "codex")
            if not provider:
                return {"success": False, "error": "Codex provider not found"}
            token = get_valid_token(_db, provider["id"])
            if not token:
                return {"success": False, "error": "Not connected to Codex. Complete OAuth flow first."}
            client = CodexClient(token, self.base_url)
            return client.test_connection()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_reasoning_capabilities(self, model, metadata=None):
        if not self.supports_reasoning_effort:
            return reasoning_capabilities()
        metadata = metadata or {}
        levels = metadata.get("supported_reasoning_levels")
        if not isinstance(levels, list):
            return reasoning_capabilities()
        return reasoning_capabilities(
            [item.get("effort") for item in levels if isinstance(item, dict)],
            metadata.get("default_reasoning_level"),
        )

    def apply_reasoning_effort(self, payload, effort):
        if effort is not None:
            payload.setdefault("reasoning", {})["effort"] = effort

    def discovery_params(self):
        return {"client_version": CODEX_CLIENT_VERSION}

    def discovery_headers(self, headers):
        from backend.provider.oauth_codex import extract_account_id
        result = dict(headers)
        result.update({"User-Agent": f"codex_cli_rs/{CODEX_CLIENT_VERSION}",
                       "originator": "codex_cli_rs"})
        account = extract_account_id(result.get("Authorization", "").removeprefix("Bearer "))
        if account:
            result["ChatGPT-Account-ID"] = account
        return result

    def parse_models(self, data):
        rows = data.get("models", data.get("data", []))
        if not isinstance(rows, list) or not rows:
            return [
                {"id": "gpt-5.6-sol", "name": "GPT-5.6 Sol"},
                {"id": "gpt-5.6-terra", "name": "GPT-5.6 Terra"},
                {"id": "gpt-5.6-luna", "name": "GPT-5.6 Luna"},
            ]
        models = []
        for item in rows:
            if isinstance(item, str):
                models.append({"id": item, "name": item})
            elif isinstance(item, dict):
                mid = item.get("id") or item.get("slug") or item.get("model") or item.get("name", "")
                models.append({**item, "id": mid, "name": mid})
        return models

    def build_payload(self, model, messages, max_tokens, temperature=None,
                      tools=None, tool_choice=None, oauth=False):
        from backend.provider.codex_client import CodexClient
        payload = {"model": model, "input": CodexClient._convert_messages(messages),
                   "stream": True, "store": False}
        if tools:
            payload["tools"] = CodexClient._convert_tools(tools)
            if tool_choice:
                payload["tool_choice"] = {"type": "function", "name": tool_choice}
        return payload


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


class CerebrasProvider(OpenAIProvider):
    """Cerebras rejects non-standard reasoning fields in input messages."""

    def prepare_messages(self, messages, enable_thinking):
        processed = super().prepare_messages(messages, enable_thinking)
        for message in processed:
            message.pop("reasoning_content", None)
        return processed


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
