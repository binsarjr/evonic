"""OpenAI Codex provider, including Responses API and SSE transport."""

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

import httpx

from backend.provider.common import _format_llm_error, _normalize_system_messages
from backend.provider.openai_provider import OpenAIProvider
from backend.provider.oauth_codex import CODEX_BASE_URL, extract_account_id
from backend.provider.provider_auth_error import ProviderAuthError
from backend.provider.reasoning_capabilities import reasoning_capabilities
from evaluator.api_logger import log_api_call

CODEX_CLIENT_VERSION = "0.153.4"
_log = logging.getLogger(__name__)


def model_supports_fast_mode(model: Optional[str]) -> bool:
    """Return whether Codex advertises Fast mode for this model family."""
    model_id = str(model or "").strip().lower()
    if "/" in model_id:
        model_id = model_id.split("/", 1)[1]
    model_id = model_id.split(":", 1)[0]
    return (
        model_id in {"gpt-5.4", "gpt-5.5", "gpt-5.6"}
        or model_id.startswith("gpt-5.6-")
    )


def _classify_stream_error(code: str) -> str:
    """Map a Responses API error/incomplete code to an llm_loop error_type.

    'provider_error' is the transient default — llm_loop retries it with backoff
    and it is fallback-eligible, which is what we want for backend hiccups.
    """
    code = (code or "").lower()
    if any(k in code for k in ("rate_limit", "usage_limit", "quota", "too_many")):
        return "rate_limit_error"
    if any(k in code for k in ("auth", "unauthorized", "token_expired", "invalid_api_key")):
        return "auth_error"
    if "timeout" in code:
        return "request_timeout"
    if "max_output_tokens" in code or "max_tokens" in code:
        # Retrying the same request verbatim is unlikely to help — 'llm_error'
        # gives one retry, then falls back to another model.
        return "llm_error"
    return "provider_error"


def _map_usage(usage) -> Dict[str, Any]:
    """Map Responses API usage (input_tokens/output_tokens) to the
    OpenAI-chat-style keys (prompt_tokens/completion_tokens) that the rest
    of the pipeline (context monitor, traces, dashboards) reads."""
    if not isinstance(usage, dict) or not usage:
        return {}
    prompt = usage.get("input_tokens", 0) or 0
    completion = usage.get("output_tokens", 0) or 0
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    mapped = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens", 0) or (prompt + completion),
    }
    if input_details or output_details:
        mapped["prompt_tokens_details"] = {
            "cached_tokens": input_details.get("cached_tokens", 0) or 0,
        }
        mapped["completion_tokens_details"] = {
            "reasoning_tokens": output_details.get("reasoning_tokens", 0) or 0,
        }
    return mapped


class OpenAiCodexProvider(OpenAIProvider):
    """Own Codex credentials, model discovery, requests and streaming responses."""

    completion_path = "/responses"
    http_client = httpx
    reasoning_host = "chatgpt.com"

    # Codex subscription catalog, verified 2026-09-08. Live metadata takes priority.
    # API model docs differ: https://developers.openai.com/api/docs/models/gpt-6-astra
    reasoning_defaults = {
        "gpt-6-astra": (["low", "medium", "high", "xhigh", "max", "ultra"], "medium"),
        "gpt-5.6-sol": (["low", "medium", "high", "xhigh", "max", "ultra"], "low"),
        "gpt-5.6-terra": (["low", "medium", "high", "xhigh", "max", "ultra"], "medium"),
        "gpt-5.6-luna": (["low", "medium", "high", "xhigh", "max"], "medium"),
        "gpt-5.5": (["low", "medium", "high", "xhigh"], "medium"),
        "gpt-5.4-mini": (["low", "medium", "high", "xhigh"], "medium"),
        "gpt-5.3-codex-spark": (["low", "medium", "high", "xhigh"], "high"),
    }

    def __init__(self, config=None, base_url="", *, access_token=None):
        # Keep the former CodexClient(token, base_url) constructor working.
        if not isinstance(config, dict):
            config = {"access_token": access_token if access_token is not None else config,
                      "base_url": base_url}
        super().__init__({**config, "base_url": config.get("base_url") or CODEX_BASE_URL})
        self.access_token = config.get("access_token")
        self._account_id = extract_account_id(self.access_token) if self.access_token else None

    def request_headers(self, discovery=False):
        from models.db import db
        from backend.provider.oauth_codex import get_valid_token
        token = self.config.get("access_token") or get_valid_token(db, self.provider or "codex")
        if not token:
            raise ProviderAuthError("Not connected. Click Connect first.")
        self.access_token = token
        self._account_id = extract_account_id(token)
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
        """Send a Codex completion and record normalized usage."""
        messages = _normalize_system_messages(messages)
        start_time = time.time()
        try:
            self.request_headers()
        except ProviderAuthError:
            return {
                "response": {"error": _format_llm_error("auth_error")},
                "duration_ms": 0,
                "success": False,
                "error_type": "auth_error",
                "error_detail": "Codex OAuth token missing or expired. Reconnect in Settings.",
            }

        if max_tokens is None:
            max_tokens = self.max_tokens
        result = self.send_request(
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

        # Usage is mapped from the Responses API's
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
        """Test if the Codex endpoint is reachable with current token."""
        try:
            self.request_headers()
            resp = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(accept="application/json"),
                params={"client_version": CODEX_CLIENT_VERSION},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", data.get("models", []))
                return {
                    "success": True,
                    "message": "Connected to Codex",
                    "available_models": len(models) if isinstance(models, list) else 0,
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_reasoning_capabilities(self, model, metadata=None):
        if not self.supports_reasoning_effort:
            return reasoning_capabilities()
        metadata = metadata or {}
        levels = metadata.get("supported_reasoning_levels")
        if not isinstance(levels, list):
            return reasoning_capabilities(*self.reasoning_defaults.get(model, ()))
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
                {"id": "gpt-6-astra", "name": "GPT-6 Astra"},
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
        payload = {"model": model, "input": self._convert_messages(messages),
                   "stream": True, "store": False}
        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice:
                payload["tool_choice"] = {"type": "function", "name": tool_choice}
        return payload

    def _is_chatgpt_backend(self) -> bool:
        """True when talking to the ChatGPT subscription backend (Codex CLI protocol)."""
        return "chatgpt.com" in self.base_url

    def _headers(
        self,
        accept: str = "text/event-stream",
        model: str = "",
        service_tier: Optional[str] = None,
    ) -> Dict[str, str]:
        if not self.access_token:
            self.request_headers()
        h = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": accept,
            "User-Agent": f"codex_cli_rs/{CODEX_CLIENT_VERSION}",
            "originator": "codex_cli_rs",
        }
        if self._account_id:
            h["ChatGPT-Account-ID"] = self._account_id
        if service_tier == "priority" and model_supports_fast_mode(model):
            h["x-codex-routing-hint"] = f"model={model};tier=priority"
        return h

    def send_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
        reasoning: bool = False,
        stream: bool = True,
        timeout: int = 120,
        tool_choice: Optional[str] = None,
        service_tier: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a request to the Codex Responses API."""
        payload = self.build_payload(
            model, messages, max_tokens, temperature, tools, tool_choice,
        )
        payload["stream"] = stream
        if reasoning:
            payload["reasoning"] = {"summary": "auto"}
        self.apply_reasoning_effort(payload, reasoning_effort)
        if service_tier == "priority" and model_supports_fast_mode(model):
            payload["service_tier"] = "priority"
        # Cap the output so a reasoning model can't spend the whole budget
        # thinking and return an empty (incomplete) response. The ChatGPT
        # Codex backend only accepts the parameter set the Codex CLI sends,
        # so the cap is limited to standard Responses API endpoints.
        if max_tokens and not self._is_chatgpt_backend():
            payload["max_output_tokens"] = max_tokens

        url = f"{self.base_url}/responses"

        try:
            if stream:
                return self._stream_response(url, payload, timeout)
            else:
                return self._blocking_response(url, payload, timeout)
        except httpx.TimeoutException:
            # Use 'request_timeout' (not 'timeout_error') so llm_loop retries on
            # the same model before falling back — reasoning turns are slow and a
            # single stall shouldn't demote us to a weaker fallback model.
            return {
                "success": False,
                "error_type": "request_timeout",
                "error": "Codex request timed out",
            }
        except httpx.ConnectError as e:
            return {
                "success": False,
                "error_type": "connection_error",
                "error": f"Cannot connect to Codex: {e}",
            }
        except Exception as e:
            _log.error("Codex request failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error_type": "unknown_error",
                "error": str(e),
            }

    def _stream_response(self, url: str, payload: Dict, timeout: int) -> Dict[str, Any]:
        """Handle SSE streaming from the Responses API."""
        # Reasoning models can pause a long time before/between SSE chunks.
        # Keep connect fast, but allow a generous read-gap for reasoning.
        _timeout = httpx.Timeout(connect=30.0, read=max(timeout, 300), write=30.0, pool=30.0)
        with httpx.Client(timeout=_timeout) as client:
            with client.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers(
                    model=payload.get("model", ""),
                    service_tier=payload.get("service_tier"),
                ),
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode(errors="replace")[:500]
                    return {
                        "success": False,
                        "error_type": self._classify_http_error(resp.status_code),
                        "error": f"HTTP {resp.status_code}: {body}",
                    }

                content_parts = []
                thinking_parts = []
                tool_calls = []
                response_id = ""
                model_used = ""
                usage: Dict[str, Any] = {}
                # A stream can end in failure without an HTTP error — usage
                # limits, filtered output, or an output-token cap hit while
                # reasoning. Track it so an empty stream is reported as an
                # error instead of a successful empty answer (which the agent
                # loop renders as "(No response)").
                stream_error: Optional[Dict[str, str]] = None
                completed = False

                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "response.created":
                        r = event.get("response", {})
                        response_id = r.get("id", "")
                        model_used = r.get("model", "")

                    elif event_type == "response.output_text.delta":
                        content_parts.append(event.get("delta", ""))

                    elif event_type == "response.reasoning_summary_text.delta":
                        thinking_parts.append(event.get("delta", ""))

                    elif event_type == "response.reasoning_summary_part.added":
                        # Separate reasoning summary parts with a blank line
                        if thinking_parts:
                            thinking_parts.append("\n\n")

                    elif event_type == "response.output_item.done":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            tool_calls.append({
                                "id": item.get("call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", "{}"),
                                },
                            })

                    elif event_type == "response.completed":
                        completed = True
                        r = event.get("response", {})
                        if not response_id:
                            response_id = r.get("id", "")
                        usage = _map_usage(r.get("usage"))

                    elif event_type == "response.failed":
                        r = event.get("response", {}) or {}
                        usage = _map_usage(r.get("usage")) or usage
                        err = r.get("error") or {}
                        code = err.get("code") or err.get("type") or ""
                        stream_error = {
                            "error_type": _classify_stream_error(code),
                            "error": (f"Codex response failed"
                                      f"{f' [{code}]' if code else ''}: "
                                      f"{err.get('message') or 'no detail'}"),
                        }

                    elif event_type == "response.incomplete":
                        r = event.get("response", {}) or {}
                        usage = _map_usage(r.get("usage")) or usage
                        reason = (r.get("incomplete_details") or {}).get("reason") or "unknown"
                        stream_error = {
                            "error_type": _classify_stream_error(reason),
                            "error": f"Codex response incomplete ({reason})",
                        }

                    elif event_type == "error":
                        err = event.get("error") if isinstance(event.get("error"), dict) else event
                        code = err.get("code") or err.get("type") or ""
                        stream_error = {
                            "error_type": _classify_stream_error(code),
                            "error": (f"Codex stream error"
                                      f"{f' [{code}]' if code else ''}: "
                                      f"{err.get('message') or 'no detail'}"),
                        }

        full_content = "".join(content_parts)
        full_thinking = "".join(thinking_parts)

        # Nothing usable came back — surface the failure so the agent loop can
        # retry or fall back, rather than treating it as an empty final answer.
        if not full_content and not tool_calls:
            if stream_error:
                _log.warning("Codex stream produced no output: %s", stream_error["error"])
                return {"success": False, **stream_error}
            if not completed:
                _log.warning("Codex stream ended without response.completed and no output")
                return {
                    "success": False,
                    "error_type": "provider_error",
                    "error": "Codex stream ended without a completed response",
                }
        elif stream_error:
            # Partial output is still usable — keep it, but leave a trace.
            _log.warning("Codex stream returned partial output: %s", stream_error["error"])

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": full_content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if full_thinking:
            message["reasoning_content"] = full_thinking

        return {
            "success": True,
            "response": {
                "id": response_id,
                "model": model_used,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop" if not tool_calls else "tool_calls",
                    }
                ],
                "usage": usage,
            },
        }

    def _blocking_response(self, url: str, payload: Dict, timeout: int) -> Dict[str, Any]:
        """Non-streaming fallback."""
        payload["stream"] = False
        resp = httpx.post(
            url,
            json=payload,
            headers=self._headers(
                accept="application/json",
                model=payload.get("model", ""),
                service_tier=payload.get("service_tier"),
            ),
            timeout=timeout,
        )

        if resp.status_code != 200:
            return {
                "success": False,
                "error_type": self._classify_http_error(resp.status_code),
                "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
            }

        data = resp.json()
        output = data.get("output", [])
        content = ""
        thinking = ""
        tool_calls = []

        for item in output:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content += part.get("text", "")
            elif item.get("type") == "reasoning":
                for part in item.get("summary", []):
                    if part.get("type") == "summary_text":
                        thinking += part.get("text", "")
            elif item.get("type") == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })

        # Same guard as the streaming path: a failed/incomplete response with no
        # output must not look like a successful empty answer.
        status = data.get("status", "")
        if not content and not tool_calls and status in ("failed", "incomplete", "cancelled"):
            if status == "incomplete":
                code = (data.get("incomplete_details") or {}).get("reason") or "unknown"
                detail = f"Codex response incomplete ({code})"
            else:
                err = data.get("error") or {}
                code = err.get("code") or err.get("type") or status
                detail = f"Codex response {status} [{code}]: {err.get('message') or 'no detail'}"
            _log.warning("Codex response produced no output: %s", detail)
            return {
                "success": False,
                "error_type": _classify_stream_error(code),
                "error": detail,
            }

        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if thinking:
            message["reasoning_content"] = thinking

        return {
            "success": True,
            "response": {
                "id": data.get("id", ""),
                "model": data.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop" if not tool_calls else "tool_calls",
                    }
                ],
                "usage": _map_usage(data.get("usage")),
            },
        }

    def stream_chunks(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
        reasoning: bool = False,
        timeout: int = 120,
        service_tier: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield SSE delta chunks for real-time streaming to the frontend."""
        payload = self.build_payload(
            model, messages, max_tokens, temperature, tools,
        )
        if reasoning:
            payload["reasoning"] = {"summary": "auto"}
        self.apply_reasoning_effort(payload, reasoning_effort)
        if service_tier == "priority" and model_supports_fast_mode(model):
            payload["service_tier"] = "priority"

        url = f"{self.base_url}/responses"

        _timeout = httpx.Timeout(connect=30.0, read=max(timeout, 300), write=30.0, pool=30.0)
        with httpx.Client(timeout=_timeout) as client:
            with client.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers(
                    model=payload.get("model", ""),
                    service_tier=payload.get("service_tier"),
                ),
            ) as resp:
                if resp.status_code != 200:
                    yield {
                        "error": True,
                        "error_type": self._classify_http_error(resp.status_code),
                        "message": f"HTTP {resp.status_code}",
                    }
                    return

                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                        yield event
                    except json.JSONDecodeError:
                        continue

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Chat Completions messages to Responses API input format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                converted.append({
                    "role": "system",
                    "content": content if isinstance(content, str) else json.dumps(content),
                })
            elif role == "user":
                if isinstance(content, list):
                    # Convert Chat Completions content block types to Responses API format.
                    # Chat Completions: "text" / "image_url" → Responses: "input_text" / "input_image"
                    converted_content = []
                    for block in content:
                        btype = block.get("type", "")
                        if btype == "text":
                            block = {**block, "type": "input_text"}
                        elif btype == "image_url":
                            # Chat Completions: {"image_url": {"url": "..."}}
                            # Responses API:    {"image_url": "..."}
                            url = block.get("image_url", {})
                            if isinstance(url, dict):
                                url = url.get("url", "")
                            block = {"type": "input_image", "image_url": url}
                        converted_content.append(block)
                    converted.append({"role": "user", "content": converted_content})
                else:
                    converted.append({"role": "user", "content": str(content)})
            elif role == "assistant":
                # Keep the assistant's own text even when the same message
                # carries tool calls — it is part of the reasoning trail the
                # model needs on the next turn. Empty content is dropped
                # entirely (the Responses API has no use for a blank item).
                text = "" if content is None else (
                    content if isinstance(content, str) else json.dumps(content)
                )
                if text.strip():
                    converted.append({"role": "assistant", "content": text})
                for tc in msg.get("tool_calls") or []:
                    converted.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", "{}"),
                    })
            elif role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": str(content),
                })

        return converted

    @staticmethod
    def _convert_tools(tools: List[Dict]) -> List[Dict]:
        """Convert Chat Completions tools to Responses API format.

        Chat Completions nests under "function":
            {"type": "function", "function": {"name", "description", "parameters"}}
        Responses API is flat:
            {"type": "function", "name", "description", "parameters", "strict": False}
        """
        converted = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                converted.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    "strict": False,
                })
            else:
                converted.append(t)
        return converted

    @staticmethod
    def _classify_http_error(status_code: int) -> str:
        if status_code == 401:
            return "auth_error"
        elif status_code == 429:
            return "rate_limit_error"
        elif status_code >= 500:
            return "api_error"
        elif status_code == 408:
            return "timeout_error"
        return "llm_error"
