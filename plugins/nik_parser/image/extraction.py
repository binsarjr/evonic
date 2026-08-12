from __future__ import annotations

import base64
import json
import re
from typing import Any

from ..parser import parse_nik

_DIGITS = re.compile(r"(?<!\d)(?:\d[\s-]*){16}(?!\d)")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_ALLOWED_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_FALLBACK_ERRORS = frozenset({
    "connection_error", "api_error", "provider_error", "rate_limit_error",
    "timeout_error", "request_timeout", "generation_timeout", "auth_error",
})


def _image_data_url(image: bytes, mime_type: str, max_bytes: int) -> str:
    if mime_type not in _ALLOWED_MIME:
        raise ValueError("UNSUPPORTED_IMAGE_TYPE")
    if not image or len(image) > min(max(1, int(max_bytes)), _MAX_IMAGE_BYTES):
        raise ValueError("IMAGE_SIZE_LIMIT_EXCEEDED")
    signatures = {"image/jpeg": image[:3] == b"\xff\xd8\xff", "image/png": image[:8] == b"\x89PNG\r\n\x1a\n", "image/webp": image[:4] == b"RIFF" and image[8:12] == b"WEBP"}
    if not signatures[mime_type]:
        raise ValueError("INVALID_IMAGE")
    return f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"


def _extract_digits(text: str) -> list[str]:
    return ["".join(match.group().split()).replace("-", "") for match in _DIGITS.finditer(text)]


def _add_vision_model(models: list[dict], seen_ids: set[str], model_id: str | None, db) -> None:
    """Append an enabled vision model from Evonic's model registry."""

    if not model_id or model_id in seen_ids:
        return
    model = db.get_model_by_id(model_id)
    if model and model.get("enabled") and model.get("vision_supported"):
        models.append(model)
        seen_ids.add(model_id)


def _vision_models(selected_id: str | None, fallback_id: str | None = None) -> tuple[list[dict], str | None]:
    from models.db import db
    models: list[dict] = []
    seen_ids: set[str] = set()
    if selected_id:
        _add_vision_model(models, seen_ids, selected_id, db)
        if not models:
            return [], "SELECTED_VISION_MODEL_UNAVAILABLE"
    _add_vision_model(models, seen_ids, fallback_id, db)
    for model in db.get_enabled_llm_models():
        model_id = model.get("id") or model.get("name", "")
        if model.get("enabled") and model.get("vision_supported") and model_id not in seen_ids:
            models.append(model)
            seen_ids.add(model_id)
    return (models, None) if models else ([], "NO_VISION_MODEL_AVAILABLE")


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_response(content: str) -> dict[str, Any]:
    try:
        return _object(json.loads(content))
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(content)
        if not match:
            return {}
        try:
            return _object(json.loads(match.group()))
        except json.JSONDecodeError:
            return {}


def extract_with_vision(image: bytes, mime_type: str, model_id: str | None = None,
                        fallback_model_id: str | None = None, max_bytes: int = 8 * 1024 * 1024) -> dict:
    data_url = _image_data_url(image, mime_type, max_bytes)
    models, error = _vision_models(model_id, fallback_model_id)
    if error:
        return {"available": False, "provider": "vision_model", "error": error}
    from backend.llm_client import LLMClient, strip_thinking_tags
    messages = [{"role": "system", "content": "Transcribe only clearly visible fields from the supplied document image. Do not infer hidden or unreadable characters or perform identity verification. Return exactly one JSON object: {\"document_fields\": {\"nik\": \"visible digits only or empty\", \"name\": \"visible text or empty\", \"birth_place_date\": \"visible text or empty\", \"gender\": \"visible text or empty\", \"address\": \"visible text or empty\", \"religion\": \"visible text or empty\", \"marital_status\": \"visible text or empty\", \"occupation\": \"visible text or empty\", \"nationality\": \"visible text or empty\", \"valid_until\": \"visible text or empty\"}, \"confidence\": 0.0}. Preserve visible text as written."}, {"role": "user", "content": [{"type": "text", "text": "Return the requested structured transcription only."}, {"type": "image_url", "image_url": {"url": data_url}}]}]
    attempted = 0
    for model in models:
        attempted += 1
        try:
            client = LLMClient(model_config=model)
            client.timeout = min(client.timeout or 120, 120)
            response = client.chat_completion(messages=messages, enable_thinking=False)
        except Exception:
            continue
        if not response.get("success"):
            if response.get("error_type") in _FALLBACK_ERRORS or not response.get("error_type"):
                continue
            return {"available": True, "provider": "vision_model", "model_id": model.get("id"), "error": "VISION_MODEL_REQUEST_REJECTED"}
        choices = response.get("response", {}).get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        content, _ = strip_thinking_tags(content)
        payload = _parse_response(content)
        fields = _object(payload.get("document_fields"))
        candidates = _extract_digits(str(fields.get("nik") or ""))
        result = {"available": True, "provider": "vision_model", "model_id": model.get("id"), "confidence": payload.get("confidence"), "document_fields": fields}
        if candidates:
            result["result"] = parse_nik(candidates[0])
        if fields or candidates:
            return result
    return {"available": True, "provider": "vision_model", "models_attempted": attempted, "error": "VISION_EXTRACTION_FAILED"}
