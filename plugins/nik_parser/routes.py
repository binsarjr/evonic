from __future__ import annotations

import json

from flask import Blueprint, jsonify, request

from .parser import parse_nik, validate_nik
from .service import detect, extract, image_crop


def _error(code: str, status: int = 400): return jsonify({"error": code, "officially_verified": False}), status

def _image():
    item = request.files.get("image")
    if not item: raise ValueError("IMAGE_REQUIRED")
    return item.read(), item.mimetype

def _coordinates():
    raw = request.form.get("coordinates")
    if not raw: return None
    try: value = json.loads(raw)
    except (TypeError, ValueError) as exc: raise ValueError("INVALID_CROP_COORDINATES") from exc
    if not isinstance(value, dict): raise ValueError("INVALID_CROP_COORDINATES")
    return value

def create_blueprint():
    bp = Blueprint("nik_parser", __name__)
    @bp.get("/api/nik/vision-models")
    def vision_models_api():
        """Return safe, current options for enabled Evonic vision models."""
        from models.db import db
        models = [{
            "id": model["id"],
            "name": model.get("name") or model["id"],
            "provider": model.get("provider") or "unknown",
            "label": f"{model.get('name') or model['id']} ({model.get('provider') or 'unknown'})"
        } for model in db.get_enabled_llm_models() if model.get("vision_supported")]
        return jsonify({
            "models": models,
            "count": len(models),
            "automatic_option": {"value": "", "label": "Automatic, use available vision model"}
        })
    @bp.post("/api/nik/parse")
    def parse_api():
        body = request.get_json(silent=True) or {}
        if "nik" not in body: return _error("NIK_REQUIRED")
        return jsonify(parse_nik(body["nik"]))
    @bp.post("/api/nik/validate")
    def validate_api():
        body = request.get_json(silent=True) or {}
        if "nik" not in body: return _error("NIK_REQUIRED")
        return jsonify(validate_nik(body["nik"]))
    @bp.post("/api/nik/detect-crop")
    def crop_api():
        try: return jsonify(detect(*_image()))
        except (ValueError, RuntimeError) as exc: return _error(str(exc))
    @bp.post("/api/nik/image-crop")
    def image_crop_api():
        try:
            image, mime = _image()
            return jsonify(image_crop(
                image, mime,
                crop_type=request.form.get("crop_type") or "nik",
                coordinates=_coordinates(),
                coordinate_space=request.form.get("coordinate_space") or "pixel",
                output_format=request.form.get("output_format") or "metadata",
                output_mime_type=request.form.get("output_mime_type") or "image/jpeg",
            ))
        except (ValueError, RuntimeError) as exc: return _error(str(exc))
    @bp.post("/api/nik/extract-image")
    def extract_api():
        try:
            image, mime = _image()
            return jsonify(extract(image, mime, model_id=request.form.get("vision_model_id") or None))
        except (ValueError, RuntimeError) as exc: return _error(str(exc))
    return bp
