from __future__ import annotations

from .image.extraction import extract_with_vision
from .image.pipeline import crop_data_url, detect_crop


def _config() -> dict:
    from backend.skills_manager import skills_manager
    return skills_manager.get_skill_config("nik_parser")


def _limits(config: dict) -> tuple[int, int]:
    return int(config.get("MAX_IMAGE_BYTES", 8 * 1024 * 1024)), int(config.get("MAX_IMAGE_PIXELS", 20_000_000))


def detect(image: bytes, mime_type: str) -> dict:
    config = _config(); max_bytes, max_pixels = _limits(config)
    crop = detect_crop(image, mime_type, max_bytes=max_bytes, max_pixels=max_pixels)
    return {"document": crop.document, "nik_region": crop.nik_region, "warnings": crop.warnings, "officially_verified": False}


def image_crop(image: bytes, mime_type: str, *, crop_type: str = "nik", coordinates: dict | None = None,
               coordinate_space: str = "pixel", output_format: str = "metadata",
               output_mime_type: str = "image/jpeg") -> dict:
    if output_format not in {"metadata", "data_uri"}:
        raise ValueError("INVALID_OUTPUT_FORMAT")
    config = _config(); max_bytes, max_pixels = _limits(config)
    crop = detect_crop(image, mime_type, crop_type, coordinates, coordinate_space, output_mime_type, max_bytes, max_pixels)
    return {
        "crop_type": crop.crop_type,
        "detected": bool(crop.region and crop.region.get("detected")),
        "bounding_box": (crop.region or {}).get("bounding_box"),
        "confidence": (crop.region or {}).get("confidence", 0.0),
        "source": (crop.region or {}).get("source"),
        "output_format": output_format,
        "mime_type": crop.mime_type,
        "data": crop_data_url(crop) if output_format == "data_uri" else None,
        "document": crop.document,
        "nik_region": crop.nik_region,
        "warnings": crop.warnings,
        "officially_verified": False,
    }


def extract(image: bytes, mime_type: str, *, model_id: str | None = None) -> dict:
    config = _config()
    max_bytes, _ = _limits(config)
    extracted = extract_with_vision(
        image, mime_type,
        model_id or config.get("DEFAULT_VISION_MODEL_ID") or None,
        config.get("FALLBACK_VISION_MODEL_ID") or None,
        max_bytes=max_bytes,
    )
    return {"officially_verified": False, "ocr": extracted}
