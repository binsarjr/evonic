from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000
_ALLOWED_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})
_CROP_TYPES = frozenset({"nik", "portrait", "signature", "custom"})
_OUTPUT_MIME = frozenset({"image/jpeg", "image/png"})
_TEMPLATES = {
    "nik": (.08, .47, .72, .13),
    "portrait": (.72, .18, .23, .55),
    "signature": (.70, .72, .25, .18),
}


@dataclass(frozen=True)
class CropResult:
    image: bytes
    mime_type: str
    document: dict
    nik_region: dict
    warnings: list[str]
    crop_type: str = "nik"
    region: dict | None = None


def _image_libs():
    try:
        import cv2
        import numpy as np
        from PIL import Image
        return cv2, np, Image
    except ImportError as exc:
        raise RuntimeError("IMAGE_PROCESSING_DEPENDENCY_UNAVAILABLE") from exc


def full_image_fallback(image_bytes: bytes, mime_type: str, max_bytes: int = MAX_BYTES, max_pixels: int = MAX_PIXELS) -> CropResult:
    """Validate an upload and retain its original bytes for vision extraction."""
    pil = _validate_image(image_bytes, mime_type, max_bytes, max_pixels)
    return CropResult(image_bytes, mime_type, {"detected": False, "source_size": {"width": pil.width, "height": pil.height}}, {"detected": False}, ["IMAGE_PROCESSING_DEPENDENCY_UNAVAILABLE", "FULL_IMAGE_SENT_TO_VISION_MODEL"])


def _validate_image(image_bytes: bytes, mime_type: str, max_bytes: int = MAX_BYTES, max_pixels: int = MAX_PIXELS):
    if mime_type not in _ALLOWED_MIME:
        raise ValueError("UNSUPPORTED_IMAGE_TYPE")
    if not image_bytes or len(image_bytes) > min(max(1, int(max_bytes)), MAX_BYTES):
        raise ValueError("IMAGE_SIZE_LIMIT_EXCEEDED")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("IMAGE_VALIDATION_DEPENDENCY_UNAVAILABLE") from exc
    try:
        check = Image.open(BytesIO(image_bytes)); check.verify()
        pil = Image.open(BytesIO(image_bytes)); pil.load()
        if pil.width * pil.height > min(max(1, int(max_pixels)), MAX_PIXELS):
            raise ValueError("IMAGE_PIXEL_LIMIT_EXCEEDED")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("INVALID_IMAGE") from exc
    return pil


def _order(points, np):
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(diffs)], points[np.argmax(sums)], points[np.argmax(diffs)]], dtype="float32")


def _rectify(image, cv2, np):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    ih, iw = image.shape[:2]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            rectangularity = area / max(cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(approx))), 1)
            if area > ih * iw * .12 and rectangularity >= .72:
                candidates.append((area * rectangularity, rectangularity, approx.reshape(4, 2).astype("float32")))
    if not candidates:
        return image, {"detected": False, "confidence": 0.0, "corners": [], "edge_density": round(float((edges > 0).mean()), 4)}, ["DOCUMENT_BOUNDARY_NOT_DETECTED"]
    _, rectangularity, points = max(candidates, key=lambda item: item[0])
    tl, tr, br, bl = _order(points, np)
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    if width < 300 or height < 180:
        return image, {"detected": False, "confidence": 0.0, "corners": points.astype(int).tolist()}, ["DOCUMENT_CORNERS_LOW_CONFIDENCE"]
    ratio = width / height
    margin = max(3, int(min(iw, ih) * .01))
    touches = any(x <= margin or y <= margin or x >= iw - margin or y >= ih - margin for x, y in points)
    confidence = min(.95, .55 + rectangularity * .3 + (.08 if 1.35 <= ratio <= 1.9 else 0))
    warnings = []
    if touches: warnings.append("DOCUMENT_TOUCHES_IMAGE_EDGE")
    if not 1.35 <= ratio <= 1.9: warnings.append("DOCUMENT_ASPECT_RATIO_UNEXPECTED")
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    metadata = {"detected": True, "confidence": round(confidence, 3), "corners": [{"x": int(x), "y": int(y)} for x, y in (tl, tr, br, bl)], "edge_density": round(float((edges > 0).mean()), 4), "aspect_ratio": round(ratio, 3)}
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height)), metadata, warnings


def _bbox(x, y, width, height, image_width, image_height):
    return {"pixel": {"x": x, "y": y, "width": width, "height": height}, "normalized": {"x": round(x / image_width, 6), "y": round(y / image_height, 6), "width": round(width / image_width, 6), "height": round(height / image_height, 6)}}


def _custom_bbox(coordinates, space, width, height):
    if not isinstance(coordinates, dict) or space not in {"pixel", "normalized"}:
        raise ValueError("INVALID_CROP_COORDINATES")
    try: values = [float(coordinates[key]) for key in ("x", "y", "width", "height")]
    except (KeyError, TypeError, ValueError) as exc: raise ValueError("INVALID_CROP_COORDINATES") from exc
    if not all(math.isfinite(value) for value in values) or min(values[:2]) < 0 or min(values[2:]) <= 0:
        raise ValueError("INVALID_CROP_COORDINATES")
    x, y, cw, ch = values
    if space == "normalized":
        if x > 1 or y > 1 or cw > 1 or ch > 1 or x + cw > 1 or y + ch > 1: raise ValueError("CROP_OUT_OF_BOUNDS")
        x, y, cw, ch = x * width, y * height, cw * width, ch * height
    x, y, cw, ch = round(x), round(y), round(cw), round(ch)
    if x + cw > width or y + ch > height: raise ValueError("CROP_OUT_OF_BOUNDS")
    return x, y, cw, ch


def _encode_pil(image, mime_type):
    if mime_type not in _OUTPUT_MIME: raise ValueError("UNSUPPORTED_OUTPUT_TYPE")
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG" if mime_type == "image/png" else "JPEG", quality=92)
    return output.getvalue()


def _auto_region(rectified, crop_type, cv2):
    h, w = rectified.shape[:2]
    tx, ty, tw, th = _TEMPLATES[crop_type]
    x, y, cw, ch = int(w * tx), int(h * ty), max(1, int(w * tw)), max(1, int(h * th))
    confidence, warnings = .5, ["TEMPLATE_CROP_USED"]
    if crop_type == "portrait":
        try:
            classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = classifier.detectMultiScale(cv2.cvtColor(rectified[y:y + ch, x:x + cw], cv2.COLOR_BGR2GRAY), 1.1, 4, minSize=(30, 30))
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda face: face[2] * face[3])
                pad_x, pad_y = int(fw * .55), int(fh * .65)
                x, y = max(0, x + fx - pad_x), max(0, y + fy - pad_y)
                cw, ch = min(w - x, fw + 2 * pad_x), min(h - y, fh + 2 * pad_y)
                confidence, warnings = .82, []
            else: warnings.append("FACE_REGION_NOT_DETECTED")
        except Exception:
            warnings.append("FACE_DETECTOR_UNAVAILABLE")
    elif crop_type == "signature":
        gray = cv2.cvtColor(rectified[y:y + ch, x:x + cw], cv2.COLOR_BGR2GRAY)
        ink_density = float((gray < 150).mean())
        confidence = min(.8, .35 + ink_density * 3) if .01 <= ink_density <= .22 else .25
        if confidence < .4: warnings.append("SIGNATURE_REGION_LOW_CONFIDENCE")
    return (x, y, cw, ch), confidence, warnings


def detect_crop(image_bytes: bytes, mime_type: str, crop_type: str = "nik", coordinates: dict | None = None,
                coordinate_space: str = "pixel", output_mime_type: str = "image/jpeg",
                max_bytes: int = MAX_BYTES, max_pixels: int = MAX_PIXELS) -> CropResult:
    if crop_type not in _CROP_TYPES: raise ValueError("INVALID_CROP_TYPE")
    pil = _validate_image(image_bytes, mime_type, max_bytes, max_pixels)
    if crop_type == "custom":
        x, y, cw, ch = _custom_bbox(coordinates, coordinate_space, pil.width, pil.height)
        cropped = pil.crop((x, y, x + cw, y + ch))
        bbox = _bbox(x, y, cw, ch, pil.width, pil.height)
        region = {"detected": True, "source": "custom", "bounding_box": bbox, "bbox_normalized": bbox["normalized"], "confidence": 1.0}
        document = {"detected": False, "source_size": {"width": pil.width, "height": pil.height}, "rectified_size": {"width": pil.width, "height": pil.height}}
        return CropResult(_encode_pil(cropped, output_mime_type), output_mime_type, document, {"detected": False}, [], crop_type, region)
    cv2, np, Image = _image_libs()
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError("INVALID_IMAGE") from exc
    if image is None: raise ValueError("INVALID_IMAGE")
    rectified, document, warnings = _rectify(image, cv2, np)
    h, w = rectified.shape[:2]
    (x, y, cw, ch), confidence, region_warnings = _auto_region(rectified, crop_type, cv2)
    crop = rectified[y:y + ch, x:x + cw]
    extension = ".png" if output_mime_type == "image/png" else ".jpg"
    if output_mime_type not in _OUTPUT_MIME: raise ValueError("UNSUPPORTED_OUTPUT_TYPE")
    params = [] if extension == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 92]
    ok, encoded = cv2.imencode(extension, crop, params)
    if not ok: raise ValueError("CROP_ENCODING_FAILED")
    blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    confidence = min(.95, confidence + document.get("confidence", 0) * .2 + min(blur / 500, .12))
    warnings.extend(region_warnings)
    if blur < 60: warnings.append("LOW_SHARPNESS")
    document["rectified_size"] = {"width": w, "height": h}
    bbox = _bbox(x, y, cw, ch, w, h)
    region = {"detected": True, "source": "detector" if not region_warnings else "template", "bounding_box": bbox, "bbox_normalized": bbox["normalized"], "confidence": round(confidence, 3)}
    nik_region = region if crop_type == "nik" else {"detected": False}
    return CropResult(encoded.tobytes(), output_mime_type, document, nik_region, warnings, crop_type, region)


def crop_data_url(result: CropResult) -> str:
    return f"data:{result.mime_type};base64,{base64.b64encode(result.image).decode('ascii')}"
