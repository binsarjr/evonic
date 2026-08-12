# NIK Parser

Performs local structural parsing only. It never proves a NIK is active, officially issued, unique, or owned by a person.

## APIs

- `POST /api/nik/parse` JSON `{ "nik": "..." }`
- `POST /api/nik/validate` JSON `{ "nik": "..." }`
- `POST /api/nik/detect-crop` multipart field `image`
- `POST /api/nik/image-crop` multipart fields: `image`, optional `crop_type` (`nik`, `portrait`, `signature`, `custom`), `coordinates` JSON for custom crops, `coordinate_space` (`pixel` or `normalized`), `output_format` (`metadata` or opt-in `data_uri`), and `output_mime_type` (`image/jpeg` or `image/png`).
- `POST /api/nik/extract-image` multipart fields: `image` and optional `vision_model_id`. Omitting the model selects automatically from enabled Evonic models that declare `vision_supported`.
- `GET /api/nik/vision-models` lists the same enabled vision-model choices used by extraction.

## Configuration

The Plugin Settings page provides Default Vision Model and Fallback Vision Model selects. They are populated only with enabled Evonic models that declare `vision_supported`. The default model is tried first, followed by the configured fallback and then other enabled vision models. A request-level `vision_model_id` overrides the configured default and is validated against the same registry.

## Vision-only extraction

Image extraction validates the upload MIME type, byte limit, and file signature without decoding it locally. The original image bytes are encoded as a data URL and sent through Evonic's canonical `LLMClient`. Extraction never imports or falls back to OpenCV, NumPy, Pillow, Tesseract, or offline OCR. A disabled, missing, or non-vision explicit model returns `SELECTED_VISION_MODEL_UNAVAILABLE`.

## Optional local crop tools

`nik_detect_crop` and `nik_image_crop` remain separate local image-processing features. They may require OpenCV, NumPy, and Pillow and can return a dependency error without affecting `nik_extract_from_image`. The plugin does not retain images or crop outputs. Metadata mode returns no image bytes. Base64 data URIs are returned only when explicitly requested and use the encoded MIME type.

The bundled region data is a minimal bootstrap snapshot. Unknown codes mean the snapshot lacks coverage, not that a code is invalid.
