import unittest
from unittest.mock import patch

from plugins.nik_parser.service import extract


class ServiceExtractionTests(unittest.TestCase):
    def test_extract_uses_vision_directly_when_local_processing_is_disabled(self):
        config = {
            "ENABLE_IMAGE_PROCESSING": False,
            "DEFAULT_VISION_MODEL_ID": "vision-a",
            "FALLBACK_VISION_MODEL_ID": "vision-b",
            "MAX_IMAGE_BYTES": 1234,
        }
        expected = {"available": True, "provider": "vision_model", "model_id": "vision-a"}
        with patch("plugins.nik_parser.service._config", return_value=config), patch(
            "plugins.nik_parser.service.extract_with_vision", return_value=expected
        ) as vision, patch("plugins.nik_parser.service.detect_crop") as local_crop:
            result = extract(b"image", "image/png")
        vision.assert_called_once_with(b"image", "image/png", "vision-a", "vision-b", max_bytes=1234)
        local_crop.assert_not_called()
        self.assertEqual(result, {"officially_verified": False, "ocr": expected})

    def test_explicit_model_overrides_configured_default(self):
        config = {"DEFAULT_VISION_MODEL_ID": "configured", "FALLBACK_VISION_MODEL_ID": "fallback"}
        with patch("plugins.nik_parser.service._config", return_value=config), patch(
            "plugins.nik_parser.service.extract_with_vision", return_value={"available": True}
        ) as vision:
            extract(b"image", "image/png", model_id="explicit")
        self.assertEqual(vision.call_args.args[2:4], ("explicit", "fallback"))


if __name__ == "__main__":
    unittest.main()
