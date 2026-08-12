import builtins
import json
import unittest
from unittest.mock import patch

from plugins.nik_parser.image.extraction import _vision_models, extract_with_vision

PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"


class FakeDB:
    def __init__(self, models):
        self.models = {model["id"]: model for model in models}

    def get_model_by_id(self, model_id):
        return self.models.get(model_id)

    def get_enabled_llm_models(self):
        return [model for model in self.models.values() if model.get("enabled")]


class FakeClient:
    responses = []
    calls = []

    def __init__(self, model_config):
        self.model_config = model_config
        self.timeout = None

    def chat_completion(self, messages, enable_thinking):
        self.calls.append((self.model_config["id"], messages, enable_thinking))
        return self.responses.pop(0)


class VisionExtractionTests(unittest.TestCase):
    def setUp(self):
        FakeClient.calls = []
        FakeClient.responses = []
        self.models = [
            {"id": "vision-a", "name": "Vision A", "enabled": True, "vision_supported": True},
            {"id": "vision-b", "name": "Vision B", "enabled": True, "vision_supported": True},
            {"id": "text", "name": "Text", "enabled": True, "vision_supported": False},
            {"id": "disabled", "name": "Disabled", "enabled": False, "vision_supported": True},
        ]

    def _patch_runtime(self):
        db = FakeDB(self.models)
        return patch.multiple("models.db.db", get_model_by_id=db.get_model_by_id,
                              get_enabled_llm_models=db.get_enabled_llm_models)

    def test_automatic_selection_uses_only_enabled_vision_models(self):
        with self._patch_runtime():
            models, error = _vision_models(None)
        self.assertIsNone(error)
        self.assertEqual([model["id"] for model in models], ["vision-a", "vision-b"])

    def test_explicit_disabled_or_nonvision_model_is_rejected(self):
        with self._patch_runtime():
            for model_id in ("disabled", "text", "missing"):
                with self.subTest(model_id=model_id):
                    models, error = _vision_models(model_id)
                    self.assertEqual(models, [])
                    self.assertEqual(error, "SELECTED_VISION_MODEL_UNAVAILABLE")

    def test_fallback_and_structured_result(self):
        payload = {"document_fields": {"nik": "3201010101010001", "name": "VISIBLE NAME"}, "confidence": 0.91}
        FakeClient.responses = [
            {"success": False, "error_type": "timeout_error"},
            {"success": True, "response": {"choices": [{"message": {"content": json.dumps(payload)}}]}},
        ]
        with self._patch_runtime(), patch("backend.llm_client.LLMClient", FakeClient):
            result = extract_with_vision(PNG, "image/png")
        self.assertEqual([call[0] for call in FakeClient.calls], ["vision-a", "vision-b"])
        self.assertEqual(result["model_id"], "vision-b")
        self.assertEqual(result["document_fields"]["name"], "VISIBLE NAME")
        self.assertEqual(result["result"]["nik_masked"], "320101******0001")
        image_url = FakeClient.calls[0][1][1]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))

    def test_extraction_does_not_import_local_image_libraries(self):
        payload = {"document_fields": {"name": "VISIBLE NAME"}, "confidence": 0.8}
        FakeClient.responses = [{"success": True, "response": {"choices": [{"message": {"content": json.dumps(payload)}}]}}]
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"cv2", "numpy", "PIL", "pytesseract"}:
                raise AssertionError(f"local image dependency imported: {name}")
            return original_import(name, *args, **kwargs)

        with self._patch_runtime(), patch("backend.llm_client.LLMClient", FakeClient), patch("builtins.__import__", guarded_import):
            result = extract_with_vision(PNG, "image/png")
        self.assertTrue(result["available"])
        self.assertEqual(result["document_fields"]["name"], "VISIBLE NAME")

    def test_invalid_image_is_rejected_without_decoder(self):
        with self.assertRaisesRegex(ValueError, "INVALID_IMAGE"):
            extract_with_vision(b"not-an-image", "image/png")


if __name__ == "__main__":
    unittest.main()
