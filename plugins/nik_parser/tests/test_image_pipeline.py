import base64
import unittest
from io import BytesIO

from PIL import Image

from plugins.nik_parser.image.pipeline import crop_data_url, detect_crop
from plugins.nik_parser.service import image_crop


def image_bytes(width=100, height=80, mime_type="image/png"):
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(
        output, format="PNG" if mime_type == "image/png" else "JPEG"
    )
    return output.getvalue()


class ImagePipelineTests(unittest.TestCase):
    def test_custom_pixel_crop_returns_both_coordinate_spaces(self):
        result = detect_crop(
            image_bytes(), "image/png", "custom",
            {"x": 10, "y": 8, "width": 40, "height": 20}, "pixel", "image/png",
        )
        self.assertEqual(result.region["bounding_box"]["pixel"], {"x": 10, "y": 8, "width": 40, "height": 20})
        self.assertEqual(result.region["bounding_box"]["normalized"], {"x": .1, "y": .1, "width": .4, "height": .25})
        self.assertEqual(result.mime_type, "image/png")

    def test_custom_normalized_crop_encodes_actual_mime(self):
        result = detect_crop(
            image_bytes(), "image/png", "custom",
            {"x": .1, "y": .25, "width": .5, "height": .5}, "normalized", "image/jpeg",
        )
        uri = crop_data_url(result)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        self.assertTrue(base64.b64decode(uri.split(",", 1)[1]).startswith(b"\xff\xd8\xff"))

    def test_metadata_omits_image_bytes(self):
        result = image_crop(
            image_bytes(), "image/png", crop_type="custom",
            coordinates={"x": 0, "y": 0, "width": 10, "height": 10},
        )
        self.assertIsNone(result["data"])
        self.assertEqual(result["output_format"], "metadata")

    def test_invalid_coordinates_are_rejected(self):
        for coordinates in (
            {"x": -1, "y": 0, "width": 1, "height": 1},
            {"x": 0, "y": 0, "width": 0, "height": 1},
            {"x": 90, "y": 0, "width": 20, "height": 1},
            {"x": float("nan"), "y": 0, "width": 1, "height": 1},
        ):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                detect_crop(image_bytes(), "image/png", "custom", coordinates)

    def test_invalid_contract_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CROP_TYPE"):
            detect_crop(image_bytes(), "image/png", "face")
        with self.assertRaisesRegex(ValueError, "INVALID_OUTPUT_FORMAT"):
            image_crop(image_bytes(), "image/png", crop_type="custom", coordinates={"x": 0, "y": 0, "width": 1, "height": 1}, output_format="path")
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_OUTPUT_TYPE"):
            detect_crop(image_bytes(), "image/png", "custom", {"x": 0, "y": 0, "width": 1, "height": 1}, output_mime_type="image/webp")


if __name__ == "__main__":
    unittest.main()
