from base64 import b64decode
from io import BytesIO

import pytest
from PIL import Image

from plugins.nik_parser.image.pipeline import detect_crop
from plugins.nik_parser.service import image_crop


def image_bytes(fmt="PNG"):
    output = BytesIO()
    Image.new("RGB", (100, 60), (220, 30, 40)).save(output, format=fmt)
    return output.getvalue()


def test_custom_pixel_crop_returns_both_coordinate_spaces():
    result = image_crop(image_bytes(), "image/png", crop_type="custom",
                        coordinates={"x": 10, "y": 12, "width": 40, "height": 24})
    assert result["detected"] is True
    assert result["bounding_box"]["pixel"] == {"x": 10, "y": 12, "width": 40, "height": 24}
    assert result["bounding_box"]["normalized"] == {"x": .1, "y": .2, "width": .4, "height": .4}
    assert result["output_format"] == "metadata"
    assert result["data"] is None


def test_custom_normalized_crop_and_png_data_uri():
    result = image_crop(image_bytes(), "image/png", crop_type="custom",
                        coordinates={"x": .1, "y": .2, "width": .4, "height": .5},
                        coordinate_space="normalized", output_format="data_uri",
                        output_mime_type="image/png")
    prefix, encoded = result["data"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")
    assert result["mime_type"] == "image/png"


@pytest.mark.parametrize("coordinates", [
    {"x": -1, "y": 0, "width": 10, "height": 10},
    {"x": 95, "y": 0, "width": 10, "height": 10},
    {"x": float("inf"), "y": 0, "width": 10, "height": 10},
])
def test_custom_crop_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        detect_crop(image_bytes(), "image/png", "custom", coordinates)


def test_custom_crop_requires_coordinates():
    with pytest.raises(ValueError, match="INVALID_CROP_COORDINATES"):
        detect_crop(image_bytes(), "image/png", "custom")


def test_invalid_output_contract_is_rejected():
    with pytest.raises(ValueError, match="INVALID_OUTPUT_FORMAT"):
        image_crop(image_bytes(), "image/png", crop_type="custom",
                   coordinates={"x": 0, "y": 0, "width": 10, "height": 10},
                   output_format="path")
