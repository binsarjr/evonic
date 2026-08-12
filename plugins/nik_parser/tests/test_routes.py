from io import BytesIO
from unittest.mock import patch

from flask import Flask
from plugins.nik_parser.routes import create_blueprint

def client():
    app = Flask(__name__); app.register_blueprint(create_blueprint()); return app.test_client()

def test_parse_route_masks_value():
    response = client().post("/api/nik/parse", json={"nik": "3201010101010001"})
    assert response.status_code == 200
    assert response.get_json()["nik_masked"] == "320101******0001"

def test_route_requires_nik():
    response = client().post("/api/nik/validate", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "NIK_REQUIRED"


def test_extract_route_forwards_only_optional_vision_model_id():
    expected = {"officially_verified": False, "ocr": {"available": True}}
    with patch("plugins.nik_parser.routes.extract", return_value=expected) as extract:
        response = client().post("/api/nik/extract-image", data={
            "image": (BytesIO(b"image"), "sample.png"),
            "provider": "offline",
            "vision_model_id": "vision-a",
        })
    assert response.status_code == 200
    assert response.get_json() == expected
    extract.assert_called_once()
    assert extract.call_args.args == (b"image", "image/png")
    assert extract.call_args.kwargs == {"model_id": "vision-a"}
