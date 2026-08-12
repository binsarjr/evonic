from plugins.nik_parser.parser import parse_nik
from plugins.nik_parser.privacy import mask_nik

def test_parse_known_structural_nik():
    result = parse_nik("3201010101010001")
    assert result["valid"] is True
    assert result["gender"] == "male"
    assert result["nik_masked"] == "320101******0001"
    assert result["region"]["status"] == "known"

def test_parse_female_and_invalid_date():
    assert parse_nik("3201014101010001")["gender"] == "female"
    assert "INVALID_BIRTH_DATE" in parse_nik("3201013102990001")["errors"]

def test_masking_and_bad_input():
    assert mask_nik("123") == "***"
    assert "NIK_MUST_HAVE_16_DIGITS" in parse_nik("not-a-nik")["errors"]
