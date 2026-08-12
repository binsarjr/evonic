from __future__ import annotations

from datetime import date

from .privacy import mask_nik, normalize_nik
from .regions import lookup, metadata


def _birth_candidates(day: int, month: int, year: int, reference_year: int | None) -> list[str]:
    candidates = []
    for century in (1900, 2000):
        try:
            value = date(century + year, month, day)
        except ValueError:
            continue
        if reference_year is None or value.year <= reference_year:
            candidates.append(value.isoformat())
    return candidates


def parse_nik(value: object, *, reference_year: int | None = None) -> dict:
    nik = normalize_nik(value)
    result = {"valid": False, "nik_masked": mask_nik(nik), "errors": [], "warnings": [], "officially_verified": False}
    if len(nik) != 16:
        result["errors"].append("NIK_MUST_HAVE_16_DIGITS")
        return result
    if not nik.isascii() or not nik.isdigit():
        result["errors"].append("NIK_MUST_CONTAIN_DIGITS_ONLY")
        return result
    region_code, encoded_day, month, year, sequence = nik[:6], int(nik[6:8]), int(nik[8:10]), int(nik[10:12]), nik[12:]
    gender = "female" if encoded_day > 40 else "male"
    day = encoded_day - 40 if gender == "female" else encoded_day
    candidates = _birth_candidates(day, month, year, reference_year)
    region = lookup(region_code)
    result.update({
        "region": region,
        "region_dataset": metadata(),
        "birth": {"encoded": nik[6:12], "day": day, "month": month, "year_two_digits": year, "candidates": candidates},
        "gender": gender,
        "sequence": sequence,
    })
    if not candidates:
        result["errors"].append("INVALID_BIRTH_DATE")
    elif len(candidates) > 1:
        result["warnings"].append("AMBIGUOUS_BIRTH_CENTURY")
    if region["status"] == "unknown":
        result["warnings"].append("UNKNOWN_REGION_CODE")
    result["valid"] = not result["errors"]
    return result


def validate_nik(value: object, *, reference_year: int | None = None) -> dict:
    parsed = parse_nik(value, reference_year=reference_year)
    return {key: parsed[key] for key in ("valid", "nik_masked", "errors", "warnings", "officially_verified", "region")}
