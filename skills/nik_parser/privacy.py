from __future__ import annotations


def normalize_nik(value: object) -> str:
    return "".join(str(value or "").split())


def mask_nik(value: object) -> str:
    nik = normalize_nik(value)
    if not nik:
        return ""
    if len(nik) <= 4:
        return "*" * len(nik)
    return f"{nik[:6]}{'*' * max(0, len(nik) - 10)}{nik[-4:]}"
