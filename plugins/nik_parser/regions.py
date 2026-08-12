from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).with_name("data") / "regions.json"


@lru_cache(maxsize=1)
def _snapshot() -> dict:
    with _DATA.open(encoding="utf-8") as handle:
        return json.load(handle)


def metadata() -> dict:
    data = _snapshot()
    raw = _DATA.read_bytes()
    return {**data["metadata"], "sha256": hashlib.sha256(raw).hexdigest()}


def lookup(code: str) -> dict:
    item = _snapshot().get("regions", {}).get(str(code), {})
    return {"status": "known", **item} if item else {"status": "unknown", "code": str(code)}
