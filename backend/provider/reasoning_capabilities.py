"""Per-model reasoning metadata and effort validation."""

import re
from typing import List, Optional, TypedDict

from backend.provider.reasoning_effort_error import ReasoningEffortError


class ReasoningCapabilities(TypedDict):
    efforts: List[str]
    default_effort: Optional[str]

def reasoning_capabilities(efforts=(), default=None) -> ReasoningCapabilities:
    """Normalize provider metadata without inventing a universal effort enum."""
    values = list(dict.fromkeys(
        value for value in efforts
        if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value)
    ))
    return {"efforts": values, "default_effort": default if default in values else None}

def validate_reasoning_effort(effort, capabilities):
    if effort is None or effort == "":
        return None
    if not isinstance(effort, str) or effort not in capabilities["efforts"]:
        raise ReasoningEffortError(
            "Reasoning effort is not supported by this provider/model. "
            "Fetch models to refresh support, or select Default (provider)."
        )
    return effort
