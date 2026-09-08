"""Per-model reasoning metadata and effort validation."""

import re
from urllib.parse import urlsplit
from typing import List, Optional, TypedDict

from backend.reasoning_effort_error import ReasoningEffortError


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


# Codex subscription catalog verified 2026-09-08; live metadata takes priority.
CODEX_REASONING_DEFAULTS = {
    'gpt-6-astra': (['low', 'medium', 'high', 'xhigh', 'max', 'ultra'], 'medium'),
    'gpt-5.6-sol': (['low', 'medium', 'high', 'xhigh', 'max', 'ultra'], 'low'),
    'gpt-5.6-terra': (['low', 'medium', 'high', 'xhigh', 'max', 'ultra'], 'medium'),
    'gpt-5.6-luna': (['low', 'medium', 'high', 'xhigh', 'max'], 'medium'),
    'gpt-5.5': (['low', 'medium', 'high', 'xhigh'], 'medium'),
    'gpt-5.4-mini': (['low', 'medium', 'high', 'xhigh'], 'medium'),
    'gpt-5.3-codex-spark': (['low', 'medium', 'high', 'xhigh'], 'high'),
}


def reasoning_format(config):
    """Identify supported direct endpoints without depending on provider classes."""
    endpoint = urlsplit(config.get('base_url') or '')
    if endpoint.scheme != 'https':
        return None
    api_format = config.get('api_format', 'openai')
    if api_format == 'codex' and endpoint.hostname == 'chatgpt.com':
        return 'codex'
    if api_format == 'anthropic' and endpoint.hostname == 'api.anthropic.com':
        return 'anthropic'
    if api_format == 'openai' and endpoint.hostname == 'api.deepseek.com':
        return 'deepseek'
    return None


def model_reasoning_capabilities(config, metadata=None):
    """Use advertised levels or known per-model defaults, never a universal enum."""
    kind = reasoning_format(config)
    model = config.get('model_name')
    metadata = metadata or {}
    if kind == 'codex':
        levels = metadata.get('supported_reasoning_levels')
        if isinstance(levels, list):
            return reasoning_capabilities(
                [item.get('effort') for item in levels if isinstance(item, dict)],
                metadata.get('default_reasoning_level'))
        return reasoning_capabilities(*CODEX_REASONING_DEFAULTS.get(model, ()))
    if kind == 'anthropic':
        capabilities = metadata.get('capabilities')
        effort = capabilities.get('effort') if isinstance(capabilities, dict) else None
        if isinstance(effort, dict) and effort.get('supported'):
            levels = [level for level in ('low', 'medium', 'high', 'xhigh', 'max')
                      if isinstance(effort.get(level), dict) and effort[level].get('supported')]
            return reasoning_capabilities(levels, 'high')
    # https://api-docs.deepseek.com/api/create-chat-completion/
    if kind == 'deepseek' and model in {
        'deepseek-v4-flash', 'deepseek-v4-pro', 'deepseek-v4-flash-vision-exp',
    }:
        return reasoning_capabilities(['low', 'high', 'max'], 'high')
    return reasoning_capabilities()


def apply_reasoning_effort(payload, config, effort):
    """Merge an already validated override into the existing request payload."""
    if effort is None:
        return
    kind = reasoning_format(config)
    if kind == 'codex':
        payload.setdefault('reasoning', {})['effort'] = effort
    elif kind == 'anthropic':
        payload.setdefault('output_config', {})['effort'] = effort
    elif kind == 'deepseek':
        payload['reasoning_effort'] = effort
    else:
        raise ReasoningEffortError('Reasoning effort is not supported by this provider.')
