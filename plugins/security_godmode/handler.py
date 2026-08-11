from __future__ import annotations

import copy
import threading

from backend.plugin_manager import (
    apply_turn_context,
    register_agent_state_summary_provider,
    register_final_response_handler,
    register_turn_context_provider,
    register_tool_request_transformer,
    register_tool_result_transformer,
    register_user_message_transformer,
    unregister_agent_state_summary_provider,
    unregister_final_response_handler,
    unregister_turn_context_provider,
    unregister_tool_request_transformer,
    unregister_tool_result_transformer,
    unregister_user_message_transformer,
)

from .backend.tools import godmode_auto, godmode_race
from .backend.tools._godmode import strategies
from .backend.tools._lib import (
    PLUGIN_VERSION,
    consume_rediscovery,
    effective_profile,
    get_activation,
    get_automatic_discovery,
    get_profile,
    get_race_on_failure,
    get_system_prompt_mode,
    mark_context_provided,
    mark_discovery,
    mark_response_evaluated,
    mark_transform_applied,
    model_family,
    profile_status,
    score_response,
    transform_policy,
    transform_text,
)

_config = {}
_discovery_locks: dict[str, threading.Lock] = {}
_runtime: dict[tuple[str, str], dict] = {}

def _truthy(value) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def _current_model(agent_id: str) -> dict:
    from models.db import db
    return db.get_agent_model(agent_id) or {}


def _discovery_cached(agent_id: str, model_id: str) -> bool:
    profile = get_profile(agent_id)
    return (
        profile.get("discovery_model_id") == model_id
        and profile.get("discovery_source_version") == PLUGIN_VERSION
        and profile.get("discovery_state") in {"ready", "none_needed", "failed"}
    )


def ensure_discovered(agent_id: str) -> None:
    if not get_automatic_discovery(agent_id):
        return
    model = _current_model(agent_id)
    model_id = str(model.get("id") or "")
    refresh = consume_rediscovery(agent_id)
    if not model_id or (_discovery_cached(agent_id, model_id) and not refresh):
        return

    lock = _discovery_locks.setdefault(agent_id, threading.Lock())
    with lock:
        if _discovery_cached(agent_id, model_id) and not refresh:
            return
        mark_discovery(agent_id, "discovering", model_id)
        result = godmode_auto.execute(
            {"id": agent_id, "model_id": model_id},
            {"model_id": model_id, "max_tokens": 2048},
        )
        if result.get("success"):
            state = "none_needed" if result.get("strategy") == "none_needed" else "ready"
            mark_discovery(agent_id, state, model_id)
        else:
            mark_discovery(agent_id, "failed", model_id)


def provide_context(agent_id: str, session_id: str):
    if not _truthy(_config.get("AUTO_CONTEXT_ENABLED", True)) or not get_activation(agent_id):
        return None
    ensure_discovered(agent_id)
    profile = effective_profile(agent_id)
    if profile.get("strategy") == "none_needed":
        return None
    mode = get_system_prompt_mode(agent_id)
    mark_context_provided(agent_id, session_id)
    return {
        "id": "security_godmode_profile",
        "tools": [],
        "system_md": profile.get("system_prompt", "") if mode != "preserve" else "",
        "system_mode": "replace" if mode == "override" else mode,
        "prefill_messages": profile.get("prefill", []),
    }


def _remember_original(agent_id: str, session_id: str, original: str, transformed: str) -> None:
    state = _runtime.setdefault((agent_id, session_id), {"raw": {}, "attempts": []})
    state["raw"][transformed] = original
    if len(state["raw"]) > 32:
        state["raw"].pop(next(iter(state["raw"])))


def transform_user_message(agent_id: str, session_id: str, text: str):
    if not _truthy(_config.get("AUTO_CONTEXT_ENABLED", True)):
        return text
    if not get_activation(agent_id) or not text:
        return text
    policy = transform_policy(agent_id)
    if policy["mode"] == "inactive":
        _remember_original(agent_id, session_id, text, text)
        return text
    transformed = transform_text(text, policy["encoding"])
    _remember_original(agent_id, session_id, text, transformed)
    mark_transform_applied(agent_id, session_id, policy["encoding"], transformed != text)
    return transformed


def _transform_tool_payload(agent_id: str, session_id: str, _tool_name: str, payload):
    """Transform strings in tool payloads only when the active policy requires it."""
    if not _truthy(_config.get("AUTO_CONTEXT_ENABLED", True)) or not get_activation(agent_id):
        return payload
    policy = transform_policy(agent_id)
    if policy["mode"] == "inactive":
        return payload

    def transform(value):
        if isinstance(value, str):
            return transform_text(value, policy["encoding"])
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, dict):
            return {key: transform(value) for key, value in value.items()}
        return value

    transformed = transform(payload)
    if transformed != payload:
        mark_transform_applied(agent_id, session_id, policy["encoding"], True)
    return transformed


def transform_tool_request(agent_id: str, session_id: str, tool_name: str, args):
    return _transform_tool_payload(agent_id, session_id, tool_name, args)


def transform_tool_result(agent_id: str, session_id: str, tool_name: str, result):
    return _transform_tool_payload(agent_id, session_id, tool_name, result)


def _candidate_profiles(agent_id: str) -> list[dict]:
    model = _current_model(agent_id)
    family = model_family(model)
    config = strategies.MODEL_STRATEGIES.get(family, strategies.DEFAULT_STRATEGY)
    candidates = []
    for name in config["order"]:
        if name == "parseltongue":
            for level, encoding in enumerate(("PLAIN", "L33T", "BUBBLE", "BRAILLE", "MORSE")):
                candidates.append({
                    "strategy": f"parseltongue_L{level}_{encoding}",
                    "system_prompt": "",
                    "prefill": list(strategies.SUBTLE_PREFILL),
                    "encoding": encoding,
                })
            continue
        if name == "prefill_only":
            candidates.append({
                "strategy": name, "system_prompt": "",
                "prefill": list(strategies.STANDARD_PREFILL), "encoding": "",
            })
            continue
        system = config.get("system_templates", {}).get(name, "")
        if system:
            candidates.append({
                "strategy": name, "system_prompt": system, "prefill": [], "encoding": "",
            })
            candidates.append({
                "strategy": name + "+prefill", "system_prompt": system,
                "prefill": list(strategies.STANDARD_PREFILL), "encoding": "",
            })
    for candidate in candidates:
        candidate.update({
            "agent_id": agent_id,
            "model_family": family,
            "model_id": str(model.get("id") or ""),
            "profile_source": "runtime",
        })
    return candidates


def _restore_newest_user(messages: list, raw: dict[str, str], encoding: str) -> None:
    target = next((message for message in reversed(messages)
                   if isinstance(message, dict) and message.get("role") == "user"), None)
    if not target:
        return

    def restore(text: str) -> str:
        original = raw.get(text, text)
        return transform_text(original, encoding) if encoding and encoding != "PLAIN" else original

    content = target.get("content")
    if isinstance(content, str):
        target["content"] = restore(content)
    elif isinstance(content, list):
        target["content"] = [
            {**part, "text": restore(part["text"])}
            if isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str) else part
            for part in content
        ]


def _retry_messages(agent_id: str, session_id: str, messages: list,
                    candidate: dict, raw: dict[str, str]) -> list:
    revised = copy.deepcopy(messages)
    mode = get_system_prompt_mode(agent_id)
    policy = transform_policy(agent_id)
    encoding = candidate.get("encoding") or (policy["encoding"] if policy["forced"] else "")
    _restore_newest_user(revised, raw, encoding)
    if encoding:
        mark_transform_applied(agent_id, session_id, encoding, encoding != "PLAIN")

    context = {
        "system_md": candidate.get("system_prompt", "") if mode != "preserve" else "",
        "system_mode": "replace" if mode == "override" else mode,
        "prefill_messages": candidate.get("prefill", []),
    }
    apply_turn_context(revised, [], [context])
    return revised


def _initial_candidate_index(candidates: list[dict], strategy: str) -> int:
    return next((index + 1 for index, item in enumerate(candidates)
                 if item["strategy"] == strategy), 0)


def _candidate_signature(agent_id: str, candidate: dict) -> tuple:
    policy = transform_policy(agent_id)
    encoding = candidate.get("encoding") or (policy["encoding"] if policy["forced"] else "")
    encoding = "" if encoding == "PLAIN" else encoding
    if candidate.get("strategy") == "none_needed":
        return "", (), encoding
    mode = get_system_prompt_mode(agent_id)
    system = candidate.get("system_prompt", "") if mode != "preserve" else ""
    prefill = tuple((message.get("role"), message.get("content"))
                    for message in candidate.get("prefill", []))
    return system, prefill, encoding


def evaluate_final_response(context: dict):
    agent_id = str(context.get("agent_id") or "")
    session_id = str(context.get("session_id") or "")
    content = str(context.get("content") or "")
    if not content or not get_activation(agent_id):
        _runtime.pop((agent_id, session_id), None)
        return None

    key = (agent_id, session_id)
    state = _runtime.setdefault(key, {"raw": {}, "attempts": []})
    raw_query = " ".join(state.get("raw", {}).values())
    scored = score_response(content, query=raw_query)
    profile = effective_profile(agent_id)
    strategy = state.get("current_strategy") or str(profile.get("strategy") or "")
    state.setdefault("attempts", []).append({
        "content": content, "score": scored["score"],
        "refused": scored["refused"], "strategy": strategy,
    })
    attempts = state["attempts"]

    if not scored["refused"]:
        retry_state = "recovered" if len(attempts) > 1 else "not_needed"
        mark_response_evaluated(agent_id, scored["score"], False, len(attempts),
                                retry_state, strategy, "not_needed")
        _runtime.pop(key, None)
        return {"content": content, "timeline": {
            "state": retry_state, "attempts": len(attempts),
            "strategy": strategy, "score": scored["score"],
        }}

    candidates = state.setdefault("candidates", _candidate_profiles(agent_id))
    state.setdefault("next_candidate", _initial_candidate_index(candidates, strategy))
    state.setdefault("seen_signatures", {_candidate_signature(agent_id, profile)})
    if int(context.get("retry_count") or 0) < 2:
        while state["next_candidate"] < len(candidates):
            candidate = candidates[state["next_candidate"]]
            state["next_candidate"] += 1
            signature = _candidate_signature(agent_id, candidate)
            if signature in state["seen_signatures"]:
                continue
            state["seen_signatures"].add(signature)
            revised = _retry_messages(
                agent_id, session_id, context.get("messages") or [], candidate,
                state.get("raw", {}),
            )
            if revised == context.get("messages"):
                continue
            state["current_strategy"] = candidate["strategy"]
            mark_response_evaluated(agent_id, scored["score"], True, len(attempts),
                                    "retrying", candidate["strategy"], "not_needed")
            return {
                "retry": True,
                "messages": revised,
                "timeline": {
                    "state": "retrying", "attempts": len(attempts),
                    "strategy": candidate["strategy"], "score": scored["score"],
                },
            }

    race_state = "disabled"
    if get_race_on_failure(agent_id):
        model_id = str(_current_model(agent_id).get("id") or "")
        race = godmode_race.execute({"id": agent_id}, {
            "prompt": raw_query, "backend": "evonic", "tier": "fast",
            "exclude_model_id": model_id, "max_tokens": 4096,
        })
        race_state = "won" if race.get("content") and not race.get("is_refusal", True) \
            else "failed" if not race.get("error") else "unavailable"
        if race_state == "won":
            attempts.append({
                "content": race["content"], "score": race.get("score", -9999),
                "refused": False, "strategy": "evonic_fast_race",
            })

    best = max(enumerate(attempts), key=lambda item: (item[1]["score"], item[0]))[1]
    retry_state = "raced" if race_state == "won" else "exhausted"
    mark_response_evaluated(agent_id, best["score"], best["refused"],
                            min(len(attempts), 3), retry_state,
                            best["strategy"], race_state)
    _runtime.pop(key, None)
    return {"content": best["content"], "timeline": {
        "state": retry_state, "attempts": min(len(attempts), 3),
        "strategy": best["strategy"], "score": best["score"],
        "race_state": race_state,
    }}


def provide_state(agent_id: str, session_id: str):
    status = profile_status(agent_id)
    active = _truthy(_config.get("AUTO_CONTEXT_ENABLED", True)) \
        and status["activation_enabled"]
    return {
        "state": "active" if active else "inactive",
        "data": {
            "source": status.get("payload_source"),
            "strategy": status.get("effective_strategy"),
            "model_family": status.get("effective_model_family"),
            "current_model_id": status.get("current_model_id"),
            "tested_model_id": status.get("model_id") or None,
            "profile_family_match": status.get("profile_family_match"),
            "transform_mode": status.get("transform_mode"),
            "encoding": status.get("effective_encoding"),
            "force_transform": status.get("force_transform"),
            "system_prompt_mode": status.get("system_prompt_mode"),
            "automatic_discovery": get_automatic_discovery(agent_id),
            "discovery_state": status.get("discovery_state") or "idle",
            "discovery_model_id": status.get("discovery_model_id") or None,
            "last_discovery_at": status.get("last_discovery_at") or None,
            "response_retry_state": status.get("response_retry_state") or "inactive",
            "response_attempt_count": status.get("response_attempt_count", 0),
            "last_response_score": status.get("last_response_score"),
            "last_response_refused": bool(status.get("last_response_refused")),
            "last_retry_strategy": status.get("last_retry_strategy") or None,
            "race_state": status.get("race_state") or "disabled",
            "last_transform_at": status.get("last_transform_at") or None,
            "last_transform_session_id": status.get("last_transform_session_id") or None,
            "last_transform_changed": bool(status.get("last_transform_changed")),
            "transform_count": status.get("transform_count", 0),
            "last_context_provided_at": status.get("last_context_provided_at") or None,
            "last_session_id": status.get("last_session_id") or None,
            "context_provided_count": status.get("context_provided_count", 0),
        },
    }


def on_enable(sdk=None):
    global _config
    _config = dict(sdk.config if sdk else {})
    register_turn_context_provider(provide_context)
    register_agent_state_summary_provider("security_godmode", provide_state)
    register_user_message_transformer("security_godmode", transform_user_message)
    register_tool_request_transformer("security_godmode", transform_tool_request)
    register_tool_result_transformer("security_godmode", transform_tool_result)
    register_final_response_handler("security_godmode", evaluate_final_response)
    if sdk:
        sdk.log("Security Godmode enabled; orchestration is programmatic and opt-in per agent.")


def on_disable(sdk=None):
    unregister_turn_context_provider(provide_context)
    unregister_agent_state_summary_provider("security_godmode")
    unregister_user_message_transformer("security_godmode")
    unregister_tool_request_transformer("security_godmode")
    unregister_tool_result_transformer("security_godmode")
    unregister_final_response_handler("security_godmode")
    _runtime.clear()
