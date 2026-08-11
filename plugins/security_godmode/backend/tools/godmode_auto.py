from __future__ import annotations

from ._godmode import parseltongue, racing, strategies
from ._lib import model_family, set_activation, set_profile
from .godmode_race import call_model, execute as race


def _scored(result: dict, query: str) -> dict:
    if result.get("error") or not str(result.get("response") or "").strip():
        return {**result, "score": -9999, "refused": True, "hedges": 0}
    score = racing.score_response(result.get("response") or "", query)
    return {**result, "score": score["score"], "refused": score["is_refusal"],
            "hedges": score["hedge_count"]}


def _attempt(strategy: str, result: dict) -> dict:
    return {"strategy": strategy, **{key: value for key, value in result.items()
                                     if key != "response"}}


def execute(agent: dict, args: dict) -> dict:
    from models.db import db

    canary = str(args.get("canary") or args.get("prompt") or strategies.QUICK_CANARY)
    model_id = str(args.get("model_id") or agent.get("model_id") or "")
    agent_id = str(agent.get("id") or "")
    model = db.get_model_by_id(model_id) if model_id else db.get_agent_model(agent_id)
    if not model or not model.get("enabled"):
        return {"success": False, "error": "the selected agent model is unavailable or disabled"}

    max_tokens = max(64, min(int(args.get("max_tokens") or 2048), 4096))
    dry_run = bool(args.get("dry_run", False))
    family = model_family(model)
    config = strategies.MODEL_STRATEGIES.get(family, strategies.DEFAULT_STRATEGY)
    attempts = []

    baseline = _scored(call_model(model, canary, max_tokens, baseline=True), canary)
    attempts.append(_attempt("baseline", baseline))
    if not baseline["refused"] and baseline["score"] > 100 and baseline["hedges"] == 0:
        if not dry_run:
            set_profile(agent_id, True, "none_needed", system_prompt="", prefill=[],
                        encoding="", model_family_name=family,
                        profile_source="auto-discovered",
                        tested_model_id=str(model.get("id") or ""))
            set_activation(agent_id, True)
        return {"success": True, "model": model.get("id"), "family": family,
                "strategy": "none_needed", "system_prompt": None, "prefill": None,
                "score": baseline["score"], "content_preview": (baseline.get("response") or "")[:300],
                "saved": not dry_run, "attempts": attempts}

    winner = None
    for strategy_name in config["order"]:
        if strategy_name == "parseltongue":
            for level in range(5):
                encoded, label = parseltongue.escalate_encoding(canary, level)
                result = _scored(call_model(model, encoded, max_tokens,
                                            prefill=strategies.SUBTLE_PREFILL), canary)
                attempts.append(_attempt(f"parseltongue_L{level}_{label}", result))
                if not result["refused"]:
                    winner = {"strategy": f"parseltongue_L{level}_{label}", "system_prompt": "",
                              "prefill": strategies.SUBTLE_PREFILL, "encoding": label,
                              "result": result}
                    break
            if winner:
                break
            continue

        system_prompt = config.get("system_templates", {}).get(strategy_name, "")
        if strategy_name == "prefill_only":
            result = _scored(call_model(model, canary, max_tokens,
                                        prefill=strategies.STANDARD_PREFILL), canary)
            attempts.append(_attempt(strategy_name, result))
            if not result["refused"]:
                winner = {"strategy": strategy_name, "system_prompt": "",
                          "prefill": strategies.STANDARD_PREFILL, "encoding": "", "result": result}
                break
            continue
        if not system_prompt:
            continue

        result = _scored(call_model(model, canary, max_tokens, system_prompt=system_prompt), canary)
        attempts.append(_attempt(strategy_name, result))
        if not result["refused"]:
            winner = {"strategy": strategy_name, "system_prompt": system_prompt,
                      "prefill": [], "encoding": "", "result": result}
            break

        combined = _scored(call_model(model, canary, max_tokens, system_prompt=system_prompt,
                                      prefill=strategies.STANDARD_PREFILL), canary)
        attempts.append(_attempt(strategy_name + "+prefill", combined))
        if not combined["refused"]:
            winner = {"strategy": strategy_name + "+prefill", "system_prompt": system_prompt,
                      "prefill": strategies.STANDARD_PREFILL, "encoding": "", "result": combined}
            break

    if winner:
        if not dry_run:
            set_profile(agent_id, True, winner["strategy"],
                        system_prompt=winner["system_prompt"], prefill=winner["prefill"],
                        encoding=winner["encoding"], model_family_name=family,
                        profile_source="auto-discovered",
                        tested_model_id=str(model.get("id") or ""))
            set_activation(agent_id, True)
        result = winner["result"]
        return {"success": True, "model": model.get("id"), "family": family,
                "strategy": winner["strategy"], "system_prompt": winner["system_prompt"],
                "prefill": winner["prefill"], "encoding": winner["encoding"],
                "score": result["score"], "content_preview": (result.get("response") or "")[:500],
                "saved": not dry_run, "attempts": attempts}

    response = {"success": False, "model": model.get("id"), "family": family,
                "strategy": None, "score": -9999, "attempts": attempts,
                "message": "All strategies failed. Try ULTRAPLINIAN mode or a different model."}
    if args.get("race_on_failure"):
        response["race"] = race(agent, {
            "prompt": canary, "backend": args.get("race_backend", "evonic"),
            "tier": args.get("race_tier", "standard"),
            "model_ids": args.get("fallback_model_ids", []),
        })
    return response
