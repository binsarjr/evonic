from __future__ import annotations

import concurrent.futures
import time

from ._godmode import racing, strategies
from ._lib import (get_system_prompt_mode, model_family, score_response, strategy_context,
                   strategy_prefill)


def call_model(model: dict, prompt: str, max_tokens: int, *, system_prompt: str = "",
               prefill: list | None = None, baseline: bool = False,
               timeout: int | None = None) -> dict:
    from backend.llm_client import LLMClient

    started = time.monotonic()
    messages = []
    if system_prompt and not baseline:
        messages.append({"role": "system", "content": system_prompt})
    if prefill and not baseline:
        messages.extend(prefill)
    messages.append({"role": "user", "content": prompt})
    try:
        model_config = dict(model)
        if timeout is not None:
            model_config["timeout"] = timeout
        client = LLMClient(model_config=model_config)
        result = client.chat_completion(
            messages=messages, tools=None, enable_thinking=False, max_tokens=max_tokens,
        )
        latency = int((time.monotonic() - started) * 1000)
        if not result.get("success"):
            return {"model_id": model.get("id"), "name": model.get("name"),
                    "error": result.get("error_detail") or result.get("error_type") or "request failed",
                    "latency_ms": latency, "score": -9999, "refused": True}
        response = client.extract_content(result)
        if not str(response or "").strip():
            return {"model_id": model.get("id"), "name": model.get("name"),
                    "error": "empty response", "latency_ms": latency,
                    "score": -9999, "refused": True, "is_refusal": True,
                    "hedges": 0, "hedge_count": 0}
        return {"model_id": model.get("id"), "name": model.get("name"), "response": response,
                **score_response(response, latency, prompt)}
    except Exception as exc:
        return {"model_id": model.get("id"), "name": model.get("name"), "error": str(exc),
                "latency_ms": int((time.monotonic() - started) * 1000), "score": -9999,
                "refused": True}


def _openrouter(args: dict, prompt: str) -> dict:
    from models.db import db

    provider = db.get_provider("openrouter") or {}
    api_key = provider.get("api_key") or ""
    if not provider.get("enabled") or not api_key:
        return {"error": "OpenRouter provider is not configured", "required_provider": "openrouter"}
    mode = str(args.get("race_type") or "ultraplinian")
    timeout = max(5, min(int(args.get("timeout") or 60), 300))
    if mode == "classic":
        return racing.race_godmode_classic(prompt, api_key=api_key, timeout=timeout)
    return racing.race_models(
        prompt,
        tier=str(args.get("tier") or "standard"),
        api_key=api_key,
        system_prompt=str(args.get("system_prompt") or "") or None,
        max_workers=max(1, min(int(args.get("max_workers") or 10), 20)),
        timeout=timeout,
        append_directive=bool(args.get("append_directive", True)),
    )


def _evonic(args: dict, prompt: str) -> dict:
    from models.db import db

    enabled = {model["id"]: model for model in db.get_enabled_llm_models()}
    requested = [str(item) for item in (args.get("model_ids") or [])]
    selected = [enabled[item] for item in requested if item in enabled] if requested else list(enabled.values())
    excluded = str(args.get("exclude_model_id") or "")
    if excluded:
        selected = [model for model in selected if str(model.get("id") or "") != excluded]
    tier_size = racing.TIER_SIZES.get(str(args.get("tier") or "standard"), 24)
    selected = selected[:tier_size]
    if not selected:
        return {"error": "no enabled Evonic models selected", "available": list(enabled)}
    strategy = str(args.get("strategy") or "")
    max_tokens = max(64, min(int(args.get("max_tokens") or 4096), 4096))
    timeout = max(5, min(int(args.get("timeout") or 60), 300))
    effective_prompt = prompt
    if bool(args.get("append_directive", True)):
        effective_prompt += racing.DEPTH_DIRECTIVE

    def run(model):
        family = model_family(model)
        selected_strategy = strategy
        if not selected_strategy:
            config = strategies.MODEL_STRATEGIES.get(family, strategies.DEFAULT_STRATEGY)
            selected_strategy = next(
                (name for name in config.get("order", []) if name != "parseltongue"),
                "prefill_only",
            )
        profile = {
            "strategy": selected_strategy,
            "system_prompt": str(args.get("system_prompt") or "")
            or strategy_context(selected_strategy, family=family),
            "prefill": strategy_prefill(selected_strategy),
            "encoding": "",
            "model_family": family,
        }
        mode = get_system_prompt_mode(str(args.get("_agent_id") or ""))
        return call_model(
            model, effective_prompt, max_tokens,
            system_prompt=profile.get("system_prompt", "") if mode != "preserve" else "",
            prefill=profile.get("prefill", []),
            timeout=timeout,
        )

    workers = max(1, min(int(args.get("max_workers") or 10), 20, len(selected)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, selected))
    results.sort(key=lambda item: item.get("score", -9999), reverse=True)
    winner = next((item for item in results if not item.get("refused") and not item.get("error")),
                  results[0] if results else None)
    return {"model": winner.get("model_id") if winner else None,
            "content": winner.get("response") if winner else None,
            "score": winner.get("score", -9999) if winner else -9999,
            "is_refusal": winner.get("refused", True) if winner else True,
            "latency_ms": winner.get("latency_ms", 0) if winner else 0,
            "hedge_count": winner.get("hedge_count", winner.get("hedges", 0)) if winner else 0,
            "all_results": results,
            "refusal_count": sum(1 for item in results if item.get("refused")),
            "total_models": len(results)}


def _bounded_result(result: dict) -> dict:
    item = {
        "model": result.get("model") or result.get("model_id"),
        "name": result.get("name") or result.get("codename"),
        "score": result.get("score", -9999),
        "latency_ms": result.get("latency_ms") if result.get("latency_ms") is not None
        else int(float(result.get("latency") or 0) * 1000),
        "is_refusal": bool(result.get("is_refusal", result.get("refused", True))),
        "hedge_count": int(result.get("hedge_count", result.get("hedges", 0)) or 0),
        "error": result.get("error"),
    }
    content = result.get("content") if result.get("content") is not None else result.get("response")
    item["content_preview"] = str(content or "")[:500]
    return item


def _normalize_race_result(result: dict) -> dict:
    result = dict(result)
    all_results = result.get("all_results") or []
    result["all_results"] = [_bounded_result(item) for item in all_results if isinstance(item, dict)]
    result["latency_ms"] = result.get("latency_ms") if result.get("latency_ms") is not None \
        else int(float(result.get("latency") or 0) * 1000)
    result["hedge_count"] = int(result.get("hedge_count", result.get("hedges", 0)) or 0)
    result.pop("latency", None)
    return result


def execute(agent: dict, args: dict) -> dict:
    prompt = str(args.get("prompt") or args.get("query") or "")
    if not prompt:
        return {"error": "prompt is required"}
    backend = str(args.get("backend") or "evonic").lower()
    if str(args.get("race_type") or "ultraplinian") == "classic" and backend != "openrouter":
        return {"error": "classic race requires backend=openrouter", "backend": backend}
    try:
        args = {**args, "_agent_id": str(agent.get("id") or "")}
        result = _openrouter(args, prompt) if backend == "openrouter" else _evonic(args, prompt)
    except Exception as exc:
        return {"error": str(exc), "backend": backend}
    if isinstance(result, dict):
        result = _normalize_race_result(result)
        result["backend"] = backend
        result["quota_warning"] = "Each model request consumes provider quota."
    return result
