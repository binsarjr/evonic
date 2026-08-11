from ._godmode.parseltongue import escalate_encoding, generate_variants, obfuscate_query


def execute(agent: dict, args: dict) -> dict:
    prompt = str(args.get("prompt") or "")
    if not prompt:
        return {"error": "prompt is required"}
    technique = str(args.get("technique") or "")
    custom_triggers = [str(item) for item in (args.get("custom_triggers") or [])]
    if args.get("escalation_level") is not None:
        level = max(0, min(int(args["escalation_level"]), 4))
        text, label = escalate_encoding(prompt, level)
        return {"escalation_level": level, "label": label,
                "text": text, "executed": False}
    if technique:
        return {"technique": technique,
                "text": obfuscate_query(prompt, technique, custom_triggers or None),
                "executed": False}
    tier = str(args.get("tier") or "standard").lower()
    if tier not in {"light", "standard", "heavy"}:
        tier = "standard"
    limit = max(1, min(int(args.get("limit") or 22), 33))
    return {"tier": tier, "variants": generate_variants(prompt, tier, custom_triggers)[:limit],
            "executed": False}
