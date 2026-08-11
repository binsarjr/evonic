from ._lib import score_response


def execute(agent: dict, args: dict) -> dict:
    return score_response(str(args.get("response") or ""), int(args.get("latency_ms") or 0),
                          str(args.get("query") or ""))
