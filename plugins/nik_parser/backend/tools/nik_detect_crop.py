from ._bridge import module, read_agent_image

def execute(agent: dict, args: dict):
    try: return module("service").detect(*read_agent_image(agent, args))
    except (ValueError, RuntimeError) as exc: return {"error": str(exc), "officially_verified": False}
