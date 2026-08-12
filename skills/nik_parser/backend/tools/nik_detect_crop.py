from ._bridge import read_agent_image
from ...service import detect


def execute(agent: dict, args: dict):
    try:
        return detect(*read_agent_image(agent, args))
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc), "officially_verified": False}
