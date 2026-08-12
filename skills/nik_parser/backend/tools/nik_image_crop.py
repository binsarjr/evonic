from ._bridge import read_agent_image
from ...service import image_crop


def execute(agent: dict, args: dict):
    try:
        return image_crop(*read_agent_image(agent, args), **args)
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc), "officially_verified": False}
