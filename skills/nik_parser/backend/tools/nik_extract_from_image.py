from ._bridge import read_agent_image
from ...service import extract


def execute(agent: dict, args: dict):
    model_id = args.get("vision_model_id")
    model_id = model_id.strip() if isinstance(model_id, str) else None
    try:
        return extract(*read_agent_image(agent, args), model_id=model_id)
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc), "officially_verified": False}
