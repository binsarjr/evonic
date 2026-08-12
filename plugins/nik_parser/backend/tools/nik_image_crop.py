from __future__ import annotations

from ._bridge import module, read_agent_image


def execute(agent: dict, args: dict):
    try:
        return module("service").image_crop(
            *read_agent_image(agent, args), crop_type=args.get("crop_type", "nik"),
            coordinates=args.get("coordinates"), coordinate_space=args.get("coordinate_space", "pixel"),
            output_format=args.get("output_format", "metadata"),
            output_mime_type=args.get("output_mime_type", "image/jpeg"),
        )
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc), "officially_verified": False}
