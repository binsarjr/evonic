from __future__ import annotations

from pathlib import Path


def read_agent_image(agent: dict, args: dict):
    path = args.get("path")
    path = path.strip() if isinstance(path, str) else ""
    if not path:
        raise ValueError("IMAGE_PATH_REQUIRED")
    if path.startswith("/_self/"):
        from backend.tools._workspace import effective_agent_id, resolve_self_path
        path = resolve_self_path(effective_agent_id(agent), path) or ""
    candidate = Path(path).resolve()
    attachments = (Path("data") / "attachments" / str(agent.get("id") or "")).resolve()
    artifacts = (Path("agents") / str(agent.get("id") or "") / "artifacts").resolve()
    if not candidate.is_file() or not any(parent == candidate or parent in candidate.parents for parent in (attachments, artifacts)):
        raise ValueError("IMAGE_PATH_NOT_AUTHORIZED")
    suffix = candidate.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if not mime:
        raise ValueError("UNSUPPORTED_IMAGE_TYPE")
    return candidate.read_bytes(), mime
