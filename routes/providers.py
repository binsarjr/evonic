from typing import Any, Dict

import requests
import httpx
from flask import Blueprint, jsonify, request

from models.db import db
from backend.provider.adapters import get_provider

providers_bp = Blueprint("providers", __name__)

_SENSITIVE_KEYS = frozenset({"api_key", "refresh_token"})


def _sanitize(provider: Dict[str, Any]) -> Dict[str, Any]:
    provider["credential_configured"] = bool(provider.get("api_key")) or (
        provider.get("credential_source") == "claude_code"
    )
    for key in _SENSITIVE_KEYS:
        provider.pop(key, None)
    return provider


@providers_bp.route("/api/providers", methods=["GET"])
def api_list_providers():
    providers = db.get_providers()
    for p in providers:
        _sanitize(p)
        p["model_count"] = len(db.get_models_by_provider(p["id"]))
    return jsonify({"providers": providers})


@providers_bp.route("/api/providers/<provider_id>", methods=["GET"])
def api_get_provider(provider_id):
    provider = db.get_provider(provider_id)
    if not provider:
        return jsonify({"error": "Provider not found"}), 404
    return jsonify(_sanitize(provider))


@providers_bp.route("/api/providers", methods=["POST"])
def api_create_provider():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"success": False, "error": "name is required"}), 400

    pid = data.get("id") or data["name"].lower().replace(" ", "_")
    existing = db.get_provider(pid)
    if existing:
        return jsonify({"success": False, "error": f"Provider '{pid}' already exists"}), 409

    try:
        new_id = db.create_provider(data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": True, "provider_id": new_id})


@providers_bp.route("/api/providers/<provider_id>", methods=["PUT"])
def api_update_provider(provider_id):
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    provider = db.get_provider(provider_id)
    if not provider:
        return jsonify({"success": False, "error": "Provider not found"}), 404

    if "api_key" in data and not data["api_key"]:
        del data["api_key"]

    success = db.update_provider(provider_id, data)
    if not success:
        return jsonify({"success": False, "error": "No changes made"}), 400

    return jsonify({"success": True})


@providers_bp.route("/api/providers/<provider_id>", methods=["DELETE"])
def api_delete_provider(provider_id):
    provider = db.get_provider(provider_id)
    if not provider:
        return jsonify({"success": False, "error": "Provider not found"}), 404

    success = db.delete_provider(provider_id)
    if not success:
        return jsonify(
            {"success": False, "error": "Cannot delete provider that has models. Remove its models first."}
        ), 409

    return jsonify({"success": True})


@providers_bp.route("/api/providers/<provider_id>/fetch-models", methods=["POST"])
def api_fetch_provider_models(provider_id):
    """Fetch available models from a provider's API endpoint."""
    provider = db.get_provider(provider_id)
    if not provider:
        return jsonify({"error": "Provider not found"}), 404

    base_url = provider.get("base_url", "").rstrip("/")
    if not base_url:
        return jsonify({"success": False, "error": "Provider has no base_url configured"}), 400

    api_format = provider.get("api_format", "openai")
    adapter = get_provider(provider)

    headers = {"Content-Type": "application/json"}
    if api_format == "codex":
        from backend.provider.oauth_codex import get_valid_token
        token = get_valid_token(db, provider_id)
        if not token:
            return jsonify({"success": False, "error": "Not connected. Click Connect first."}), 401
        headers["Authorization"] = f"Bearer {token}"
    elif api_format == "anthropic":
        from backend.provider.claude_code import auth_headers, resolve_credential
        token, oauth = resolve_credential(db, provider_id)
        if not token:
            return jsonify({"success": False, "error": "Not connected. Set up credentials first."}), 401
        headers = auth_headers(token, oauth)
    else:
        api_key = provider.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = adapter.fetch_models(headers, timeout=15)
        if resp.status_code != 200:
            return jsonify({
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "status_code": resp.status_code,
            })

        discovered = adapter.parse_models(resp.json())
        models = [{"id": m["id"], "name": m["name"]} for m in discovered]

        # Mark which ones are already added
        existing = {m["model_name"] for m in db.get_models_by_provider(provider_id)}
        for m in models:
            m["already_added"] = m["id"] in existing

        return jsonify({"success": True, "models": models, "total": len(models)})

    except (requests.exceptions.Timeout, httpx.TimeoutException):
        return jsonify({"success": False, "error": "Connection timed out"}), 408
    except (requests.exceptions.ConnectionError, httpx.ConnectError) as e:
        return jsonify({"success": False, "error": f"Connection error: {str(e)[:200]}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Error: {str(e)[:200]}"}), 500


@providers_bp.route("/api/providers/<provider_id>/test", methods=["POST"])
def api_test_provider(provider_id):
    """Test connection to a provider's endpoint."""
    provider = db.get_provider(provider_id)
    if not provider:
        return jsonify({"error": "Provider not found"}), 404

    base_url = provider.get("base_url", "").rstrip("/")
    if not base_url:
        return jsonify({"success": False, "error": "Provider has no base_url configured"}), 400

    api_format = provider.get("api_format", "openai")
    adapter = get_provider(provider)

    headers = {}
    if api_format == "codex":
        from backend.provider.oauth_codex import get_valid_token
        token = get_valid_token(db, provider_id)
        if not token:
            return jsonify({"success": False, "error": "Not connected. Click Connect first."}), 401
        headers["Authorization"] = f"Bearer {token}"
    elif api_format == "anthropic":
        from backend.provider.claude_code import auth_headers, resolve_credential
        token, oauth = resolve_credential(db, provider_id)
        if not token:
            return jsonify({"success": False, "error": "Not connected. Set up credentials first."}), 401
        headers = auth_headers(token, oauth)
    else:
        api_key = provider.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = adapter.fetch_models(headers, timeout=10)
        if resp.status_code == 200:
            try:
                data = resp.json()
                if api_format == "ollama":
                    models = data.get("models", [])
                else:
                    models = data.get("data", data.get("models", []))
                count = len(models) if isinstance(models, list) else "?"
            except Exception:
                count = "?"
            return jsonify({"success": True, "message": f"Connected ({count} models available)"})
        else:
            return jsonify({
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "status_code": resp.status_code,
            })
    except (requests.exceptions.Timeout, httpx.TimeoutException):
        return jsonify({"success": False, "error": "Connection timed out"}), 408
    except (requests.exceptions.ConnectionError, httpx.ConnectError) as e:
        return jsonify({"success": False, "error": f"Connection error: {str(e)[:200]}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Error: {str(e)[:200]}"}), 500


@providers_bp.route("/api/providers/<provider_id>/add-model", methods=["POST"])
def api_add_model_from_provider(provider_id):
    """Quick-add a model from the provider's discovered models list."""
    provider = db.get_provider(provider_id)
    if not provider:
        return jsonify({"error": "Provider not found"}), 404

    data = request.get_json()
    model_name = data.get("model_name") if data else None
    if not model_name:
        return jsonify({"success": False, "error": "model_name is required"}), 400

    display_name = data.get("name") or model_name.split("/")[-1]

    model_data = {
        "name": display_name,
        "type": provider.get("type", "remote"),
        "provider": provider_id,
        "base_url": "",
        "api_key": "",
        "model_name": model_name,
        "max_tokens": data.get("max_tokens", 32768),
        "timeout": data.get("timeout", 60),
        "thinking": data.get("thinking", 0),
        "thinking_budget": data.get("thinking_budget", 0),
        "enabled": 1,
        "api_format": provider.get("api_format", "openai"),
    }

    try:
        new_id = db.create_model(model_data)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 409

    return jsonify({"success": True, "model_id": new_id})
