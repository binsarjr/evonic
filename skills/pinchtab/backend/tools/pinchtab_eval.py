"""pinchtab_eval — evaluate JavaScript expression in a browser tab."""

from ._pinchtab_api import _api


def execute(agent: dict, args: dict) -> dict:
    """Evaluate a JavaScript expression in a browser tab.

    Uses PinchTab's POST /evaluate endpoint. The expression runs in
    the page context of the specified tab and its return value is
    serialized and returned.  Use this for cookie injection, DOM
    inspection, or any custom JavaScript execution.

    Args:
        tab_id: ID of the tab (required).
        expression: JavaScript expression to evaluate (required).

    Returns:
        Evaluation result from PinchTab.
    """
    tab_id = args.get("tab_id", "")
    expression = args.get("expression", "")

    if not tab_id:
        return {"error": "tab_id is required."}
    if not expression:
        return {"error": "expression is required."}

    result = _api("POST", "/evaluate", {
        "tabId": tab_id,
        "expression": expression,
    })

    if "error" in result:
        return result

    return {
        "message": "Expression evaluated successfully.",
        "result": result,
    }
