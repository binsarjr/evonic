import unittest
from unittest.mock import patch

from backend.plugin_lifecycle import PluginManager


class PluginAgentSettingsTests(unittest.TestCase):
    def test_select_only_accepts_declared_options(self):
        manager = PluginManager.__new__(PluginManager)
        manager.get_agent_settings_schema = lambda _plugin_id: [{
            "name": "mode",
            "label": "Mode",
            "type": "select",
            "options": ["preserve", {"value": "append", "label": "Append"}],
        }]

        with patch("models.db.db.set_setting") as save:
            self.assertEqual(manager.set_agent_plugin_settings(
                "demo", "agent-1", {"mode": "append"}
            ), {"success": True})
            save.assert_called_once_with(
                "plugin_agent_setting:demo:agent-1:mode", "append"
            )
            save.reset_mock()
            result = manager.set_agent_plugin_settings(
                "demo", "agent-1", {"mode": "invalid"}
            )

        self.assertEqual(result, {"error": "Invalid value for Mode"})
        save.assert_not_called()
