"""Regression tests for NIK tools registered exclusively by the skill."""

import json
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def test_nik_skill_is_available_and_plugin_has_no_tool_definitions(monkeypatch):
    from backend.plugin_lifecycle import PluginManager
    from backend.skills_manager import SkillsManager

    monkeypatch.setattr('backend.skills_manager.SKILLS_DIR', str(ROOT / 'skills'))
    monkeypatch.setattr('backend.plugin_lifecycle.PLUGINS_DIR', str(ROOT / 'plugins'))
    skills = SkillsManager()
    plugins = PluginManager()

    with patch('models.db.db.get_setting', return_value=None):
        skill = skills.get_skill('nik_parser')
        assert skill and skill['enabled'] is True
        assert skill['variables']
        assert {tool['function']['name'] for tool in skills.get_all_skill_tool_defs()
                if tool['_skill_id'] == 'nik_parser'} == {
            'nik_parse', 'nik_validate', 'nik_detect_crop', 'nik_image_crop',
            'nik_extract_from_image',
        }

    with patch('backend.plugin_lifecycle.PluginManager.list_plugins', return_value=[{
        'id': 'nik_parser', 'enabled': True, '_dir': str(ROOT / 'plugins/nik_parser'),
    }]):
        assert plugins.get_all_plugin_tool_defs() == []

    manifest = json.loads((ROOT / 'plugins/nik_parser/plugin.json').read_text())
    assert 'tools_file' not in manifest
