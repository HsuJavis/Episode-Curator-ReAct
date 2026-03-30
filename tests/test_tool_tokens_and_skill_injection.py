"""TDD tests for:
1. _tool_tokens should reflect active tools only (not all tools)
2. SkillLoaderPlugin should be registered in create_agent()
3. Skill metadata should be injected into LLM context

Written BEFORE fix per TDD methodology.
"""

import json
import time

import pytest

from cli_app import TUIPlugin, _estimate_tokens
from react_agent import AgentContext, SkillPluginManager
from system_tools import SystemToolsPlugin
from tool_registry import ToolRegistryPlugin


pytestmark = pytest.mark.tui


# ============================================================
# 1. _tool_tokens should reflect active tools only
# ============================================================

class TestToolTokenEstimation:
    """Context gauge tool token count should match active tool set."""

    def test_tool_tokens_only_counts_active(self):
        """_tool_tokens should be based on active tools, not all tools."""
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        # Only load_tools + unload_tools are active (2 tools)
        active = mgr.get_active_tool_definitions()
        all_tools = mgr.get_all_tool_definitions()
        assert len(active) < len(all_tools), "Deferred tools should not be active"

        # Token estimate for active should be much smaller than all
        active_tokens = _estimate_tokens(json.dumps(active))
        all_tokens = _estimate_tokens(json.dumps(all_tools))
        assert active_tokens < all_tokens

    def test_tool_tokens_updates_after_load(self):
        """After loading tools, token count should increase."""
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        active_before = mgr.get_active_tool_definitions()
        tokens_before = _estimate_tokens(json.dumps(active_before))

        mgr.load_tools(["read", "bash", "grep", "write", "search"])
        active_after = mgr.get_active_tool_definitions()
        tokens_after = _estimate_tokens(json.dumps(active_after))

        assert tokens_after > tokens_before

    def test_tool_tokens_decreases_after_unload(self):
        """After unloading tools, token count should decrease."""
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        mgr.load_tools(["read", "bash", "grep"])
        active_loaded = mgr.get_active_tool_definitions()
        tokens_loaded = _estimate_tokens(json.dumps(active_loaded))

        mgr.unload_tools(["read", "bash", "grep"])
        active_unloaded = mgr.get_active_tool_definitions()
        tokens_unloaded = _estimate_tokens(json.dumps(active_unloaded))

        assert tokens_unloaded < tokens_loaded

    def test_context_event_tool_tokens_reflects_active(self):
        """TUIPlugin context event should report tool tokens based on active set."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        # Simulate: only 2 active tools (load_tools, unload_tools)
        # _tool_tokens should be small (not 2.1k from 20 tool schemas)
        plugin._system_base_tokens = 100
        plugin._tool_tokens = 50  # small — only 2 tools active

        ctx = AgentContext(
            user_query="test",
            messages=[{"role": "user", "content": "test"}],
            metadata={"_base_system_prompt": "You are helpful."},
            iteration=1,
            total_input_tokens=1000,
            total_output_tokens=200,
            start_time=time.time(),
            tool_call_history=[],
        )
        plugin.on_token_usage(ctx, 1000, 200)

        status_events = [e for e in events if e.kind == "status"]
        assert len(status_events) >= 1
        context = status_events[0].data.get("context", {})
        # Tool tokens in context should be the small value (50), not a big one
        assert context["tools"] == 50


# ============================================================
# 2. SkillLoaderPlugin registration in create_agent
# ============================================================

class TestToolTokensDynamic:
    """Integration: _calc_active_tool_tokens reflects real active set."""

    def test_calc_active_changes_with_load_unload(self):
        """_calc_active_tool_tokens should change when tools are loaded/unloaded."""
        from unittest.mock import MagicMock
        app = __import__("cli_app").EpisodeCuratorApp()
        mock_agent = MagicMock()
        mock_agent._manager = SkillPluginManager()
        mock_agent._manager.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mock_agent._manager)
        mock_agent._manager.register(registry)
        app._agent = mock_agent

        tokens_before = app._calc_active_tool_tokens()

        mock_agent._manager.load_tools(["read", "write", "bash", "grep", "search"])
        tokens_loaded = app._calc_active_tool_tokens()
        assert tokens_loaded > tokens_before, (
            f"Expected more tokens after loading 5 tools: {tokens_before} → {tokens_loaded}"
        )

        mock_agent._manager.unload_tools(["read", "write", "bash", "grep", "search"])
        tokens_unloaded = app._calc_active_tool_tokens()
        assert tokens_unloaded < tokens_loaded, (
            f"Expected fewer tokens after unloading: {tokens_loaded} → {tokens_unloaded}"
        )
        assert tokens_unloaded == tokens_before


class TestSkillLoaderRegistration:
    """create_agent should register SkillLoaderPlugin."""

    @pytest.mark.llm
    def test_create_agent_includes_skill_loader(self, api_key, tmp_storage):
        """create_agent should register SkillLoaderPlugin."""
        from episode_curator import create_agent
        agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            storage_dir=tmp_storage,
            api_key=api_key,
        )
        plugin_names = [p.name for p in agent._manager._plugins]
        assert "skill_loader" in plugin_names, (
            f"SkillLoaderPlugin not registered. Plugins: {plugin_names}"
        )

    def test_skill_catalog_injected_on_agent_start(self, tmp_path):
        """Skill metadata should be in system_prompt_extra after on_agent_start."""
        from skill_loader import SkillManager, SkillLoaderPlugin

        # Create a test skill
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill for analysis\n---\n\nDo the thing.\n"
        )

        mgr = SkillPluginManager()
        skill_mgr = SkillManager(str(tmp_path / "skills"))
        skill_plugin = SkillLoaderPlugin(skill_mgr)
        mgr.register(skill_plugin)

        ctx = AgentContext(
            user_query="test",
            messages=[{"role": "user", "content": "test"}],
            metadata={},
            iteration=0,
            total_input_tokens=0,
            total_output_tokens=0,
            start_time=time.time(),
            tool_call_history=[],
        )
        mgr.dispatch_on_agent_start(ctx)

        extra = ctx.metadata.get("system_prompt_extra", "")
        assert "test-skill" in extra
        assert "A test skill for analysis" in extra

    def test_skill_and_tool_catalogs_coexist(self, tmp_path):
        """Both skill and deferred tool catalogs should be in system_prompt_extra."""
        from skill_loader import SkillManager, SkillLoaderPlugin

        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Does something cool\n---\n\nContent.\n"
        )

        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())

        skill_mgr = SkillManager(str(tmp_path / "skills"))
        skill_plugin = SkillLoaderPlugin(skill_mgr)
        mgr.register(skill_plugin)

        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        ctx = AgentContext(
            user_query="test",
            messages=[{"role": "user", "content": "test"}],
            metadata={},
            iteration=0,
            total_input_tokens=0,
            total_output_tokens=0,
            start_time=time.time(),
            tool_call_history=[],
        )
        mgr.dispatch_on_agent_start(ctx)

        extra = ctx.metadata.get("system_prompt_extra", "")
        # Both should be present
        assert "my-skill" in extra, f"Skill catalog missing. Extra:\n{extra[:500]}"
        assert "Deferred tools" in extra, f"Tool catalog missing. Extra:\n{extra[:500]}"
