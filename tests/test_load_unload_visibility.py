"""TDD tests for load/unload visibility in context detail panels.

Issues to reproduce:
1. tools panel (Ctrl+T) uses get_all_tool_definitions() — no difference after load/unload
2. system panel (Ctrl+D) contains deferred tool catalog descriptions —
   overlaps with tools panel content
3. tools panel should show ACTIVE tools (reflecting load/unload state)
4. system panel should show system prompt + facts/index, NOT tool descriptions

Written BEFORE fix per TDD methodology.
"""

import json
import time

import pytest

from cli_app import EpisodeCuratorApp, TUIEvent, TUIPlugin
from react_agent import AgentContext, SkillPluginManager
from system_tools import SystemToolsPlugin
from tool_registry import ToolRegistryPlugin


pytestmark = pytest.mark.tui


# ============================================================
# Unit: ToolRegistryPlugin catalog in system_prompt_extra
# ============================================================

class TestToolCatalogPlacement:
    """Deferred tool catalog should be in system_prompt_extra for LLM,
    but context detail system panel should separate system prompt from tool catalog."""

    def test_system_prompt_extra_contains_deferred_catalog(self):
        """on_agent_start should inject deferred tool catalog into system_prompt_extra."""
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        ctx = AgentContext(
            user_query="test",
            messages=[{"role": "user", "content": "test"}],
            metadata={"_base_system_prompt": "You are helpful."},
            iteration=0,
            total_input_tokens=0,
            total_output_tokens=0,
            start_time=time.time(),
            tool_call_history=[],
        )
        mgr.dispatch_on_agent_start(ctx)

        extra = ctx.metadata.get("system_prompt_extra", "")
        # Deferred catalog SHOULD be in system_prompt_extra (for LLM)
        assert "Deferred tools" in extra
        assert "read" in extra  # system tool listed

    def test_load_changes_active_set(self):
        """After load_tools, tools should move from deferred to active."""
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        # Initially, system tools are deferred
        active_before = mgr.get_active_tool_definitions()
        active_names_before = {t["name"] for t in active_before}
        assert "read" not in active_names_before
        assert "bash" not in active_names_before

        # Load some tools
        mgr.load_tools(["read", "bash"])
        active_after = mgr.get_active_tool_definitions()
        active_names_after = {t["name"] for t in active_after}
        assert "read" in active_names_after
        assert "bash" in active_names_after

    def test_unload_removes_from_active(self):
        """After unload_tools, tools should be removed from active set."""
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        mgr.load_tools(["read", "bash", "grep"])
        active = {t["name"] for t in mgr.get_active_tool_definitions()}
        assert "read" in active

        mgr.unload_tools(["read"])
        active_after = {t["name"] for t in mgr.get_active_tool_definitions()}
        assert "read" not in active_after
        assert "bash" in active_after  # still loaded

    def test_active_count_changes_on_load_unload(self):
        """Active tool count should change when loading/unloading."""
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        count_initial = len(mgr.get_active_tool_definitions())

        mgr.load_tools(["read", "write", "bash", "grep", "search"])
        count_loaded = len(mgr.get_active_tool_definitions())
        assert count_loaded == count_initial + 5

        mgr.unload_tools(["read", "write", "bash"])
        count_unloaded = len(mgr.get_active_tool_definitions())
        assert count_unloaded == count_loaded - 3


# ============================================================
# TUI: tools panel should reflect active/deferred status
# ============================================================

class TestToolsPanelReflectsLoadState:
    """Ctrl+T tools panel should show which tools are active vs deferred."""

    def test_build_tools_content_shows_status(self):
        """_build_tools_content should show [✓] active and [·] deferred tools."""
        app = EpisodeCuratorApp()
        # Simulate agent with manager
        from unittest.mock import MagicMock
        mock_agent = MagicMock()
        mock_agent._manager = SkillPluginManager()
        mock_agent._manager.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mock_agent._manager)
        mock_agent._manager.register(registry)
        app._agent = mock_agent

        content = app._build_tools_content()
        assert "[✓]" in content  # load_tools/unload_tools are active
        assert "[·]" in content  # system tools are deferred
        assert "Active tools" in content
        assert "Deferred tools" in content

    def test_build_tools_content_after_load(self):
        """After loading tools, they should move from [·] to [✓]."""
        app = EpisodeCuratorApp()
        from unittest.mock import MagicMock
        mock_agent = MagicMock()
        mock_agent._manager = SkillPluginManager()
        mock_agent._manager.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mock_agent._manager)
        mock_agent._manager.register(registry)
        app._agent = mock_agent

        # Before load
        content_before = app._build_tools_content()
        assert "[·] read:" in content_before

        # Load read
        mock_agent._manager.load_tools(["read"])
        content_after = app._build_tools_content()
        assert "[✓] read:" in content_after
        assert "[·] read:" not in content_after

    def test_build_tools_content_after_unload(self):
        """After unloading, tools should go back to [·]."""
        app = EpisodeCuratorApp()
        from unittest.mock import MagicMock
        mock_agent = MagicMock()
        mock_agent._manager = SkillPluginManager()
        mock_agent._manager.register(SystemToolsPlugin())
        registry = ToolRegistryPlugin(mock_agent._manager)
        mock_agent._manager.register(registry)
        app._agent = mock_agent

        mock_agent._manager.load_tools(["read", "bash"])
        mock_agent._manager.unload_tools(["read"])

        content = app._build_tools_content()
        assert "[·] read:" in content  # unloaded
        assert "[✓] bash:" in content  # still loaded


# ============================================================
# TUI: system panel should NOT contain tool descriptions
# ============================================================

class TestSystemPanelSeparation:
    """System panel (Ctrl+D) should show system prompt + context,
    NOT duplicate tool catalog descriptions that belong in tools panel."""

    def test_context_content_event_separates_system_and_tools(self):
        """TUIPlugin context_content event should provide tools separately."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(
            user_query="test",
            messages=[{"role": "user", "content": "test"}],
            metadata={
                "_base_system_prompt": "You are a helpful assistant.",
                "system_prompt_extra": (
                    "Known facts:\n- User uses Python\n\n"
                    "Deferred tools (use load_tools):\n"
                    "- read: Read a file\n"
                    "- bash: Execute shell"
                ),
            },
            iteration=1,
            total_input_tokens=1000,
            total_output_tokens=200,
            start_time=time.time(),
            tool_call_history=[],
        )
        plugin.on_token_usage(ctx, 1000, 200)

        context_events = [e for e in events if e.kind == "context_content"]
        assert len(context_events) >= 1
        system_content = context_events[0].data.get("system", "")

        # System should contain the base prompt and facts
        assert "helpful assistant" in system_content
        # System should contain context info (facts, index)
        assert "Known facts" in system_content

    def test_context_content_includes_tools_field(self):
        """context_content event should include a 'tools' field for tools panel."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(
            user_query="test",
            messages=[{"role": "user", "content": "test"}],
            metadata={
                "_base_system_prompt": "You are helpful.",
                "system_prompt_extra": "Deferred tools:\n- read: Read file",
            },
            iteration=1,
            total_input_tokens=1000,
            total_output_tokens=200,
            start_time=time.time(),
            tool_call_history=[],
        )
        plugin.on_token_usage(ctx, 1000, 200)

        context_events = [e for e in events if e.kind == "context_content"]
        assert len(context_events) >= 1
        # Should include tools info in the event
        data = context_events[0].data
        assert "tools" in data, (
            f"context_content event should include 'tools' field. Keys: {list(data.keys())}"
        )


# ============================================================
# Close-book compression should be visible in msgs panel
# ============================================================

class TestCloseBookVisibility:
    """After unload_tools, compressed tool_results should be visible in msgs panel."""

    def test_compression_visible_in_messages(self):
        """After unload, tool_result content should show [compressed] marker."""
        mgr = SkillPluginManager()
        sys_plugin = SystemToolsPlugin()
        mgr.register(sys_plugin)
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        ctx = AgentContext(
            user_query="test",
            messages=[
                {"role": "user", "content": "read a file"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_001", "name": "read",
                     "input": {"file_path": "/tmp/test.txt"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_001",
                     "content": "A" * 500},  # >200 chars
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I read the file."},
                ]},
            ],
            metadata={},
            iteration=1,
            total_input_tokens=5000,
            total_output_tokens=500,
            start_time=time.time(),
            tool_call_history=[],
        )

        # Load then unload to trigger compression
        mgr.load_tools(["read"])
        result = mgr.route_tool_call("unload_tools", {"names": ["read"]})
        registry.after_action(ctx,
                              {"name": "unload_tools", "input": {"names": ["read"]}},
                              result)

        # Check: tool_result should be compressed
        tool_result_block = ctx.messages[2]["content"][0]
        assert "compressed" in tool_result_block["content"]
        assert "load_tools" in tool_result_block["content"]

    def test_reexpand_restores_content(self):
        """After re-loading, compressed content should be restored."""
        mgr = SkillPluginManager()
        sys_plugin = SystemToolsPlugin()
        mgr.register(sys_plugin)
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        original = "B" * 500
        ctx = AgentContext(
            user_query="test",
            messages=[
                {"role": "user", "content": "grep something"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_002", "name": "grep",
                     "input": {"pattern": "test", "path": "."}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_002",
                     "content": original},
                ]},
            ],
            metadata={},
            iteration=1,
            total_input_tokens=5000,
            total_output_tokens=500,
            start_time=time.time(),
            tool_call_history=[],
        )

        # Load → unload → compress
        mgr.load_tools(["grep"])
        mgr.unload_tools(["grep"])
        registry.after_action(ctx,
                              {"name": "unload_tools", "input": {"names": ["grep"]}},
                              "ok")

        compressed = ctx.messages[2]["content"][0]["content"]
        assert "compressed" in compressed

        # Re-load → expand
        mgr.load_tools(["grep"])
        registry.after_action(ctx,
                              {"name": "load_tools", "input": {"names": ["grep"]}},
                              "ok")

        restored = ctx.messages[2]["content"][0]["content"]
        assert restored == original
