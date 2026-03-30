"""TDD tests for context panel redesign and skill/msgs visibility.

Issues:
1. Skill content in messages should be compressible via close-book
2. msgs panel should show tool_result content (not just metadata)
3. ContextUsagePanel should be a single stacked bar with color-coded segments

Written BEFORE fix per TDD methodology.
"""

import json
import time

import pytest

from cli_app import ContextUsagePanel, EpisodeCuratorApp, TUIEvent, TUIPlugin, _estimate_tokens
from react_agent import AgentContext, SkillPluginManager
from system_tools import SystemToolsPlugin
from tool_registry import ToolRegistryPlugin


pytestmark = pytest.mark.tui


# ============================================================
# 1. Skill content compressible via close-book
# ============================================================

class TestSkillCloseBook:
    """execute_skill results should be compressible via unload_tools."""

    def test_execute_skill_result_compressed_on_unload(self):
        """When execute_skill is unloaded, its large tool_result should be compressed."""
        mgr = SkillPluginManager()
        sys_plugin = SystemToolsPlugin()
        mgr.register(sys_plugin)
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        # Simulate: LLM called execute_skill, got a big skill body back
        skill_body = "# Data Analysis\n" + "Lorem ipsum dolor sit amet. " * 50  # ~1400 chars
        ctx = AgentContext(
            user_query="help me analyze data",
            messages=[
                {"role": "user", "content": "help me analyze data"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_skill_1", "name": "execute_skill",
                     "input": {"skill_name": "data-analysis"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_skill_1",
                     "content": skill_body},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I've loaded the data analysis skill."},
                ]},
            ],
            metadata={},
            iteration=2,
            total_input_tokens=5000,
            total_output_tokens=500,
            start_time=time.time(),
            tool_call_history=[],
        )

        # Load then unload execute_skill
        mgr.load_tools(["execute_skill"])
        mgr.unload_tools(["execute_skill"])
        registry.after_action(ctx,
                              {"name": "unload_tools", "input": {"names": ["execute_skill"]}},
                              "ok")

        # The tool_result should be compressed
        tool_result = ctx.messages[2]["content"][0]
        assert "compressed" in tool_result["content"]
        assert "load_tools" in tool_result["content"]
        assert len(tool_result["content"]) < len(skill_body)

    def test_execute_skill_result_reexpands_on_load(self):
        """Re-loading execute_skill should restore the original content."""
        mgr = SkillPluginManager()
        sys_plugin = SystemToolsPlugin()
        mgr.register(sys_plugin)
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        skill_body = "# Full Skill Content\n" + "Detailed instructions. " * 30
        ctx = AgentContext(
            user_query="test",
            messages=[
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_s2", "name": "execute_skill",
                     "input": {"skill_name": "test"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_s2",
                     "content": skill_body},
                ]},
            ],
            metadata={},
            iteration=1,
            total_input_tokens=3000,
            total_output_tokens=200,
            start_time=time.time(),
            tool_call_history=[],
        )

        # Compress
        mgr.load_tools(["execute_skill"])
        mgr.unload_tools(["execute_skill"])
        registry.after_action(ctx,
                              {"name": "unload_tools", "input": {"names": ["execute_skill"]}},
                              "ok")
        assert "compressed" in ctx.messages[2]["content"][0]["content"]

        # Re-expand
        mgr.load_tools(["execute_skill"])
        registry.after_action(ctx,
                              {"name": "load_tools", "input": {"names": ["execute_skill"]}},
                              "ok")
        assert ctx.messages[2]["content"][0]["content"] == skill_body


# ============================================================
# 2. msgs panel should show tool_result content summary
# ============================================================

class TestMsgsPanelToolResultVisibility:
    """msgs panel should show tool_result content (truncated), not just metadata."""

    def test_tool_result_shows_content_preview(self):
        """tool_result blocks should include content preview, not just [tool_result:id]."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(
            user_query="test",
            messages=[
                {"role": "user", "content": "read my file"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_r1", "name": "read",
                     "input": {"file_path": "/tmp/test.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_r1",
                     "content": "1\tdef hello():\n2\t    return 'world'\n3\t"},
                ]},
            ],
            metadata={"_base_system_prompt": "You are helpful."},
            iteration=1,
            total_input_tokens=1000,
            total_output_tokens=200,
            start_time=time.time(),
            tool_call_history=[],
        )
        plugin.on_token_usage(ctx, 1000, 200)

        ctx_events = [e for e in events if e.kind == "context_content"]
        msgs = ctx_events[0].data.get("msgs", "")
        # tool_result content preview should appear, not just [tool_result:tu_r1]
        assert "hello" in msgs or "def " in msgs, (
            f"msgs should show tool_result content preview, not just metadata. Got:\n{msgs}"
        )

    def test_tool_result_shows_compressed_marker(self):
        """Compressed tool_result should show [compressed] marker in msgs."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(
            user_query="test",
            messages=[
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_c1", "name": "grep",
                     "input": {"pattern": "foo", "path": "."}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_c1",
                     "content": "[grep result compressed] file.py:1: foo... (500 chars — use load_tools to re-expand)"},
                ]},
            ],
            metadata={"_base_system_prompt": "prompt"},
            iteration=1,
            total_input_tokens=500,
            total_output_tokens=100,
            start_time=time.time(),
            tool_call_history=[],
        )
        plugin.on_token_usage(ctx, 500, 100)

        ctx_events = [e for e in events if e.kind == "context_content"]
        msgs = ctx_events[0].data.get("msgs", "")
        assert "compressed" in msgs, (
            f"msgs should show compressed marker. Got:\n{msgs}"
        )


# ============================================================
# 3. ContextUsagePanel — single stacked bar with segments
# ============================================================

class TestContextUsagePanelRedesign:
    """Context panel should be a single stacked bar with color-coded segments."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    def test_stacked_bar_has_all_segments(self):
        """Rendered output should show a single bar with system/tools/msgs segments."""
        panel = ContextUsagePanel()
        panel.system_tokens = 2000
        panel.tool_tokens = 1000
        panel.message_tokens = 7000
        panel.threshold = 200000

        rendered = panel.render()
        # Should contain a single combined bar, not three separate ones
        # Should show total usage percentage
        assert "5%" in rendered  # 10k/200k = 5%
        # Should have category breakdown below the bar
        assert "system" in rendered.lower()
        assert "tools" in rendered.lower()
        assert "msgs" in rendered.lower()

    def test_shows_threshold(self):
        """Should show total vs threshold (e.g. 200k)."""
        panel = ContextUsagePanel()
        panel.system_tokens = 5000
        panel.tool_tokens = 2000
        panel.message_tokens = 3000
        panel.threshold = 200000

        rendered = panel.render()
        assert "200" in rendered  # threshold

    def test_shows_category_token_counts(self):
        """Each category should show token count and percentage."""
        panel = ContextUsagePanel()
        panel.system_tokens = 4000
        panel.tool_tokens = 2000
        panel.message_tokens = 4000
        panel.threshold = 200000

        rendered = panel.render()
        # Should show individual token counts
        assert "4.0k" in rendered or "4000" in rendered  # system or msgs
        assert "2.0k" in rendered or "2000" in rendered  # tools

    def test_percentages_of_threshold(self):
        """Percentages should be relative to threshold, not total used."""
        panel = ContextUsagePanel()
        panel.system_tokens = 20000
        panel.tool_tokens = 10000
        panel.message_tokens = 70000
        panel.threshold = 200000

        rendered = panel.render()
        # Total is 100k / 200k = 50%
        assert "50%" in rendered

    @pytest.mark.asyncio
    async def test_context_updates_from_event(self, app):
        """status event should still update context usage panel values."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("status", {
                "iteration": 1, "max_iterations": 30,
                "input_tokens": 10000, "output_tokens": 1000,
                "elapsed": 3.0, "episode_count": 2,
                "context": {"system": 2500, "tools": 1500, "messages": 6000},
            }))
            await pilot.pause()

            ctx = app.query_one("#context-usage", ContextUsagePanel)
            assert ctx.system_tokens == 2500
            assert ctx.tool_tokens == 1500
            assert ctx.message_tokens == 6000
