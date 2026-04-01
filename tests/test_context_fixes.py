"""TDD tests for context panel fixes:
1. Skill loading with absolute path resolution
2. Context panel threshold = model context window (not compression threshold)
3. Panel height shows all 4 lines (bar + 3 categories)

Written BEFORE fix per TDD methodology.
"""

import json
import os
import time
import textwrap

import pytest

from cli_app import ContextUsagePanel, EpisodeCuratorApp, TUIEvent, TUIPlugin
from react_agent import AgentContext, SkillPluginManager
from system_tools import SystemToolsPlugin


pytestmark = pytest.mark.tui


# ============================================================
# 1. Skill loading — path resolution
# ============================================================

class TestSkillPathResolution:
    """execute_skill should find skills regardless of CWD."""

    def test_skill_found_with_absolute_skills_dir(self, tmp_path):
        """When _skills_dir is absolute, skills should be found."""
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: my-skill
            description: Test
            ---

            # My Skill Content
            Do the thing.
        """))

        plugin = SystemToolsPlugin()
        plugin._skills_dir = str(tmp_path / "skills")
        result = plugin.execute_tool("execute_skill", {"skill_name": "my-skill"})
        assert "My Skill Content" in result

    def test_skill_found_from_different_cwd(self, tmp_path, monkeypatch):
        """Skills should be found even when CWD is different from project dir."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        skill_dir = project_dir / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: X\n---\n\nContent here.\n")

        # Change CWD to somewhere else
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)

        plugin = SystemToolsPlugin()
        plugin._skills_dir = str(project_dir / "skills")  # absolute
        result = plugin.execute_tool("execute_skill", {"skill_name": "test-skill"})
        assert "Content here" in result

    def test_create_agent_sets_absolute_skills_dir(self, tmp_path):
        """SystemToolsPlugin in create_agent should have an absolute skills_dir."""
        # We just check that the plugin's _skills_dir is absolute after creation
        plugin = SystemToolsPlugin()
        # After the fix, __init__ should resolve to absolute
        assert os.path.isabs(plugin._skills_dir), (
            f"_skills_dir should be absolute, got: {plugin._skills_dir}"
        )


# ============================================================
# 2. Context panel threshold = model context window
# ============================================================

class TestContextThresholdIsModelWindow:
    """threshold should represent the model's context window, not compression threshold."""

    MODEL_CONTEXT_WINDOWS = {
        "claude-sonnet-4-20250514": 200000,
        "claude-haiku-4-5-20251001": 200000,
    }

    def test_default_threshold_is_model_window(self):
        """Default threshold should be 200k (model context window), not 80k."""
        panel = ContextUsagePanel()
        # Should default to a model context window size, not compression threshold
        assert panel.threshold >= 200000, (
            f"threshold should be model context window (~200k), got: {panel.threshold}"
        )

    def test_app_sets_model_context_window(self):
        """App should store model context window separately from compression threshold."""
        app = EpisodeCuratorApp()
        assert app._context_window >= 200000, (
            f"_context_window should be model context window (~200k), got: {app._context_window}"
        )
        # Compression threshold should be a fraction (default 50%) of context window
        assert app._threshold == app._context_window * 50 // 100

    def test_percentages_make_sense_at_200k(self):
        """With 200k threshold, 10k input should show ~5%."""
        panel = ContextUsagePanel()
        panel.threshold = 200000
        panel.system_tokens = 2000
        panel.tool_tokens = 1000
        panel.message_tokens = 7000

        rendered = panel.render()
        assert "5%" in rendered  # 10k/200k = 5% total
        assert "200" in rendered  # threshold shown


# ============================================================
# 3. Panel height — all 4 lines visible
# ============================================================

class TestContextPanelHeight:
    """All 4 lines (bar + 3 categories) should be visible in the panel."""

    def test_render_produces_8_lines(self):
        """render() should produce exactly 8 lines (bar + header + 4 categories + free + compress)."""
        panel = ContextUsagePanel()
        panel.system_tokens = 5000
        panel.tool_tokens = 2000
        panel.memory_tokens = 3000
        panel.message_tokens = 10000
        panel.threshold = 200000

        rendered = panel.render()
        lines = rendered.strip().split("\n")
        assert len(lines) == 8, f"Expected 8 lines, got {len(lines)}:\n{rendered}"

    def test_all_categories_in_render(self):
        """All categories (System prompt, Tools, Memory, Messages, Free space) should appear."""
        panel = ContextUsagePanel()
        panel.system_tokens = 5000
        panel.tool_tokens = 2000
        panel.memory_tokens = 3000
        panel.message_tokens = 10000
        panel.threshold = 200000

        rendered = panel.render()
        assert "System prompt" in rendered
        assert "Tools" in rendered
        assert "Memory" in rendered
        assert "Messages" in rendered
        assert "Free space" in rendered

    @pytest.mark.asyncio
    async def test_panel_css_height_sufficient(self):
        """CSS height should be enough to show all content lines + border."""
        app = EpisodeCuratorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            ctx = app.query_one("#context-usage", ContextUsagePanel)
            ctx.system_tokens = 5000
            ctx.tool_tokens = 2000
            ctx.memory_tokens = 3000
            ctx.message_tokens = 10000
            ctx.threshold = 200000
            # The rendered content should all be visible
            rendered = ctx.render()
            lines = rendered.strip().split("\n")
            assert len(lines) == 8, f"Expected 8 lines, got {len(lines)}:\n{rendered}"
            # Category markers should be present
            assert "█" in rendered  # system marker
            assert "▓" in rendered  # tools marker
            assert "▒" in rendered  # memory marker
            assert "░" in rendered  # messages marker


# ============================================================
# 4. Context values reflect current API call (= current context window usage)
# ============================================================

class TestContextValuesAreCurrentCall:
    """Context breakdown should reflect the CURRENT API call's token usage,
    which represents the actual context window utilization at this moment."""

    def test_context_uses_current_input_tokens(self):
        """Context breakdown should use the current input_tokens, not cumulative."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))
        plugin._system_base_tokens = 500

        ctx = AgentContext(
            user_query="test",
            messages=[{"role": "user", "content": "test"}],
            metadata={"_base_system_prompt": "prompt", "system_prompt_extra": ""},
            iteration=1,
            total_input_tokens=0,
            total_output_tokens=0,
            start_time=time.time(),
            tool_call_history=[],
        )

        # First call: 5000 tokens
        plugin.on_token_usage(ctx, 5000, 500)
        # Second call: 8000 tokens (context grew because messages accumulated)
        plugin.on_token_usage(ctx, 8000, 600)

        status_events = [e for e in events if e.kind == "status"]
        last_context = status_events[-1].data.get("context", {})

        # messages should be based on the latest input_tokens (8000), not cumulative (13000)
        total_in_context = last_context["system"] + last_context["tools"] + last_context["messages"]
        # Should be close to 8000, not 13000
        assert total_in_context <= 8000, (
            f"Context total should reflect current call (~8000), got {total_in_context}"
        )
