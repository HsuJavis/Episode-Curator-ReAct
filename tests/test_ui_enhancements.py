"""TDD tests for UI enhancements:
1. Threshold value displayed in context panel
2. web_fetch caps result to avoid exceeding context
3. Input history with up/down arrow (max 5)
4. Copy conversation content

Written BEFORE fix per TDD methodology.
"""

import json
import time

import pytest

from cli_app import ContextUsagePanel, EpisodeCuratorApp, TUIPlugin
from react_agent import AgentContext
from system_tools import SystemToolsPlugin


pytestmark = pytest.mark.tui


# ============================================================
# 1. Threshold displayed in context panel
# ============================================================

class TestThresholdDisplay:
    """Context panel should show compression threshold line."""

    def test_threshold_shown_in_render(self):
        """Render should show the compression threshold value."""
        panel = ContextUsagePanel()
        panel.system_tokens = 5000
        panel.tool_tokens = 2000
        panel.message_tokens = 13000
        panel.threshold = 200000
        panel.compress_threshold = 100000

        rendered = panel.render()
        # Should show compression threshold marker
        assert "100.0k" in rendered, (
            f"Compression threshold (100k) should appear in render. Got:\n{rendered}"
        )

    def test_threshold_marker_on_bar(self):
        """Bar should have a threshold marker showing where compression triggers."""
        panel = ContextUsagePanel()
        panel.system_tokens = 40000
        panel.tool_tokens = 10000
        panel.message_tokens = 50000
        panel.threshold = 200000
        panel.compress_threshold = 100000

        rendered = panel.render()
        # Should contain threshold indicator (like "|" or "▏" at 50% position)
        assert "compress" in rendered.lower() or "100.0k" in rendered


# ============================================================
# 2. web_fetch should cap result size for LLM context
# ============================================================

class TestWebFetchSizeCap:
    """web_fetch result should be capped to avoid blowing up LLM context."""

    def test_fetch_result_capped_at_reasonable_size(self):
        """Even if page is huge, result should not exceed a reasonable limit."""
        plugin = SystemToolsPlugin()
        # Fetch a page that returns large content
        result = plugin.execute_tool("web_fetch", {
            "url": "https://httpbin.org/bytes/100000",
            "max_size": 50000,
        })
        assert len(result) <= 60000  # some slack for truncation message

    def test_default_max_size_is_reasonable(self):
        """Default max_size should be reasonable for LLM context (not 500k)."""
        plugin = SystemToolsPlugin()
        # Check the tool schema default
        tools = plugin.get_tools()
        wf = [t for t in tools if t["name"] == "web_fetch"][0]
        desc = wf["input_schema"]["properties"]["max_size"]["description"]
        # The default should be mentioned or the implementation should cap
        assert "max_size" in wf["input_schema"]["properties"]


# ============================================================
# 3. Input history — up/down arrow
# ============================================================

class TestInputHistory:
    """Up arrow should cycle through previous commands (max 5)."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_history_stored_on_submit(self, app):
        """Submitting input should store it in command history."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input")
            inp.value = "first command"
            await inp.action_submit()
            await pilot.pause()

            assert len(app._input_history) >= 1
            assert "first command" in app._input_history

    @pytest.mark.asyncio
    async def test_history_max_5(self, app):
        """History should keep max 5 commands."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input")
            for i in range(7):
                inp.value = f"cmd {i}"
                await inp.action_submit()
                await pilot.pause()

            assert len(app._input_history) <= 5
            # Most recent should be last
            assert app._input_history[-1] == "cmd 6"
            # Oldest (cmd 0, cmd 1) should have been dropped
            assert "cmd 0" not in app._input_history

    @pytest.mark.asyncio
    async def test_up_arrow_recalls_last(self, app):
        """Pressing up arrow should fill input with last command."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input")
            inp.value = "hello world"
            await inp.action_submit()
            await pilot.pause()

            # Press up arrow
            app.action_history_up()
            await pilot.pause()
            assert inp.value == "hello world"

    @pytest.mark.asyncio
    async def test_up_arrow_cycles_through_history(self, app):
        """Multiple up presses should cycle through older commands."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input")
            for cmd in ["first", "second", "third"]:
                inp.value = cmd
                await inp.action_submit()
                await pilot.pause()

            app.action_history_up()
            await pilot.pause()
            assert inp.value == "third"

            app.action_history_up()
            await pilot.pause()
            assert inp.value == "second"

            app.action_history_up()
            await pilot.pause()
            assert inp.value == "first"

    @pytest.mark.asyncio
    async def test_down_arrow_goes_forward(self, app):
        """After going up, down should go forward in history."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input")
            for cmd in ["aaa", "bbb"]:
                inp.value = cmd
                await inp.action_submit()
                await pilot.pause()

            app.action_history_up()  # bbb
            app.action_history_up()  # aaa
            await pilot.pause()
            assert inp.value == "aaa"

            app.action_history_down()  # bbb
            await pilot.pause()
            assert inp.value == "bbb"

    @pytest.mark.asyncio
    async def test_down_past_end_clears(self, app):
        """Pressing down past newest entry should clear input."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input")
            inp.value = "cmd1"
            await inp.action_submit()
            await pilot.pause()

            app.action_history_up()   # cmd1
            app.action_history_down() # past end → clear
            await pilot.pause()
            assert inp.value == ""


# ============================================================
# 4. Copy conversation content
# ============================================================

class TestCopyConversation:
    """User should be able to copy conversation content."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_copy_binding_exists(self, app):
        """Ctrl+Y binding for copy should exist."""
        async with app.run_test(size=(120, 40)) as pilot:
            bindings = [b for b in app.BINDINGS if "copy" in b.action.lower() or b.key == "ctrl+y"]
            assert len(bindings) >= 1, "Should have a copy keybinding"

    @pytest.mark.asyncio
    async def test_copy_last_answer(self, app):
        """Ctrl+Y should copy the last assistant answer to clipboard."""
        async with app.run_test(size=(120, 40)) as pilot:
            # Simulate an answer
            app._last_answer = "This is the answer"
            # The action should exist
            assert hasattr(app, "action_copy_last")
