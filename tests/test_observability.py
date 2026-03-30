"""TDD tests for observability (logging) and arrow key scope fix.

Requirements:
1. Structured logging to file — every agent event logged with context
2. Up/down arrow only works when input is focused, not in select mode
3. Token estimation uses actual API-reported values, not heuristic

Written BEFORE fix per TDD methodology.
"""

import json
import logging
import os
import time

import pytest

from cli_app import EpisodeCuratorApp, TUIEvent, TUIPlugin
from react_agent import AgentContext
from textual.widgets import Input, TextArea


pytestmark = pytest.mark.tui


# ============================================================
# 1. Logging — all events should be logged to file
# ============================================================

class TestLogging:
    """Agent should log all events to a structured log file."""

    def test_react_agent_has_logger(self):
        """react_agent module should have a logger."""
        import react_agent
        assert hasattr(react_agent, 'logger')

    def test_episode_curator_has_logger(self):
        """episode_curator module should have a logger."""
        import episode_curator
        assert hasattr(episode_curator, 'logger')

    def test_system_tools_has_logger(self):
        """system_tools module should have a logger."""
        import system_tools
        assert hasattr(system_tools, 'logger')

    def test_cli_app_has_logger(self):
        """cli_app module should have a logger."""
        import cli_app
        assert hasattr(cli_app, 'logger')

    def test_log_file_created_on_init(self, tmp_path):
        """Agent logging should write to a file."""
        import react_agent
        log_file = tmp_path / "agent.log"
        handler = logging.FileHandler(str(log_file))
        react_agent.logger.addHandler(handler)
        react_agent.logger.setLevel(logging.DEBUG)
        react_agent.logger.info("test log entry")
        handler.flush()
        assert log_file.exists()
        content = log_file.read_text()
        assert "test log entry" in content
        react_agent.logger.removeHandler(handler)


# ============================================================
# 2. Arrow keys — only in input focus
# ============================================================

class TestArrowKeyScope:
    """Up/down should only trigger history when input is focused."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_history_up_only_when_input_focused(self, app):
        """action_history_up should do nothing if input is not focused."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input", Input)
            inp.value = "cmd1"
            await inp.action_submit()
            await pilot.pause()

            # Enter select mode — TextArea gets focus
            app.action_toggle_select_mode()
            await pilot.pause()

            # Up arrow should NOT fill input (TextArea is focused)
            app.action_history_up()
            await pilot.pause()
            assert inp.value == "", (
                f"Input should stay empty in select mode, got: '{inp.value}'"
            )

    @pytest.mark.asyncio
    async def test_history_works_when_input_focused(self, app):
        """action_history_up should work when input IS focused."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input", Input)
            inp.value = "my command"
            await inp.action_submit()
            await pilot.pause()

            inp.focus()
            await pilot.pause()
            app.action_history_up()
            await pilot.pause()
            assert inp.value == "my command"

    @pytest.mark.asyncio
    async def test_arrow_bindings_not_priority(self, app):
        """Arrow key bindings should NOT have priority=True."""
        async with app.run_test(size=(120, 40)) as pilot:
            arrow_bindings = [b for b in app.BINDINGS if b.key in ("up", "down")]
            for b in arrow_bindings:
                assert not b.priority, (
                    f"Arrow binding '{b.key}' should not have priority=True"
                )
