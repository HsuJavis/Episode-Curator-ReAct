"""TDD tests for select & copy mode.

Requirements:
- Ctrl+S enters selection mode (TextArea overlay with conversation text)
- In selection mode: mouse select + Ctrl+C copies
- Esc exits selection mode
- Ctrl+Q quits (replaces Ctrl+C as quit)
- Plain text log maintained alongside RichLog

Written BEFORE fix per TDD methodology.
"""

import pytest

from cli_app import EpisodeCuratorApp, TUIEvent
from textual.widgets import Input, TextArea


pytestmark = pytest.mark.tui


class TestSelectionMode:
    """Ctrl+S toggles selection mode with a read-only TextArea."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_select_area_exists_hidden(self, app):
        """Selection TextArea should exist but be hidden by default."""
        async with app.run_test(size=(120, 40)) as pilot:
            area = app.query_one("#select-area", TextArea)
            assert area is not None
            assert area.display is False or "select-visible" not in area.classes

    @pytest.mark.asyncio
    async def test_ctrl_s_shows_select_area(self, app):
        """Ctrl+S should show the selection TextArea."""
        async with app.run_test(size=(120, 40)) as pilot:
            app.action_toggle_select_mode()
            await pilot.pause()
            area = app.query_one("#select-area", TextArea)
            assert "select-visible" in area.classes

    @pytest.mark.asyncio
    async def test_esc_exits_select_mode(self, app):
        """Esc should hide the selection TextArea."""
        async with app.run_test(size=(120, 40)) as pilot:
            app.action_toggle_select_mode()
            await pilot.pause()
            area = app.query_one("#select-area", TextArea)
            assert "select-visible" in area.classes

            app.action_toggle_select_mode()
            await pilot.pause()
            assert "select-visible" not in area.classes

    @pytest.mark.asyncio
    async def test_select_area_contains_conversation(self, app):
        """TextArea should contain the plain text conversation log."""
        async with app.run_test(size=(120, 40)) as pilot:
            # Simulate conversation
            app._plain_log.append("user: hello")
            app._plain_log.append("assistant: world")

            app.action_toggle_select_mode()
            await pilot.pause()
            area = app.query_one("#select-area", TextArea)
            assert "hello" in area.text
            assert "world" in area.text

    @pytest.mark.asyncio
    async def test_select_area_is_read_only(self, app):
        """TextArea should be read-only."""
        async with app.run_test(size=(120, 40)) as pilot:
            area = app.query_one("#select-area", TextArea)
            assert area.read_only is True


class TestPlainLog:
    """A plain text log should be maintained alongside RichLog."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_plain_log_exists(self, app):
        """App should have a _plain_log list."""
        async with app.run_test(size=(120, 40)) as pilot:
            assert hasattr(app, "_plain_log")
            assert isinstance(app._plain_log, list)

    @pytest.mark.asyncio
    async def test_user_message_in_plain_log(self, app):
        """Submitting input should add to plain log."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input", Input)
            inp.value = "test message"
            await inp.action_submit()
            await pilot.pause()

            assert any("test message" in line for line in app._plain_log)

    @pytest.mark.asyncio
    async def test_answer_in_plain_log(self, app):
        """Answer event should add to plain log."""
        async with app.run_test(size=(120, 40)) as pilot:
            from cli_app import TUIEvent
            app._handle_event(TUIEvent("answer", {"text": "my answer here"}))
            await pilot.pause()

            assert any("my answer here" in line for line in app._plain_log)

    @pytest.mark.asyncio
    async def test_thought_in_plain_log(self, app):
        """Thought events should be in plain log."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("thought", {"text": "thinking about it"}))
            await pilot.pause()
            assert any("thinking about it" in line for line in app._plain_log)

    @pytest.mark.asyncio
    async def test_action_in_plain_log(self, app):
        """Action events should be in plain log."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("action", {"tool": "read", "input": {"file_path": "/tmp/x"}}))
            await pilot.pause()
            assert any("read" in line for line in app._plain_log)

    @pytest.mark.asyncio
    async def test_observation_in_plain_log(self, app):
        """Observation events should be in plain log."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("observation", {"result": "file content here"}))
            await pilot.pause()
            assert any("file content" in line for line in app._plain_log)


class TestThresholdPctArg:
    """--threshold-pct CLI arg should be passed to App."""

    def test_app_accepts_threshold_pct(self):
        app = EpisodeCuratorApp(threshold_pct=10)
        assert app._threshold_pct == 10
        assert app._threshold == app._context_window * 10 // 100

    def test_default_threshold_pct_is_50(self):
        app = EpisodeCuratorApp()
        assert app._threshold_pct == 50


class TestQuitBinding:
    """Ctrl+Q should quit, Ctrl+C should not quit."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_ctrl_q_binding_exists(self, app):
        """Ctrl+Q should be bound to quit."""
        async with app.run_test(size=(120, 40)) as pilot:
            quit_bindings = [b for b in app.BINDINGS
                             if b.action == "quit" and b.key == "ctrl+q"]
            assert len(quit_bindings) >= 1

    @pytest.mark.asyncio
    async def test_ctrl_c_not_quit(self, app):
        """Ctrl+C should NOT be bound to quit anymore."""
        async with app.run_test(size=(120, 40)) as pilot:
            ctrl_c_quit = [b for b in app.BINDINGS
                           if b.action == "quit" and b.key == "ctrl+c"]
            assert len(ctrl_c_quit) == 0, "Ctrl+C should not be quit"
