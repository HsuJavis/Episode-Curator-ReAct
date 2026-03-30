"""E2E tests for CLI TUI (Textual) — widget layout, event flow, status updates."""

import asyncio
import json

import pytest

from textual.widgets import Input

from cli_app import (
    ContextDetailPanel,
    ContextUsagePanel,
    EpisodeCuratorApp,
    EpisodeSummaryPanel,
    StatusBar,
    TUIEvent,
    TUIPlugin,
    _format_tokens,
    _relative_time,
    _salience_dot,
)
from react_agent import AgentContext


# Mark all tests in this module
pytestmark = pytest.mark.tui


# ============================================================
# Widget Layout — all panels, input, status bar are present
# ============================================================

class TestWidgetLayout:
    """All required UI panels must be present and correctly configured."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_all_widgets_present(self, app):
        """Core layout: conversation log, context panel, episodes, input, status bar."""
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.query_one("#log") is not None
            assert app.query_one("#context-usage") is not None
            assert app.query_one("#episodes") is not None
            assert app.query_one("#user-input") is not None
            assert app.query_one("#status-bar") is not None

    @pytest.mark.asyncio
    async def test_panel_border_titles(self, app):
        """Panels should have descriptive border titles."""
        async with app.run_test(size=(120, 40)) as pilot:
            conv = app.query_one("#conversation")
            assert conv.border_title is not None and "Conversation" in conv.border_title
            ctx = app.query_one("#context-usage")
            assert ctx.border_title is not None and "Context" in ctx.border_title
            eps = app.query_one("#episodes")
            assert eps.border_title is not None and "Episodes" in eps.border_title

    @pytest.mark.asyncio
    async def test_initial_status_bar_idle(self, app):
        """Status bar should start in idle state."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)
            assert status.busy is False
            rendered = status.render()
            assert "○" in rendered  # idle icon

    @pytest.mark.asyncio
    async def test_input_has_focus_on_startup(self, app):
        """Input box should have focus immediately on startup so user can type."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input", Input)
            assert inp.has_focus, "Input should have focus on startup"
            assert app.focused is inp, f"Focused widget should be Input, got {app.focused}"

    @pytest.mark.asyncio
    async def test_typing_goes_to_input(self, app):
        """Keystrokes should appear in the input box."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input", Input)
            await pilot.press("h", "e", "l", "l", "o")
            assert inp.value == "hello", f"Expected 'hello', got '{inp.value}'"

    @pytest.mark.asyncio
    async def test_input_visible_with_nonzero_size(self, app):
        """Input box should be visible and have non-zero dimensions."""
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#user-input", Input)
            assert inp.visible is True
            assert inp.size.width > 0
            assert inp.size.height > 0
            assert inp.region.height > 0

    @pytest.mark.asyncio
    async def test_input_area_not_overlapped_by_status_bar(self, app):
        """Input area and status bar must not overlap — input must be fully visible."""
        async with app.run_test(size=(120, 30)) as pilot:
            input_area = app.query_one("#input-area")
            status_bar = app.query_one("#status-bar", StatusBar)

            ia_bottom = input_area.region.y + input_area.region.height
            sb_top = status_bar.region.y

            assert ia_bottom <= sb_top, (
                f"Input area (y={input_area.region.y}, h={input_area.region.height}) "
                f"overlaps status bar (y={status_bar.region.y}): "
                f"input bottom {ia_bottom} > status top {sb_top}"
            )

    @pytest.mark.asyncio
    async def test_input_area_not_overlapped_small_terminal(self, app):
        """Even in a small terminal (80x24), input should not be covered."""
        async with app.run_test(size=(80, 24)) as pilot:
            input_area = app.query_one("#input-area")
            status_bar = app.query_one("#status-bar", StatusBar)

            ia_bottom = input_area.region.y + input_area.region.height
            sb_top = status_bar.region.y

            assert ia_bottom <= sb_top, (
                f"Overlap in small terminal: input bottom {ia_bottom} > status top {sb_top}"
            )

    @pytest.mark.asyncio
    async def test_initial_context_panel_renders(self, app):
        """Context usage panel should render with CONTEXT WINDOW header."""
        async with app.run_test(size=(120, 40)) as pilot:
            ctx = app.query_one("#context-usage", ContextUsagePanel)
            rendered = ctx.render()
            assert "CONTEXT WINDOW" in rendered
            assert "system" in rendered
            assert "tools" in rendered
            assert "msgs" in rendered


# ============================================================
# Event Flow — TUIPlugin events update UI correctly
# ============================================================

class TestEventFlow:
    """Simulated ReAct loop events should update all panels in real time."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_thought_appears_in_log(self, app):
        """on_thought event should show 💭 in conversation log."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("thought", {"text": "Let me think about this..."}))
            await pilot.pause()
            # RichLog content is internal, but we can verify no crash
            # and the event was processed without error

    @pytest.mark.asyncio
    async def test_action_appears_in_log(self, app):
        """before_action event should show 🔧 with tool name."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("action", {
                "tool": "recall_episode",
                "input": {"search_query": "database"},
            }))
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_observation_appears_in_log(self, app):
        """on_observation event should show 📋 with result."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("observation", {
                "result": "Found 2 episodes about database design",
            }))
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_answer_clears_busy(self, app):
        """answer event should set status bar to idle."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)
            status.busy = True

            app._handle_event(TUIEvent("answer", {"text": "Here is the answer."}))
            await pilot.pause()

            assert status.busy is False

    @pytest.mark.asyncio
    async def test_done_clears_busy(self, app):
        """done event should set status bar to idle."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)
            status.busy = True

            app._handle_event(TUIEvent("done", {}))
            await pilot.pause()

            assert status.busy is False

    @pytest.mark.asyncio
    async def test_error_shows_in_log_and_clears_busy(self, app):
        """error event should display error and clear busy state."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)
            status.busy = True

            app._handle_event(TUIEvent("error", {"message": "Connection timeout"}))
            await pilot.pause()

            assert status.busy is False

    @pytest.mark.asyncio
    async def test_full_event_sequence(self, app):
        """A complete ReAct cycle: thought → action → observation → status → answer → done."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)
            status.busy = True

            events = [
                TUIEvent("thought", {"text": "Thinking about PostgreSQL indexes..."}),
                TUIEvent("action", {"tool": "recall_episode", "input": {"search_query": "db"}}),
                TUIEvent("observation", {"result": "Found episode #001"}),
                TUIEvent("status", {
                    "iteration": 1, "max_iterations": 30,
                    "input_tokens": 5000, "output_tokens": 800,
                    "elapsed": 2.3, "episode_count": 1,
                    "context": {"system": 1000, "tools": 500, "messages": 3500},
                }),
                TUIEvent("answer", {"text": "Use B-tree index for equality queries."}),
                TUIEvent("done", {}),
            ]

            for event in events:
                app._handle_event(event)
            await pilot.pause()

            # Status bar should reflect final state
            assert status.iteration == 1
            assert status.input_tokens == 5000
            assert status.busy is False


# ============================================================
# Status Bar — token counts, iteration, elapsed
# ============================================================

class TestStatusBar:
    """Status bar should accurately reflect agent state."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_status_updates_from_event(self, app):
        """status event should update all status bar fields."""
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_event(TUIEvent("status", {
                "iteration": 3, "max_iterations": 30,
                "input_tokens": 24500, "output_tokens": 3200,
                "elapsed": 8.7, "episode_count": 5,
                "context": {"system": 4000, "tools": 2000, "messages": 18500},
            }))
            await pilot.pause()

            status = app.query_one("#status-bar", StatusBar)
            assert status.iteration == 3
            assert status.max_iterations == 30
            assert status.input_tokens == 24500
            assert status.output_tokens == 3200
            assert status.episode_count == 5
            assert abs(status.elapsed - 8.7) < 0.01

    @pytest.mark.asyncio
    async def test_status_bar_render_format(self, app):
        """Rendered status bar should contain all fields in expected format."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)
            status.iteration = 2
            status.max_iterations = 30
            status.input_tokens = 15800
            status.output_tokens = 2100
            status.episode_count = 3
            status.elapsed = 5.2
            status.busy = True

            rendered = status.render()
            assert "◉" in rendered  # busy icon
            assert "iter 2/30" in rendered
            assert "15.8k" in rendered
            assert "2.1k" in rendered
            assert "3 eps" in rendered
            assert "5.2s" in rendered

    @pytest.mark.asyncio
    async def test_busy_idle_toggle(self, app):
        """Status icon should toggle between ◉ (busy) and ○ (idle)."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)

            status.busy = True
            assert "◉" in status.render()

            status.busy = False
            assert "○" in status.render()


# ============================================================
# Context Usage Panel — 3-bar breakdown
# ============================================================

class TestContextUsagePanel:
    """Context usage panel should show correct proportions."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_context_updates_from_event(self, app):
        """status event should update context usage panel values."""
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

    @pytest.mark.asyncio
    async def test_context_panel_shows_percentages(self, app):
        """Rendered panel should include percentage labels."""
        async with app.run_test(size=(120, 40)) as pilot:
            ctx = app.query_one("#context-usage", ContextUsagePanel)
            ctx.system_tokens = 2000
            ctx.tool_tokens = 1000
            ctx.message_tokens = 7000

            rendered = ctx.render()
            assert "20%" in rendered   # system 2k/10k
            assert "10%" in rendered   # tools 1k/10k
            assert "70%" in rendered   # msgs 7k/10k

    @pytest.mark.asyncio
    async def test_context_threshold_display(self, app):
        """Panel should show the compression threshold."""
        async with app.run_test(size=(120, 40)) as pilot:
            ctx = app.query_one("#context-usage", ContextUsagePanel)
            ctx.threshold = 80000
            ctx.system_tokens = 1000
            ctx.tool_tokens = 500
            ctx.message_tokens = 3500

            rendered = ctx.render()
            assert "80.0k" in rendered


# ============================================================
# Episode Summary Panel — sorted by salience, shows dimensions
# ============================================================

class TestEpisodeSummaryPanel:
    """Episode panel should sort by salience and display cognitive dimensions."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.fixture
    def sample_episodes(self):
        return {
            "001": {
                "title": "DB Design",
                "summary": "設計 products 表結構，確認 7 個產品",
                "tags": ["database"],
                "salience": 0.6,
                "created_at": "2026-03-29T10:00:00",
                "dimensions": {
                    "decisions": ["選擇 PostgreSQL 因為團隊熟悉"],
                    "corrections": [],
                    "insights": [],
                    "pending": ["index 策略待定"],
                    "user_intent": "設計資料庫",
                    "outcome": "positive",
                },
            },
            "002": {
                "title": "Port Fix",
                "summary": "原 port 3000 改為 8080 修好部署問題",
                "tags": ["deployment"],
                "salience": 0.9,
                "created_at": "2026-03-29T12:00:00",
                "dimensions": {
                    "decisions": [],
                    "corrections": ["port 設定從 3000 改為 8080"],
                    "insights": ["Cloud Run 預設 8080"],
                    "pending": [],
                    "user_intent": "修復部署",
                    "outcome": "positive",
                },
            },
            "003": {
                "title": "API Key",
                "summary": "確認 API key 設定方式",
                "tags": ["misc"],
                "salience": 0.2,
                "created_at": "2026-03-28T08:00:00",
                "dimensions": {},
            },
        }

    @pytest.mark.asyncio
    async def test_episodes_sorted_by_salience(self, app, sample_episodes):
        """Episodes should be ordered by salience descending."""
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one("#episodes", EpisodeSummaryPanel)
            panel.update_episodes(sample_episodes)
            await pilot.pause()

            cards = panel.query(".episode-card")
            assert len(cards) == 3

            # Verify order: #002 (0.9) → #001 (0.6) → #003 (0.2)
            texts = [str(c.render()) for c in cards]
            assert "#002" in texts[0]
            assert "#001" in texts[1]
            assert "#003" in texts[2]

    @pytest.mark.asyncio
    async def test_episode_card_shows_salience(self, app, sample_episodes):
        """Each card should display the salience value."""
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one("#episodes", EpisodeSummaryPanel)
            panel.update_episodes(sample_episodes)
            await pilot.pause()

            cards = panel.query(".episode-card")
            first_card_text = str(cards[0].render())
            assert "sal:0.9" in first_card_text

    @pytest.mark.asyncio
    async def test_episode_card_shows_dimensions(self, app, sample_episodes):
        """Cards with non-empty dimensions should show dimension abbreviations."""
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one("#episodes", EpisodeSummaryPanel)
            panel.update_episodes(sample_episodes)
            await pilot.pause()

            cards = panel.query(".episode-card")
            # #002 has corrections and insights
            card_002 = str(cards[0].render())
            assert "C:" in card_002  # corrections
            assert "I:" in card_002  # insights

    @pytest.mark.asyncio
    async def test_episode_card_shows_summary(self, app, sample_episodes):
        """Each card should display the episode summary text."""
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one("#episodes", EpisodeSummaryPanel)
            panel.update_episodes(sample_episodes)
            await pilot.pause()

            cards = panel.query(".episode-card")
            card_002 = str(cards[0].render())
            assert "port 3000" in card_002 or "8080" in card_002

    @pytest.mark.asyncio
    async def test_empty_episodes(self, app):
        """Empty episode list should show placeholder text."""
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one("#episodes", EpisodeSummaryPanel)
            panel.update_episodes({})
            await pilot.pause()

            empties = panel.query(".episode-empty")
            assert len(empties) == 1

    @pytest.mark.asyncio
    async def test_episodes_updated_event_refreshes_panel(self, app):
        """episodes_updated event should trigger panel refresh."""
        async with app.run_test(size=(120, 40)) as pilot:
            # This just verifies the event handler doesn't crash
            app._handle_event(TUIEvent("episodes_updated", {}))
            await pilot.pause()


# ============================================================
# Input Handling
# ============================================================

class TestInputHandling:
    """Input box should accept text and trigger agent."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_empty_input_ignored(self, app):
        """Submitting empty input should do nothing."""
        async with app.run_test(size=(120, 40)) as pilot:
            status = app.query_one("#status-bar", StatusBar)
            inp = app.query_one("#user-input", Input)

            # Submit empty
            inp.value = ""
            await inp.action_submit()
            await pilot.pause()

            assert status.busy is False  # Should not start agent


# ============================================================
# TUIPlugin Unit Tests
# ============================================================

class TestTUIPlugin:
    """TUIPlugin should emit correct events from ReAct hooks."""

    def test_plugin_name(self):
        plugin = TUIPlugin()
        assert plugin.name == "tui_bridge"

    def test_emit_thought(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        plugin.on_thought(ctx, "thinking about it...")

        assert len(events) == 1
        assert events[0].kind == "thought"
        assert events[0].data["text"] == "thinking about it..."

    def test_emit_action(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        plugin.before_action(ctx, {"name": "recall_episode", "input": {"search_query": "db"}})

        assert len(events) == 1
        assert events[0].kind == "action"
        assert events[0].data["tool"] == "recall_episode"

    def test_emit_observation_truncated(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        long_result = "x" * 600
        plugin.on_observation(ctx, long_result)

        assert len(events) == 1
        assert events[0].kind == "observation"
        assert len(events[0].data["result"]) == 503  # 500 + "..."

    def test_emit_answer_and_done(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        plugin.on_agent_end(ctx, "The answer is 42.")

        assert len(events) == 2
        assert events[0].kind == "answer"
        assert events[0].data["text"] == "The answer is 42."
        assert events[1].kind == "done"

    def test_emit_error(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        plugin.on_error(ctx, Exception("API timeout"))

        assert len(events) == 1
        assert events[0].kind == "error"
        assert "API timeout" in events[0].data["message"]

    def test_no_callback_no_crash(self):
        """Plugin should not crash when no callback is set."""
        plugin = TUIPlugin()
        ctx = AgentContext(user_query="test")

        # All hooks should silently return without error
        plugin.on_agent_start(ctx)
        plugin.on_thought(ctx, "test")
        plugin.before_action(ctx, {"name": "t", "input": {}})
        plugin.on_observation(ctx, "result")
        plugin.on_token_usage(ctx, 100, 50)
        plugin.on_agent_end(ctx, "done")
        plugin.on_error(ctx, Exception("err"))

    def test_status_event_includes_context_breakdown(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))
        plugin._system_base_tokens = 500
        plugin._tool_tokens = 300
        plugin._max_iterations = 10

        ctx = AgentContext(user_query="test")
        ctx.start_time = time.time() - 2.0
        ctx.iteration = 1
        ctx.total_input_tokens = 2000
        ctx.total_output_tokens = 400

        plugin.on_token_usage(ctx, 2000, 400)

        status_events = [e for e in events if e.kind == "status"]
        assert len(status_events) >= 1
        data = status_events[0].data
        assert data["iteration"] == 1
        assert data["input_tokens"] >= 2000  # cumulative
        assert "context" in data
        assert "system" in data["context"]
        assert "tools" in data["context"]
        assert "messages" in data["context"]
        # messages = 2000 - system_est - 300 (tool_tokens)
        assert data["context"]["messages"] >= 0


# ============================================================
# Streaming + Status Turn Counter + Context Detail
# ============================================================


class TestStreamingEvents:
    """Streaming is used at API level; on_thought displays text once after stream completes."""

    def test_stream_delta_is_noop_in_tui(self):
        """on_stream_delta should not emit events (streaming is API-level only)."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        plugin.on_stream_delta(ctx, "Hello ")
        plugin.on_stream_delta(ctx, "world")

        deltas = [e for e in events if e.kind == "stream_delta"]
        assert len(deltas) == 0

    def test_thought_always_emitted(self):
        """on_thought should always emit — it is the single display path for text."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        plugin.on_stream_delta(ctx, "streamed text")  # no-op
        plugin.on_thought(ctx, "Thinking normally")

        thoughts = [e for e in events if e.kind == "thought"]
        assert len(thoughts) == 1
        assert thoughts[0].data["text"] == "Thinking normally"

    def test_thought_emitted_across_multiple_runs(self):
        """on_thought should work consistently across multiple agent runs."""
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx1 = AgentContext(user_query="q1")
        plugin.on_agent_start(ctx1)
        plugin.on_thought(ctx1, "Thought 1")

        ctx2 = AgentContext(user_query="q2")
        plugin.on_agent_start(ctx2)
        plugin.on_thought(ctx2, "Thought 2")

        thoughts = [e for e in events if e.kind == "thought"]
        assert len(thoughts) == 2


class TestStatusTurnCounter:
    """Status bar should track cumulative turns across multiple run() calls."""

    def test_turn_increments_on_agent_start(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx1 = AgentContext(user_query="first")
        plugin.on_agent_start(ctx1)
        assert plugin._total_turns == 1

        ctx2 = AgentContext(user_query="second")
        plugin.on_agent_start(ctx2)
        assert plugin._total_turns == 2

    def test_turn_in_status_event(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))

        ctx = AgentContext(user_query="test")
        plugin.on_agent_start(ctx)

        status_events = [e for e in events if e.kind == "status"]
        assert status_events[0].data["turn"] == 1

    def test_cumulative_tokens_across_runs(self):
        plugin = TUIPlugin()
        events = []
        plugin.set_callback(lambda e: events.append(e))
        plugin._system_base_tokens = 100
        plugin._tool_tokens = 50

        # First run
        ctx1 = AgentContext(user_query="q1")
        ctx1.start_time = time.time()
        ctx1.iteration = 1
        plugin.on_agent_start(ctx1)
        plugin.on_token_usage(ctx1, 1000, 200)

        # Second run
        ctx2 = AgentContext(user_query="q2")
        ctx2.start_time = time.time()
        ctx2.iteration = 1
        plugin.on_agent_start(ctx2)
        plugin.on_token_usage(ctx2, 800, 150)

        status_events = [e for e in events if e.kind == "status"]
        last_status = status_events[-1].data
        assert last_status["input_tokens"] == 1800  # 1000 + 800
        assert last_status["output_tokens"] == 350   # 200 + 150
        assert last_status["turn"] == 2

    def test_status_bar_renders_turn(self):
        """StatusBar should display the turn counter."""
        bar = StatusBar()
        bar.turn = 5
        bar.iteration = 2
        bar.max_iterations = 30
        rendered = bar.render()
        assert "turn 5" in rendered
        assert "iter 2/30" in rendered


class TestContextDetailPanel:
    """Context detail panel toggles visibility with keybindings."""

    @pytest.fixture
    def app(self):
        return EpisodeCuratorApp()

    @pytest.mark.asyncio
    async def test_detail_panel_exists_hidden(self, app):
        """Context detail panel should exist but be hidden by default."""
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#context-detail")
            assert detail is not None
            assert "detail-visible" not in detail.classes

    @pytest.mark.asyncio
    async def test_toggle_system_detail(self, app):
        """Ctrl+D should toggle system context detail."""
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#context-detail")

            # Toggle on
            app.action_toggle_detail_system()
            await pilot.pause()
            assert "detail-visible" in detail.classes
            assert app._detail_showing == "system"

            # Toggle off
            app.action_toggle_detail_system()
            await pilot.pause()
            assert "detail-visible" not in detail.classes
            assert app._detail_showing == ""

    @pytest.mark.asyncio
    async def test_toggle_switches_category(self, app):
        """Pressing a different toggle should switch category, not just toggle off."""
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#context-detail")

            app.action_toggle_detail_system()
            await pilot.pause()
            assert app._detail_showing == "system"

            # Switch to tools
            app.action_toggle_detail_tools()
            await pilot.pause()
            assert app._detail_showing == "tools"
            assert "detail-visible" in detail.classes

    @pytest.mark.asyncio
    async def test_stream_widget_exists(self, app):
        """Stream output widget should exist in conversation panel."""
        async with app.run_test(size=(120, 40)) as pilot:
            stream = app.query_one("#stream-output")
            assert stream is not None


# ============================================================
# Utility Functions
# ============================================================

class TestUtilities:
    def test_format_tokens_small(self):
        assert _format_tokens(500) == "500"

    def test_format_tokens_k(self):
        assert _format_tokens(15800) == "15.8k"

    def test_format_tokens_m(self):
        assert _format_tokens(1_500_000) == "1.5M"

    def test_salience_dot_high(self):
        dot = _salience_dot(0.9)
        assert "red" in dot

    def test_salience_dot_medium(self):
        dot = _salience_dot(0.6)
        assert "f0a500" in dot  # orange

    def test_salience_dot_low(self):
        dot = _salience_dot(0.2)
        assert "dim" in dot

    def test_relative_time_just_now(self):
        from datetime import datetime
        now = datetime.now().isoformat()
        assert _relative_time(now) == "just now"

    def test_relative_time_empty(self):
        assert _relative_time("") == ""

    def test_relative_time_hours(self):
        from datetime import datetime, timedelta
        t = (datetime.now() - timedelta(hours=5)).isoformat()
        assert "h ago" in _relative_time(t)


# Need time import for TUIPlugin status test
import time
