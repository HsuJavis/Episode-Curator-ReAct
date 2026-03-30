"""Episode Curator ReAct Agent — Terminal UI (Textual)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Input, RichLog, Static

from react_agent import AgentContext, SkillPlugin


# ============================================================
# Model Context Windows
# ============================================================

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-sonnet-4-6-20250627": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-opus-4-6-20250724": 1_000_000,
}

DEFAULT_CONTEXT_WINDOW = 200_000


def get_model_context_window(model: str) -> int:
    """Return the context window size for a model name."""
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # Prefix match for versioned model names
    for name, window in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(name.rsplit("-", 1)[0]):
            return window
    return DEFAULT_CONTEXT_WINDOW


# ============================================================
# TUIPlugin — Bridge ReAct hooks → Textual App
# ============================================================

@dataclass
class TUIEvent:
    kind: str  # thought, action, observation, answer, status, episodes_updated, done, error
    data: dict = field(default_factory=dict)


class TUIPlugin(SkillPlugin):
    """Captures ReAct loop events and forwards them to the TUI."""

    def __init__(self):
        self._callback: Callable[[TUIEvent], None] | None = None
        self._system_base_tokens: int = 0
        self._tool_tokens: int = 0
        self._last_episode_count: int = 0
        self._cumulative_input_tokens: int = 0
        self._cumulative_output_tokens: int = 0
        self._total_turns: int = 0
        self._get_tools_content: Callable[[], str] | None = None
        self._get_active_tool_tokens: Callable[[], int] | None = None

    def set_callback(self, cb: Callable[[TUIEvent], None]):
        self._callback = cb

    def _emit(self, kind: str, **data):
        if self._callback:
            self._callback(TUIEvent(kind=kind, data=data))

    @property
    def name(self) -> str:
        return "tui_bridge"

    def on_agent_start(self, ctx: AgentContext) -> None:
        self._last_episode_count = self._count_episodes()
        self._total_turns += 1
        self._emit("status", iteration=0, max_iterations=0,
                   input_tokens=self._cumulative_input_tokens,
                   output_tokens=self._cumulative_output_tokens,
                   elapsed=0.0, episode_count=self._last_episode_count,
                   turn=self._total_turns,
                   context={"system": 0, "tools": 0, "messages": 0})

    def on_thought(self, ctx: AgentContext, thought: str) -> Optional[str]:
        self._emit("thought", text=thought)
        return None

    def before_action(self, ctx: AgentContext, tool_call: dict) -> Optional[dict]:
        self._emit("action", tool=tool_call.get("name", "?"),
                   input=tool_call.get("input", {}))
        return None

    def on_observation(self, ctx: AgentContext, observation: str) -> Optional[str]:
        truncated = observation[:500] + "..." if len(observation) > 500 else observation
        self._emit("observation", result=truncated)
        return None

    def on_stream_delta(self, ctx: AgentContext, delta: str) -> None:
        # Streaming is used at the API level for responsiveness.
        # Text display is handled by on_thought (once, after stream completes).
        pass

    def on_token_usage(self, ctx: AgentContext, input_tokens: int, output_tokens: int) -> None:
        # Emit raw context content for detail panel — include base system prompt
        base_prompt = ctx.metadata.get("_base_system_prompt", "")
        extra = ctx.metadata.get("system_prompt_extra", "")
        system_content = base_prompt
        if extra:
            system_content = f"{base_prompt}\n\n── system_prompt_extra ──\n{extra}" if base_prompt else extra
        def _msg_preview(m: dict) -> dict:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str):
                return {"role": role, "content": content[:200]}
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict):
                        t = b.get("type", "?")
                        if t == "tool_result":
                            tu_id = b.get("tool_use_id", "")
                            body = b.get("content", "")
                            preview = body[:80] if isinstance(body, str) else str(body)[:80]
                            parts.append(f"[tool_result:{tu_id}] {preview}")
                        else:
                            parts.append(f"[{t}]")
                    elif hasattr(b, "type"):
                        if b.type == "text":
                            parts.append(getattr(b, "text", "")[:100])
                        elif b.type == "tool_use":
                            parts.append(f"[tool_use:{getattr(b, 'name', '?')}]")
                        else:
                            parts.append(f"[{b.type}]")
                return {"role": role, "content": " | ".join(parts)[:300]}
            return {"role": role, "content": str(content)[:200]}

        msgs_content = json.dumps(
            [_msg_preview(m) for m in ctx.messages[-10:]],
            ensure_ascii=False, indent=1,
        ) if ctx.messages else "[]"
        # Build tools content with active/deferred status from agent manager
        tools_content = ""
        if self._get_tools_content:
            tools_content = self._get_tools_content()
        self._emit("context_content",
                   system=system_content,
                   msgs=msgs_content,
                   tools=tools_content)
        elapsed = time.time() - ctx.start_time if ctx.start_time else 0.0
        self._cumulative_input_tokens += input_tokens
        self._cumulative_output_tokens += output_tokens

        # Estimate context breakdown — tool tokens reflect current active set
        extra = ctx.metadata.get("system_prompt_extra", "")
        system_est = _estimate_tokens(extra) + self._system_base_tokens
        tool_est = self._get_active_tool_tokens() if self._get_active_tool_tokens else self._tool_tokens
        context = {
            "system": system_est,
            "tools": tool_est,
            "messages": max(0, input_tokens - system_est - tool_est),
        }

        cur_count = self._count_episodes()
        self._emit("status",
                   iteration=ctx.iteration,
                   max_iterations=getattr(self, '_max_iterations', 30),
                   input_tokens=self._cumulative_input_tokens,
                   output_tokens=self._cumulative_output_tokens,
                   elapsed=elapsed,
                   episode_count=cur_count,
                   turn=self._total_turns,
                   context=context)

        # Detect compression (new episodes created)
        if cur_count > self._last_episode_count:
            self._last_episode_count = cur_count
            self._emit("episodes_updated")

    def on_agent_end(self, ctx: AgentContext, final_answer: str) -> Optional[str]:
        self._emit("answer", text=final_answer)
        self._emit("done")
        return None

    def on_error(self, ctx: AgentContext, error: Exception) -> Optional[str]:
        self._emit("error", message=str(error))
        return None

    def _count_episodes(self) -> int:
        try:
            from episode_curator import EpisodeStore
            import os
            store_dir = os.environ.get("EPISODE_STORE_DIR",
                                       os.path.expanduser("~/.episode_store"))
            store = EpisodeStore(store_dir)
            return len(store._index)
        except Exception:
            return 0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3) if text else 0


# ============================================================
# Custom Widgets
# ============================================================

class ContextUsagePanel(Widget):
    """Displays context window token usage as a single stacked bar with color-coded segments."""

    system_tokens: reactive[int] = reactive(0)
    tool_tokens: reactive[int] = reactive(0)
    message_tokens: reactive[int] = reactive(0)
    threshold: reactive[int] = reactive(200000)
    compress_threshold: reactive[int] = reactive(100000)

    def render(self) -> str:
        total = self.system_tokens + self.tool_tokens + self.message_tokens
        cap = max(self.threshold, 1)
        usage_pct = total / cap

        bar_width = 30

        # Calculate segment widths proportional to threshold
        sys_w = int(self.system_tokens / cap * bar_width)
        tool_w = int(self.tool_tokens / cap * bar_width)
        msg_w = int(self.message_tokens / cap * bar_width)
        used_w = sys_w + tool_w + msg_w
        if used_w > bar_width:
            msg_w = max(0, msg_w - (used_w - bar_width))
        empty_w = bar_width - sys_w - tool_w - msg_w

        # Build bar with compress threshold marker
        bar_chars = list("█" * sys_w + "▓" * tool_w + "▒" * msg_w + "░" * empty_w)
        ct_pos = int(self.compress_threshold / cap * bar_width)
        if 0 < ct_pos < bar_width:
            bar_chars[ct_pos] = "▏"
        bar = "".join(bar_chars)
        pct_str = f"{usage_pct:.0%}"

        def cat_line(label: str, value: int, marker: str) -> str:
            pct = value / cap * 100 if cap > 1 else 0
            return f"  {marker} {label:<7} {_format_tokens(value):>6}  {pct:4.1f}%"

        lines = [
            f" {bar} {_format_tokens(total):>6} / {_format_tokens(cap)} ({pct_str})",
            cat_line("system", self.system_tokens, "█"),
            cat_line("tools", self.tool_tokens, "▓"),
            cat_line("msgs", self.message_tokens, "▒"),
            f"  ▏ compress {_format_tokens(self.compress_threshold):>6}",
        ]
        return "\n".join(lines)


class ContextDetailPanel(VerticalScroll):
    """Expandable panel showing full context content for each category."""

    def update_detail(self, category: str, content: str):
        from rich.markup import escape
        self.remove_children()
        header = f" [bold #5dade2]{category.upper()}[/] context detail"
        safe = escape(content[:2000])
        if len(content) > 2000:
            safe += f"\n... ({len(content)} chars total)"
        self.mount(Static(f"{header}\n[dim]{safe}[/]", classes="context-detail-content"))


class EpisodeSummaryPanel(VerticalScroll):
    """Scrollable list of episode summary cards."""

    def update_episodes(self, episodes: dict):
        self.remove_children()
        if not episodes:
            self.mount(Static(" [dim]No episodes yet[/]", classes="episode-empty"))
            return

        # Sort: salience desc, then created_at desc
        sorted_eps = sorted(
            episodes.items(),
            key=lambda x: (x[1].get("salience", 0.5), x[1].get("created_at", "")),
            reverse=True,
        )

        for ep_id, entry in sorted_eps:
            card = self._render_card(ep_id, entry)
            self.mount(Static(card, classes="episode-card"))

    @staticmethod
    def _render_card(ep_id: str, entry: dict) -> str:
        salience = entry.get("salience", 0.5)
        dot = _salience_dot(salience)
        rel_time = _relative_time(entry.get("created_at", ""))
        summary = entry.get("summary", "")
        if len(summary) > 60:
            summary = summary[:57] + "..."

        dims = entry.get("dimensions", {})
        dim_parts = []
        for key in ("decisions", "corrections", "insights", "pending"):
            items = dims.get(key, [])
            if items:
                dim_parts.append(f"[dim]{key[0].upper()}: {items[0][:30]}[/]")

        lines = [f" {dot} [bold]#{ep_id}[/] [dim]{rel_time}[/]  sal:{salience:.1f}"]
        lines.append(f"   {summary}")
        if dim_parts:
            lines.append(f"   {' · '.join(dim_parts[:2])}")

        return "\n".join(lines)


class StatusBar(Static):
    """Bottom status line — mission control readout."""

    turn: reactive[int] = reactive(0)
    iteration: reactive[int] = reactive(0)
    max_iterations: reactive[int] = reactive(30)
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)
    episode_count: reactive[int] = reactive(0)
    elapsed: reactive[float] = reactive(0.0)
    busy: reactive[bool] = reactive(False)

    def render(self) -> str:
        status_icon = "◉" if self.busy else "○"
        parts = [
            f" {status_icon}",
            f"turn {self.turn}",
            f"iter {self.iteration}/{self.max_iterations}",
            f"in: {_format_tokens(self.input_tokens)}",
            f"out: {_format_tokens(self.output_tokens)}",
            f"{self.episode_count} eps",
            f"{self.elapsed:.1f}s",
        ]
        return " │ ".join(parts)


# ============================================================
# Main App
# ============================================================

class EpisodeCuratorApp(App):
    """Episode Curator ReAct Agent — Terminal Dashboard."""

    TITLE = "Episode Curator ReAct Agent"

    CSS = """
    Screen {
        background: #0f1923;
        color: #c8d6e5;
    }

    #root-layout {
        height: 1fr;
    }

    #main-area {
        height: 1fr;
    }

    #conversation {
        width: 2fr;
        border: solid #1e3a5f;
        border-title-color: #5dade2;
        background: #0f1923;
        padding: 0 1;
    }

    #conversation RichLog {
        background: #0f1923;
        scrollbar-size: 1 1;
    }

    #stream-output {
        height: auto;
        max-height: 8;
        color: #c8d6e5;
        background: #0f1923;
        padding: 0 2;
        display: none;
    }

    #stream-output.streaming {
        display: block;
    }

    #sidebar {
        width: 1fr;
        min-width: 36;
    }

    #context-usage {
        height: 9;
        border: solid #1e3a5f;
        border-title-color: #5dade2;
        background: #111d2b;
        padding: 0 0;
    }

    #episodes {
        border: solid #1e3a5f;
        border-title-color: #5dade2;
        background: #111d2b;
        padding: 0 0;
        height: 1fr;
        scrollbar-size: 1 1;
    }

    #context-detail {
        display: none;
        height: 12;
        border: solid #1e3a5f;
        border-title-color: #5dade2;
        background: #111d2b;
        padding: 0 0;
        scrollbar-size: 1 1;
    }

    #context-detail.detail-visible {
        display: block;
    }

    .context-detail-content {
        padding: 0 1;
    }

    .episode-card {
        padding: 0 0;
        margin: 0 0 0 0;
    }

    .episode-empty {
        padding: 1 1;
    }

    #input-area {
        height: auto;
        max-height: 3;
        background: #0d1520;
        border-top: solid #1e3a5f;
        padding: 0 1;
        margin-bottom: 1;
    }

    #input-area Input {
        background: #152238;
        color: #e0e8f0;
        border: tall #1e3a5f;
    }

    #input-area Input:focus {
        border: tall #5dade2;
    }

    #status-bar {
        height: 1;
        background: #0a1018;
        color: #5dade2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+l", "clear_log", "Clear", show=True),
        Binding("ctrl+d", "toggle_detail_system", "Sys", show=True),
        Binding("ctrl+t", "toggle_detail_tools", "Tools", show=True),
        Binding("ctrl+b", "toggle_detail_msgs", "Msgs", show=True),
        Binding("ctrl+y", "copy_last", "Copy", show=True),
        Binding("up", "history_up", "", show=False, priority=True),
        Binding("down", "history_down", "", show=False, priority=True),
    ]

    def __init__(self, agent=None, store=None, threshold_pct: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent
        self._store = store
        self._threshold_pct = threshold_pct  # compression threshold as % of context window
        self._context_window: int = DEFAULT_CONTEXT_WINDOW  # updated in _init_agent
        self._threshold = self._context_window * threshold_pct // 100  # compression threshold
        self._tui_plugin: TUIPlugin | None = None
        self._history: list[dict] = []
        self._context_contents: dict[str, str] = {"system": "", "tools": "", "msgs": ""}
        self._detail_showing: str = ""  # "" = hidden, "system"/"tools"/"msgs" = showing
        self._input_history: list[str] = []   # command history (max 5)
        self._history_index: int = -1         # -1 = not browsing
        self._last_answer: str = ""           # last assistant answer for copy

    def compose(self) -> ComposeResult:
        with Vertical(id="root-layout"):
            yield Horizontal(
                Vertical(
                    RichLog(highlight=True, markup=True, wrap=True, id="log"),
                    Static("", id="stream-output"),
                    id="conversation",
                ),
                Vertical(
                    ContextUsagePanel(id="context-usage"),
                    ContextDetailPanel(id="context-detail"),
                    EpisodeSummaryPanel(id="episodes"),
                    id="sidebar",
                ),
                id="main-area",
            )
            yield Horizontal(
                Input(placeholder="Type your message...", id="user-input"),
                id="input-area",
            )
            yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        self.query_one("#conversation", Vertical).border_title = "◈ Conversation"
        self.query_one("#context-usage").border_title = "◈ Context"
        self.query_one("#episodes").border_title = "◈ Episodes"

        ctx_panel = self.query_one("#context-usage", ContextUsagePanel)
        ctx_panel.threshold = self._context_window
        ctx_panel.compress_threshold = self._threshold

        # Load initial episodes
        self._refresh_episodes()

        # Welcome message
        log = self.query_one("#log", RichLog)
        log.write("[dim]─── Episode Curator ReAct Agent ───[/]")
        log.write("[dim]Type a message below to start.[/]\n")

        # Focus input so user can start typing immediately
        self.query_one("#user-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return

        # Store in command history (max 5)
        self._input_history.append(message)
        if len(self._input_history) > 5:
            self._input_history = self._input_history[-5:]
        self._history_index = -1

        event.input.value = ""
        log = self.query_one("#log", RichLog)
        log.write(f"\n[bold #a8d8ea]▎ user[/]  {message}")

        status = self.query_one("#status-bar", StatusBar)
        status.busy = True

        self._run_agent(message)

    @work(thread=True)
    def _run_agent(self, message: str) -> None:
        if self._agent is None:
            self._init_agent()

        try:
            answer = self._agent.run(message, list(self._history))
            self._history.append({"role": "user", "content": message})
            self._history.append({"role": "assistant", "content": answer})
        except Exception as e:
            self.call_from_thread(self._on_error, str(e))

    def _init_agent(self):
        import os
        from episode_curator import create_agent, EpisodeStore, Curator

        storage_dir = os.environ.get("EPISODE_STORE_DIR",
                                     os.path.expanduser("~/.episode_store"))
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        worker_model = os.environ.get("WORKER_MODEL", "claude-haiku-4-5-20251001")
        curator_model = os.environ.get("CURATOR_MODEL", "claude-haiku-4-5-20251001")

        # Resolve context window from model and compute compression threshold
        self._context_window = get_model_context_window(worker_model)
        self._threshold = self._context_window * self._threshold_pct // 100

        # Update context panel with model's actual context window
        try:
            ctx_panel = self.query_one("#context-usage", ContextUsagePanel)
            ctx_panel.threshold = self._context_window
            ctx_panel.compress_threshold = self._threshold
        except Exception:
            pass

        self._agent = create_agent(
            worker_model=worker_model,
            curator_model=curator_model,
            storage_dir=storage_dir,
            api_key=api_key,
            threshold=self._threshold,
        )
        self._store = EpisodeStore(storage_dir)

        # Inject TUIPlugin
        self._tui_plugin = TUIPlugin()
        self._tui_plugin._max_iterations = self._agent.max_iterations

        # Estimate static token costs
        import json as _json
        sys_prompt = getattr(self._agent, 'system_prompt', '')
        tools = self._agent._manager.get_all_tool_definitions()
        self._tui_plugin._system_base_tokens = _estimate_tokens(sys_prompt)
        self._tui_plugin._tool_tokens = _estimate_tokens(_json.dumps(tools))

        self._tui_plugin._get_tools_content = self._build_tools_content
        self._tui_plugin._get_active_tool_tokens = self._calc_active_tool_tokens
        self._tui_plugin.set_callback(self._on_tui_event_from_thread)
        self._agent.register_skill(self._tui_plugin)

        # Populate initial tools content for detail panel
        self._context_contents["tools"] = self._build_tools_content()

    def _on_tui_event_from_thread(self, event: TUIEvent):
        """Called from the agent thread — dispatch to main thread."""
        self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: TUIEvent):
        log = self.query_one("#log", RichLog)
        kind = event.kind
        data = event.data

        if kind == "thought":
            text = data.get("text", "")
            if len(text) > 200:
                text = text[:197] + "..."
            log.write(f"[dim italic #6c7a89]  💭 {text}[/]")

        elif kind == "action":
            tool = data.get("tool", "?")
            inp = data.get("input", {})
            inp_str = json.dumps(inp, ensure_ascii=False)
            if len(inp_str) > 80:
                inp_str = inp_str[:77] + "..."
            log.write(f"[bold #f0a500]  🔧 {tool}[/][#d4a017]({inp_str})[/]")

        elif kind == "observation":
            result = data.get("result", "")
            if len(result) > 150:
                result = result[:147] + "..."
            log.write(f"[#4ecca3]  📋 {result}[/]")

        elif kind == "answer":
            text = data.get("text", "")
            self._last_answer = text
            log.write(f"\n[bold #e0e8f0]▎ assistant[/]  {text}\n")
            status = self.query_one("#status-bar", StatusBar)
            status.busy = False

        elif kind == "status":
            self._update_status(data)

        elif kind == "episodes_updated":
            self._refresh_episodes()

        elif kind == "error":
            msg = data.get("message", "Unknown error")
            log.write(f"\n[bold red]  ✗ Error: {msg}[/]\n")
            status = self.query_one("#status-bar", StatusBar)
            status.busy = False

        elif kind == "context_content":
            self._context_contents["system"] = data.get("system", "")
            self._context_contents["msgs"] = data.get("msgs", "")
            tools_data = data.get("tools", "")
            if tools_data:
                self._context_contents["tools"] = tools_data
            # Update detail panel if visible
            if self._detail_showing:
                detail = self.query_one("#context-detail", ContextDetailPanel)
                detail.update_detail(self._detail_showing,
                                     self._context_contents.get(self._detail_showing, ""))

        elif kind == "done":
            status = self.query_one("#status-bar", StatusBar)
            status.busy = False
            self._refresh_episodes()

    def _update_status(self, data: dict):
        status = self.query_one("#status-bar", StatusBar)
        status.turn = data.get("turn", 0)
        status.iteration = data.get("iteration", 0)
        status.max_iterations = data.get("max_iterations", 30)
        status.input_tokens = data.get("input_tokens", 0)
        status.output_tokens = data.get("output_tokens", 0)
        status.episode_count = data.get("episode_count", 0)
        status.elapsed = data.get("elapsed", 0.0)

        ctx = data.get("context", {})
        ctx_panel = self.query_one("#context-usage", ContextUsagePanel)
        ctx_panel.system_tokens = ctx.get("system", 0)
        ctx_panel.tool_tokens = ctx.get("tools", 0)
        ctx_panel.message_tokens = ctx.get("messages", 0)

    def _refresh_episodes(self):
        try:
            import os
            from episode_curator import EpisodeStore
            storage_dir = os.environ.get("EPISODE_STORE_DIR",
                                         os.path.expanduser("~/.episode_store"))
            store = EpisodeStore(storage_dir)
            panel = self.query_one("#episodes", EpisodeSummaryPanel)
            panel.update_episodes(store._index)
        except Exception:
            pass

    def _on_error(self, msg: str):
        log = self.query_one("#log", RichLog)
        log.write(f"\n[bold red]  ✗ Error: {msg}[/]\n")
        status = self.query_one("#status-bar", StatusBar)
        status.busy = False

    def _toggle_detail(self, category: str):
        detail = self.query_one("#context-detail", ContextDetailPanel)
        if self._detail_showing == category:
            # Toggle off
            detail.remove_class("detail-visible")
            detail.border_title = ""
            self._detail_showing = ""
        else:
            # Show this category
            self._detail_showing = category
            detail.border_title = f"◈ {category.upper()} Detail (Ctrl+D/T/B)"
            detail.update_detail(category, self._context_contents.get(category, "(no data yet)"))
            detail.add_class("detail-visible")

    def action_toggle_detail_system(self):
        self._toggle_detail("system")

    def _calc_active_tool_tokens(self) -> int:
        """Estimate tokens for currently active tools only."""
        if not self._agent:
            return 0
        active = self._agent._manager.get_active_tool_definitions()
        return _estimate_tokens(json.dumps(active)) if active else 0

    def _build_tools_content(self) -> str:
        """Build tools panel content with active/deferred status."""
        if not self._agent:
            return "(agent not initialized)"
        catalog = self._agent._manager.get_tool_catalog()
        active = [t for t in catalog if t["loaded"]]
        deferred = [t for t in catalog if not t["loaded"]]
        lines = [f"Active tools ({len(active)}):"]
        for t in active:
            lines.append(f"  [✓] {t['name']}: {t['description'][:80]}")
        lines.append(f"\nDeferred tools ({len(deferred)}):")
        for t in deferred:
            lines.append(f"  [·] {t['name']}: {t['description'][:80]}")
        return "\n".join(lines)

    def action_toggle_detail_tools(self):
        if self._agent:
            self._context_contents["tools"] = self._build_tools_content()
        self._toggle_detail("tools")

    def action_toggle_detail_msgs(self):
        self._toggle_detail("msgs")

    def action_clear_log(self):
        log = self.query_one("#log", RichLog)
        log.clear()

    def action_history_up(self):
        """Navigate to previous command in history."""
        if not self._input_history:
            return
        inp = self.query_one("#user-input", Input)
        if self._history_index == -1:
            self._history_index = len(self._input_history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        inp.value = self._input_history[self._history_index]
        inp.cursor_position = len(inp.value)

    def action_history_down(self):
        """Navigate to next command in history."""
        inp = self.query_one("#user-input", Input)
        if self._history_index == -1:
            return
        if self._history_index < len(self._input_history) - 1:
            self._history_index += 1
            inp.value = self._input_history[self._history_index]
            inp.cursor_position = len(inp.value)
        else:
            self._history_index = -1
            inp.value = ""

    def action_copy_last(self):
        """Copy the last assistant answer to clipboard."""
        if self._last_answer:
            import pyperclip
            try:
                pyperclip.copy(self._last_answer)
                log = self.query_one("#log", RichLog)
                log.write("[dim]  ✂ Copied last answer to clipboard.[/]")
            except Exception:
                # pyperclip not installed or clipboard unavailable — use Textual fallback
                try:
                    import subprocess
                    proc = subprocess.Popen(
                        ["pbcopy"] if __import__("sys").platform == "darwin" else ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE,
                    )
                    proc.communicate(self._last_answer.encode("utf-8"))
                    log = self.query_one("#log", RichLog)
                    log.write("[dim]  ✂ Copied last answer to clipboard.[/]")
                except Exception:
                    log = self.query_one("#log", RichLog)
                    log.write("[dim yellow]  ✂ Copy failed — clipboard not available.[/]")

    def action_quit(self):
        self.exit()


# ============================================================
# Utilities
# ============================================================

def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _salience_dot(salience: float) -> str:
    if salience >= 0.8:
        return "[bold red]●[/]"
    elif salience >= 0.6:
        return "[#f0a500]●[/]"
    elif salience >= 0.4:
        return "[yellow]●[/]"
    else:
        return "[dim]●[/]"


def _relative_time(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        created = datetime.fromisoformat(iso_str)
        delta = datetime.now() - created
        secs = delta.total_seconds()
        if secs < 60:
            return "just now"
        elif secs < 3600:
            return f"{int(secs / 60)}m ago"
        elif secs < 86400:
            return f"{int(secs / 3600)}h ago"
        elif secs < 604800:
            return f"{int(secs / 86400)}d ago"
        else:
            return f"{int(secs / 604800)}w ago"
    except Exception:
        return ""


# ============================================================
# Entry point
# ============================================================

def main():
    app = EpisodeCuratorApp()
    app.run()


if __name__ == "__main__":
    main()
