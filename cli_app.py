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
        self._streamed_current: bool = False  # suppress on_thought when streamed

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
        self._streamed_current = False
        self._emit("status", iteration=0, max_iterations=0,
                   input_tokens=self._cumulative_input_tokens,
                   output_tokens=self._cumulative_output_tokens,
                   elapsed=0.0, episode_count=self._last_episode_count,
                   turn=self._total_turns,
                   context={"system": 0, "tools": 0, "messages": 0})

    def on_thought(self, ctx: AgentContext, thought: str) -> Optional[str]:
        if not self._streamed_current:
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
        self._streamed_current = True
        self._emit("stream_delta", text=delta)

    def on_token_usage(self, ctx: AgentContext, input_tokens: int, output_tokens: int) -> None:
        # Emit raw context content for detail panel — include base system prompt
        base_prompt = ctx.metadata.get("_base_system_prompt", "")
        extra = ctx.metadata.get("system_prompt_extra", "")
        system_content = base_prompt
        if extra:
            system_content = f"{base_prompt}\n\n── system_prompt_extra ──\n{extra}" if base_prompt else extra
        msgs_content = json.dumps(
            [{"role": m.get("role", "?"), "content": str(m.get("content", ""))[:200]}
             for m in ctx.messages[-10:]],  # last 10 messages, truncated
            ensure_ascii=False, indent=1,
        ) if ctx.messages else "[]"
        self._emit("context_content",
                   system=system_content,
                   msgs=msgs_content)
        elapsed = time.time() - ctx.start_time if ctx.start_time else 0.0
        self._cumulative_input_tokens += input_tokens
        self._cumulative_output_tokens += output_tokens

        # Estimate context breakdown
        extra = ctx.metadata.get("system_prompt_extra", "")
        system_est = _estimate_tokens(extra) + self._system_base_tokens
        context = {
            "system": system_est,
            "tools": self._tool_tokens,
            "messages": max(0, input_tokens - system_est - self._tool_tokens),
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
    """Displays context window token breakdown as horizontal gauges."""

    system_tokens: reactive[int] = reactive(0)
    tool_tokens: reactive[int] = reactive(0)
    message_tokens: reactive[int] = reactive(0)
    threshold: reactive[int] = reactive(80000)

    def render(self) -> str:
        total = self.system_tokens + self.tool_tokens + self.message_tokens
        if total == 0:
            total = 1  # avoid div-by-zero

        bar_width = 22

        def gauge(label: str, value: int, color_char: str) -> str:
            pct = value / total if total > 1 else 0
            filled = int(pct * bar_width)
            empty = bar_width - filled
            bar = color_char * filled + "░" * empty
            tok_str = _format_tokens(value)
            return f" {label:<8} {bar} {tok_str:>6} ({pct:4.0%})"

        lines = [
            " ┈ CONTEXT WINDOW ┈",
            gauge("system", self.system_tokens, "█"),
            gauge("tools", self.tool_tokens, "▓"),
            gauge("msgs", self.message_tokens, "▒"),
            f" {'─' * (bar_width + 20)}",
            f" total    {_format_tokens(self.system_tokens + self.tool_tokens + self.message_tokens):>28} / {_format_tokens(self.threshold)}",
        ]
        return "\n".join(lines)


class ContextDetailPanel(VerticalScroll):
    """Expandable panel showing full context content for each category."""

    def update_detail(self, category: str, content: str):
        self.remove_children()
        header = f" [bold #5dade2]{category.upper()}[/] context detail"
        lines = content[:2000]  # cap display
        if len(content) > 2000:
            lines += f"\n... ({len(content)} chars total)"
        self.mount(Static(f"{header}\n[dim]{lines}[/]", classes="context-detail-content"))


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
        height: 8;
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
    ]

    def __init__(self, agent=None, store=None, threshold: int = 80000, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent
        self._store = store
        self._threshold = threshold
        self._tui_plugin: TUIPlugin | None = None
        self._history: list[dict] = []
        self._context_contents: dict[str, str] = {"system": "", "tools": "", "msgs": ""}
        self._detail_showing: str = ""  # "" = hidden, "system"/"tools"/"msgs" = showing

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
        ctx_panel.threshold = self._threshold

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

        self._tui_plugin.set_callback(self._on_tui_event_from_thread)
        self._agent.register_skill(self._tui_plugin)

        # Populate initial tools content for detail panel
        tools = self._agent._manager.get_all_tool_definitions()
        self._context_contents["tools"] = f"Registered tools ({len(tools)}):\n" + "\n".join(
            f"  - {t['name']}: {t.get('description', '')[:80]}" for t in tools
        )

    def _on_tui_event_from_thread(self, event: TUIEvent):
        """Called from the agent thread — dispatch to main thread."""
        self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: TUIEvent):
        log = self.query_one("#log", RichLog)
        stream_widget = self.query_one("#stream-output", Static)
        kind = event.kind
        data = event.data

        if kind == "stream_delta":
            text = data.get("text", "")
            if not hasattr(self, '_stream_buffer'):
                self._stream_buffer = ""
            self._stream_buffer += text
            stream_widget.update(self._stream_buffer)
            stream_widget.add_class("streaming")
            return

        elif kind == "thought":
            # If we were streaming, flush buffer as thought
            self._flush_stream(log, stream_widget, style="thought")
            text = data.get("text", "")
            if len(text) > 200:
                text = text[:197] + "..."
            log.write(f"[dim italic #6c7a89]  💭 {text}[/]")

        elif kind == "action":
            self._flush_stream(log, stream_widget, style="thought")
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
            self._flush_stream(log, stream_widget, style="answer")
            text = data.get("text", "")
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
            # Update detail panel if visible
            if self._detail_showing:
                detail = self.query_one("#context-detail", ContextDetailPanel)
                detail.update_detail(self._detail_showing,
                                     self._context_contents.get(self._detail_showing, ""))

        elif kind == "done":
            status = self.query_one("#status-bar", StatusBar)
            status.busy = False
            self._refresh_episodes()

    def _flush_stream(self, log: RichLog, stream_widget: Static, style: str = "thought"):
        """Flush accumulated stream buffer to the log and hide stream widget."""
        buf = getattr(self, '_stream_buffer', '')
        if buf.strip():
            if style == "answer":
                pass  # answer event will write the final text
            else:
                if len(buf) > 200:
                    buf = buf[:197] + "..."
                log.write(f"[dim italic #6c7a89]  💭 {buf}[/]")
        self._stream_buffer = ""
        stream_widget.update("")
        stream_widget.remove_class("streaming")

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

    def action_toggle_detail_tools(self):
        # Build tools content from agent if available
        if self._agent:
            import json as _json
            tools = self._agent._manager.get_all_tool_definitions()
            names = [t["name"] for t in tools]
            self._context_contents["tools"] = f"Registered tools ({len(names)}):\n" + "\n".join(
                f"  - {t['name']}: {t.get('description', '')[:80]}" for t in tools
            )
        self._toggle_detail("tools")

    def action_toggle_detail_msgs(self):
        self._toggle_detail("msgs")

    def action_clear_log(self):
        log = self.query_one("#log", RichLog)
        log.clear()

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
