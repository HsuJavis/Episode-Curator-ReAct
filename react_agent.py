# react_agent.py — Base ReAct Agent + SkillPlugin Hook System

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic


@dataclass
class AgentContext:
    """Shared state passed through the ReAct loop."""
    user_query: str
    messages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    iteration: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: float = 0.0
    tool_call_history: list = field(default_factory=list)


class SkillPlugin(ABC):
    """Abstract base class for ReAct Agent plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def get_tools(self) -> list[dict]:
        return []

    def is_deferred(self) -> bool:
        """If True, tools start unloaded — only metadata in catalog until load_tools."""
        return False

    def execute_tool(self, name: str, tool_input: dict) -> Any:
        raise NotImplementedError(f"Tool {name} not implemented")

    def on_agent_start(self, ctx: AgentContext) -> None:
        pass

    def on_thought(self, ctx: AgentContext, thought: str) -> Optional[str]:
        return None

    def before_action(self, ctx: AgentContext, tool_call: dict) -> Optional[dict]:
        return None

    def after_action(self, ctx: AgentContext, tool_call: dict, result: Any) -> Optional[Any]:
        return None

    def on_observation(self, ctx: AgentContext, observation: str) -> Optional[str]:
        return None

    def on_error(self, ctx: AgentContext, error: Exception) -> Optional[str]:
        return None

    def on_agent_end(self, ctx: AgentContext, final_answer: str) -> Optional[str]:
        return None

    def on_token_usage(self, ctx: AgentContext, input_tokens: int, output_tokens: int) -> None:
        pass


class SkillPluginManager:
    """Manages plugin registration, tool routing, and hook dispatch."""

    def __init__(self):
        self._plugins: list[SkillPlugin] = []
        self._tool_map: dict[str, SkillPlugin] = {}
        self._all_tool_defs: dict[str, dict] = {}  # name -> definition
        self._active_tools: set[str] = set()  # currently loaded tool names
        self._deferred_tools: set[str] = set()  # tools from deferred plugins

    def register(self, plugin: SkillPlugin) -> None:
        deferred = plugin.is_deferred()
        for tool_def in plugin.get_tools():
            tool_name = tool_def["name"]
            if tool_name in self._tool_map:
                existing = self._tool_map[tool_name].name
                raise ValueError(
                    f"Tool name conflict: '{tool_name}' already registered by plugin '{existing}'"
                )
            self._tool_map[tool_name] = plugin
            self._all_tool_defs[tool_name] = tool_def
            if deferred:
                self._deferred_tools.add(tool_name)
            else:
                self._active_tools.add(tool_name)
        self._plugins.append(plugin)

    def get_all_tool_definitions(self) -> list[dict]:
        """Return all tool definitions (backward compat)."""
        tools = []
        for plugin in self._plugins:
            tools.extend(plugin.get_tools())
        return tools

    def get_active_tool_definitions(self) -> list[dict]:
        """Return only currently active (loaded) tool definitions."""
        return [self._all_tool_defs[name] for name in self._active_tools
                if name in self._all_tool_defs]

    def load_tools(self, names: list[str]) -> str:
        """Load deferred tools into the active set (open book)."""
        loaded = []
        errors = []
        for name in names:
            if name not in self._all_tool_defs:
                errors.append(f"Unknown tool: '{name}'")
            elif name in self._active_tools:
                loaded.append(name)  # idempotent
            else:
                self._active_tools.add(name)
                loaded.append(name)
        if errors:
            return f"Error: {'; '.join(errors)}"
        return f"Loaded tools: {', '.join(loaded)}"

    def unload_tools(self, names: list[str]) -> str:
        """Unload tools from the active set, keeping metadata (close book)."""
        unloaded = []
        errors = []
        for name in names:
            if name not in self._all_tool_defs:
                errors.append(f"Unknown tool: '{name}'")
            elif name not in self._deferred_tools:
                errors.append(f"Cannot unload non-deferred tool: '{name}'")
            else:
                self._active_tools.discard(name)
                unloaded.append(name)
        if errors:
            return f"Error: {'; '.join(errors)}"
        return f"Unloaded tools: {', '.join(unloaded)}"

    def get_tool_catalog(self) -> list[dict]:
        """Return metadata for all tools (name, description, loaded status)."""
        catalog = []
        for name, tool_def in self._all_tool_defs.items():
            catalog.append({
                "name": name,
                "description": tool_def.get("description", ""),
                "loaded": name in self._active_tools,
            })
        return catalog

    def route_tool_call(self, name: str, tool_input: dict) -> Any:
        plugin = self._tool_map.get(name)
        if plugin is None:
            raise ValueError(f"Unknown tool: '{name}'")
        return plugin.execute_tool(name, tool_input)

    def dispatch_on_agent_start(self, ctx: AgentContext) -> None:
        for plugin in self._plugins:
            plugin.on_agent_start(ctx)

    def dispatch_on_thought(self, ctx: AgentContext, thought: str) -> str:
        for plugin in self._plugins:
            modified = plugin.on_thought(ctx, thought)
            if modified is not None:
                thought = modified
        return thought

    def dispatch_before_action(self, ctx: AgentContext, tool_call: dict) -> dict:
        for plugin in self._plugins:
            modified = plugin.before_action(ctx, tool_call)
            if modified is not None:
                tool_call = modified
        return tool_call

    def dispatch_after_action(self, ctx: AgentContext, tool_call: dict, result: Any) -> Any:
        for plugin in self._plugins:
            modified = plugin.after_action(ctx, tool_call, result)
            if modified is not None:
                result = modified
        return result

    def dispatch_on_observation(self, ctx: AgentContext, observation: str) -> str:
        for plugin in self._plugins:
            modified = plugin.on_observation(ctx, observation)
            if modified is not None:
                observation = modified
        return observation

    def dispatch_on_error(self, ctx: AgentContext, error: Exception) -> Optional[str]:
        for plugin in self._plugins:
            result = plugin.on_error(ctx, error)
            if result is not None:
                return result
        return None

    def dispatch_on_agent_end(self, ctx: AgentContext, final_answer: str) -> str:
        for plugin in self._plugins:
            modified = plugin.on_agent_end(ctx, final_answer)
            if modified is not None:
                final_answer = modified
        return final_answer

    def dispatch_on_token_usage(self, ctx: AgentContext, input_tokens: int, output_tokens: int) -> None:
        for plugin in self._plugins:
            plugin.on_token_usage(ctx, input_tokens, output_tokens)


def _resolve_auth(api_key: str | None = None) -> dict:
    """Resolve Anthropic client auth: api_key param → env var → OAuth credentials."""
    if api_key:
        return {"api_key": api_key}
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"api_key": os.environ["ANTHROPIC_API_KEY"]}
    token = _read_oauth_token()
    if token:
        return {"api_key": token}
    raise ValueError(
        "No API key found. Set ANTHROPIC_API_KEY env var, pass api_key parameter, "
        "or ensure ~/.claude/.credentials.json contains a valid OAuth token."
    )


def _read_oauth_token() -> str | None:
    """Read fresh OAuth token from ~/.claude/.credentials.json."""
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if creds_path.exists():
        creds = json.loads(creds_path.read_text())
        oauth = creds.get("claudeAiOauth", {})
        return oauth.get("accessToken")
    return None


def _is_oauth_auth(api_key: str | None) -> bool:
    """Check if we're using OAuth (no explicit api_key or env var)."""
    return api_key is None and not os.environ.get("ANTHROPIC_API_KEY")


class ReActAgent:
    """Core ReAct Agent engine with plugin support."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. Use the provided tools to help answer questions. "
        "Think step by step, use tools when needed, and provide clear final answers."
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 10,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._uses_oauth = _is_oauth_auth(api_key)
        auth = _resolve_auth(api_key)
        self._client = anthropic.Anthropic(**auth)
        self._manager = SkillPluginManager()

    def _refresh_client_if_needed(self):
        """Re-read OAuth token from disk (Claude Code may have refreshed it)."""
        if not self._uses_oauth:
            return
        token = _read_oauth_token()
        if token and token != self._client.api_key:
            self._client = anthropic.Anthropic(api_key=token)

    def register_skill(self, plugin: SkillPlugin) -> ReActAgent:
        self._manager.register(plugin)
        return self

    def run(self, user_query: str, conversation_history: list | None = None) -> str:
        ctx = AgentContext(
            user_query=user_query,
            messages=list(conversation_history or []),
            start_time=time.time(),
        )
        ctx.messages.append({"role": "user", "content": user_query})
        self._manager.dispatch_on_agent_start(ctx)
        final_answer = self._react_loop(ctx)
        return final_answer

    def _react_loop(self, ctx: AgentContext) -> str:
        while ctx.iteration < self.max_iterations:
            ctx.iteration += 1

            # Refresh active tool definitions each iteration (dynamic loading)
            tools = self._manager.get_active_tool_definitions()

            # Build system prompt with plugin extras
            system = self.system_prompt
            extra = ctx.metadata.get("system_prompt_extra", "")
            if extra:
                system = f"{system}\n\n{extra}"

            # Refresh OAuth token if needed (Claude Code may have rotated it)
            self._refresh_client_if_needed()

            # API call
            api_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system,
                "messages": ctx.messages,
            }
            if tools:
                api_params["tools"] = tools

            response = self._client.messages.create(**api_params)

            # Track token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            ctx.total_input_tokens += input_tokens
            ctx.total_output_tokens += output_tokens
            self._manager.dispatch_on_token_usage(ctx, input_tokens, output_tokens)

            # Append assistant response
            ctx.messages.append({"role": "assistant", "content": response.content})

            # Process response blocks
            tool_uses = []
            final_text = ""

            for block in response.content:
                if block.type == "text":
                    thought = self._manager.dispatch_on_thought(ctx, block.text)
                    final_text = thought
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # If no tool calls and stop_reason is end_turn, we're done
            if not tool_uses and response.stop_reason == "end_turn":
                final_answer = self._manager.dispatch_on_agent_end(ctx, final_text)
                # Stop hook can force continuation
                if isinstance(final_answer, dict) and final_answer.get("continue"):
                    msg = final_answer.get("message", "Please continue.")
                    ctx.messages.append({"role": "user", "content": msg})
                    continue
                return final_answer if isinstance(final_answer, str) else final_text

            # Execute tool calls
            if tool_uses:
                tool_results = []
                for tool_use in tool_uses:
                    tc = {"name": tool_use.name, "input": tool_use.input, "id": tool_use.id}
                    tc = self._manager.dispatch_before_action(ctx, tc)

                    if tc.get("_blocked"):
                        result = tc["_blocked"]
                    else:
                        try:
                            result = self._manager.route_tool_call(tc["name"], tc["input"])
                            result = self._manager.dispatch_after_action(ctx, tc, result)
                        except Exception as e:
                            error_msg = self._manager.dispatch_on_error(ctx, e)
                            result = error_msg or f"Error: {e}"

                    result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    observation = self._manager.dispatch_on_observation(ctx, result_str)

                    ctx.tool_call_history.append({
                        "name": tc["name"],
                        "input": tc["input"],
                        "result": observation,
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": observation,
                    })

                ctx.messages.append({"role": "user", "content": tool_results})

        # Max iterations reached
        fallback = final_text or "I've reached the maximum number of iterations without a final answer."
        result = self._manager.dispatch_on_agent_end(ctx, fallback)
        return result if isinstance(result, str) else fallback
