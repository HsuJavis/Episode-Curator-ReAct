# hook_manager.py — User-mountable hook system (PreToolUse/PostToolUse/Stop)

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from react_agent import AgentContext, SkillPlugin


@dataclass
class HookResult:
    """Result from running a hook."""
    allow: bool = True              # PreToolUse: allow execution; Stop: allow stop
    force_continue: bool = False    # Stop only: force agent to continue
    system_message: str | None = None


class HookManager:
    """Loads and executes user-configured hooks."""

    def __init__(self, config_path: str | None = None):
        self.config: dict = self._load_config(config_path)

    def _load_config(self, explicit_path: str | None) -> dict:
        paths_to_try = []
        if explicit_path:
            paths_to_try.append(explicit_path)
        else:
            # Project root
            paths_to_try.append(os.path.join(os.getcwd(), "hooks.json"))
            # Home directory
            paths_to_try.append(os.path.expanduser("~/.claude/hooks.json"))

        for p in paths_to_try:
            if os.path.isfile(p):
                try:
                    return json.loads(Path(p).read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    return {}
        return {}

    def _find_matching_hooks(self, event: str, tool_name: str = "") -> list[dict]:
        """Find hooks matching a tool name for a given event."""
        entries = self.config.get(event, [])
        matched_hooks = []
        for entry in entries:
            matcher = entry.get("matcher", "*")
            if self._matches(matcher, tool_name):
                matched_hooks.extend(entry.get("hooks", []))
        return matched_hooks

    @staticmethod
    def _matches(matcher: str, tool_name: str) -> bool:
        if matcher == "*":
            return True
        # Pipe-separated alternatives: "Write|bash|read"
        alternatives = [m.strip() for m in matcher.split("|")]
        for alt in alternatives:
            if alt == tool_name:
                return True
            try:
                if re.fullmatch(alt, tool_name):
                    return True
            except re.error:
                pass
        return False

    def _run_hook(self, hook: dict, context: dict) -> HookResult:
        """Execute a single hook and parse its output."""
        hook_type = hook.get("type", "command")
        if hook_type != "command":
            return HookResult(allow=True)

        command = hook.get("command", "")
        timeout = hook.get("timeout", 30)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=json.dumps(context, ensure_ascii=False),
            )
        except subprocess.TimeoutExpired:
            return HookResult(allow=False, system_message="Hook timed out")
        except Exception as e:
            return HookResult(allow=False, system_message=f"Hook error: {e}")

        # Exit code 2 = blocking error
        if proc.returncode == 2:
            msg = proc.stderr.strip() or "Blocked by hook"
            return HookResult(allow=False, system_message=msg)

        # Try to parse JSON output
        stdout = proc.stdout.strip()
        if stdout:
            try:
                data = json.loads(stdout)
                cont = data.get("continue", True)
                sys_msg = data.get("systemMessage")

                # For Stop hooks: "continue": true means force agent to keep going
                return HookResult(
                    allow=cont,
                    force_continue=cont if "Stop" in str(context.get("event", "")) else False,
                    system_message=sys_msg,
                )
            except json.JSONDecodeError:
                pass

        return HookResult(allow=True)

    def run_pre_tool_use(self, tool_name: str, tool_input: dict) -> HookResult:
        hooks = self._find_matching_hooks("PreToolUse", tool_name)
        if not hooks:
            return HookResult(allow=True)

        context = {"event": "PreToolUse", "tool_name": tool_name, "tool_input": tool_input}
        for hook in hooks:
            result = self._run_hook(hook, context)
            if not result.allow:
                return result  # First block wins
        return HookResult(allow=True, system_message=result.system_message if hooks else None)

    def run_post_tool_use(self, tool_name: str, tool_input: dict, tool_result: str) -> HookResult:
        hooks = self._find_matching_hooks("PostToolUse", tool_name)
        if not hooks:
            return HookResult(allow=True)

        context = {"event": "PostToolUse", "tool_name": tool_name,
                   "tool_input": tool_input, "tool_result": tool_result[:1000]}
        last_result = HookResult(allow=True)
        for hook in hooks:
            last_result = self._run_hook(hook, context)
        return last_result

    def run_stop(self, final_answer: str) -> HookResult:
        entries = self.config.get("Stop", [])
        if not entries:
            return HookResult(allow=True)

        context = {"event": "Stop", "final_answer": final_answer[:1000]}
        for entry in entries:
            for hook in entry.get("hooks", []):
                result = self._run_hook(hook, context)
                if result.allow:  # "continue": true → force continue
                    return HookResult(allow=False, force_continue=True,
                                      system_message=result.system_message)
        return HookResult(allow=True)  # All hooks said continue=false → allow stop


class HookManagerPlugin(SkillPlugin):
    """Bridges HookManager into the SkillPlugin system."""

    def __init__(self, hook_manager: HookManager):
        self._hook_mgr = hook_manager

    @property
    def name(self) -> str:
        return "hooks"

    def before_action(self, ctx: AgentContext, tool_call: dict) -> Optional[dict]:
        result = self._hook_mgr.run_pre_tool_use(
            tool_call.get("name", ""),
            tool_call.get("input", {}),
        )
        if not result.allow:
            # Block execution by setting _blocked key
            blocked = dict(tool_call)
            blocked["_blocked"] = result.system_message or "Blocked by hook"
            return blocked
        return None  # Allow through

    def after_action(self, ctx: AgentContext, tool_call: dict, result: Any) -> Optional[Any]:
        hook_result = self._hook_mgr.run_post_tool_use(
            tool_call.get("name", ""),
            tool_call.get("input", {}),
            str(result)[:1000],
        )
        if hook_result.system_message:
            # Append system message to result
            return f"{result}\n\n[Hook: {hook_result.system_message}]"
        return None

    def on_agent_end(self, ctx: AgentContext, final_answer: str) -> Optional[str]:
        result = self._hook_mgr.run_stop(final_answer)
        if result.force_continue:
            return {"continue": True, "message": result.system_message or "Please continue."}
        return None
