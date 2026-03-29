"""TDD tests for HookManager — written BEFORE implementation."""

import json
import os
import stat
import sys
import textwrap

import pytest

from hook_manager import HookManager, HookManagerPlugin, HookResult
from react_agent import AgentContext, SkillPluginManager


# ============================================================
# Config Loading
# ============================================================

class TestConfigLoading:
    def test_load_from_explicit_path(self, tmp_path):
        config = {"PreToolUse": [{"matcher": "bash", "hooks": []}]}
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg_path))
        assert "PreToolUse" in mgr.config

    def test_load_from_project_root(self, tmp_path, monkeypatch):
        config = {"PostToolUse": []}
        (tmp_path / "hooks.json").write_text(json.dumps(config))
        monkeypatch.chdir(tmp_path)
        mgr = HookManager()
        assert "PostToolUse" in mgr.config

    def test_missing_config_returns_empty(self, tmp_path):
        mgr = HookManager(config_path=str(tmp_path / "nonexistent.json"))
        assert mgr.config == {}

    def test_invalid_json_returns_empty(self, tmp_path):
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text("not json{{{")
        mgr = HookManager(config_path=str(cfg_path))
        assert mgr.config == {}

    def test_config_structure_validation(self, tmp_path):
        config = {
            "PreToolUse": [
                {"matcher": "Write", "hooks": [{"type": "command", "command": "echo ok"}]}
            ]
        }
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg_path))
        assert len(mgr.config["PreToolUse"]) == 1


# ============================================================
# Matching
# ============================================================

class TestMatching:
    def _make_mgr(self, tmp_path, event, matcher):
        config = {event: [{"matcher": matcher, "hooks": [{"type": "command", "command": "echo matched"}]}]}
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))
        return HookManager(config_path=str(cfg_path))

    def test_matcher_exact_tool_name(self, tmp_path):
        mgr = self._make_mgr(tmp_path, "PreToolUse", "bash")
        hooks = mgr._find_matching_hooks("PreToolUse", "bash")
        assert len(hooks) >= 1

    def test_matcher_pipe_alternatives(self, tmp_path):
        mgr = self._make_mgr(tmp_path, "PreToolUse", "Write|bash")
        assert len(mgr._find_matching_hooks("PreToolUse", "Write")) >= 1
        assert len(mgr._find_matching_hooks("PreToolUse", "bash")) >= 1
        assert len(mgr._find_matching_hooks("PreToolUse", "read")) == 0

    def test_no_matching_hooks(self, tmp_path):
        mgr = self._make_mgr(tmp_path, "PreToolUse", "bash")
        hooks = mgr._find_matching_hooks("PreToolUse", "read")
        assert len(hooks) == 0


# ============================================================
# PreToolUse Hooks
# ============================================================

class TestPreToolUse:
    def _make_hook_script(self, tmp_path, stdout_json):
        script = tmp_path / "hook.sh"
        script.write_text(f"#!/bin/bash\necho '{json.dumps(stdout_json)}'")
        script.chmod(0o755)
        return str(script)

    def test_pre_tool_use_allows(self, tmp_path):
        script = self._make_hook_script(tmp_path, {"continue": True})
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [{"type": "command", "command": script}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_pre_tool_use("bash", {"command": "echo hi"})
        assert result.allow is True

    def test_pre_tool_use_blocks(self, tmp_path):
        script = self._make_hook_script(tmp_path, {"continue": False})
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [{"type": "command", "command": script}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_pre_tool_use("bash", {"command": "rm -rf /"})
        assert result.allow is False

    def test_pre_tool_use_system_message(self, tmp_path):
        script = self._make_hook_script(tmp_path, {"continue": True, "systemMessage": "Validated OK"})
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [{"type": "command", "command": script}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_pre_tool_use("bash", {"command": "echo ok"})
        assert result.system_message == "Validated OK"

    def test_pre_tool_use_failure_blocks(self, tmp_path):
        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/bash\nexit 2")
        script.chmod(0o755)
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [{"type": "command", "command": str(script)}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_pre_tool_use("bash", {"command": "echo"})
        assert result.allow is False

    def test_pre_tool_use_timeout(self, tmp_path):
        script = tmp_path / "slow.sh"
        script.write_text("#!/bin/bash\nsleep 30")
        script.chmod(0o755)
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [
            {"type": "command", "command": str(script), "timeout": 1}
        ]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_pre_tool_use("bash", {"command": "echo"})
        assert result.allow is False  # Timeout → block

    def test_pre_tool_use_no_matching_hooks(self, tmp_path):
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [{"type": "command", "command": "echo"}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_pre_tool_use("read", {"file_path": "test.txt"})
        assert result.allow is True  # No matching hooks → allow


# ============================================================
# PostToolUse Hooks
# ============================================================

class TestPostToolUse:
    def test_post_tool_use_runs(self, tmp_path):
        script = tmp_path / "post.sh"
        script.write_text('#!/bin/bash\necho \'{"continue": true}\'')
        script.chmod(0o755)
        config = {"PostToolUse": [{"matcher": "read", "hooks": [{"type": "command", "command": str(script)}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_post_tool_use("read", {"file_path": "x"}, "file content")
        assert result.allow is True

    def test_post_tool_use_system_message(self, tmp_path):
        script = tmp_path / "post.sh"
        script.write_text('#!/bin/bash\necho \'{"continue": true, "systemMessage": "Logged"}\'')
        script.chmod(0o755)
        config = {"PostToolUse": [{"matcher": "read", "hooks": [{"type": "command", "command": str(script)}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_post_tool_use("read", {}, "content")
        assert result.system_message == "Logged"


# ============================================================
# Stop Hooks
# ============================================================

class TestStopHook:
    def test_stop_allows(self, tmp_path):
        script = tmp_path / "stop.sh"
        script.write_text('#!/bin/bash\necho \'{"continue": false}\'')
        script.chmod(0o755)
        config = {"Stop": [{"hooks": [{"type": "command", "command": str(script)}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_stop("Final answer")
        assert result.allow is True  # continue=false means allow stop

    def test_stop_forces_continue(self, tmp_path):
        script = tmp_path / "stop.sh"
        script.write_text('#!/bin/bash\necho \'{"continue": true, "systemMessage": "Not done yet"}\'')
        script.chmod(0o755)
        config = {"Stop": [{"hooks": [{"type": "command", "command": str(script)}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))
        mgr = HookManager(config_path=str(cfg))
        result = mgr.run_stop("Partial answer")
        assert result.force_continue is True
        assert result.system_message == "Not done yet"


# ============================================================
# HookManagerPlugin (SkillPlugin integration)
# ============================================================

class TestHookManagerPlugin:
    def test_plugin_name(self, tmp_path):
        mgr = HookManager(config_path=str(tmp_path / "nope.json"))
        plugin = HookManagerPlugin(mgr)
        assert plugin.name == "hooks"

    def test_registers_with_plugin_manager(self, tmp_path):
        hook_mgr = HookManager(config_path=str(tmp_path / "nope.json"))
        plugin = HookManagerPlugin(hook_mgr)
        pm = SkillPluginManager()
        pm.register(plugin)
        # Should not raise, no tools to register

    def test_before_action_blocks_tool(self, tmp_path):
        script = tmp_path / "block.sh"
        script.write_text('#!/bin/bash\necho \'{"continue": false}\'')
        script.chmod(0o755)
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [{"type": "command", "command": str(script)}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))

        hook_mgr = HookManager(config_path=str(cfg))
        plugin = HookManagerPlugin(hook_mgr)
        ctx = AgentContext(user_query="test")

        tc = {"name": "bash", "input": {"command": "rm -rf /"}, "id": "toolu_001"}
        result = plugin.before_action(ctx, tc)
        assert result is not None
        assert "_blocked" in result

    def test_before_action_allows_tool(self, tmp_path):
        script = tmp_path / "allow.sh"
        script.write_text('#!/bin/bash\necho \'{"continue": true}\'')
        script.chmod(0o755)
        config = {"PreToolUse": [{"matcher": "bash", "hooks": [{"type": "command", "command": str(script)}]}]}
        cfg = tmp_path / "hooks.json"
        cfg.write_text(json.dumps(config))

        hook_mgr = HookManager(config_path=str(cfg))
        plugin = HookManagerPlugin(hook_mgr)
        ctx = AgentContext(user_query="test")

        tc = {"name": "bash", "input": {"command": "echo hi"}, "id": "toolu_001"}
        result = plugin.before_action(ctx, tc)
        assert result is None  # None = don't modify, allow through
