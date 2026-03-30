"""TDD tests for extended SystemToolsPlugin — web_fetch, task mgmt, process mgmt, system_info, execute_skill.

Written BEFORE implementation per TDD methodology.
"""

import json
import os
import signal
import sys
import textwrap
import time

import pytest

from system_tools import SystemToolsPlugin
from react_agent import SkillPluginManager, AgentContext


# ============================================================
# Fixture: fresh plugin
# ============================================================

@pytest.fixture
def plugin():
    return SystemToolsPlugin()


@pytest.fixture
def ctx():
    """Minimal AgentContext for hooks."""
    return AgentContext(
        user_query="test",
        messages=[],
        metadata={},
        iteration=0,
        total_input_tokens=0,
        total_output_tokens=0,
        start_time=time.time(),
        tool_call_history=[],
    )


# ============================================================
# 0. Registration — new tools exist alongside original 6
# ============================================================

class TestExtendedRegistration:
    def test_new_tools_exist(self, plugin):
        """All new tools should be registered."""
        names = {t["name"] for t in plugin.get_tools()}
        expected_new = {
            "web_fetch", "task_create", "task_get", "task_list",
            "task_update", "task_stop", "task_output", "task_delete",
            "process_spawn", "process_kill", "process_list", "process_status",
            "system_info", "execute_skill",
        }
        assert expected_new.issubset(names), f"Missing: {expected_new - names}"

    def test_original_tools_still_exist(self, plugin):
        names = {t["name"] for t in plugin.get_tools()}
        original = {"read", "write", "grep", "search", "bash", "web_search"}
        assert original.issubset(names)

    def test_total_tool_count(self, plugin):
        tools = plugin.get_tools()
        # 6 original + 14 new = 20
        assert len(tools) == 20

    def test_all_new_tools_have_schema(self, plugin):
        for tool in plugin.get_tools():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"


# ============================================================
# 1. web_fetch — HTTP GET with urllib
# ============================================================

class TestWebFetchTool:
    def test_fetch_valid_url(self, plugin):
        """Fetch a known stable URL."""
        result = plugin.execute_tool("web_fetch", {"url": "https://httpbin.org/get"})
        data = json.loads(result)
        assert "url" in data or "headers" in data

    def test_fetch_with_timeout(self, plugin):
        """Should respect timeout parameter."""
        # httpbin delay — 5 second delay but 1 second timeout
        result = plugin.execute_tool("web_fetch", {
            "url": "https://httpbin.org/delay/5",
            "timeout": 1,
        })
        assert "error" in result.lower() or "timeout" in result.lower()

    def test_fetch_invalid_url(self, plugin):
        result = plugin.execute_tool("web_fetch", {"url": "https://nonexistent.invalid.tld/"})
        assert "error" in result.lower()

    def test_fetch_empty_url(self, plugin):
        result = plugin.execute_tool("web_fetch", {"url": ""})
        assert "error" in result.lower()

    def test_fetch_max_size_truncation(self, plugin):
        """Large responses should be truncated."""
        # httpbin /bytes/N returns N random bytes
        result = plugin.execute_tool("web_fetch", {
            "url": "https://httpbin.org/bytes/500000",
            "max_size": 1000,
        })
        # Should be truncated or error about size
        assert len(result) < 100000

    def test_fetch_returns_text(self, plugin):
        """Should return text content."""
        result = plugin.execute_tool("web_fetch", {"url": "https://httpbin.org/html"})
        assert "Herman Melville" in result or "<html" in result.lower() or "error" in result.lower()


# ============================================================
# 2. Task Management — background task lifecycle
# ============================================================

class TestTaskManagement:
    def test_task_create_returns_id(self, plugin):
        """task_create should return a task ID."""
        result = plugin.execute_tool("task_create", {
            "command": "echo hello",
            "name": "test_echo",
        })
        data = json.loads(result)
        assert "task_id" in data
        assert data["task_id"]

    def test_task_get_existing(self, plugin):
        """task_get should return task info."""
        create_result = json.loads(plugin.execute_tool("task_create", {
            "command": "sleep 10",
            "name": "long_task",
        }))
        task_id = create_result["task_id"]

        result = plugin.execute_tool("task_get", {"task_id": task_id})
        data = json.loads(result)
        assert data["task_id"] == task_id
        assert data["name"] == "long_task"
        assert "status" in data

        # Cleanup
        plugin.execute_tool("task_stop", {"task_id": task_id})

    def test_task_get_nonexistent(self, plugin):
        result = plugin.execute_tool("task_get", {"task_id": "nonexistent_999"})
        assert "error" in result.lower() or "not found" in result.lower()

    def test_task_list_empty(self, plugin):
        """Fresh plugin should have empty task list."""
        # Create a fresh plugin to avoid pollution
        p = SystemToolsPlugin()
        result = p.execute_tool("task_list", {})
        data = json.loads(result)
        assert "tasks" in data
        assert isinstance(data["tasks"], list)

    def test_task_list_after_create(self, plugin):
        create_result = json.loads(plugin.execute_tool("task_create", {
            "command": "sleep 10",
            "name": "listed_task",
        }))
        task_id = create_result["task_id"]

        result = plugin.execute_tool("task_list", {})
        data = json.loads(result)
        ids = [t["task_id"] for t in data["tasks"]]
        assert task_id in ids

        plugin.execute_tool("task_stop", {"task_id": task_id})

    def test_task_stop(self, plugin):
        create_result = json.loads(plugin.execute_tool("task_create", {
            "command": "sleep 60",
            "name": "to_stop",
        }))
        task_id = create_result["task_id"]

        result = plugin.execute_tool("task_stop", {"task_id": task_id})
        assert "stop" in result.lower() or "terminated" in result.lower()

        # Verify stopped
        get_result = json.loads(plugin.execute_tool("task_get", {"task_id": task_id}))
        assert get_result["status"] in ("stopped", "completed", "terminated")

    def test_task_output(self, plugin):
        """task_output should capture stdout."""
        create_result = json.loads(plugin.execute_tool("task_create", {
            "command": "echo task_output_test",
            "name": "output_task",
        }))
        task_id = create_result["task_id"]

        # Wait briefly for command to finish
        time.sleep(0.5)

        result = plugin.execute_tool("task_output", {"task_id": task_id})
        assert "task_output_test" in result

    def test_task_delete(self, plugin):
        create_result = json.loads(plugin.execute_tool("task_create", {
            "command": "echo delete_me",
            "name": "to_delete",
        }))
        task_id = create_result["task_id"]
        time.sleep(0.3)

        result = plugin.execute_tool("task_delete", {"task_id": task_id})
        assert "delete" in result.lower() or "removed" in result.lower()

        # Should not be listed anymore
        list_result = json.loads(plugin.execute_tool("task_list", {}))
        ids = [t["task_id"] for t in list_result["tasks"]]
        assert task_id not in ids

    def test_task_update(self, plugin):
        """task_update should update task metadata/name."""
        create_result = json.loads(plugin.execute_tool("task_create", {
            "command": "sleep 10",
            "name": "original_name",
        }))
        task_id = create_result["task_id"]

        result = plugin.execute_tool("task_update", {
            "task_id": task_id,
            "name": "updated_name",
        })
        assert "update" in result.lower()

        get_result = json.loads(plugin.execute_tool("task_get", {"task_id": task_id}))
        assert get_result["name"] == "updated_name"

        plugin.execute_tool("task_stop", {"task_id": task_id})


# ============================================================
# 3. Process Management — spawn/kill/list/status
# ============================================================

class TestProcessManagement:
    def test_process_spawn_returns_pid(self, plugin):
        result = plugin.execute_tool("process_spawn", {
            "command": "sleep 30",
            "name": "test_sleep",
        })
        data = json.loads(result)
        assert "pid" in data
        assert isinstance(data["pid"], int)

        # Cleanup
        plugin.execute_tool("process_kill", {"pid": data["pid"]})

    def test_process_kill(self, plugin):
        spawn_result = json.loads(plugin.execute_tool("process_spawn", {
            "command": "sleep 60",
            "name": "to_kill",
        }))
        pid = spawn_result["pid"]

        result = plugin.execute_tool("process_kill", {"pid": pid})
        assert "kill" in result.lower() or "terminated" in result.lower()

    def test_process_kill_nonexistent(self, plugin):
        result = plugin.execute_tool("process_kill", {"pid": 999999999})
        assert "error" in result.lower() or "not found" in result.lower()

    def test_process_list(self, plugin):
        spawn_result = json.loads(plugin.execute_tool("process_spawn", {
            "command": "sleep 30",
            "name": "listed_proc",
        }))
        pid = spawn_result["pid"]

        result = plugin.execute_tool("process_list", {})
        data = json.loads(result)
        assert "processes" in data
        pids = [p["pid"] for p in data["processes"]]
        assert pid in pids

        plugin.execute_tool("process_kill", {"pid": pid})

    def test_process_status(self, plugin):
        spawn_result = json.loads(plugin.execute_tool("process_spawn", {
            "command": "sleep 30",
            "name": "status_proc",
        }))
        pid = spawn_result["pid"]

        result = plugin.execute_tool("process_status", {"pid": pid})
        data = json.loads(result)
        assert data["pid"] == pid
        assert data["status"] == "running"

        plugin.execute_tool("process_kill", {"pid": pid})

    def test_process_status_after_kill(self, plugin):
        spawn_result = json.loads(plugin.execute_tool("process_spawn", {
            "command": "sleep 60",
            "name": "kill_then_status",
        }))
        pid = spawn_result["pid"]
        plugin.execute_tool("process_kill", {"pid": pid})
        time.sleep(0.3)

        result = plugin.execute_tool("process_status", {"pid": pid})
        data = json.loads(result)
        assert data["status"] in ("terminated", "not_found", "stopped")


# ============================================================
# 4. System Info
# ============================================================

class TestSystemInfo:
    def test_returns_json(self, plugin):
        result = plugin.execute_tool("system_info", {})
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_os_info(self, plugin):
        data = json.loads(plugin.execute_tool("system_info", {}))
        assert "os" in data
        assert "platform" in data

    def test_contains_python_version(self, plugin):
        data = json.loads(plugin.execute_tool("system_info", {}))
        assert "python_version" in data
        assert sys.version.split()[0] in data["python_version"]

    def test_contains_cpu_info(self, plugin):
        data = json.loads(plugin.execute_tool("system_info", {}))
        assert "cpu_count" in data
        assert isinstance(data["cpu_count"], int)

    def test_contains_cwd(self, plugin):
        data = json.loads(plugin.execute_tool("system_info", {}))
        assert "cwd" in data

    def test_contains_memory_info(self, plugin):
        data = json.loads(plugin.execute_tool("system_info", {}))
        # At minimum, should have some memory field
        assert "memory" in data or "memory_total" in data


# ============================================================
# 5. Execute Skill
# ============================================================

class TestExecuteSkill:
    def test_execute_existing_skill(self, plugin, tmp_path, ctx):
        """Should load and return a skill's content."""
        # Create a skill directory
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: test-skill
            description: A test skill
            ---

            # Test Skill
            Do the test thing.
        """))

        # Plugin needs to know where skills are
        plugin._skills_dir = str(tmp_path / "skills")
        result = plugin.execute_tool("execute_skill", {"skill_name": "test-skill"})
        assert "Test Skill" in result
        assert "Do the test thing" in result

    def test_execute_nonexistent_skill(self, plugin):
        result = plugin.execute_tool("execute_skill", {"skill_name": "nonexistent-skill"})
        assert "error" in result.lower() or "not found" in result.lower()

    def test_execute_skill_returns_content(self, plugin, tmp_path):
        skill_dir = tmp_path / "skills" / "greet"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: greet
            description: Greeting skill
            ---

            Say hello to the user warmly.
        """))

        plugin._skills_dir = str(tmp_path / "skills")
        result = plugin.execute_tool("execute_skill", {"skill_name": "greet"})
        assert "Say hello" in result


# ============================================================
# 6. create_agent should register SystemToolsPlugin
# ============================================================

class TestCreateAgentRegistration:
    @pytest.mark.llm
    def test_create_agent_includes_system_tools(self, api_key, tmp_storage):
        from episode_curator import create_agent
        agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            storage_dir=tmp_storage,
            api_key=api_key,
        )
        all_tools = agent._manager.get_all_tool_definitions()
        all_names = {t["name"] for t in all_tools}
        # System tools should be registered (possibly deferred)
        plugin_names = [p.name for p in agent._manager._plugins]
        assert "system_tools" in plugin_names, (
            f"SystemToolsPlugin not registered. Plugins: {plugin_names}"
        )

    @pytest.mark.llm
    def test_create_agent_includes_tool_registry(self, api_key, tmp_storage):
        from episode_curator import create_agent
        agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            storage_dir=tmp_storage,
            api_key=api_key,
        )
        plugin_names = [p.name for p in agent._manager._plugins]
        assert "tool_registry" in plugin_names
