"""TDD tests for SystemToolsPlugin — written BEFORE implementation."""

import json
import os
import stat

import pytest

# This import will fail until system_tools.py is created
from system_tools import SystemToolsPlugin
from react_agent import SkillPluginManager


# ============================================================
# Plugin Registration
# ============================================================

class TestPluginRegistration:
    def test_plugin_name(self):
        plugin = SystemToolsPlugin()
        assert plugin.name == "system_tools"

    def test_get_tools_returns_6_tools(self):
        plugin = SystemToolsPlugin()
        tools = plugin.get_tools()
        assert len(tools) == 6

    def test_all_tools_have_required_schema_keys(self):
        plugin = SystemToolsPlugin()
        for tool in plugin.get_tools():
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "input_schema" in tool, f"Tool missing 'input_schema': {tool}"
            assert tool["input_schema"]["type"] == "object"

    def test_tool_names_are_correct(self):
        plugin = SystemToolsPlugin()
        names = {t["name"] for t in plugin.get_tools()}
        assert names == {"read", "write", "grep", "search", "bash", "web_search"}

    def test_registers_with_plugin_manager(self):
        mgr = SkillPluginManager()
        plugin = SystemToolsPlugin()
        mgr.register(plugin)
        all_tools = mgr.get_all_tool_definitions()
        assert len(all_tools) == 6

    def test_no_conflict_with_episode_curator(self):
        from episode_curator import EpisodeStore, Curator, EpisodeCuratorPlugin
        mgr = SkillPluginManager()
        store = EpisodeStore.__new__(EpisodeStore)
        store._index = {}
        store._facts = []
        curator = Curator.__new__(Curator)
        ep_plugin = EpisodeCuratorPlugin(store, curator)
        sys_plugin = SystemToolsPlugin()
        mgr.register(ep_plugin)
        mgr.register(sys_plugin)  # Should not raise


# ============================================================
# Read Tool
# ============================================================

class TestReadTool:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("read", {"file_path": str(f)})
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_with_offset_and_limit(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 21)), encoding="utf-8")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("read", {"file_path": str(f), "offset": 5, "limit": 3})
        assert "line5" in result
        assert "line7" in result
        assert "line8" not in result

    def test_read_nonexistent_file(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("read", {"file_path": "/nonexistent/path.txt"})
        assert "error" in result.lower() or "not found" in result.lower()

    def test_read_binary_file(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("read", {"file_path": str(f)})
        assert "error" in result.lower() or "binary" in result.lower()

    def test_read_directory(self, tmp_path):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("read", {"file_path": str(tmp_path)})
        assert "error" in result.lower() or "directory" in result.lower()


# ============================================================
# Write Tool
# ============================================================

class TestWriteTool:
    def test_write_creates_new_file(self, tmp_path):
        target = tmp_path / "new.txt"
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("write", {"file_path": str(target), "content": "hello world"})
        assert target.exists()
        assert target.read_text() == "hello world"
        assert "success" in result.lower() or "wrote" in result.lower()

    def test_write_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "existing.txt"
        target.write_text("old content")
        plugin = SystemToolsPlugin()
        plugin.execute_tool("write", {"file_path": str(target), "content": "new content"})
        assert target.read_text() == "new content"

    def test_write_creates_intermediate_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "file.txt"
        plugin = SystemToolsPlugin()
        plugin.execute_tool("write", {"file_path": str(target), "content": "deep"})
        assert target.exists()
        assert target.read_text() == "deep"

    def test_write_permission_denied(self, tmp_path):
        if os.name == "nt":
            pytest.skip("Permission test not reliable on Windows")
        target = tmp_path / "readonly_dir" / "file.txt"
        ro_dir = tmp_path / "readonly_dir"
        ro_dir.mkdir()
        ro_dir.chmod(0o444)
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("write", {"file_path": str(target), "content": "fail"})
        assert "error" in result.lower() or "permission" in result.lower()
        ro_dir.chmod(0o755)  # Cleanup


# ============================================================
# Grep Tool
# ============================================================

class TestGrepTool:
    def test_grep_finds_pattern_in_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    return 'world'\n\ndef goodbye():\n    pass\n")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("grep", {"pattern": "def \\w+", "path": str(tmp_path)})
        assert "hello" in result
        assert "goodbye" in result

    def test_grep_regex_pattern(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("ERROR: disk full\nINFO: ok\nERROR: timeout\n")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("grep", {"pattern": "^ERROR:", "path": str(f)})
        assert "disk full" in result
        assert "timeout" in result
        assert "INFO" not in result

    def test_grep_with_include_glob(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\n")
        (tmp_path / "b.txt").write_text("import os\n")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("grep", {
            "pattern": "import", "path": str(tmp_path), "include": "*.py"
        })
        assert "a.py" in result
        assert "b.txt" not in result

    def test_grep_no_matches(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("nothing here")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("grep", {"pattern": "zzzzz", "path": str(tmp_path)})
        assert "no match" in result.lower() or result.strip() == ""

    def test_grep_nonexistent_path(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("grep", {"pattern": "test", "path": "/nonexistent"})
        assert "error" in result.lower() or "not found" in result.lower()


# ============================================================
# Search Tool
# ============================================================

class TestSearchTool:
    def test_search_finds_files_by_glob(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("search", {"pattern": "*.py", "path": str(tmp_path)})
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_search_with_path_restriction(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.py").write_text("")
        (tmp_path / "outer.py").write_text("")
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("search", {"pattern": "*.py", "path": str(sub)})
        assert "inner.py" in result
        assert "outer.py" not in result

    def test_search_no_matches(self, tmp_path):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("search", {"pattern": "*.xyz", "path": str(tmp_path)})
        assert "no match" in result.lower() or result.strip() == ""


# ============================================================
# Bash Tool
# ============================================================

class TestBashTool:
    def test_bash_simple_command(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("bash", {"command": "echo hello"})
        assert "hello" in result

    def test_bash_returns_stdout(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("bash", {"command": "echo stdout_test"})
        assert "stdout_test" in result

    def test_bash_returns_stderr_on_failure(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("bash", {"command": "ls /nonexistent_dir_12345"})
        assert "error" in result.lower() or "no such" in result.lower()

    def test_bash_timeout(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("bash", {"command": "sleep 10", "timeout": 1})
        assert "timeout" in result.lower() or "timed out" in result.lower()

    def test_bash_output_truncated(self):
        plugin = SystemToolsPlugin()
        # Generate output > 100KB
        result = plugin.execute_tool("bash", {
            "command": "python3 -c \"print('x' * 200000)\"",
            "timeout": 10,
        })
        assert len(result) <= 110000  # Some slack for truncation message


# ============================================================
# Web Search Tool
# ============================================================

class TestWebSearchTool:
    def test_web_search_returns_structured_result(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("web_search", {"query": "Python programming"})
        # Should return a JSON string with results
        data = json.loads(result)
        assert "results" in data

    def test_web_search_empty_query(self):
        plugin = SystemToolsPlugin()
        result = plugin.execute_tool("web_search", {"query": ""})
        assert "error" in result.lower() or "empty" in result.lower()


# ============================================================
# Integration — route through PluginManager
# ============================================================

class TestSystemToolsIntegration:
    def test_route_tool_calls(self, tmp_path):
        mgr = SkillPluginManager()
        plugin = SystemToolsPlugin()
        mgr.register(plugin)

        f = tmp_path / "test.txt"
        f.write_text("hello world")

        result = mgr.route_tool_call("read", {"file_path": str(f)})
        assert "hello world" in result

    def test_unknown_tool_raises(self):
        plugin = SystemToolsPlugin()
        with pytest.raises(ValueError, match="Unknown tool"):
            plugin.execute_tool("nonexistent", {})
