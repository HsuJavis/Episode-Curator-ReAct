"""TDD tests for MCP client support — written BEFORE implementation."""

import json
import os
import signal
import sys
import time

import pytest

from mcp_client import MCPManager, MCPPlugin
from react_agent import SkillPluginManager


FAKE_SERVER = os.path.join(os.path.dirname(__file__), "fake_mcp_server.py")


# ============================================================
# Config Loading
# ============================================================


class TestMCPConfigLoading:
    def test_load_from_explicit_path(self, tmp_path):
        config = {
            "mcpServers": {
                "test": {
                    "command": sys.executable,
                    "args": [FAKE_SERVER],
                }
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))
        mgr = MCPManager(config_path=str(config_file))
        assert "test" in mgr.server_configs
        assert mgr.server_configs["test"]["command"] == sys.executable

    def test_missing_config_returns_empty(self, tmp_path):
        mgr = MCPManager(config_path=str(tmp_path / "nonexistent.json"))
        assert mgr.server_configs == {}

    def test_config_with_multiple_servers(self, tmp_path):
        config = {
            "mcpServers": {
                "server_a": {"command": "python", "args": ["a.py"]},
                "server_b": {"command": "python", "args": ["b.py"]},
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))
        mgr = MCPManager(config_path=str(config_file))
        assert len(mgr.server_configs) == 2
        assert "server_a" in mgr.server_configs
        assert "server_b" in mgr.server_configs

    def test_env_var_expansion(self, tmp_path):
        config = {
            "mcpServers": {
                "test": {
                    "command": "${HOME}/bin/server",
                    "args": [],
                }
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))
        mgr = MCPManager(config_path=str(config_file))
        home = os.environ.get("HOME", "")
        assert mgr.server_configs["test"]["command"] == f"{home}/bin/server"


# ============================================================
# MCPManager
# ============================================================


class TestMCPManager:
    def _make_manager(self, tmp_path, server_name="test"):
        config = {
            "mcpServers": {
                server_name: {
                    "command": sys.executable,
                    "args": [FAKE_SERVER],
                }
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))
        return MCPManager(config_path=str(config_file))

    def test_start_server(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        ok = mgr.start_server("test", mgr.server_configs["test"])
        assert ok is True
        assert "test" in mgr.servers
        mgr.shutdown()

    def test_discover_tools(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.start_server("test", mgr.server_configs["test"])
        tools = mgr.discover_tools("test")
        assert len(tools) == 1
        tool = tools[0]
        assert tool["name"] == "mcp__test__echo"
        assert "input_schema" in tool
        assert "description" in tool
        mgr.shutdown()

    def test_call_tool(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.start_server("test", mgr.server_configs["test"])
        result = mgr.call_tool("test", "echo", {"text": "hello"})
        assert "Echo: hello" in result
        mgr.shutdown()

    def test_tool_name_prefixing(self, tmp_path):
        mgr = self._make_manager(tmp_path, server_name="myserver")
        mgr.start_server("myserver", mgr.server_configs["myserver"])
        tools = mgr.discover_tools("myserver")
        assert tools[0]["name"] == "mcp__myserver__echo"
        mgr.shutdown()

    def test_server_crash_handled(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.start_server("test", mgr.server_configs["test"])
        # Kill the server process
        proc = mgr.servers["test"]
        proc.terminate()
        proc.wait()
        result = mgr.call_tool("test", "echo", {"text": "hello"})
        assert "error" in result.lower()
        mgr.shutdown()

    def test_shutdown_stops_servers(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.start_server("test", mgr.server_configs["test"])
        proc = mgr.servers["test"]
        mgr.shutdown()
        # Process should be terminated
        assert proc.poll() is not None

    def test_inputSchema_to_input_schema(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.start_server("test", mgr.server_configs["test"])
        tools = mgr.discover_tools("test")
        tool = tools[0]
        # Must have snake_case key for Anthropic API
        assert "input_schema" in tool
        # Must NOT have camelCase key
        assert "inputSchema" not in tool
        mgr.shutdown()


# ============================================================
# MCPPlugin
# ============================================================


class TestMCPPlugin:
    def _make_plugin(self, tmp_path):
        config = {
            "mcpServers": {
                "test": {
                    "command": sys.executable,
                    "args": [FAKE_SERVER],
                }
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))
        mgr = MCPManager(config_path=str(config_file))
        mgr.start_all()
        return MCPPlugin(mgr)

    def test_plugin_name(self, tmp_path):
        plugin = self._make_plugin(tmp_path)
        assert plugin.name == "mcp"
        plugin._manager.shutdown()

    def test_get_tools_returns_discovered(self, tmp_path):
        plugin = self._make_plugin(tmp_path)
        tools = plugin.get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "mcp__test__echo"
        plugin._manager.shutdown()

    def test_execute_tool_routes(self, tmp_path):
        plugin = self._make_plugin(tmp_path)
        result = plugin.execute_tool("mcp__test__echo", {"text": "world"})
        assert "Echo: world" in result
        plugin._manager.shutdown()

    def test_registers_with_plugin_manager(self, tmp_path):
        plugin = self._make_plugin(tmp_path)
        mgr = SkillPluginManager()
        mgr.register(plugin)
        all_tools = mgr.get_all_tool_definitions()
        assert len(all_tools) == 1
        assert all_tools[0]["name"] == "mcp__test__echo"
        plugin._manager.shutdown()

    def test_no_servers_empty_tools(self, tmp_path):
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))
        mgr = MCPManager(config_path=str(config_file))
        mgr.start_all()
        plugin = MCPPlugin(mgr)
        assert plugin.get_tools() == []
        mgr.shutdown()

    def test_unknown_tool_raises(self, tmp_path):
        plugin = self._make_plugin(tmp_path)
        with pytest.raises(ValueError, match="Unknown MCP tool"):
            plugin.execute_tool("mcp__test__nonexistent", {})
        plugin._manager.shutdown()
