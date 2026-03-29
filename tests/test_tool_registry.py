# tests/test_tool_registry.py — TDD tests for dynamic tool loading (open/close book)

from __future__ import annotations

import pytest

from react_agent import AgentContext, SkillPlugin, SkillPluginManager


# ── Test Doubles ──────────────────────────────────────────────


class AlwaysActivePlugin(SkillPlugin):
    """A plugin whose tools are always active (is_deferred=False, the default)."""

    @property
    def name(self) -> str:
        return "always_active"

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "recall_episode",
                "description": "Recall an episode from memory",
                "input_schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            }
        ]

    def execute_tool(self, name, tool_input):
        return f"recalled: {tool_input.get('id')}"


class DeferredPlugin(SkillPlugin):
    """A plugin whose tools start unloaded (is_deferred=True)."""

    @property
    def name(self) -> str:
        return "deferred_tools"

    def is_deferred(self) -> bool:
        return True

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "read",
                "description": "Read a file's contents",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
            {
                "name": "grep",
                "description": "Search file contents with regex",
                "input_schema": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
            {
                "name": "bash",
                "description": "Execute shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        ]

    def execute_tool(self, name, tool_input):
        return f"executed: {name}"


# ── SkillPlugin.is_deferred ──────────────────────────────────


class TestIsDeferredDefault:
    """is_deferred() defaults to False for backward compatibility."""

    def test_default_is_false(self):
        plugin = AlwaysActivePlugin()
        assert plugin.is_deferred() is False

    def test_deferred_plugin_returns_true(self):
        plugin = DeferredPlugin()
        assert plugin.is_deferred() is True


# ── SkillPluginManager: active tool tracking ─────────────────


class TestActiveToolTracking:
    """Manager tracks which tools are active vs deferred."""

    def test_non_deferred_tools_are_always_active(self):
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "recall_episode" in names

    def test_deferred_tools_start_inactive(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "read" not in names
        assert "grep" not in names
        assert "bash" not in names

    def test_get_all_tool_definitions_still_returns_everything(self):
        """Backward compat: get_all_tool_definitions returns all tools."""
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        mgr.register(DeferredPlugin())
        all_tools = mgr.get_all_tool_definitions()
        names = {t["name"] for t in all_tools}
        assert names == {"recall_episode", "read", "grep", "bash"}

    def test_mixed_plugins_active_only_non_deferred(self):
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        mgr.register(DeferredPlugin())
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "recall_episode" in names
        assert "read" not in names


# ── load_tools / unload_tools ────────────────────────────────


class TestLoadUnloadTools:
    """Dynamic loading and unloading of deferred tools."""

    def test_load_single_tool(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        result = mgr.load_tools(["read"])
        assert "read" in result  # confirmation message
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "read" in names
        assert "grep" not in names

    def test_load_multiple_tools(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        mgr.load_tools(["read", "grep"])
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "read" in names
        assert "grep" in names
        assert "bash" not in names

    def test_unload_tool(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        mgr.load_tools(["read", "grep"])
        result = mgr.unload_tools(["read"])
        assert "read" in result  # confirmation message
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "read" not in names
        assert "grep" in names

    def test_unload_all_deferred(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        mgr.load_tools(["read", "grep", "bash"])
        mgr.unload_tools(["read", "grep", "bash"])
        active = mgr.get_active_tool_definitions()
        assert len(active) == 0

    def test_load_unknown_tool_returns_error(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        result = mgr.load_tools(["nonexistent"])
        assert "error" in result.lower() or "unknown" in result.lower()

    def test_unload_non_deferred_tool_rejected(self):
        """Cannot unload always-active tools."""
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        result = mgr.unload_tools(["recall_episode"])
        assert "error" in result.lower() or "cannot" in result.lower()
        # Tool should still be active
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "recall_episode" in names

    def test_load_already_loaded_is_idempotent(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        mgr.load_tools(["read"])
        mgr.load_tools(["read"])  # no error
        active = mgr.get_active_tool_definitions()
        names = [t["name"] for t in active]
        assert names.count("read") == 1

    def test_route_tool_call_works_for_loaded_tool(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        mgr.load_tools(["read"])
        result = mgr.route_tool_call("read", {"file_path": "/tmp/test"})
        assert "executed: read" in result

    def test_route_tool_call_works_for_unloaded_tool(self):
        """Routing still works even for unloaded tools (safety net)."""
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        # Don't load — but route should still work
        result = mgr.route_tool_call("read", {"file_path": "/tmp/test"})
        assert "executed: read" in result


# ── get_tool_catalog ─────────────────────────────────────────


class TestGetToolCatalog:
    """Catalog returns metadata for all tools (name, description, loaded status)."""

    def test_catalog_includes_all_tools(self):
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        mgr.register(DeferredPlugin())
        catalog = mgr.get_tool_catalog()
        names = {t["name"] for t in catalog}
        assert names == {"recall_episode", "read", "grep", "bash"}

    def test_catalog_shows_loaded_status(self):
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        mgr.register(DeferredPlugin())
        mgr.load_tools(["read"])
        catalog = mgr.get_tool_catalog()
        by_name = {t["name"]: t for t in catalog}
        assert by_name["recall_episode"]["loaded"] is True
        assert by_name["read"]["loaded"] is True
        assert by_name["grep"]["loaded"] is False
        assert by_name["bash"]["loaded"] is False

    def test_catalog_includes_description(self):
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        catalog = mgr.get_tool_catalog()
        by_name = {t["name"]: t for t in catalog}
        assert "file" in by_name["read"]["description"].lower()


# ── ToolRegistryPlugin ───────────────────────────────────────


class TestToolRegistryPlugin:
    """Plugin that provides load_tools/unload_tools and injects catalog."""

    def test_import(self):
        from tool_registry import ToolRegistryPlugin
        assert ToolRegistryPlugin is not None

    def test_provides_two_tools(self):
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        plugin = ToolRegistryPlugin(mgr)
        tools = plugin.get_tools()
        names = {t["name"] for t in tools}
        assert names == {"load_tools", "unload_tools"}

    def test_is_not_deferred(self):
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        plugin = ToolRegistryPlugin(mgr)
        assert plugin.is_deferred() is False

    def test_load_tools_execution(self):
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        result = plugin.execute_tool("load_tools", {"names": ["read"]})
        assert "read" in result
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "read" in names

    def test_unload_tools_execution(self):
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read", "grep"])
        result = plugin.execute_tool("unload_tools", {"names": ["read"]})
        assert "read" in result
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "read" not in names
        assert "grep" in names

    def test_injects_catalog_on_agent_start(self):
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        ctx = AgentContext(user_query="test")
        plugin.on_agent_start(ctx)
        extra = ctx.metadata.get("system_prompt_extra", "")
        # Catalog should list deferred tools
        assert "read" in extra
        assert "grep" in extra
        assert "bash" in extra
        assert "load_tools" in extra  # instructions to use load_tools

    def test_catalog_not_injected_when_no_deferred_tools(self):
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        ctx = AgentContext(user_query="test")
        plugin.on_agent_start(ctx)
        extra = ctx.metadata.get("system_prompt_extra", "")
        # No deferred tools, so no catalog injected
        assert "load_tools" not in extra


# ── SystemToolsPlugin / MCPPlugin deferred ───────────────────


class TestExistingPluginsDeferred:
    """SystemToolsPlugin and MCPPlugin should be deferred."""

    def test_system_tools_is_deferred(self):
        from system_tools import SystemToolsPlugin
        plugin = SystemToolsPlugin()
        assert plugin.is_deferred() is True

    def test_system_tools_start_inactive(self):
        from system_tools import SystemToolsPlugin
        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        active = mgr.get_active_tool_definitions()
        names = {t["name"] for t in active}
        assert "read" not in names
        assert "bash" not in names

    def test_mcp_plugin_is_deferred(self):
        from mcp_client import MCPPlugin, MCPManager
        # MCPPlugin with no servers is fine for testing is_deferred
        import tempfile, json
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        config_file = tmp / ".mcp.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))
        manager = MCPManager(config_path=str(config_file))
        plugin = MCPPlugin(manager)
        assert plugin.is_deferred() is True


# ── Integration: _react_loop uses dynamic tools ─────────────


class TestReactLoopDynamicTools:
    """_react_loop should use get_active_tool_definitions each iteration."""

    def test_run_with_deferred_plugin_excludes_from_api(self):
        """When a deferred plugin is registered, its tools aren't sent to API initially."""
        from react_agent import ReActAgent
        # We can't easily test the API call without mocking, but we can verify
        # that get_active_tool_definitions is what would be used
        mgr = SkillPluginManager()
        mgr.register(AlwaysActivePlugin())
        mgr.register(DeferredPlugin())
        active = mgr.get_active_tool_definitions()
        all_tools = mgr.get_all_tool_definitions()
        assert len(active) < len(all_tools)
        assert len(active) == 1  # only recall_episode
        assert len(all_tools) == 4
