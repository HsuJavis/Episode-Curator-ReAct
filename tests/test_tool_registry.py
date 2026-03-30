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


# ── Close Book: context compression on unload ────────────────


def _make_tool_use_block(tool_id, name, tool_input):
    """Create a mock tool_use block (simulates Anthropic API response object)."""
    class MockToolUse:
        def __init__(self, id, name, input):
            self.type = "tool_use"
            self.id = id
            self.name = name
            self.input = input
    return MockToolUse(tool_id, name, tool_input)


def _make_text_block(text):
    """Create a mock text block."""
    class MockText:
        def __init__(self, text):
            self.type = "text"
            self.text = text
    return MockText(text)


def _build_conversation_with_tool_calls():
    """Build a realistic ctx.messages with tool_use/tool_result pairs.

    Simulates:
    1. User asks question
    2. Assistant thinks + calls read tool
    3. User sends tool_result with large file content
    4. Assistant thinks + calls grep tool
    5. User sends tool_result with grep results
    6. Assistant gives answer
    """
    large_file_content = "line " * 500  # ~2500 chars of file content
    grep_results = "match: " * 200      # ~1400 chars of grep results

    messages = [
        # [0] user question
        {"role": "user", "content": "Search for bugs in main.py"},
        # [1] assistant thinks + calls read
        {"role": "assistant", "content": [
            _make_text_block("Let me read the file first."),
            _make_tool_use_block("tu_001", "read", {"file_path": "main.py"}),
        ]},
        # [2] tool_result for read
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_001", "content": large_file_content},
        ]},
        # [3] assistant thinks + calls grep
        {"role": "assistant", "content": [
            _make_text_block("Now let me search for error patterns."),
            _make_tool_use_block("tu_002", "grep", {"pattern": "error|bug", "path": "."}),
        ]},
        # [4] tool_result for grep
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_002", "content": grep_results},
        ]},
        # [5] assistant final answer
        {"role": "assistant", "content": [
            _make_text_block("I found 3 potential bugs in main.py."),
        ]},
    ]
    return messages, large_file_content, grep_results


class TestCloseBookCompression:
    """When unload_tools is called, tool_result content should be compressed."""

    def test_unload_compresses_tool_results(self):
        """After unload, tool_result content should be shortened."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read", "grep"])

        messages, large_content, grep_content = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        # Unload via after_action hook (simulating tool execution)
        tc = {"name": "unload_tools", "input": {"names": ["read", "grep"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Unloaded tools: read, grep")

        # Check tool_result for read (message[2]) is compressed
        read_result = ctx.messages[2]["content"][0]
        assert len(read_result["content"]) < len(large_content)
        assert "compressed" in read_result["content"].lower() or len(read_result["content"]) < 200

        # Check tool_result for grep (message[4]) is compressed
        grep_result = ctx.messages[4]["content"][0]
        assert len(grep_result["content"]) < len(grep_content)

    def test_unload_preserves_tool_result_structure(self):
        """Compressed tool_results should still have type and tool_use_id."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages, _, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")

        read_result = ctx.messages[2]["content"][0]
        assert read_result["type"] == "tool_result"
        assert read_result["tool_use_id"] == "tu_001"

    def test_unload_only_compresses_targeted_tools(self):
        """Only unloaded tools' results should be compressed, not others."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read", "grep"])

        messages, large_content, grep_content = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        # Only unload 'read', not 'grep'
        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")

        # read result (msg[2]) should be compressed
        read_result = ctx.messages[2]["content"][0]
        assert len(read_result["content"]) < len(large_content)

        # grep result (msg[4]) should NOT be compressed
        grep_result = ctx.messages[4]["content"][0]
        assert grep_result["content"] == grep_content

    def test_unload_does_not_touch_non_tool_messages(self):
        """User text messages and assistant text should be untouched."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages, _, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)
        original_user_msg = ctx.messages[0]["content"]

        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")

        assert ctx.messages[0]["content"] == original_user_msg

    def test_unload_keeps_metadata_summary(self):
        """Compressed result should contain tool name and brief summary of what was returned."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages, _, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")

        read_result = ctx.messages[2]["content"][0]
        content = read_result["content"]
        # Should mention the tool or have some preview of original content
        assert "read" in content.lower() or "line" in content.lower()

    def test_small_results_not_compressed(self):
        """Tool results under threshold should not be compressed."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        small_content = "just 3 lines"
        messages = [
            {"role": "user", "content": "read a small file"},
            {"role": "assistant", "content": [
                _make_tool_use_block("tu_s1", "read", {"file_path": "tiny.txt"}),
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_s1", "content": small_content},
            ]},
        ]
        ctx = AgentContext(user_query="test", messages=messages)

        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")

        # Small content should remain unchanged
        result = ctx.messages[2]["content"][0]
        assert result["content"] == small_content

    def test_non_unload_tool_call_does_not_compress(self):
        """after_action for other tools (not unload_tools) should not compress."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages, large_content, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        # Simulate a load_tools call (not unload)
        tc = {"name": "load_tools", "input": {"names": ["bash"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Loaded tools: bash")

        # read result should NOT be compressed
        read_result = ctx.messages[2]["content"][0]
        assert read_result["content"] == large_content


# ── Re-expand: load_tools restores compressed results ────────


class TestReExpandOnLoad:
    """When load_tools is called, previously compressed tool_results should be restored."""

    def test_unload_saves_originals_to_metadata(self):
        """Compression should save original content to ctx.metadata for later re-expand."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages, large_content, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_meta"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")

        # Original content should be saved in metadata
        compressed_store = ctx.metadata.get("_compressed_results", {})
        assert "tu_001" in compressed_store
        assert compressed_store["tu_001"] == large_content

    def test_load_restores_compressed_results(self):
        """load_tools should restore previously compressed tool_results."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages, large_content, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        # Close book
        tc_unload = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_u1"}
        plugin.after_action(ctx, tc_unload, "Unloaded tools: read")

        # Verify compressed
        read_result = ctx.messages[2]["content"][0]
        assert len(read_result["content"]) < len(large_content)

        # Re-open book
        tc_load = {"name": "load_tools", "input": {"names": ["read"]}, "id": "tu_l1"}
        plugin.after_action(ctx, tc_load, "Loaded tools: read")

        # Verify restored
        read_result = ctx.messages[2]["content"][0]
        assert read_result["content"] == large_content

    def test_full_close_reopen_cycle(self):
        """Full cycle: load → use → close → re-open → content restored."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read", "grep"])

        messages, large_content, grep_content = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        # Close both
        tc = {"name": "unload_tools", "input": {"names": ["read", "grep"]}, "id": "tu_u"}
        plugin.after_action(ctx, tc, "Unloaded tools: read, grep")

        # Both compressed
        assert len(ctx.messages[2]["content"][0]["content"]) < len(large_content)
        assert len(ctx.messages[4]["content"][0]["content"]) < len(grep_content)

        # Re-open only read
        tc = {"name": "load_tools", "input": {"names": ["read"]}, "id": "tu_l"}
        plugin.after_action(ctx, tc, "Loaded tools: read")

        # read restored, grep still compressed
        assert ctx.messages[2]["content"][0]["content"] == large_content
        assert len(ctx.messages[4]["content"][0]["content"]) < len(grep_content)

        # Re-open grep
        tc = {"name": "load_tools", "input": {"names": ["grep"]}, "id": "tu_l2"}
        plugin.after_action(ctx, tc, "Loaded tools: grep")

        # Both restored
        assert ctx.messages[4]["content"][0]["content"] == grep_content

    def test_load_without_prior_compress_is_noop(self):
        """load_tools on tools that were never compressed should not crash."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)

        messages, large_content, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        # Load without prior unload — should be harmless
        tc = {"name": "load_tools", "input": {"names": ["read"]}, "id": "tu_l"}
        plugin.after_action(ctx, tc, "Loaded tools: read")

        # Content unchanged
        assert ctx.messages[2]["content"][0]["content"] == large_content

    def test_compressed_store_cleaned_after_restore(self):
        """After re-expand, the stored originals should be removed from metadata."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages, large_content, _ = _build_conversation_with_tool_calls()
        ctx = AgentContext(user_query="test", messages=messages)

        # Close
        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_u"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")
        assert "tu_001" in ctx.metadata.get("_compressed_results", {})

        # Re-open
        tc = {"name": "load_tools", "input": {"names": ["read"]}, "id": "tu_l"}
        plugin.after_action(ctx, tc, "Loaded tools: read")

        # Store should be cleaned
        assert "tu_001" not in ctx.metadata.get("_compressed_results", {})

    def test_small_results_not_in_compressed_store(self):
        """Small results that weren't compressed should not be in the store."""
        from tool_registry import ToolRegistryPlugin
        mgr = SkillPluginManager()
        mgr.register(DeferredPlugin())
        plugin = ToolRegistryPlugin(mgr)
        mgr.register(plugin)
        mgr.load_tools(["read"])

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": [
                _make_tool_use_block("tu_tiny", "read", {"file_path": "tiny.txt"}),
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_tiny", "content": "short"},
            ]},
        ]
        ctx = AgentContext(user_query="test", messages=messages)

        tc = {"name": "unload_tools", "input": {"names": ["read"]}, "id": "tu_u"}
        plugin.after_action(ctx, tc, "Unloaded tools: read")

        # Small content not compressed, so not in store
        assert "tu_tiny" not in ctx.metadata.get("_compressed_results", {})
