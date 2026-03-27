"""Tests for AgentContext, SkillPlugin, and SkillPluginManager."""

import pytest
from react_agent import AgentContext, SkillPlugin, SkillPluginManager


# --- Test Plugins ---

class CalculatorPlugin(SkillPlugin):
    @property
    def name(self) -> str:
        return "calculator"

    def get_tools(self) -> list[dict]:
        return [{
            "name": "add",
            "description": "Add two numbers",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        }]

    def execute_tool(self, name, tool_input):
        if name == "add":
            return tool_input["a"] + tool_input["b"]
        raise ValueError(f"Unknown tool: {name}")


class UpperCasePlugin(SkillPlugin):
    """Plugin that uppercases thoughts and observations."""
    @property
    def name(self) -> str:
        return "uppercase"

    def on_thought(self, ctx, thought):
        return thought.upper()

    def on_observation(self, ctx, observation):
        return observation.upper()


class ConflictPlugin(SkillPlugin):
    """Plugin with a conflicting tool name."""
    @property
    def name(self) -> str:
        return "conflict"

    def get_tools(self) -> list[dict]:
        return [{
            "name": "add",
            "description": "Conflicting add tool",
            "input_schema": {"type": "object", "properties": {}},
        }]


# --- Tests ---

class TestAgentContext:
    def test_default_values(self):
        ctx = AgentContext(user_query="test")
        assert ctx.user_query == "test"
        assert ctx.messages == []
        assert ctx.metadata == {}
        assert ctx.iteration == 0
        assert ctx.total_input_tokens == 0
        assert ctx.total_output_tokens == 0
        assert ctx.tool_call_history == []

    def test_custom_values(self):
        ctx = AgentContext(
            user_query="hello",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"key": "value"},
            iteration=5,
        )
        assert ctx.iteration == 5
        assert len(ctx.messages) == 1
        assert ctx.metadata["key"] == "value"


class TestSkillPluginManager:
    def test_register_plugin(self):
        mgr = SkillPluginManager()
        plugin = CalculatorPlugin()
        mgr.register(plugin)
        assert len(mgr._plugins) == 1

    def test_tool_name_conflict(self):
        mgr = SkillPluginManager()
        mgr.register(CalculatorPlugin())
        with pytest.raises(ValueError, match="Tool name conflict"):
            mgr.register(ConflictPlugin())

    def test_get_all_tool_definitions(self):
        mgr = SkillPluginManager()
        mgr.register(CalculatorPlugin())
        tools = mgr.get_all_tool_definitions()
        assert len(tools) == 1
        assert tools[0]["name"] == "add"

    def test_route_tool_call(self):
        mgr = SkillPluginManager()
        mgr.register(CalculatorPlugin())
        result = mgr.route_tool_call("add", {"a": 3, "b": 4})
        assert result == 7

    def test_route_unknown_tool(self):
        mgr = SkillPluginManager()
        mgr.register(CalculatorPlugin())
        with pytest.raises(ValueError, match="Unknown tool"):
            mgr.route_tool_call("multiply", {"a": 3, "b": 4})

    def test_dispatch_on_thought_chaining(self):
        mgr = SkillPluginManager()
        mgr.register(UpperCasePlugin())
        ctx = AgentContext(user_query="test")
        result = mgr.dispatch_on_thought(ctx, "hello world")
        assert result == "HELLO WORLD"

    def test_dispatch_on_observation_chaining(self):
        mgr = SkillPluginManager()
        mgr.register(UpperCasePlugin())
        ctx = AgentContext(user_query="test")
        result = mgr.dispatch_on_observation(ctx, "result: 7")
        assert result == "RESULT: 7"

    def test_dispatch_before_action_passthrough(self):
        mgr = SkillPluginManager()
        mgr.register(CalculatorPlugin())
        ctx = AgentContext(user_query="test")
        tc = {"name": "add", "input": {"a": 1, "b": 2}, "id": "test"}
        result = mgr.dispatch_before_action(ctx, tc)
        assert result == tc  # No modification

    def test_dispatch_on_token_usage(self):
        mgr = SkillPluginManager()
        calls = []

        class TrackingPlugin(SkillPlugin):
            @property
            def name(self):
                return "tracker"
            def on_token_usage(self, ctx, input_tokens, output_tokens):
                calls.append((input_tokens, output_tokens))

        mgr.register(TrackingPlugin())
        ctx = AgentContext(user_query="test")
        mgr.dispatch_on_token_usage(ctx, 100, 50)
        assert calls == [(100, 50)]

    def test_multiple_plugins_order(self):
        mgr = SkillPluginManager()
        order = []

        class Plugin1(SkillPlugin):
            @property
            def name(self):
                return "p1"
            def on_agent_start(self, ctx):
                order.append("p1")

        class Plugin2(SkillPlugin):
            @property
            def name(self):
                return "p2"
            def on_agent_start(self, ctx):
                order.append("p2")

        mgr.register(Plugin1())
        mgr.register(Plugin2())
        ctx = AgentContext(user_query="test")
        mgr.dispatch_on_agent_start(ctx)
        assert order == ["p1", "p2"]
