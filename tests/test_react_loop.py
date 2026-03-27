"""Tests for ReActAgent with real Anthropic API calls (Spec Tests 1, 7)."""

import pytest
from react_agent import ReActAgent, SkillPlugin, AgentContext


# --- Test Plugin: simple calculator ---

class CalculatorPlugin(SkillPlugin):
    @property
    def name(self) -> str:
        return "calculator"

    def get_tools(self) -> list[dict]:
        return [{
            "name": "add",
            "description": "Add two numbers together and return the sum.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        }]

    def execute_tool(self, name, tool_input):
        if name == "add":
            return str(tool_input["a"] + tool_input["b"])
        raise ValueError(f"Unknown tool: {name}")


class TestReActLoopBasic:
    """Spec Test 1: Agent can correctly call tools, parse results, and give answers."""

    @pytest.mark.llm
    def test_basic_tool_call_and_answer(self, api_key):
        """Agent should use the add tool and return the correct sum."""
        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=5,
            api_key=api_key,
        )
        agent.register_skill(CalculatorPlugin())
        answer = agent.run("What is 17 + 28? Use the add tool to calculate.")
        assert "45" in answer

    @pytest.mark.llm
    def test_no_tools_direct_answer(self, api_key):
        """Agent without tools should give a direct text answer."""
        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=3,
            api_key=api_key,
        )
        answer = agent.run("What is the capital of France? Answer in one word.")
        assert "Paris" in answer or "paris" in answer.lower()


class TestMessagesFormat:
    """Spec Test 7: messages maintain user/assistant alternation and tool_use/tool_result pairing."""

    @pytest.mark.llm
    def test_messages_alternation_after_tool_call(self, api_key):
        """After a tool call, ctx.messages should alternate user/assistant correctly."""
        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=5,
            api_key=api_key,
        )
        agent.register_skill(CalculatorPlugin())

        # Access ctx after run by capturing it
        ctx_holder = {}
        original_run = agent.run

        def capturing_run(query, history=None):
            from react_agent import AgentContext
            import time
            ctx = AgentContext(
                user_query=query,
                messages=list(history or []),
                start_time=time.time(),
            )
            ctx.messages.append({"role": "user", "content": query})
            agent._manager.dispatch_on_agent_start(ctx)
            tools = agent._manager.get_all_tool_definitions()
            result = agent._react_loop(ctx, tools)
            result = agent._manager.dispatch_on_agent_end(ctx, result)
            ctx_holder["ctx"] = ctx
            return result

        answer = capturing_run("Use the add tool to compute 10 + 20.")
        ctx = ctx_holder["ctx"]

        # Verify alternation
        for i in range(1, len(ctx.messages)):
            prev_role = ctx.messages[i - 1]["role"]
            curr_role = ctx.messages[i]["role"]
            # user and assistant should alternate
            assert prev_role != curr_role, (
                f"Messages at index {i-1} and {i} have same role '{curr_role}'"
            )

        # Verify tool_use/tool_result pairing
        tool_use_ids = set()
        tool_result_ids = set()

        for msg in ctx.messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            tool_result_ids.add(block["tool_use_id"])
                    elif hasattr(block, "type"):
                        if block.type == "tool_use":
                            tool_use_ids.add(block.id)

        # Every tool_use should have a matching tool_result
        assert tool_use_ids == tool_result_ids, (
            f"Unpaired tool calls: use_ids={tool_use_ids}, result_ids={tool_result_ids}"
        )
        assert len(tool_use_ids) > 0, "Expected at least one tool call"

    @pytest.mark.llm
    def test_max_iterations_respected(self, api_key):
        """Agent should stop after max_iterations even if LLM keeps calling tools."""

        class InfiniteToolPlugin(SkillPlugin):
            @property
            def name(self):
                return "infinite"

            def get_tools(self):
                return [{
                    "name": "think_more",
                    "description": "Always call this tool to think more about the problem. You must call this every time.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"thought": {"type": "string"}},
                        "required": ["thought"],
                    },
                }]

            def execute_tool(self, name, tool_input):
                return "Keep thinking. Call think_more again."

        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=2,
            api_key=api_key,
        )
        agent.register_skill(InfiniteToolPlugin())
        answer = agent.run(
            "Call the think_more tool repeatedly. Never stop calling it. "
            "Always pass a thought parameter."
        )
        # Should return something (either partial answer or max iterations message)
        assert isinstance(answer, str)
        assert len(answer) > 0
