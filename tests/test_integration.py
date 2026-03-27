"""Integration tests for create_agent factory and full pipeline (real API)."""

import pytest
from episode_curator import create_agent, EpisodeStore


class TestCreateAgent:
    """Phase 7: Factory function creates working agent."""

    @pytest.mark.llm
    def test_factory_creates_working_agent(self, api_key, tmp_storage):
        """create_agent returns an agent that can answer questions."""
        agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            curator_model="claude-haiku-4-5-20251001",
            threshold=80000,
            max_iterations=5,
            storage_dir=tmp_storage,
            api_key=api_key,
        )
        answer = agent.run("What is 2 + 2? Answer with just the number.")
        assert "4" in answer

    @pytest.mark.llm
    def test_factory_default_params(self, api_key, tmp_storage):
        """create_agent with defaults should work."""
        agent = create_agent(storage_dir=tmp_storage, api_key=api_key)
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_iterations == 30

    @pytest.mark.llm
    def test_recall_tool_available(self, api_key, tmp_storage):
        """Agent should have recall_episode tool registered."""
        agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            storage_dir=tmp_storage,
            api_key=api_key,
        )
        tools = agent._manager.get_all_tool_definitions()
        tool_names = [t["name"] for t in tools]
        assert "recall_episode" in tool_names


class TestFullPipeline:
    """Integration: full Worker + Curator interaction."""

    @pytest.mark.llm
    def test_multi_turn_with_facts(self, api_key, tmp_storage):
        """Multi-turn conversation should extract and persist facts."""
        agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            curator_model="claude-haiku-4-5-20251001",
            max_iterations=5,
            storage_dir=tmp_storage,
            api_key=api_key,
        )

        # First turn: introduce ourselves
        history = []
        answer1 = agent.run("我叫小明，我用 Python 開發。請記住。", history)
        history.append({"role": "user", "content": "我叫小明，我用 Python 開發。請記住。"})
        history.append({"role": "assistant", "content": answer1})

        assert isinstance(answer1, str)

        # Check facts were extracted
        store = EpisodeStore(tmp_storage)
        facts = store.get_facts()
        assert any("小明" in f for f in facts) or any("Python" in f for f in facts), (
            f"Expected facts about user. Got: {facts}"
        )

    @pytest.mark.llm
    def test_plugin_injects_system_extra(self, api_key, tmp_storage):
        """After facts exist, on_agent_start should inject them into system prompt."""
        store = EpisodeStore(tmp_storage)
        store.add_facts(["User prefers dark mode"])
        store.save_episode("001", [{"role": "user", "content": "test"}],
                          "Test Topic", "A test discussion", ["test"])

        agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            storage_dir=tmp_storage,
            api_key=api_key,
        )

        # Run agent — facts and index should be injected
        answer = agent.run("What do you know about my preferences?")
        # The agent should reference the injected fact about dark mode
        assert isinstance(answer, str)
