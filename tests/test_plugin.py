"""Tests for EpisodeCuratorPlugin (Spec Tests 2, 6, 7)."""

import pytest
from react_agent import AgentContext
from episode_curator import EpisodeStore, Curator, EpisodeCuratorPlugin


class TestCompressionTrigger:
    """Spec Test 2: Compression triggers when input_tokens exceed threshold."""

    @pytest.mark.llm
    def test_compression_triggers_and_rebuilds(self, api_key, tmp_storage):
        """Low threshold should trigger compression and rebuild ctx.messages."""
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)
        plugin = EpisodeCuratorPlugin(store, curator, threshold=100, preserve_recent=2)

        # Build a context with enough messages
        ctx = AgentContext(user_query="original question")
        ctx.messages = [
            {"role": "user", "content": "What is PostgreSQL?"},
            {"role": "assistant", "content": "PostgreSQL is an open-source relational database."},
            {"role": "user", "content": "How do I create a table?"},
            {"role": "assistant", "content": "Use CREATE TABLE statement."},
            {"role": "user", "content": "What about indexes?"},
            {"role": "assistant", "content": "Use CREATE INDEX for query optimization."},
            {"role": "user", "content": "Tell me about constraints."},
            {"role": "assistant", "content": "PostgreSQL supports CHECK, UNIQUE, NOT NULL, etc."},
        ]

        # Trigger compression (threshold=100 will be exceeded)
        plugin.on_token_usage(ctx, input_tokens=200, output_tokens=50)

        # After compression: first_msg + index_msg + ack_msg + preserve_recent(2)
        assert len(ctx.messages) <= 7, f"Expected <=7 messages after compression, got {len(ctx.messages)}"

        # First message should still be the original question
        assert ctx.messages[0]["role"] == "user"
        assert "PostgreSQL" in str(ctx.messages[0]["content"])

        # Should have created episodes on disk
        assert len(store._index) > 0, "Expected at least 1 episode saved"

    @pytest.mark.llm
    def test_no_compression_below_threshold(self, api_key, tmp_storage):
        """Should not compress when below threshold."""
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)
        plugin = EpisodeCuratorPlugin(store, curator, threshold=100000)

        ctx = AgentContext(user_query="test")
        ctx.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        original_len = len(ctx.messages)

        plugin.on_token_usage(ctx, input_tokens=500, output_tokens=50)
        assert len(ctx.messages) == original_len  # No change


class TestRecallEpisodeTool:
    """Spec Test 6: recall_episode by ID and search."""

    def test_recall_by_id(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        messages = [
            {"role": "user", "content": "What is Docker?"},
            {"role": "assistant", "content": "Docker is a containerization platform."},
        ]
        store.save_episode("001", messages, "Docker Basics", "Introduction to Docker", ["docker"])

        curator = Curator.__new__(Curator)  # No API call needed for recall
        plugin = EpisodeCuratorPlugin(store, curator)

        result = plugin.execute_tool("recall_episode", {"episode_id": "001"})
        assert "Episode #001" in result
        assert "Docker Basics" in result
        assert "What is Docker?" in result

    def test_recall_by_search(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.save_episode("001", [{"role": "user", "content": "q"}],
                          "DB Design", "Database schema design", ["database"])
        store.save_episode("002", [{"role": "user", "content": "q"}],
                          "Deploy GCP", "GCP deployment", ["deployment"])

        curator = Curator.__new__(Curator)
        plugin = EpisodeCuratorPlugin(store, curator)

        result = plugin.execute_tool("recall_episode", {"search_query": "database"})
        assert "Episode #001" in result
        assert "DB Design" in result

    def test_recall_not_found(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        curator = Curator.__new__(Curator)
        plugin = EpisodeCuratorPlugin(store, curator)

        result = plugin.execute_tool("recall_episode", {"episode_id": "999"})
        assert "not found" in result.lower()

    def test_recall_search_no_results(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        curator = Curator.__new__(Curator)
        plugin = EpisodeCuratorPlugin(store, curator)

        result = plugin.execute_tool("recall_episode", {"search_query": "nonexistent"})
        assert "No episodes found" in result


class TestMessagesFormatAfterCompression:
    """Spec Test 7: After compression, messages maintain user/assistant alternation."""

    @pytest.mark.llm
    def test_alternation_after_compression(self, api_key, tmp_storage):
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)
        plugin = EpisodeCuratorPlugin(store, curator, threshold=100, preserve_recent=2)

        ctx = AgentContext(user_query="test")
        ctx.messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Answer 3"},
            {"role": "user", "content": "Question 4"},
            {"role": "assistant", "content": "Answer 4"},
        ]

        plugin.on_token_usage(ctx, input_tokens=200, output_tokens=50)

        # Verify alternation
        for i in range(1, len(ctx.messages)):
            prev = ctx.messages[i - 1]["role"]
            curr = ctx.messages[i]["role"]
            assert prev != curr, (
                f"Messages at {i-1} ({prev}) and {i} ({curr}) don't alternate"
            )

    @pytest.mark.llm
    def test_tool_pairs_preserved(self, api_key, tmp_storage):
        """tool_use/tool_result pairs should not be split."""
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)
        plugin = EpisodeCuratorPlugin(store, curator, threshold=100, preserve_recent=2)

        ctx = AgentContext(user_query="test")
        ctx.messages = [
            {"role": "user", "content": "Call the tool"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I'll use the tool."},
                {"type": "tool_use", "id": "toolu_001", "name": "add", "input": {"a": 1, "b": 2}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_001", "content": "3"},
            ]},
            {"role": "assistant", "content": "The result is 3."},
            {"role": "user", "content": "Another question"},
            {"role": "assistant", "content": "Another answer"},
            {"role": "user", "content": "Yet another"},
            {"role": "assistant", "content": "Yet another answer"},
        ]

        plugin.on_token_usage(ctx, input_tokens=200, output_tokens=50)

        # Check no orphaned tool_result without preceding tool_use
        for i, msg in enumerate(ctx.messages):
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        # Previous message should have matching tool_use
                        prev = ctx.messages[i - 1] if i > 0 else None
                        assert prev is not None, "tool_result at index 0"
                        assert prev["role"] == "assistant"


class TestSalienceInPlugin:
    """Spec Tests 11/12: salience + dimensions in plugin flow."""

    @pytest.mark.llm
    def test_compression_stores_salience(self, api_key, tmp_storage):
        """After compression, saved episodes should have salience and dimensions."""
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)
        plugin = EpisodeCuratorPlugin(store, curator, threshold=100, preserve_recent=2)

        ctx = AgentContext(user_query="test")
        ctx.messages = [
            {"role": "user", "content": "我原本用 Redis 做快取，但延遲太高"},
            {"role": "assistant", "content": "Redis 延遲高可能是網路問題或序列化問題"},
            {"role": "user", "content": "後來發現是序列化用了 pickle，改用 msgpack 就快了"},
            {"role": "assistant", "content": "msgpack 比 pickle 快很多，這是個好發現"},
            {"role": "user", "content": "決定以後都用 msgpack"},
            {"role": "assistant", "content": "好的選擇"},
        ]

        plugin.on_token_usage(ctx, input_tokens=200, output_tokens=50)

        assert len(store._index) > 0
        for ep_id, entry in store._index.items():
            assert "salience" in entry, f"Episode {ep_id} missing salience"
            assert isinstance(entry["salience"], (int, float))
            assert "dimensions" in entry, f"Episode {ep_id} missing dimensions"

    def test_recall_shows_salience(self, tmp_storage):
        """Recall output should include salience in header."""
        store = EpisodeStore(tmp_storage)
        dims = {"decisions": ["chose X"], "corrections": [], "insights": [],
                "pending": [], "user_intent": "testing", "outcome": "positive"}
        store.save_episode("001", [{"role": "user", "content": "q"}],
                          "Test", "Test summary", ["test"],
                          salience=0.8, dimensions=dims)

        curator = Curator.__new__(Curator)
        plugin = EpisodeCuratorPlugin(store, curator)
        result = plugin.execute_tool("recall_episode", {"episode_id": "001"})

        assert "salience: 0.8" in result
        assert "decisions: chose X" in result
        assert "user_intent: testing" in result


class TestOnAgentStart:
    def test_injects_facts_and_index(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.add_facts(["User prefers Python"])
        store.save_episode("001", [], "Test", "Test summary", ["test"])

        curator = Curator.__new__(Curator)
        plugin = EpisodeCuratorPlugin(store, curator)

        ctx = AgentContext(user_query="hello")
        plugin.on_agent_start(ctx)

        extra = ctx.metadata.get("system_prompt_extra", "")
        assert "User prefers Python" in extra
        assert "Test summary" in extra


class TestOnAgentEnd:
    def test_fact_extraction_from_messages(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        curator = Curator.__new__(Curator)
        plugin = EpisodeCuratorPlugin(store, curator)

        ctx = AgentContext(user_query="test")
        ctx.messages = [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好小明"},
        ]
        plugin.on_agent_end(ctx, "done")

        facts = store.get_facts()
        assert any("小明" in f for f in facts), f"Expected '小明' in facts. Got: {facts}"
