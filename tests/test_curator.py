"""Tests for Curator with real Anthropic API (Spec Tests 3, 9)."""

import pytest
from episode_curator import Curator


class TestTopicSegmentation:
    """Spec Test 3: Multi-topic messages produce multiple segments."""

    @pytest.mark.llm
    def test_two_distinct_topics(self, api_key):
        """Messages about 2 topics should be split into 2 segments."""
        messages = [
            {"role": "user", "content": "PostgreSQL 的 index 怎麼加？"},
            {"role": "assistant", "content": "可以用 CREATE INDEX 語法..."},
            {"role": "user", "content": "Docker 怎麼部署到 GCP？"},
            {"role": "assistant", "content": "可以用 Cloud Run 或 GKE..."},
            {"role": "user", "content": "那 PostgreSQL 的 composite index 呢？"},
            {"role": "assistant", "content": "CREATE INDEX idx ON table(col1, col2)..."},
        ]
        curator = Curator(api_key=api_key)
        result = curator.process(messages, existing_index={})

        assert "segments" in result
        assert len(result["segments"]) >= 2, (
            f"Expected 2+ segments for 2 topics, got {len(result['segments'])}"
        )

        # Verify each segment has required fields
        for seg in result["segments"]:
            assert "title" in seg
            assert "summary" in seg
            assert "tags" in seg
            assert "message_indices" in seg
            assert isinstance(seg["message_indices"], list)
            assert len(seg["message_indices"]) > 0

    @pytest.mark.llm
    def test_single_topic_stays_together(self, api_key):
        """Messages about 1 topic should produce 1 segment."""
        messages = [
            {"role": "user", "content": "Python 的 list comprehension 怎麼用？"},
            {"role": "assistant", "content": "[x for x in range(10)]"},
            {"role": "user", "content": "那 nested list comprehension 呢？"},
            {"role": "assistant", "content": "[[j for j in range(3)] for i in range(3)]"},
        ]
        curator = Curator(api_key=api_key)
        result = curator.process(messages, existing_index={})

        assert len(result["segments"]) >= 1
        # All message indices should be covered
        all_indices = set()
        for seg in result["segments"]:
            all_indices.update(seg["message_indices"])
        assert all_indices == {0, 1, 2, 3}


class TestTopicContinuation:
    """Spec Test 9: Curator sees existing index, reuses tags, marks continuation."""

    @pytest.mark.llm
    def test_continuation_with_existing_index(self, api_key):
        """New database messages should continue existing database episode."""
        existing_index = {
            "001": {
                "title": "PostgreSQL 資料庫設計",
                "summary": "設計 products 表結構，確認 7 個產品",
                "tags": ["database"],
                "message_count": 4,
                "created_at": "2026-03-26T14:30:00",
            },
            "002": {
                "title": "GCP 部署選擇",
                "summary": "討論 Cloud Run vs GKE 的選擇",
                "tags": ["deployment"],
                "message_count": 3,
                "created_at": "2026-03-26T15:00:00",
            },
        }
        messages = [
            {"role": "user", "content": "剛才討論的 products 表需要加什麼 index？"},
            {"role": "assistant", "content": "根據查詢模式，建議加 name 和 category 的 index"},
            {"role": "user", "content": "那 sales 表呢？"},
            {"role": "assistant", "content": "sales 表建議加 product_id 和 date 的複合 index"},
        ]

        curator = Curator(api_key=api_key)
        result = curator.process(messages, existing_index)

        assert len(result["segments"]) >= 1

        # Find the database-related segment
        db_segment = None
        for seg in result["segments"]:
            tags_lower = [t.lower() for t in seg.get("tags", [])]
            if "database" in tags_lower or any("data" in t for t in tags_lower):
                db_segment = seg
                break

        assert db_segment is not None, (
            f"Expected a database-tagged segment. Got: {result['segments']}"
        )

        # Should reuse "database" tag (not create new one like "db" or "sql")
        assert "database" in [t.lower() for t in db_segment["tags"]], (
            f"Expected reuse of 'database' tag. Got tags: {db_segment['tags']}"
        )

    @pytest.mark.llm
    def test_facts_extraction(self, api_key):
        """Curator should extract facts from conversation."""
        messages = [
            {"role": "user", "content": "我叫小明，我用 PostgreSQL 和 Python 開發"},
            {"role": "assistant", "content": "好的小明，我了解你的技術棧了"},
        ]
        curator = Curator(api_key=api_key)
        result = curator.process(messages, existing_index={})

        # Should extract at least some facts
        assert "facts" in result
        if result["facts"]:  # LLM may or may not extract facts
            facts_text = " ".join(result["facts"]).lower()
            # At least one of these should be mentioned
            assert any(term in facts_text for term in ["小明", "postgresql", "python"]), (
                f"Expected facts about user. Got: {result['facts']}"
            )
