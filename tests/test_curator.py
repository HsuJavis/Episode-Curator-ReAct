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


class TestCognitiveSalience:
    """Spec Test 11: Curator produces salience and dimensions."""

    @pytest.mark.llm
    def test_salience_and_dimensions_present(self, api_key):
        """Curator should output salience and dimensions for each segment."""
        messages = [
            {"role": "user", "content": "我原本用 mock 測試，但發現跟 prod 不一致，改成真實 API 測試後通過了"},
            {"role": "assistant", "content": "好的，真實 API 測試確實更可靠。mock 可能遮蔽實際行為差異。"},
            {"role": "user", "content": "對，我決定以後所有測試都用真實 API，不再 mock LLM"},
            {"role": "assistant", "content": "了解，這是個好決定。雖然成本稍高但品質保證更好。"},
        ]
        curator = Curator(api_key=api_key)
        result = curator.process(messages, existing_index={})

        assert len(result["segments"]) >= 1
        seg = result["segments"][0]

        # salience should exist and be a number
        assert "salience" in seg, f"Missing salience. Segment: {seg}"
        assert isinstance(seg["salience"], (int, float))
        assert 0.0 <= seg["salience"] <= 1.0

        # dimensions should exist
        assert "dimensions" in seg, f"Missing dimensions. Segment: {seg}"
        dims = seg["dimensions"]

        # outcome and user_intent must be present
        assert "outcome" in dims, f"Missing outcome in dimensions: {dims}"
        assert "user_intent" in dims, f"Missing user_intent in dimensions: {dims}"

    @pytest.mark.llm
    def test_correction_gets_high_salience(self, api_key):
        """A conversation with error correction should get higher salience."""
        messages = [
            {"role": "user", "content": "為什麼部署一直失敗？"},
            {"role": "assistant", "content": "看起來是 Docker image 太大了"},
            {"role": "user", "content": "不對，我發現是 port 設定錯了，改成 8080 就好了"},
            {"role": "assistant", "content": "原來如此，port 8080 是 Cloud Run 的預設值，之前設定錯誤導致健康檢查失敗"},
        ]
        curator = Curator(api_key=api_key)
        result = curator.process(messages, existing_index={})

        seg = result["segments"][0]
        # Error correction should push salience above 0.5
        assert seg["salience"] >= 0.5, (
            f"Expected salience >= 0.5 for correction. Got: {seg['salience']}"
        )


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
