"""Tests for EpisodeStore (Spec Tests 4, 5, 6 partial, 8)."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from episode_curator import EpisodeStore


class TestEpisodeImmutability:
    """Spec Test 4: Episodes are immutable — save then load returns identical content."""

    def test_save_and_load_identical(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        messages = [
            {"role": "user", "content": "What is PostgreSQL?"},
            {"role": "assistant", "content": "PostgreSQL is a relational database."},
        ]
        store.save_episode("001", messages, "PostgreSQL 介紹", "討論 PostgreSQL 基礎", ["database"])
        loaded = store.load_episode("001")
        assert loaded["messages"] == messages
        assert loaded["title"] == "PostgreSQL 介紹"
        assert loaded["summary"] == "討論 PostgreSQL 基礎"
        assert loaded["tags"] == ["database"]

    def test_overwrite_raises_error(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        messages = [{"role": "user", "content": "hello"}]
        store.save_episode("001", messages, "Test", "test summary", ["test"])
        with pytest.raises(FileExistsError, match="immutable"):
            store.save_episode("001", messages, "Test2", "another summary", ["test"])

    def test_reload_from_disk(self, tmp_storage):
        """Verify data persists across store instances."""
        store1 = EpisodeStore(tmp_storage)
        messages = [{"role": "user", "content": "data persists"}]
        store1.save_episode("001", messages, "Persist", "test persistence", ["test"])

        store2 = EpisodeStore(tmp_storage)
        loaded = store2.load_episode("001")
        assert loaded["messages"] == messages

    def test_load_nonexistent(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        assert store.load_episode("999") is None


class TestGlobalIndexNoDecay:
    """Spec Test 5: After 5 compressions, episode 001's summary is unchanged."""

    def test_summary_unchanged_after_multiple_saves(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        original_summary = "設計 products 表結構，確認 7 個產品"

        store.save_episode("001", [{"role": "user", "content": "q1"}],
                          "PostgreSQL 資料庫設計", original_summary, ["database"])

        # Simulate 5 more episodes (as if from 5 compression cycles)
        for i in range(2, 7):
            store.save_episode(
                f"{i:03d}",
                [{"role": "user", "content": f"q{i}"}],
                f"Topic {i}",
                f"Summary for topic {i}",
                ["misc"],
            )

        # Verify episode 001's summary in global index
        index_text = store.build_global_index()
        assert original_summary in index_text

        # Verify in raw index
        assert store._index["001"]["summary"] == original_summary


class TestRecallById:
    """Spec Test 6 (partial): recall episode by ID returns full content."""

    def test_recall_full_content(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        messages = [
            {"role": "user", "content": "How to add index?"},
            {"role": "assistant", "content": "Use CREATE INDEX..."},
            {"role": "user", "content": "What about composite index?"},
            {"role": "assistant", "content": "CREATE INDEX idx ON table(col1, col2)..."},
        ]
        store.save_episode("001", messages, "DB Index", "Adding indexes to PostgreSQL", ["database"])

        loaded = store.load_episode("001")
        assert len(loaded["messages"]) == 4
        assert loaded["messages"][0]["content"] == "How to add index?"
        assert loaded["messages"][3]["content"] == "CREATE INDEX idx ON table(col1, col2)..."

    def test_search_by_tag(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.save_episode("001", [{"role": "user", "content": "q"}],
                          "DB Design", "Database design discussion", ["database", "postgresql"])
        store.save_episode("002", [{"role": "user", "content": "q"}],
                          "Deploy GCP", "GCP deployment choices", ["deployment", "gcp"])

        results = store.search_episodes("database")
        assert len(results) >= 1
        assert results[0]["id"] == "001"
        assert results[0]["score"] >= 3  # tag match

    def test_search_by_title(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.save_episode("001", [{"role": "user", "content": "q"}],
                          "PostgreSQL Index Strategy", "Adding indexes", ["database"])

        results = store.search_episodes("index")
        assert len(results) >= 1
        assert results[0]["id"] == "001"

    def test_search_with_time_filter(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.save_episode("001", [{"role": "user", "content": "q"}],
                          "Recent", "Recent topic", ["test"])

        # Should find it (just created)
        results = store.search_episodes("recent", recent_hours=1)
        assert len(results) == 1

        # Manually backdate episode in index to 48h ago
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        store._index["001"]["created_at"] = old_time
        results = store.search_episodes("recent", recent_hours=1)
        assert len(results) == 0


class TestFactsPersistence:
    """Spec Test 8: Facts persist across sessions, deduplicate, and cap at 50."""

    def test_add_and_persist(self, tmp_storage):
        store1 = EpisodeStore(tmp_storage)
        store1.add_facts(["使用者叫小明", "用 PostgreSQL"])

        store2 = EpisodeStore(tmp_storage)
        assert "使用者叫小明" in store2.get_facts()
        assert "用 PostgreSQL" in store2.get_facts()

    def test_deduplication(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.add_facts(["fact A", "fact B"])
        store.add_facts(["fact A", "fact C"])  # fact A is duplicate
        facts = store.get_facts()
        assert facts.count("fact A") == 1
        assert "fact B" in facts
        assert "fact C" in facts

    def test_cap_at_50(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        many_facts = [f"fact {i}" for i in range(60)]
        store.add_facts(many_facts)
        assert len(store.get_facts()) == 50

    def test_empty_facts_ignored(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.add_facts(["valid", "", "  ", "also valid"])
        facts = store.get_facts()
        assert len(facts) == 2
        assert "valid" in facts
        assert "also valid" in facts


class TestAutoIncrementId:
    def test_auto_increment(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        assert store._next_episode_id() == "001"
        store.save_episode("001", [], "T1", "S1", ["t"])
        assert store._next_episode_id() == "002"
        store.save_episode("002", [], "T2", "S2", ["t"])
        assert store._next_episode_id() == "003"


class TestFormatTime:
    def test_just_now(self):
        now = datetime.now().isoformat()
        assert EpisodeStore.format_time(now) == "剛才"

    def test_minutes_ago(self):
        t = (datetime.now() - timedelta(minutes=30)).isoformat()
        result = EpisodeStore.format_time(t)
        assert "分鐘前" in result

    def test_hours_ago(self):
        t = (datetime.now() - timedelta(hours=5)).isoformat()
        result = EpisodeStore.format_time(t)
        assert "小時前" in result

    def test_days_ago(self):
        t = (datetime.now() - timedelta(days=3)).isoformat()
        result = EpisodeStore.format_time(t)
        assert "天前" in result


class TestBuildGlobalIndex:
    def test_grouped_by_tag(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.save_episode("001", [], "DB Design", "Design tables", ["database"])
        store.save_episode("002", [], "Deploy", "Deploy to GCP", ["deployment"])

        index = store.build_global_index()
        assert "## [database]" in index
        assert "## [deployment]" in index
        assert "#001" in index
        assert "#002" in index

    def test_empty_index(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        assert store.build_global_index() == ""

    def test_continues_episode(self, tmp_storage):
        store = EpisodeStore(tmp_storage)
        store.save_episode("001", [], "DB Design", "Design tables", ["database"])
        store.save_episode("002", [], "DB Index", "接續 #001：添加 index", ["database"],
                          continues_episode="001")

        assert store._index["002"].get("continues_episode") == "001"
