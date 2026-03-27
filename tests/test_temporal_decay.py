"""Tests for Temporal Resolution Decay (Spec Test 10)."""

from datetime import datetime, timedelta

import pytest
from episode_curator import EpisodeStore, Curator


class TestTemporalResolutionDecay:
    """Spec Test 10: build_global_index respects 3-tier temporal display."""

    def _save_episode_with_time(self, store, ep_id, title, summary, tags, hours_ago):
        """Helper: save episode and backdate its created_at."""
        store.save_episode(ep_id, [{"role": "user", "content": "q"}], title, summary, tags)
        # Backdate
        created_at = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
        store._index[ep_id]["created_at"] = created_at
        store._save_json(store._index_path, store._index)
        # Also update episode file
        ep_path = store._episodes_dir / f"{ep_id}.json"
        import json
        ep_data = json.loads(ep_path.read_text())
        ep_data["created_at"] = created_at
        ep_path.write_text(json.dumps(ep_data, ensure_ascii=False, indent=2))

    def test_recent_episodes_shown_individually(self, tmp_storage):
        """Episodes <48h should appear as individual lines."""
        store = EpisodeStore(tmp_storage)
        self._save_episode_with_time(store, "001", "Recent 1", "Recent topic one", ["database"], 1)
        self._save_episode_with_time(store, "002", "Recent 2", "Recent topic two", ["database"], 12)

        index = store.build_global_index()
        assert "#001" in index
        assert "#002" in index
        assert "Recent topic one" in index
        assert "Recent topic two" in index

    def test_midrange_episodes_show_as_daily(self, tmp_storage):
        """Episodes 48h-2w should appear as daily digests."""
        store = EpisodeStore(tmp_storage)
        # 3 days ago
        self._save_episode_with_time(store, "001", "Mid 1", "Midrange topic one", ["deploy"], 72)
        self._save_episode_with_time(store, "002", "Mid 2", "Midrange topic two", ["deploy"], 73)

        index = store.build_global_index()
        # Should NOT show individual #001, #002
        assert "近期匯總" in index
        # Should show date and count
        assert "段" in index

    def test_old_episodes_show_as_weekly(self, tmp_storage):
        """Episodes >2w should appear as weekly digests."""
        store = EpisodeStore(tmp_storage)
        # 3 weeks ago
        self._save_episode_with_time(store, "001", "Old 1", "Old topic one", ["misc"], 504)
        self._save_episode_with_time(store, "002", "Old 2", "Old topic two", ["misc"], 510)

        index = store.build_global_index()
        assert "歷史匯總" in index
        assert "段" in index

    def test_mixed_ages_three_tiers(self, tmp_storage):
        """Mix of recent, mid-range, and old episodes should show all three tiers."""
        store = EpisodeStore(tmp_storage)
        self._save_episode_with_time(store, "001", "Recent", "Recent DB work", ["database"], 2)
        self._save_episode_with_time(store, "002", "Mid", "Mid-range deploy", ["deploy"], 96)
        self._save_episode_with_time(store, "003", "Old", "Old design work", ["design"], 600)

        index = store.build_global_index()
        # Recent tier: individual line
        assert "#001" in index
        assert "Recent DB work" in index
        # Mid tier
        assert "近期匯總" in index
        # Old tier
        assert "歷史匯總" in index

    @pytest.mark.llm
    def test_digest_generation(self, api_key, tmp_storage):
        """Digests should be generated for past periods when curator is provided."""
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)

        # Create episodes from 3 days ago (same day, same tag)
        self._save_episode_with_time(store, "001", "DB Schema", "Designed user table", ["database"], 72)
        self._save_episode_with_time(store, "002", "DB Index", "Added indexes to user table", ["database"], 73)

        # Generate digests
        count = store._check_and_generate_digests(curator, max_new=3)
        assert count >= 1, "Expected at least 1 digest generated"

        # Digest should be in digest_index
        assert len(store._digest_index) >= 1

        # Build index with digests
        index = store.build_global_index()
        assert "近期匯總" in index

    @pytest.mark.llm
    def test_digest_immutable(self, api_key, tmp_storage):
        """Once generated, digests should not be regenerated."""
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)

        self._save_episode_with_time(store, "001", "Topic A", "Summary A", ["test"], 72)

        # First generation
        count1 = store._check_and_generate_digests(curator, max_new=3)
        digest_index_after_first = dict(store._digest_index)

        # Second generation — should not create new digests
        count2 = store._check_and_generate_digests(curator, max_new=3)
        assert count2 == 0, "Digests should not be regenerated"
        assert store._digest_index == digest_index_after_first

    @pytest.mark.llm
    def test_max_3_digests_per_call(self, api_key, tmp_storage):
        """At most 3 digests should be generated per call."""
        store = EpisodeStore(tmp_storage)
        curator = Curator(api_key=api_key)

        # Create 5 different day+tag combos (all in 48h-2w range)
        for i in range(5):
            hours = 72 + (i * 24)
            self._save_episode_with_time(
                store, f"{i+1:03d}", f"Topic {i}", f"Summary {i}", [f"tag{i}"], hours
            )

        count = store._check_and_generate_digests(curator, max_new=3)
        assert count <= 3, f"Expected max 3 digests, got {count}"
