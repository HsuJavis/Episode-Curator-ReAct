# episode_curator.py — Episode Curator Plugin

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import anthropic
from react_agent import AgentContext, ReActAgent, SkillPlugin, _resolve_auth


# ============================================================
# EpisodeStore — Disk storage layer
# ============================================================

class EpisodeStore:
    """Manages episodes, index, facts, and digests on disk."""

    def __init__(self, storage_dir: str | None = None):
        self._base = Path(storage_dir or os.path.expanduser("~/.episode_store"))
        self._episodes_dir = self._base / "episodes"
        self._digests_daily_dir = self._base / "digests" / "daily"
        self._digests_weekly_dir = self._base / "digests" / "weekly"
        self._index_path = self._base / "index.json"
        self._facts_path = self._base / "facts.json"
        self._digest_index_path = self._base / "digest_index.json"

        # Create directories
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        self._digests_daily_dir.mkdir(parents=True, exist_ok=True)
        self._digests_weekly_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self._index: dict = self._load_json(self._index_path, {})
        self._facts: list[str] = self._load_json(self._facts_path, [])
        self._digest_index: dict = self._load_json(self._digest_index_path, {})

    def _load_json(self, path: Path, default):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return default

    def _save_json(self, path: Path, data):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _next_episode_id(self) -> str:
        if not self._index:
            return "001"
        max_id = max(int(k) for k in self._index.keys())
        return f"{max_id + 1:03d}"

    def save_episode(
        self,
        episode_id: str,
        messages: list[dict],
        title: str,
        summary: str,
        tags: list[str],
        continues_episode: str | None = None,
    ) -> None:
        """Save an episode (immutable — refuses to overwrite)."""
        ep_path = self._episodes_dir / f"{episode_id}.json"
        if ep_path.exists():
            raise FileExistsError(f"Episode {episode_id} already exists (immutable)")

        ep_data = {
            "id": episode_id,
            "messages": messages,
            "title": title,
            "summary": summary,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
        }
        self._save_json(ep_path, ep_data)

        # Update index
        index_entry = {
            "title": title,
            "summary": summary,
            "tags": tags,
            "message_count": len(messages),
            "created_at": ep_data["created_at"],
        }
        if continues_episode:
            index_entry["continues_episode"] = continues_episode
        self._index[episode_id] = index_entry
        self._save_json(self._index_path, self._index)

    def load_episode(self, episode_id: str) -> dict | None:
        """Load full episode data (100% lossless)."""
        ep_path = self._episodes_dir / f"{episode_id}.json"
        if not ep_path.exists():
            return None
        return json.loads(ep_path.read_text(encoding="utf-8"))

    def search_episodes(
        self, query: str, limit: int = 5, recent_hours: float | None = None
    ) -> list[dict]:
        """Search episodes by title/summary/tags with scoring."""
        query_lower = query.lower()
        results = []

        now = datetime.now()
        for ep_id, entry in self._index.items():
            # Time filter
            if recent_hours is not None:
                created = datetime.fromisoformat(entry["created_at"])
                if (now - created).total_seconds() > recent_hours * 3600:
                    continue

            score = 0
            # Tag match: 3 points
            for tag in entry.get("tags", []):
                if query_lower in tag.lower():
                    score += 3
            # Title match: 2 points
            if query_lower in entry.get("title", "").lower():
                score += 2
            # Summary match: 1 point
            if query_lower in entry.get("summary", "").lower():
                score += 1

            if score > 0:
                results.append({"id": ep_id, "score": score, **entry})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def build_global_index(self) -> str:
        """Build global index string — concatenation of all episode summaries.

        This is the simple version (no temporal decay). Phase 6 adds decay.
        """
        if not self._index:
            return ""

        # Group by tag
        tag_groups: dict[str, list] = {}
        for ep_id, entry in sorted(self._index.items()):
            for tag in entry.get("tags", ["untagged"]):
                tag_groups.setdefault(tag, []).append((ep_id, entry))

        lines = []
        for tag in sorted(tag_groups.keys()):
            lines.append(f"## [{tag}]")
            for ep_id, entry in tag_groups[tag]:
                rel_time = self.format_time(entry["created_at"])
                lines.append(f"  #{ep_id} ({rel_time}) — {entry['summary']}")
        return "\n".join(lines)

    def add_facts(self, facts: list[str]) -> None:
        """Add permanent facts (deduplicated, max 50)."""
        existing = set(self._facts)
        for fact in facts:
            fact = fact.strip()
            if fact and fact not in existing:
                self._facts.append(fact)
                existing.add(fact)

        # Cap at 50
        self._facts = self._facts[:50]
        self._save_json(self._facts_path, self._facts)

    def get_facts(self) -> list[str]:
        return list(self._facts)

    @staticmethod
    def format_time(created_at: str) -> str:
        """Compute relative time string from ISO datetime."""
        created = datetime.fromisoformat(created_at)
        now = datetime.now()
        delta = now - created

        seconds = delta.total_seconds()
        if seconds < 60:
            return "剛才"
        elif seconds < 3600:
            return f"{int(seconds / 60)}分鐘前"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}小時前"
        elif seconds < 604800:
            return f"{int(seconds / 86400)}天前"
        elif seconds < 2592000:
            return f"{int(seconds / 604800)}週前"
        else:
            return f"{int(seconds / 2592000)}月前"
