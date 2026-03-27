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


# ============================================================
# Curator — LLM-based topic segmentation
# ============================================================

class Curator:
    """Second LLM that segments messages by topic."""

    SYSTEM_PROMPT = """你是一個對話整理助手。你的工作是將一段對話訊息按主題分段。

規則：
1. 按主題分段，同一主題的訊息即使不相鄰也歸為一段
2. 參考已有的主題索引，盡量複用已有的 tag
3. 如果新訊息是已有主題的延續，summary 開頭標註「接續 #xxx：」
4. continues_episode 是可選欄位，填入被延續的 episode ID
5. 如果是全新主題，不填 continues_episode
6. 提取重要的永久事實（使用者偏好、技術棧等）

你必須回傳 JSON（不要包含 markdown code fence）：
{
  "segments": [
    {
      "title": "主題名",
      "summary": "一句話摘要（如是延續，開頭加「接續 #xxx：」）",
      "tags": ["tag1"],
      "message_indices": [0, 1, 2],
      "continues_episode": "001"
    }
  ],
  "facts": ["事實1", "事實2"]
}"""

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001"):
        auth = _resolve_auth(api_key)
        self._client = anthropic.Anthropic(**auth)
        self._model = model

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages with indices for Curator input."""
        lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle structured content (tool_use, tool_result, etc.)
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            parts.append(f"[tool_use: {block.get('name', '?')}]")
                        elif block.get("type") == "tool_result":
                            parts.append(f"[tool_result: {str(block.get('content', ''))[:100]}]")
                    elif hasattr(block, "type"):
                        if block.type == "text":
                            parts.append(block.text)
                        elif block.type == "tool_use":
                            parts.append(f"[tool_use: {block.name}]")
                content = " | ".join(parts) if parts else str(content)
            # Truncate long content
            if len(str(content)) > 200:
                content = str(content)[:200] + "..."
            lines.append(f"[{i}] [{role}] {content}")
        return "\n".join(lines)

    def _format_existing_index(self, existing_index: dict) -> str:
        """Format existing index for Curator context."""
        if not existing_index:
            return "（尚無已有主題）"
        lines = ["已有主題索引："]
        for ep_id, entry in sorted(existing_index.items()):
            tags = ", ".join(entry.get("tags", []))
            lines.append(f"  #{ep_id} [{tags}] {entry.get('title', '')}")
        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from Curator response, handling code fences."""
        # Strip markdown code fences
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return json.loads(text)

    def process(self, messages: list[dict], existing_index: dict) -> dict:
        """Segment messages by topic.

        Returns: {"segments": [...], "facts": [...]}
        """
        formatted_msgs = self._format_messages(messages)
        formatted_index = self._format_existing_index(existing_index)

        user_content = f"{formatted_index}\n\n請處理以下新的對話記錄：\n{formatted_msgs}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        try:
            result = self._parse_json_response(response_text)
        except (json.JSONDecodeError, ValueError):
            # Fallback: single segment with all messages
            result = {
                "segments": [{
                    "title": "對話記錄",
                    "summary": "混合主題對話",
                    "tags": ["misc"],
                    "message_indices": list(range(len(messages))),
                }],
                "facts": [],
            }

        # Validate structure
        if "segments" not in result:
            result["segments"] = []
        if "facts" not in result:
            result["facts"] = []

        return result
