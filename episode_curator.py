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

    def generate_digest(self, summaries: list[str], tag: str) -> str:
        """Generate a digest summary from multiple episode summaries."""
        content = f"將以下 [{tag}] 主題的 episode 摘要歸納為一句話：\n" + "\n".join(f"- {s}" for s in summaries)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            system="你是摘要助手。將多條摘要歸納為一句話概述。只回傳歸納結果，不要其他內容。",
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return summaries[0] if summaries else ""


# ============================================================
# EpisodeCuratorPlugin — SkillPlugin implementation
# ============================================================

class EpisodeCuratorPlugin(SkillPlugin):
    """Plugin that manages context window via episode-based compression."""

    RECALL_TOOL = {
        "name": "recall_episode",
        "description": (
            "Retrieve a previously stored conversation episode. "
            "Use episode_id for direct lookup, or search_query to search by topic. "
            "Optionally filter by recent_hours."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": "Direct episode ID (e.g. '001')",
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find episodes by topic",
                },
                "recent_hours": {
                    "type": "number",
                    "description": "Only search episodes from the last N hours",
                },
            },
        },
    }

    # Keyword patterns for rule-based fact extraction
    FACT_PATTERNS = [
        r"(?:我(?:叫|是|的名字是))\s*(\S+)",
        r"(?:我(?:用|使用|偏好))\s*(.+?)(?:[,，。]|$)",
        r"(?:我們(?:用|使用))\s*(.+?)(?:[,，。]|$)",
        r"(?:技術棧|tech stack).*?[:：]\s*(.+?)(?:[,，。]|$)",
    ]

    def __init__(
        self,
        store: EpisodeStore,
        curator: Curator,
        threshold: int = 80000,
        preserve_recent: int = 6,
    ):
        self._store = store
        self._curator = curator
        self._threshold = threshold
        self._preserve_recent = preserve_recent

    @property
    def name(self) -> str:
        return "episode_curator"

    def get_tools(self) -> list[dict]:
        return [self.RECALL_TOOL]

    def execute_tool(self, name: str, tool_input: dict) -> Any:
        if name != "recall_episode":
            raise ValueError(f"Unknown tool: {name}")
        return self._recall_episode(tool_input)

    def _recall_episode(self, tool_input: dict) -> str:
        episode_id = tool_input.get("episode_id")
        search_query = tool_input.get("search_query")
        recent_hours = tool_input.get("recent_hours")

        if episode_id:
            ep = self._store.load_episode(episode_id)
            if ep is None:
                return f"Episode #{episode_id} not found."
            return self._format_episode_output(ep)

        if search_query:
            results = self._store.search_episodes(search_query, limit=3, recent_hours=recent_hours)
            if not results:
                return f"No episodes found for '{search_query}'."
            parts = []
            for r in results:
                ep = self._store.load_episode(r["id"])
                if ep:
                    parts.append(self._format_episode_output(ep))
            return "\n\n".join(parts)

        return "Please provide either episode_id or search_query."

    def _format_episode_output(self, ep: dict) -> str:
        header = f"── Episode #{ep['id']}: {ep['title']} | {ep.get('created_at', '')} ──"
        msgs = []
        for msg in ep.get("messages", []):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block["text"])
                    elif isinstance(block, dict):
                        parts.append(str(block))
                content = " ".join(parts)
            msgs.append(f"[{role}] {content}")
        return header + "\n" + "\n".join(msgs)

    # --- Hooks ---

    def on_agent_start(self, ctx: AgentContext) -> None:
        """Inject facts and global index into system_prompt_extra."""
        parts = []

        facts = self._store.get_facts()
        if facts:
            parts.append("已知事實：\n" + "\n".join(f"- {f}" for f in facts))

        global_index = self._store.build_global_index()
        if global_index:
            parts.append("對話歷史索引：\n" + global_index)

        if parts:
            ctx.metadata["system_prompt_extra"] = "\n\n".join(parts)

    def on_token_usage(self, ctx: AgentContext, input_tokens: int, output_tokens: int) -> None:
        """Core compression logic — triggered when input_tokens exceed threshold."""
        if input_tokens < self._threshold:
            return

        messages = ctx.messages
        if len(messages) <= self._preserve_recent + 1:
            return  # Not enough messages to compress

        # Keep first message (original question) and last N messages
        first_msg = messages[0]
        cut_point = self._find_safe_cut_point(messages, len(messages) - self._preserve_recent)
        to_archive = messages[1:cut_point]
        to_keep = messages[cut_point:]

        if not to_archive:
            return

        # Call Curator to segment
        result = self._curator.process(to_archive, self._store._index)

        # Save each segment as an episode
        for segment in result.get("segments", []):
            indices = segment.get("message_indices", [])
            if not indices:
                continue
            ep_messages = [to_archive[i] for i in indices if i < len(to_archive)]
            if not ep_messages:
                continue

            ep_id = self._store._next_episode_id()
            self._store.save_episode(
                episode_id=ep_id,
                messages=ep_messages,
                title=segment.get("title", "Untitled"),
                summary=segment.get("summary", ""),
                tags=segment.get("tags", ["misc"]),
                continues_episode=segment.get("continues_episode"),
            )

        # Save facts
        if result.get("facts"):
            self._store.add_facts(result["facts"])

        # Rebuild ctx.messages — maintain user/assistant alternation
        global_index = self._store.build_global_index()
        ctx.messages = [
            first_msg,
            {"role": "assistant", "content": [
                {"type": "text", "text": "好的，讓我來處理你的問題。"}
            ]},
            {"role": "user", "content": f"以下是之前的對話摘要索引：\n{global_index}"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "了解，我已掌握之前的對話脈絡。"}
            ]},
            *to_keep,
        ]

        # Update system_prompt_extra
        parts = []
        facts = self._store.get_facts()
        if facts:
            parts.append("已知事實：\n" + "\n".join(f"- {f}" for f in facts))
        if global_index:
            parts.append("對話歷史索引：\n" + global_index)
        if parts:
            ctx.metadata["system_prompt_extra"] = "\n\n".join(parts)

    def on_agent_end(self, ctx: AgentContext, final_answer: str) -> Optional[str]:
        """Rule-based fact extraction (no LLM call)."""
        for msg in ctx.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            extracted = []
            for pattern in self.FACT_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    match = match.strip()
                    if match and len(match) > 1:
                        extracted.append(match)
            if extracted:
                self._store.add_facts(extracted)
        return None

    @staticmethod
    def _find_safe_cut_point(messages: list, target: int) -> int:
        """Find a safe cut point that doesn't split tool_use/tool_result pairs."""
        cut = target

        # Scan backward to ensure we don't split a tool_use/tool_result pair
        while cut > 1:
            msg = messages[cut - 1]
            content = msg.get("content", [])

            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if has_tool_result:
                    cut -= 1
                    continue

                has_tool_use = any(
                    (isinstance(b, dict) and b.get("type") == "tool_use") or
                    (hasattr(b, "type") and b.type == "tool_use")
                    for b in content
                )
                if has_tool_use:
                    cut -= 1
                    continue

            break

        return max(cut, 1)


# ============================================================
# Factory function
# ============================================================

def create_agent(
    worker_model: str = "claude-sonnet-4-20250514",
    curator_model: str = "claude-haiku-4-5-20251001",
    threshold: int = 80000,
    max_iterations: int = 30,
    storage_dir: str | None = None,
    api_key: str | None = None,
) -> ReActAgent:
    """Create a ReActAgent with the EpisodeCuratorPlugin pre-configured."""
    store = EpisodeStore(storage_dir)
    curator = Curator(api_key=api_key, model=curator_model)
    plugin = EpisodeCuratorPlugin(store, curator, threshold=threshold)

    agent = ReActAgent(
        model=worker_model,
        max_iterations=max_iterations,
        api_key=api_key,
    )
    agent.register_skill(plugin)
    return agent
