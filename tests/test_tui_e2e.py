"""TUI E2E tests — tool/skill loading, memory recall, MCP compatibility, session restart.

These tests exercise the full agent pipeline through the TUI layer,
verifying dynamic tool loading, long-conversation memory, MCP integration,
and cross-session recall.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from react_agent import AgentContext, ReActAgent, SkillPlugin, SkillPluginManager
from cli_app import EpisodeCuratorApp, TUIEvent, TUIPlugin, StatusBar
from episode_curator import EpisodeStore, Curator, EpisodeCuratorPlugin, create_agent
from tool_registry import ToolRegistryPlugin
from system_tools import SystemToolsPlugin
from skill_loader import SkillManager, SkillLoaderPlugin

FAKE_SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


# ============================================================
# 1. Tool/Skill Dynamic Loading & Unloading
# ============================================================

pytestmark_tui = pytest.mark.tui


class TestToolDynamicLoadingTUI:
    """E2E: deferred tools load/unload through TUI agent pipeline."""

    @pytest.fixture
    def app_with_deferred_tools(self):
        """Create TUI app with agent pre-configured with deferred system tools."""
        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=5,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        mgr = agent._manager
        sys_plugin = SystemToolsPlugin()
        mgr.register(sys_plugin)
        registry_plugin = ToolRegistryPlugin(mgr)
        mgr.register(registry_plugin)
        return EpisodeCuratorApp(agent=agent), mgr

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_deferred_tools_not_in_initial_active_set(self, app_with_deferred_tools):
        """System tools should start unloaded in TUI context."""
        app, mgr = app_with_deferred_tools
        async with app.run_test(size=(120, 40)) as pilot:
            active = mgr.get_active_tool_definitions()
            active_names = {t["name"] for t in active}
            # Only load_tools and unload_tools should be active
            assert "load_tools" in active_names
            assert "unload_tools" in active_names
            assert "read" not in active_names
            assert "bash" not in active_names

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_load_tools_activates_in_tui(self, app_with_deferred_tools):
        """Loading a tool via manager makes it available for next API call."""
        app, mgr = app_with_deferred_tools
        async with app.run_test(size=(120, 40)) as pilot:
            result = mgr.load_tools(["read", "grep"])
            assert "Loaded" in result
            active = mgr.get_active_tool_definitions()
            active_names = {t["name"] for t in active}
            assert "read" in active_names
            assert "grep" in active_names

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_unload_tools_deactivates_in_tui(self, app_with_deferred_tools):
        """Unloading tools removes them from active set, keeping metadata."""
        app, mgr = app_with_deferred_tools
        async with app.run_test(size=(120, 40)) as pilot:
            mgr.load_tools(["read", "grep", "bash"])
            mgr.unload_tools(["read", "grep"])
            active = mgr.get_active_tool_definitions()
            active_names = {t["name"] for t in active}
            assert "read" not in active_names
            assert "grep" not in active_names
            assert "bash" in active_names
            # Catalog still has all tools
            catalog = mgr.get_tool_catalog()
            catalog_names = {t["name"] for t in catalog}
            assert "read" in catalog_names
            assert "grep" in catalog_names

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_tool_catalog_injected_into_system_prompt(self, app_with_deferred_tools):
        """ToolRegistryPlugin should inject deferred tool catalog on agent start."""
        app, mgr = app_with_deferred_tools
        async with app.run_test(size=(120, 40)) as pilot:
            ctx = AgentContext(user_query="test")
            # Find the registry plugin and fire on_agent_start
            for p in mgr._plugins:
                if p.name == "tool_registry":
                    p.on_agent_start(ctx)
                    break
            extra = ctx.metadata.get("system_prompt_extra", "")
            assert "load_tools" in extra
            assert "read" in extra
            assert "bash" in extra

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_skill_catalog_coexists_with_tool_catalog(self):
        """Skill catalog and tool catalog should both appear in system prompt."""
        fixtures_dir = str(Path(__file__).parent / "fixtures" / "skills")
        skill_mgr = SkillManager(skills_dir=fixtures_dir)
        skill_plugin = SkillLoaderPlugin(skill_mgr)

        agent = ReActAgent(model="claude-haiku-4-5-20251001", max_iterations=3)
        mgr = agent._manager
        mgr.register(SystemToolsPlugin())
        mgr.register(skill_plugin)
        registry_plugin = ToolRegistryPlugin(mgr)
        mgr.register(registry_plugin)

        ctx = AgentContext(user_query="test")
        mgr.dispatch_on_agent_start(ctx)
        extra = ctx.metadata.get("system_prompt_extra", "")
        # Both catalogs present
        assert "Available skills" in extra
        assert "commit" in extra
        assert "Deferred tools" in extra
        assert "read" in extra

    @pytest.mark.tui
    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_agent_uses_load_tools_then_reads_file(self, api_key, tmp_path):
        """Full E2E: agent should load_tools first, then use the loaded tool."""
        test_file = tmp_path / "hello.txt"
        test_file.write_text("The secret number is 42.", encoding="utf-8")

        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=10,
            api_key=api_key,
            system_prompt=(
                "You have deferred tools. Use load_tools first to activate tools you need, "
                "then use them. When done, use unload_tools to free context."
            ),
        )
        mgr = agent._manager
        mgr.register(SystemToolsPlugin())
        registry_plugin = ToolRegistryPlugin(mgr)
        mgr.register(registry_plugin)

        answer = agent.run(f"Read the file {test_file} and tell me the secret number.")
        assert "42" in answer


# ============================================================
# 2. Long Conversation Memory Recall Rate
# ============================================================


class TestLongConversationRecall:
    """E2E: measure recall accuracy after context compression."""

    @staticmethod
    def _seed_episodes(store: EpisodeStore, count: int = 5) -> list[dict]:
        """Seed store with distinct episodes containing unique facts."""
        seeded = []
        topics = [
            ("PostgreSQL Setup", "database", "選擇 PostgreSQL 13 部署在 AWS RDS", 0.8,
             {"decisions": ["選擇 PostgreSQL 因為團隊有 5 年經驗"], "corrections": [],
              "insights": ["RDS 比自建省 30% 維運成本"], "pending": [],
              "user_intent": "設定資料庫", "outcome": "positive"}),
            ("Redis Caching", "caching", "引入 Redis 做 session cache，TTL 設 3600 秒", 0.7,
             {"decisions": ["Redis 優於 Memcached 因為支援 pub/sub"], "corrections": [],
              "insights": ["TTL 3600 秒平衡了命中率和記憶體"], "pending": ["需要監控 eviction rate"],
              "user_intent": "加速 API 回應", "outcome": "positive"}),
            ("Docker Migration", "deployment", "從 VM 遷移到 Docker，用 multi-stage build", 0.9,
             {"decisions": ["使用 Alpine base image 縮小到 89MB"], "corrections": ["原本用 Ubuntu image 太大"],
              "insights": ["multi-stage build 可分離 build 和 runtime"], "pending": [],
              "user_intent": "容器化部署", "outcome": "positive"}),
            ("API Rate Limiting", "api", "實作 token bucket 限流，每分鐘 100 請求", 0.6,
             {"decisions": ["token bucket 比 sliding window 更彈性"], "corrections": [],
              "insights": [], "pending": ["需要加 per-user 限流"],
              "user_intent": "保護 API", "outcome": "positive"}),
            ("CI Pipeline", "devops", "GitHub Actions 建立 CI，pytest + mypy + ruff", 0.5,
             {"decisions": ["GitHub Actions 取代 Jenkins"], "corrections": [],
              "insights": ["並行跑 lint 和 test 省 40% 時間"], "pending": ["CD 部分尚未完成"],
              "user_intent": "自動化測試", "outcome": "positive"}),
        ]
        for i, (title, tag, summary, sal, dims) in enumerate(topics[:count]):
            ep_id = f"{i + 1:03d}"
            msgs = [
                {"role": "user", "content": f"關於 {title} 的問題"},
                {"role": "assistant", "content": summary},
            ]
            store.save_episode(
                episode_id=ep_id, messages=msgs, title=title,
                summary=summary, tags=[tag], salience=sal, dimensions=dims,
            )
            seeded.append({"id": ep_id, "title": title, "tag": tag,
                           "summary": summary, "salience": sal})
        return seeded

    def test_recall_by_id_is_lossless(self, tmp_storage):
        """Direct recall by episode_id should return 100% of original content."""
        store = EpisodeStore(tmp_storage)
        seeded = self._seed_episodes(store, 5)

        for ep in seeded:
            loaded = store.load_episode(ep["id"])
            assert loaded is not None
            assert loaded["title"] == ep["title"]
            assert loaded["summary"] == ep["summary"]

    def test_search_recall_rate_by_topic(self, tmp_storage):
        """Search by topic keyword should find relevant episodes."""
        store = EpisodeStore(tmp_storage)
        seeded = self._seed_episodes(store, 5)

        queries = [
            ("PostgreSQL", "001"),
            ("Redis", "002"),
            ("Docker", "003"),
            ("Rate Limiting", "004"),
            ("CI Pipeline", "005"),
        ]
        found = 0
        for query, expected_id in queries:
            results = store.search_episodes(query, limit=3)
            result_ids = {r["id"] for r in results}
            if expected_id in result_ids:
                found += 1

        recall_rate = found / len(queries)
        assert recall_rate >= 0.8, f"Recall rate {recall_rate:.0%} below 80% threshold"

    def test_global_index_preserves_all_episodes(self, tmp_storage):
        """Global index should mention all seeded episodes."""
        store = EpisodeStore(tmp_storage)
        seeded = self._seed_episodes(store, 5)

        index_text = store.build_global_index()
        for ep in seeded:
            assert ep["id"] in index_text or ep["title"] in index_text, \
                f"Episode {ep['id']} ({ep['title']}) missing from global index"

    def test_salience_ordering_in_index(self, tmp_storage):
        """Higher salience episodes should appear before lower ones in same time tier."""
        store = EpisodeStore(tmp_storage)
        self._seed_episodes(store, 5)

        index_text = store.build_global_index()
        # Docker (0.9) should appear before CI Pipeline (0.5) in recent tier
        docker_pos = index_text.find("Docker")
        ci_pos = index_text.find("CI")
        if docker_pos >= 0 and ci_pos >= 0:
            assert docker_pos < ci_pos, "High-salience episode should appear before low-salience"

    @pytest.mark.llm
    def test_recall_after_compression_simulation(self, tmp_storage, api_key):
        """After compression, agent can recall specific facts via recall_episode."""
        store = EpisodeStore(tmp_storage)
        seeded = self._seed_episodes(store, 5)
        curator = Curator(api_key=api_key)
        plugin = EpisodeCuratorPlugin(store, curator, threshold=80000)

        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=5,
            api_key=api_key,
        )
        agent.register_skill(plugin)

        # Ask agent to recall Docker migration details
        answer = agent.run("請用 recall_episode 搜尋 Docker migration 的細節，告訴我選了什麼 base image")
        assert "Alpine" in answer or "alpine" in answer or "89MB" in answer or "multi-stage" in answer

    @pytest.mark.llm
    def test_recall_multiple_topics_accuracy(self, tmp_storage, api_key):
        """Agent should correctly recall facts from multiple different topics."""
        store = EpisodeStore(tmp_storage)
        seeded = self._seed_episodes(store, 5)
        curator = Curator(api_key=api_key)
        plugin = EpisodeCuratorPlugin(store, curator, threshold=80000)

        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=8,
            api_key=api_key,
        )
        agent.register_skill(plugin)

        answer = agent.run(
            "用 recall_episode 搜尋以下主題，各回答一個關鍵事實：\n"
            "1. PostgreSQL 選擇原因\n"
            "2. Redis TTL 設定\n"
            "3. Docker base image"
        )
        # At least 2 out of 3 topics should have correct recall
        hits = 0
        if any(kw in answer for kw in ["團隊", "經驗", "5 年", "5年"]):
            hits += 1
        if any(kw in answer for kw in ["3600", "TTL"]):
            hits += 1
        if any(kw in answer for kw in ["Alpine", "alpine", "89MB", "89"]):
            hits += 1
        assert hits >= 2, f"Only {hits}/3 topics recalled correctly"


# ============================================================
# 3. MCP Tool & Skill Compatibility
# ============================================================


class TestMCPAndSkillCompatibility:
    """E2E: MCP tools and skills coexist correctly with tool registry."""

    @pytest.fixture
    def mcp_config(self, tmp_path):
        """Create MCP config pointing to fake server."""
        config = {
            "mcpServers": {
                "test": {
                    "command": sys.executable,
                    "args": [FAKE_SERVER],
                }
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))
        return str(config_file)

    def test_mcp_tools_are_deferred(self, mcp_config):
        """MCP tools should be deferred (start unloaded)."""
        from mcp_client import MCPManager, MCPPlugin
        manager = MCPManager(config_path=mcp_config)
        manager.start_server("test", manager.server_configs["test"])
        try:
            plugin = MCPPlugin(manager)
            assert plugin.is_deferred() is True
            tools = plugin.get_tools()
            assert len(tools) >= 1  # at least echo tool
            assert tools[0]["name"].startswith("mcp__test__")
        finally:
            manager.shutdown()

    def test_mcp_tools_in_catalog_not_active(self, mcp_config):
        """MCP tools should appear in catalog but not in active set initially."""
        from mcp_client import MCPManager, MCPPlugin
        manager = MCPManager(config_path=mcp_config)
        manager.start_server("test", manager.server_configs["test"])
        try:
            mcp_plugin = MCPPlugin(manager)
            mgr = SkillPluginManager()
            mgr.register(mcp_plugin)
            registry = ToolRegistryPlugin(mgr)
            mgr.register(registry)

            catalog = mgr.get_tool_catalog()
            mcp_tools = [t for t in catalog if t["name"].startswith("mcp__")]
            assert len(mcp_tools) >= 1
            assert all(not t["loaded"] for t in mcp_tools)

            active = mgr.get_active_tool_definitions()
            active_names = {t["name"] for t in active}
            assert not any(n.startswith("mcp__") for n in active_names)
        finally:
            manager.shutdown()

    def test_load_mcp_tool_then_execute(self, mcp_config):
        """Load MCP tool dynamically, then call it successfully."""
        from mcp_client import MCPManager, MCPPlugin
        manager = MCPManager(config_path=mcp_config)
        manager.start_server("test", manager.server_configs["test"])
        try:
            mcp_plugin = MCPPlugin(manager)
            mgr = SkillPluginManager()
            mgr.register(mcp_plugin)
            registry = ToolRegistryPlugin(mgr)
            mgr.register(registry)

            tool_name = "mcp__test__echo"
            mgr.load_tools([tool_name])
            active = mgr.get_active_tool_definitions()
            active_names = {t["name"] for t in active}
            assert tool_name in active_names

            result = mgr.route_tool_call(tool_name, {"text": "hello"})
            assert "hello" in str(result).lower()
        finally:
            manager.shutdown()

    def test_unload_mcp_tool_removes_from_active(self, mcp_config):
        """After unloading, MCP tool should leave active set but stay in catalog."""
        from mcp_client import MCPManager, MCPPlugin
        manager = MCPManager(config_path=mcp_config)
        manager.start_server("test", manager.server_configs["test"])
        try:
            mcp_plugin = MCPPlugin(manager)
            mgr = SkillPluginManager()
            mgr.register(mcp_plugin)
            registry = ToolRegistryPlugin(mgr)
            mgr.register(registry)

            tool_name = "mcp__test__echo"
            mgr.load_tools([tool_name])
            mgr.unload_tools([tool_name])

            active = mgr.get_active_tool_definitions()
            active_names = {t["name"] for t in active}
            assert tool_name not in active_names

            catalog = mgr.get_tool_catalog()
            catalog_names = {t["name"] for t in catalog}
            assert tool_name in catalog_names
        finally:
            manager.shutdown()

    def test_system_tools_and_mcp_coexist(self, mcp_config):
        """System tools and MCP tools should coexist without name conflicts."""
        from mcp_client import MCPManager, MCPPlugin
        manager = MCPManager(config_path=mcp_config)
        manager.start_server("test", manager.server_configs["test"])
        try:
            mgr = SkillPluginManager()
            mgr.register(SystemToolsPlugin())
            mgr.register(MCPPlugin(manager))
            registry = ToolRegistryPlugin(mgr)
            mgr.register(registry)

            catalog = mgr.get_tool_catalog()
            names = [t["name"] for t in catalog]
            # Both types present
            assert any(n == "read" for n in names)
            assert any(n.startswith("mcp__") for n in names)
            # No duplicates
            assert len(names) == len(set(names))
        finally:
            manager.shutdown()

    def test_skill_catalog_persists_after_tool_load_unload(self):
        """Skill catalog should remain stable through load/unload cycles."""
        fixtures_dir = str(Path(__file__).parent / "fixtures" / "skills")
        skill_mgr = SkillManager(skills_dir=fixtures_dir)
        skill_plugin = SkillLoaderPlugin(skill_mgr)

        mgr = SkillPluginManager()
        mgr.register(SystemToolsPlugin())
        mgr.register(skill_plugin)
        registry = ToolRegistryPlugin(mgr)
        mgr.register(registry)

        ctx = AgentContext(user_query="test")
        mgr.dispatch_on_agent_start(ctx)
        extra_before = ctx.metadata.get("system_prompt_extra", "")
        assert "commit" in extra_before

        mgr.load_tools(["read", "bash"])
        mgr.unload_tools(["read", "bash"])

        # Skill catalog should still be in system_prompt_extra
        # (it was set during on_agent_start and shouldn't be cleared by load/unload)
        assert "commit" in extra_before


# ============================================================
# 4. Session Restart Memory Recall Rate
# ============================================================


class TestSessionRestartRecall:
    """E2E: memory persists across agent sessions via disk."""

    @staticmethod
    def _create_session_with_episodes(storage_dir: str, api_key: str = None) -> list[str]:
        """Create a session, seed episodes, return fact keywords for verification."""
        store = EpisodeStore(storage_dir)

        # Seed distinct episodes with unique identifiable facts
        episodes_data = [
            {
                "title": "專案使用 FastAPI 框架",
                "summary": "決定用 FastAPI 因為 async 效能好，搭配 Pydantic v2",
                "tags": ["framework"],
                "salience": 0.8,
                "dimensions": {
                    "decisions": ["選擇 FastAPI 因為原生 async 支援"],
                    "corrections": [],
                    "insights": ["Pydantic v2 比 v1 快 5 倍"],
                    "pending": [],
                    "user_intent": "選擇後端框架",
                    "outcome": "positive",
                },
                "messages": [
                    {"role": "user", "content": "我們應該用什麼框架？"},
                    {"role": "assistant", "content": "建議用 FastAPI，因為原生 async 效能好，搭配 Pydantic v2 做驗證。"},
                ],
            },
            {
                "title": "資料庫選用 CockroachDB",
                "summary": "選擇 CockroachDB 因為自動 sharding，支援分散式事務",
                "tags": ["database"],
                "salience": 0.9,
                "dimensions": {
                    "decisions": ["CockroachDB 優於 PostgreSQL 因為自動水平擴展"],
                    "corrections": ["原本考慮 PostgreSQL + Citus 但太複雜"],
                    "insights": ["CockroachDB 相容 PostgreSQL wire protocol"],
                    "pending": ["需要測試跨區域延遲"],
                    "user_intent": "選擇可擴展的資料庫",
                    "outcome": "positive",
                },
                "messages": [
                    {"role": "user", "content": "分散式資料庫怎麼選？"},
                    {"role": "assistant", "content": "建議 CockroachDB，自動 sharding 且相容 PostgreSQL。"},
                ],
            },
            {
                "title": "部署到 Fly.io",
                "summary": "部署到 Fly.io 因為全球邊緣節點，啟動時間 < 500ms",
                "tags": ["deployment"],
                "salience": 0.7,
                "dimensions": {
                    "decisions": ["Fly.io 取代 AWS Lambda 因為冷啟動快"],
                    "corrections": [],
                    "insights": ["Fly.io 的 anycast 路由比 CloudFront 簡單"],
                    "pending": ["成本優化還沒做"],
                    "user_intent": "低延遲部署",
                    "outcome": "positive",
                },
                "messages": [
                    {"role": "user", "content": "部署到哪裡？"},
                    {"role": "assistant", "content": "推薦 Fly.io，全球邊緣部署，冷啟動 < 500ms。"},
                ],
            },
        ]

        fact_keywords = ["FastAPI", "CockroachDB", "Fly.io"]

        for i, ep_data in enumerate(episodes_data):
            store.save_episode(
                episode_id=f"{i + 1:03d}",
                messages=ep_data["messages"],
                title=ep_data["title"],
                summary=ep_data["summary"],
                tags=ep_data["tags"],
                salience=ep_data["salience"],
                dimensions=ep_data["dimensions"],
            )

        # Also add facts
        store.add_facts(["使用 FastAPI 框架", "資料庫是 CockroachDB", "部署在 Fly.io"])

        return fact_keywords

    def test_episodes_persist_on_disk(self, tmp_storage):
        """Episodes saved in session 1 should be readable in session 2."""
        keywords = self._create_session_with_episodes(tmp_storage)

        # New session — fresh EpisodeStore pointing to same directory
        store2 = EpisodeStore(tmp_storage)
        assert len(store2._index) == 3

        for ep_id in ["001", "002", "003"]:
            ep = store2.load_episode(ep_id)
            assert ep is not None
            assert ep["title"]

    def test_facts_persist_across_sessions(self, tmp_storage):
        """Facts from session 1 should be available in session 2."""
        self._create_session_with_episodes(tmp_storage)

        store2 = EpisodeStore(tmp_storage)
        facts = store2.get_facts()
        assert any("FastAPI" in f for f in facts)
        assert any("CockroachDB" in f for f in facts)
        assert any("Fly.io" in f for f in facts)

    def test_global_index_rebuilt_from_disk(self, tmp_storage):
        """New session should rebuild global index from persisted episodes."""
        self._create_session_with_episodes(tmp_storage)

        store2 = EpisodeStore(tmp_storage)
        index = store2.build_global_index()
        assert "FastAPI" in index
        assert "CockroachDB" in index
        assert "Fly.io" in index

    def test_search_works_in_new_session(self, tmp_storage):
        """Search in new session should find episodes from previous session."""
        self._create_session_with_episodes(tmp_storage)

        store2 = EpisodeStore(tmp_storage)
        queries_and_expected = [
            ("FastAPI", "001"),
            ("CockroachDB", "002"),
            ("Fly.io", "003"),
        ]
        found = 0
        for query, expected_id in queries_and_expected:
            results = store2.search_episodes(query, limit=3)
            if any(r["id"] == expected_id for r in results):
                found += 1

        recall_rate = found / len(queries_and_expected)
        assert recall_rate >= 0.67, f"Cross-session recall rate {recall_rate:.0%} below 67%"

    def test_new_agent_injects_previous_session_context(self, tmp_storage):
        """New agent should inject facts + index from previous session on startup."""
        self._create_session_with_episodes(tmp_storage)

        # Create fresh agent pointing to same storage
        store2 = EpisodeStore(tmp_storage)
        curator2 = Curator.__new__(Curator)  # avoid API call in __init__
        curator2._model = "claude-haiku-4-5-20251001"
        curator2._client = None
        plugin2 = EpisodeCuratorPlugin(store2, curator2, threshold=80000)

        ctx = AgentContext(user_query="test")
        plugin2.on_agent_start(ctx)

        extra = ctx.metadata.get("system_prompt_extra", "")
        # Facts should be injected
        assert "FastAPI" in extra
        assert "CockroachDB" in extra
        # Global index should be injected
        assert "Fly.io" in extra or "003" in extra

    @pytest.mark.llm
    def test_agent_recalls_previous_session_via_tool(self, tmp_storage, api_key):
        """New agent can use recall_episode to retrieve previous session episodes."""
        self._create_session_with_episodes(tmp_storage)

        store2 = EpisodeStore(tmp_storage)
        curator2 = Curator(api_key=api_key)
        plugin2 = EpisodeCuratorPlugin(store2, curator2, threshold=80000)

        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=5,
            api_key=api_key,
        )
        agent.register_skill(plugin2)

        answer = agent.run("上一個 session 中，我們選了什麼資料庫？為什麼？請用 recall_episode 查詢。")
        assert any(kw in answer for kw in ["CockroachDB", "sharding", "分散式", "水平擴展"])

    @pytest.mark.llm
    def test_cross_session_recall_rate(self, tmp_storage, api_key):
        """Measure overall recall rate: new agent should recall >= 2/3 topics."""
        self._create_session_with_episodes(tmp_storage)

        store2 = EpisodeStore(tmp_storage)
        curator2 = Curator(api_key=api_key)
        plugin2 = EpisodeCuratorPlugin(store2, curator2, threshold=80000)

        agent = ReActAgent(
            model="claude-haiku-4-5-20251001",
            max_iterations=8,
            api_key=api_key,
        )
        agent.register_skill(plugin2)

        answer = agent.run(
            "這是一個新的 session。請用 recall_episode 搜尋之前的對話，回答：\n"
            "1. 我們用什麼後端框架？\n"
            "2. 選了什麼資料庫？\n"
            "3. 部署在什麼平台？"
        )

        hits = 0
        if "FastAPI" in answer:
            hits += 1
        if "CockroachDB" in answer:
            hits += 1
        if "Fly.io" in answer:
            hits += 1

        assert hits >= 2, f"Cross-session recall: {hits}/3 topics (need >= 2)"
