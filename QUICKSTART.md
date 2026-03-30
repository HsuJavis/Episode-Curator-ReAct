# Quickstart

## 安裝

```bash
# 基本安裝（API 模式）
uv pip install -e .

# 含 TUI 介面
uv pip install -e ".[tui]"

# 含測試工具
uv pip install -e ".[dev]"
```

## 設定 API Key

二擇一：

```bash
# 方法 1: 環境變數
export ANTHROPIC_API_KEY="sk-ant-..."

# 方法 2: Claude Max OAuth（自動從 ~/.claude/.credentials.json 讀取）
# 不需要額外設定
```

## 使用方式

### 1. Python API — 最簡單

```python
from episode_curator import create_agent

agent = create_agent()
answer = agent.run("幫我解釋 Python 的 GIL")
print(answer)
```

### 2. Python API — 帶 System Tools + Dynamic Loading

```python
from react_agent import ReActAgent
from system_tools import SystemToolsPlugin
from tool_registry import ToolRegistryPlugin

agent = ReActAgent(max_iterations=15)
mgr = agent._manager
mgr.register(SystemToolsPlugin())          # read/write/grep/bash (deferred)
mgr.register(ToolRegistryPlugin(mgr))      # load_tools/unload_tools

# Agent 會自動: load_tools → 使用 → unload_tools
answer = agent.run("讀取 pyproject.toml 告訴我專案名稱")
print(answer)
```

### 3. Python API — 完整配置

```python
from episode_curator import create_agent, EpisodeStore, Curator, EpisodeCuratorPlugin
from system_tools import SystemToolsPlugin
from tool_registry import ToolRegistryPlugin
from skill_loader import SkillManager, SkillLoaderPlugin
from hook_manager import HookManagerPlugin

agent = create_agent(
    worker_model="claude-sonnet-4-20250514",
    curator_model="claude-haiku-4-5-20251001",
    threshold=80000,       # 80K tokens 觸發壓縮
    max_iterations=30,
    storage_dir="./my_episodes",
)

# 加掛插件
mgr = agent._manager
mgr.register(SystemToolsPlugin())
mgr.register(SkillLoaderPlugin(SkillManager("skills/")))
mgr.register(ToolRegistryPlugin(mgr))
mgr.register(HookManagerPlugin("hooks.json"))

# 多輪對話
history = []
while True:
    query = input("> ")
    answer = agent.run(query, history)
    print(answer)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
```

### 4. TUI 介面

```bash
# 直接啟動
python cli_app.py

# 自訂 episode 存放位置
EPISODE_STORE_DIR=./my_episodes python cli_app.py
```

### 5. 跑測試

```bash
# 全部非 LLM 測試（不需要 API key）
uv run pytest tests/ --ignore=tests/test_e2e.py -k "not llm"

# 含 LLM 測試（需要 API key）
uv run pytest tests/ --ignore=tests/test_e2e.py

# 只跑特定套件
uv run pytest tests/test_tool_registry.py -v
uv run pytest tests/test_tui_e2e.py -v -k "not llm"
```

## 核心概念

```
Worker LLM (Sonnet)              Curator LLM (Haiku)
├─ 跑 ReAct loop                ├─ 監控 token 用量
├─ 用 load_tools 載入工具        ├─ 超過門檻時壓縮
├─ 用 unload_tools 釋放 context  ├─ 按主題存 episode
└─ 用 recall_episode 取回記憶    └─ 全局索引不衰減

Open Book:  load_tools(["read"]) → schema 載入 → 工具可用
Close Book: unload_tools(["read"]) → schema 移除 + tool_result 壓縮
Re-open:    load_tools(["read"]) → schema 載入 + tool_result 還原
```

## 檔案結構

```
react_agent.py       ← 核心引擎 + 插件系統
episode_curator.py   ← Episode 記憶管理
system_tools.py      ← 檔案/shell 工具 (deferred)
tool_registry.py     ← 動態工具載入 (open/close book)
hook_manager.py      ← Hook 攔截系統
mcp_client.py        ← MCP 協議客戶端 (deferred)
skill_loader.py      ← SKILL.md 技能載入
cli_app.py           ← TUI 介面
```
