# CLAUDE.md — Episode Curator ReAct Agent

## 專案概述

建立一個帶有智慧上下文管理的 ReAct Agent。核心問題：LLM 的 context window 有限，但對話可能無限長。解法：用第二個 LLM（Curator）在幕後管理上下文，把舊對話按主題存成 episode，Worker 永遠不會爆 context。

技術棧：Python 3.11+、Anthropic SDK (`anthropic`)、無其他外部依賴。

## 架構

核心 + 插件：

```
react_agent.py          ← 基礎 ReAct Agent + SkillPlugin Hook 系統
episode_curator.py      ← Episode Curator 插件，解決 context window 問題
system_tools.py         ← System Tools 插件 (read/write/grep/search/bash/web_search)
tool_registry.py        ← 動態工具載入插件 (load_tools/unload_tools, Open/Close Book)
hook_manager.py         ← Hook Manager 插件 (PreToolUse/PostToolUse/Stop)
mcp_client.py           ← MCP Client 插件 (JSON-RPC over stdio)
skill_loader.py         ← Skill Loader 插件 (SKILL.md frontmatter)
cli_app.py              ← CLI TUI 介面 (Textual) + TUI Bridge 插件
web_app.py              ← 最小 Web 介面 (E2E 測試用)
```

### 雙 LLM 架構

```
Worker LLM (Sonnet)                    Curator LLM (Haiku)
├─ 正常跑 ReAct loop                  ├─ 掛在 on_token_usage Hook
├─ 看到的 messages:                    ├─ input_tokens 超過門檻時觸發
│  [原始問題, 全局索引, 最新幾條]      ├─ 按主題切分舊訊息 → 存 episode
└─ 可用 recall_episode 取回原文        ├─ 每個主題寫一行摘要 → index
                                       └─ 重建 ctx.messages
```

Worker 不知道 Curator 的存在。它只看到 ctx.messages 變短了但內容連貫。

### 磁碟結構 (SOURCE OF TRUTH)

```
~/.episode_store/
├── episodes/
│   ├── 001.json    ← 完整 messages 原文（immutable，永不修改）
│   ├── 002.json
│   └── ...
├── index.json      ← {"001": {"title":"...", "summary":"...", "tags":[...], "created_at":"..."}}
└── facts.json      ← ["使用者叫小明", "用 PostgreSQL", ...]
```

關鍵設計原則：**ctx.messages 是 VIEW，磁碟是 SOURCE OF TRUTH**。每個 episode 只被摘要一次（存入 index 時），全局索引是拼接所有 episode 的一行摘要，不是壓縮上一版索引。所以不存在「摘要的摘要」衰減問題。

## 檔案 1: react_agent.py

### 核心類

**AgentContext** — 在 ReAct loop 中傳遞的共享狀態：
```python
@dataclass
class AgentContext:
    user_query: str
    messages: list           # 完整對話歷史（Plugin 可直接修改）
    metadata: dict           # 插件間共享資料（如 system_prompt_extra）
    iteration: int
    total_input_tokens: int
    total_output_tokens: int
    start_time: float
    tool_call_history: list
```

**SkillPlugin** — 插件抽象基類，7 個可覆寫的 Hook + 工具定義/執行：
```python
class SkillPlugin(ABC):
    name: str                                    # 唯一名稱
    get_tools() -> list[dict]                    # 工具定義（Anthropic JSON Schema 格式）
    execute_tool(name, input) -> Any             # 工具執行
    on_agent_start(ctx)                          # Agent 啟動
    on_thought(ctx, thought) -> Optional[str]    # 模型思考（可修改）
    before_action(ctx, tool_call) -> Optional[ToolCall]  # 工具執行前（可攔截）
    after_action(ctx, tool_call, result) -> Optional[ToolResult]  # 工具執行後
    on_observation(ctx, observation) -> Optional[str]     # 觀察結果（可修改）
    on_error(ctx, error) -> Optional[str]        # 錯誤處理
    on_agent_end(ctx, final_answer) -> Optional[str]     # Agent 結束
    on_token_usage(ctx, input_tokens, output_tokens)     # Token 用量統計
    on_stream_delta(ctx, delta)                          # Streaming 文字 delta
```

**SkillPluginManager** — 插件註冊、工具路由、Hook 調度：
- `register(plugin)`: 註冊插件，檢查工具名稱衝突
- `get_all_tool_definitions()`: 彙總所有插件工具 → 傳給 Anthropic API 的 `tools` 參數
- `route_tool_call(name, input)`: 根據工具名稱找到對應插件執行
- `dispatch_*()`: 按註冊順序觸發所有插件的 Hook，支援鏈式修改

**ReActAgent** — 核心引擎：
- `__init__(model, max_iterations, max_tokens, system_prompt, api_key)`
- `register_skill(plugin)`: 註冊插件（支援鏈式）
- `run(user_query, conversation_history)`: 主入口
- `_react_loop(ctx, tools)`: ReAct 循環

### _react_loop 的 API 呼叫細節

每次迭代呼叫同一個 API：`client.messages.create()`

```python
# system prompt 動態組合（支援插件注入記憶等額外內容）
system = self.system_prompt
extra = ctx.metadata.get("system_prompt_extra", "")
if extra:
    system = f"{system}\n\n{extra}"

api_params = {
    "model": self.model,
    "max_tokens": self.max_tokens,
    "system": system,       # 動態，每輪可能不同
    "messages": ctx.messages,  # 累積成長，Curator 會修改
    "tools": tools,          # 所有插件的工具定義
}
response = client.messages.create(**api_params)
```

判斷循環結束：`stop_reason == "end_turn"` 且沒有 `tool_use` block。

tool_result 回饋格式（Anthropic 要求）：
```python
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_xxx", "content": "結果"}]}
```

## 檔案 2: episode_curator.py

### EpisodeStore

儲存層，操作磁碟上的 episodes、index、facts。

關鍵方法：
- `save_episode(id, messages, title, summary, tags, salience, dimensions)`: 存完整原文 + index 條目（immutable），salience 和 dimensions 一併存入
- `load_episode(id)`: 讀原文（100% 無損）
- `build_global_index()`: 拼接所有 episode 摘要，按 tag 分組，帶相對時間。**同一時間層內按 salience 降序排列**
- `search_episodes(query, limit, recent_hours)`: 搜尋 title/summary/tags，**salience 作為加權因子影響排名**，支援時間過濾
- `add_facts(facts)`: 新增永久事實（去重，最多 50 條）

index.json 的每個條目結構：
```json
{
  "title": "PostgreSQL 資料庫設計",
  "summary": "設計 products 表結構，確認 7 個產品，討論 index 策略",
  "tags": ["database", "postgresql"],
  "message_count": 6,
  "created_at": "2026-03-26T14:30:00",    ← 絕對時間，永不修改
  "salience": 0.6,                         ← 認知權重（磁碟上只標記不排序）
  "dimensions": {                          ← 多維度認知摘要
    "decisions": ["選擇 PostgreSQL 因為團隊熟悉"],
    "corrections": [],
    "insights": ["發現需要 7 個產品表"],
    "pending": ["index 策略待定"],
    "user_intent": "設計資料庫結構",
    "outcome": "positive"
  }
}
```

`build_global_index()` 中的 `format_time()` 在每次呼叫時即時計算相對時間（「3天前」「2小時前」），不修改 created_at。

### Curator

第二個 LLM，職責：把一段 messages 按主題切分，每段寫 title + summary + tags + **多維度認知摘要** + **salience 分數**。

#### 認知加權摘要 — Cognitive Salience Scoring

人對一段對話的記憶不是均勻的。認知科學告訴我們某些片段天生更「難忘」。Curator 在摘要時必須捕捉這些高認知權重的片段，產出結構化的多維度摘要和 salience 分數。

**6 個維度與認知科學原則對應：**

| 維度 | 內容 | 認知科學原則 |
|------|------|-------------|
| `decisions` | 做了什麼決定 + 為什麼 | **Levels of Processing** — 深層處理（理解 why）比淺層（記住 what）更持久 |
| `corrections` | 錯誤修正過程（tried X failed, switched to Y） | **Schema Violation + Emotional Tagging** — 打破預期 + 情緒喚起，編碼最強 |
| `insights` | 洞察/發現 | **Von Restorff Effect** — 在相似事物中，異常的那個記得最清楚 |
| `pending` | 還剩什麼沒做 | **Zeigarnik Effect** — 未完成任務的認知張力使記憶特別持久 |
| `user_intent` | 使用者到底想要什麼 | **Goal-directed Memory** — 與目標綁定的記憶檢索效率最高 |
| `outcome` | positive / negative / neutral | **Peak-End Rule** — 結尾決定整體記憶評價 |

**Salience 分數（0.0 ~ 1.0）：**

| 範圍 | 語義 | 典型場景 |
|------|------|---------|
| 0.0–0.3 | 低 — 普通資訊交換 | 確認型回覆、簡單查詢 |
| 0.4–0.6 | 中 — 一般討論 | 包含一般決策或中等複雜度討論 |
| 0.7–0.9 | 高 — 重要轉折 | 錯誤修正、重要決策、方向轉變 |
| 1.0 | 極高 — 關鍵突破 | 重大架構決策、嚴重 bug 修復、突破性發現 |

**Salience 的使用原則：磁碟上只標記不排序，填入 Worker context 前才按 salience 降序排列。**

**主題延續感知**：Curator 壓縮時必須看到現有的 index（已有哪些 episode 和 tag）。這樣它能：
- 複用已有的 tag（不自創新 tag，保持一致性）
- 如果新訊息是某個已有主題的延續，在 summary 開頭標註「接續 #xxx」
- 不合併 episode（episode 是 immutable 的），而是建立新 episode 並標註延續關係

```python
# process() 簽名帶 existing_index
def process(self, messages: list[dict], existing_index: dict) -> dict:
```

Curator 收到的輸入包含兩部分：
```
"已有主題索引：
  #001 [database] PostgreSQL 資料庫設計
  #002 [deployment] GCP 部署選擇

請處理以下新的對話記錄：
  [0] [user] 資料庫的 index 怎麼加？
  [1] [asst] 根據查詢模式...
  ..."
```

Curator 輸出中，如果是延續主題：
```json
{
  "segments": [{
    "title": "資料庫 index 和 sales 表",
    "summary": "接續 #001：為 products 表加入查詢 index，新增 sales 表",
    "tags": ["database"],
    "message_indices": [0, 1, 2, 3],
    "continues_episode": "001",
    "salience": 0.6,
    "dimensions": {
      "decisions": ["決定為 products 表加 B-tree index 因為查詢以等值比對為主"],
      "corrections": [],
      "insights": ["發現 GIN index 不適合此場景"],
      "pending": ["sales 表的 index 策略尚未決定"],
      "user_intent": "優化資料庫查詢效能",
      "outcome": "positive"
    }
  }],
  "facts": [...]
}
```

`continues_episode` 是可選欄位，存入 index 用於展示延續關係。全局索引中同一 tag 下的 episode 按時間排列，Worker 自然看到主題的演進脈絡。

Plugin 呼叫時傳入 existing_index：
```python
# on_token_usage 中
result = self._curator.process(to_archive, self._store._index)
```

System prompt 要求 Curator 輸出 JSON：
```json
{
  "segments": [
    {
      "title": "主題名",
      "summary": "一句話摘要（如是延續，開頭加「接續 #xxx：」）",
      "tags": ["tag1"],
      "message_indices": [0, 1, 2, 5],
      "continues_episode": "001",
      "salience": 0.7,
      "dimensions": {
        "decisions": ["決定 X 因為 Y"],
        "corrections": ["原本用 A 失敗，改用 B"],
        "insights": ["發現 Z 的性價比最高"],
        "pending": ["尚未處理 error handling"],
        "user_intent": "使用者想建立高效的 CI pipeline",
        "outcome": "positive"
      }
    }
  ],
  "facts": ["事實1", "事實2"]
}
```

System prompt 中必須包含以下指引：
- 按主題分段，同一主題的訊息即使不相鄰也歸為一段
- 參考已有的主題索引，盡量複用已有的 tag
- 如果新訊息是已有主題的延續，summary 開頭標註「接續 #xxx：」
- `continues_episode` 是可選欄位，填入被延續的 episode ID
- 如果是全新主題，不填 `continues_episode`
- 為每個 segment 評估 `salience` 分數（0.0~1.0），越高表示認知權重越大
- 填寫 `dimensions` 物件，包含 6 個維度：`decisions`（列表）、`corrections`（列表）、`insights`（列表）、`pending`（列表）、`user_intent`（字串）、`outcome`（"positive"/"negative"/"neutral"）
- dimensions 中的列表維度可以為空陣列，但 `user_intent` 和 `outcome` 必填
- salience 評分依據：包含錯誤修正 +0.3、包含重要決策 +0.2、包含方向轉變 +0.2、包含正向確認 +0.1、基線 0.3

messages 格式化時帶編號 `[0] [user] ...`，讓 Curator 能用 `message_indices` 引用。

Curator model 用 Haiku（便宜快速），max_tokens=1200（多段 + dimensions 需要更多 output）。

### EpisodeCuratorPlugin

掛進 ReActAgent 的 SkillPlugin，用 3 個 Hook + 1 個 Tool。

**Hook: on_agent_start** — 注入 facts 和全局索引到 `ctx.metadata["system_prompt_extra"]`
- 讀 facts.json → 格式化為「已知事實」
- 呼叫 build_global_index() → 格式化為按 tag 分組的目錄

**Hook: on_token_usage** — 核心。每次 Worker API 呼叫後觸發。
- `input_tokens < threshold` → 不做事
- `input_tokens >= threshold` → 觸發壓縮：
  1. 保留 messages[0]（原始問題）和最後 N 條
  2. 中間的 messages 交給 Curator 按主題切分
  3. 每個 segment 根據 `message_indices` 取出對應訊息，存成獨立 episode
  4. facts 存入 facts.json
  5. 重建 ctx.messages = [原始問題, 全局索引, assistant 確認, 最新 N 條]
  6. 全局索引是從 index.json 拼接的（不是壓縮上一版）

重建後的 messages 必須維持 user/assistant 交替：
```python
ctx.messages = [
    first_msg,                    # user: 原始問題
    {"role": "user", "content": global_index},  # user: 索引
    {"role": "assistant", "content": [{"type": "text", "text": "了解，我已掌握之前的對話脈絡。"}]},
    *to_keep,                     # 保留的最新訊息
]
```

**Hook: on_agent_end** — 規則提取 facts（不呼叫 LLM），關鍵字模式匹配。

**Tool: recall_episode** — Worker 主動取回 episode 原文。
- `episode_id`: 直接讀取
- `search_query`: 搜尋 title/summary/tags（tag 3 分，title 2 分，summary 1 分）
- `recent_hours`: 可選，只搜最近 N 小時
- 原文回傳帶時間戳：`── Episode #001: PostgreSQL 資料庫設計 | 2026-03-26T14:30 ──`

## 檔案 3: system_tools.py

### SystemToolsPlugin

提供 6 個系統工具讓 Worker LLM 操作檔案系統和 shell：

| 工具 | 功能 | 關鍵參數 |
|------|------|---------|
| `read` | 讀取檔案，回傳帶行號的內容 | `file_path`, `offset`, `limit` |
| `write` | 寫入檔案，自動建立目錄 | `file_path`, `content` |
| `grep` | 用 regex 搜尋檔案內容 | `pattern`, `path`, `include`（glob 過濾） |
| `search` | 用 glob 模式尋找檔案 | `pattern`, `path` |
| `bash` | 執行 shell 命令 | `command`, `timeout`（預設 30s） |
| `web_search` | 網路搜尋（stub） | `query` |

設計原則：
- 全部用 Python stdlib（pathlib, subprocess, re, glob），零外部依賴
- bash 輸出上限 100KB，超過截斷
- grep 最多回傳 500 個 match，掃描最多 1000 個檔案
- 錯誤一律回傳 `Error: ...` 字串（不拋異常），讓 LLM 能讀懂

## 檔案 4: hook_manager.py

### HookManager

使用者可透過 `hooks.json` 掛載前/後攔截邏輯。支援三種事件：

| 事件 | 觸發時機 | 效果 |
|------|---------|------|
| `PreToolUse` | 工具執行前 | 可阻擋（`continue: false`）或放行 |
| `PostToolUse` | 工具執行後 | 可附加系統訊息 |
| `Stop` | Agent 準備結束時 | 可強制續跑（`continue: true`） |

**hooks.json 格式**：
```json
{
  "PreToolUse": [
    {
      "matcher": "Write|bash",
      "hooks": [
        {"type": "command", "command": "bash validate.sh", "timeout": 30}
      ]
    }
  ],
  "PostToolUse": [...],
  "Stop": [...]
}
```

**Matcher 語法**：
- 精確名稱：`"bash"`
- 多選：`"Write|bash|read"`（pipe 分隔）
- 萬用：`"*"`

**Hook 執行**：用 `subprocess.run()` 跑 command，stdin 傳入 JSON context，stdout 讀取 JSON 結果：
```json
{"continue": true, "systemMessage": "Validation passed"}
```

**HookManagerPlugin**：掛進 SkillPlugin 系統：
- `before_action` hook → 呼叫 `run_pre_tool_use`，若 `continue=false` → 設 `tc["_blocked"]`
- `after_action` hook → 呼叫 `run_post_tool_use`
- `on_agent_end` hook → 呼叫 `run_stop`，若 `continue=true` → 回傳 `{"continue": True}`

### react_agent.py 擴充

**`_blocked` tool 支援**（`_react_loop` 內）：
```python
tc = self._manager.dispatch_before_action(ctx, tc)
if tc.get("_blocked"):
    result = tc["_blocked"]  # 跳過 route_tool_call
```

**Stop hook 續跑**：
```python
final_answer = self._manager.dispatch_on_agent_end(ctx, final_text)
if isinstance(final_answer, dict) and final_answer.get("continue"):
    ctx.messages.append({"role": "user", "content": final_answer.get("message")})
    continue  # 重新進入 while loop
```

## 檔案 5: mcp_client.py

### MCPManager

讀取 `.mcp.json`，啟動 MCP server 子程序，透過 JSON-RPC over stdio 通訊。

**`.mcp.json` 格式**：
```json
{
  "server-name": {
    "command": "python",
    "args": ["server.py"],
    "env": {"KEY": "value"}
  }
}
```

**JSON-RPC 協議**（line-delimited JSON over stdin/stdout）：
```json
→ {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
← {"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}

→ {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}}}
← {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "hi"}]}}
```

**工具命名**：`mcp__{server}__{tool}`（例如 `mcp__filesystem__read_file`）

**Schema 轉換**：MCP 用 `inputSchema`（camelCase），Anthropic API 用 `input_schema`（snake_case），自動轉換。

### MCPPlugin

- `get_tools()` 回傳所有 server 的工具定義（已轉換格式、已加前綴）
- `execute_tool()` 從工具名稱解析 server，呼叫 `manager.call_tool()`

## 檔案 6: tool_registry.py

### ToolRegistryPlugin

動態工具載入插件（Open/Close Book）。讓 Worker LLM 按需載入工具 schema，不需要的工具只保留 metadata。

**核心概念**：
- **Deferred tools**：`is_deferred() → True` 的插件，工具定義不會自動載入
- **Tool catalog**：所有工具的 name + description，永遠在 system prompt
- **Active tools**：已載入的工具 schema，會傳給 Anthropic API

**SkillPlugin 新增方法**：
```python
def is_deferred(self) -> bool:
    """如果 True，工具啟動時不載入 schema，只保留 metadata。"""
    return False  # 預設不延遲
```

**SkillPluginManager 新增方法**：
- `get_active_tool_definitions()`: 只回傳已載入的工具定義
- `load_tools(names)`: 載入工具 schema（Open Book）
- `unload_tools(names)`: 卸載工具 schema，保留 metadata（Close Book）
- `get_tool_catalog()`: 回傳所有工具的 `{name, description, loaded}` 列表

**ToolRegistryPlugin 提供 2 個工具**：
| 工具 | 功能 |
|------|------|
| `load_tools` | 載入指定工具的 schema，使其可用 |
| `unload_tools` | 卸載工具 schema，釋放 context 空間 |

**on_agent_start**：將 deferred 工具目錄注入 `system_prompt_extra`。

**Close Book — Context 壓縮**：

`unload_tools` 不只移除 schema，還會掃描 `ctx.messages` 壓縮對應工具的 `tool_result` 內容：

1. 建立 `tool_use_id → tool_name` 映射表（從 assistant messages 的 tool_use blocks）
2. 掃描 user messages 的 tool_result blocks，找到對應的已卸載工具
3. 若 content > 200 chars，壓縮為：`[{tool_name} result compressed] {前100字}... ({原始長度} chars — use load_tools to re-expand)`
4. 小於門檻的 result 不壓縮（overhead 不值得）
5. 不影響非目標工具的 result，不影響純文字訊息

觸發時機：`after_action` hook 偵測到 `unload_tools` 完成後自動執行。

**Re-expand — 重新展開**：

`load_tools` 時自動還原先前壓縮的 `tool_result` 內容：

1. 壓縮時原始內容存入 `ctx.metadata["_compressed_results"][tool_use_id]`
2. `load_tools` 的 `after_action` 掃描 messages，找到對應的壓縮 result
3. 用存儲的原始內容還原 `block["content"]`，並從 store 中移除
4. 若沒有先前壓縮記錄（從未 close 過），load 是 no-op

**_react_loop 變更**：每次迭代呼叫 `get_active_tool_definitions()` 而非靜態 tools 列表，支援動態載入。

**Deferred 插件**：SystemToolsPlugin 和 MCPPlugin 的 `is_deferred()` 回傳 True。

## 檔案 7（原 6）: skill_loader.py

### SkillManager

從 `skills/` 目錄載入 SKILL.md 檔案，相容 Claude Code skill 格式。

**目錄結構**：
```
skills/
  commit/
    SKILL.md
  review-pr/
    SKILL.md
```

**SKILL.md 格式**（YAML frontmatter）：
```markdown
---
name: commit
description: This skill helps create well-structured git commits.
---

# Commit Skill
When creating a commit:
1. Stage only relevant files
2. Write a concise commit message
```

**Frontmatter 解析**：用 regex `^---\n(.*?)\n---` 提取，不依賴 pyyaml。

### SkillLoaderPlugin

- `on_agent_start`：建立 skill catalog 字串，附加到 `system_prompt_extra`
- 格式：`Available skills:\n- commit: This skill helps...\n- review-pr: This skill reviews...`

## 檔案 7: cli_app.py

### TUI 介面（Textual）

```
┌───────────────────────────────────┬────────────────────────────┐
│  ◈ Conversation                   │  ◈ Context                 │
│  [user] question                  │  system ████░░░░  25%      │
│  💭 thinking...                   │  tools  ██░░░░░░  12%      │
│  🔧 recall_episode({query:db})    │  msgs   ██████░░  63%      │
│  📋 Found 3 episodes...           ├────────────────────────────┤
│  [assistant] answer               │  ◈ Context Detail (Ctrl+D) │
│  ▌streaming text appears here...  │  system_prompt_extra: ...  │
│                                   ├────────────────────────────┤
│                                   │  ◈ Episodes                │
│                                   │  ● #002 sal:0.9 (2h前)    │
│                                   │    Port fix                │
│                                   │  ● #001 sal:0.6 (1d前)    │
│                                   │    DB design               │
├───────────────────────────────────┴────────────────────────────┤
│  > Type your message...                                        │
├────────────────────────────────────────────────────────────────┤
│  ○ │ turn 3 │ iter 2/30 │ in: 15.8k │ out: 2.1k │ 3 eps     │
└────────────────────────────────────────────────────────────────┘
```

### TUIPlugin

Hook-only 插件（無工具），捕捉 ReAct loop 事件推送到 Textual App：

| Hook | 推送事件 |
|------|---------|
| `on_stream_delta` | 即時文字 → streaming Static widget |
| `on_thought` | 💭 thought 文字 → 左側面板（streaming 時 suppress） |
| `before_action` | 🔧 tool name + input → 左側面板 |
| `on_observation` | 📋 tool result → 左側面板 |
| `on_token_usage` | status bar + context usage + context_content 更新 |
| `on_agent_end` | ✓ 最終答案 → 左側面板 |

**Streaming**：`_react_loop` 用 `client.messages.stream()` 替代 `create()`，文字 delta 即時推送到 TUI 的 `#stream-output` Static widget。完成後 widget 隱藏，最終文字寫入 RichLog。

**即時更新**：Agent 在 `asyncio.to_thread()` 中跑（blocking sync），TUIPlugin 用 `app.call_from_thread()` 推事件到 Textual 主線程。

**Context Detail Panel**：按 Ctrl+D/T/B 切換顯示 system/tools/msgs 的完整內容，預設隱藏。system 包含 base prompt + system_prompt_extra。

### Status Line

```
 ◉ │ turn 3 │ iter 2/30 │ in: 15.8k / 80k │ out: 2.1k │ 3 eps │ 4.7s
 ↑     ↑         ↑              ↑              ↑          ↑       ↑
 │     │         │              │              │          │       └─ 經過時間
 │     │         │              │              │          └─ episode 數量
 │     │         │              │              └─ 累計輸出 tokens
 │     │         │              └─ 累計輸入 tokens / 壓縮門檻
 │     │         └─ 當前 run 的 ReAct loop 輪次 / 最大輪數
 │     └─ 累計對話 turn 數（跨多次 agent.run()）
 └─ ◉ busy / ○ idle
```

### 工廠函式

```python
def create_agent(
    worker_model="claude-sonnet-4-20250514",    # Worker 用 Sonnet
    curator_model="claude-haiku-4-5-20251001",  # Curator 用 Haiku（便宜 10x）
    threshold=80000,                             # 80K tokens 觸發壓縮
    max_iterations=30,                           # 有 Curator 可以設更高
    storage_dir=None,                            # 預設 ~/.episode_store/
    api_key=None,
) -> ReActAgent:
```

## 關鍵設計決策（實作時必須遵守）

1. **Episode 是 immutable 的**。寫入後不修改，不刪除。這是整個架構能避免摘要疊摘要衰減的原因。

2. **Index 中每條摘要只生成一次**。不會在後續壓縮中被「再摘要」。全局索引是拼接，不是壓縮。

3. **Curator 按主題切分，不按位置切**。同一批訊息可能產出 1~N 個 episode。`message_indices` 不要求連續，同一主題的訊息即使在對話中不相鄰也會被歸在一起。

4. **磁碟存絕對時間，顯示算相對時間**。`created_at` 是 ISO format，`format_time()` 每次呼叫時即時計算「3天前」，不修改資料。

5. **Worker 不知道 Curator 存在**。它只看到 ctx.messages 和一個 recall_episode 工具。所有上下文管理對 Worker 完全透明。

6. **重建 messages 時維持 user/assistant 交替**。Anthropic API 的格式要求。

7. **tool_use 和 tool_result 必須配對**。壓縮時不能把一對的中間切斷。preserve_recent 保留最後 N 條確保配對完整。

8. **同一主題的後續對話不合併進已有 episode，而是建立新 episode 並標註延續**。Curator 壓縮時必須看到現有 index，複用已有 tag，在 summary 中標註「接續 #xxx」。Episode immutable 原則不可打破。

9. **全局索引的時間解析度隨時間自然衰減**。近期逐條、中期按天匯總、遠期按週/月匯總。磁碟上的 episode 不動，匯總是展示層的事。

10. **認知加權摘要**。Curator 為每個 segment 產出 salience（0.0~1.0）和 6 維度 dimensions。磁碟上 index.json 存 salience 和 dimensions 但不影響排序。填入 Worker context 時（`build_global_index()`）同一時間層內按 salience 降序排列。`search_episodes()` 的評分也受 salience 加權。向後相容：舊 episode 的 salience 預設 0.5，dimensions 預設空物件。

### 時間解析度衰減 — Temporal Resolution Decay

人類記得昨天每小時做了什麼，但上個月只記得幾件大事。全局索引也應該如此。

三層解析度：

| 時間範圍 | 展示粒度 | 索引中每條的來源 |
|----------|----------|-----------------|
| 最近 48 小時 | 每個 episode 獨立一行 | 直接從 index.json |
| 48h ~ 2 週 | 同一天+同 tag 合併一行 | 從 digests/daily/ |
| 2 週以上 | 同一週/月+同 tag 合併一行 | 從 digests/weekly/ |

磁碟結構新增：
```
~/.episode_store/
├── episodes/          ← 原始 episode（immutable）
├── index.json         ← 逐條索引（immutable）
├── digests/           ← 匯總摘要（也是 immutable）
│   ├── daily/         ← 日匯總：{date}_{tag}.json
│   └── weekly/        ← 週匯總：{year}-W{week}_{tag}.json
├── digest_index.json  ← 匯總索引
└── facts.json
```

日匯總/週匯總的生成規則：
- 只為**已經過去的**時間段生成（今天不生成日匯總，本週不生成週匯總）
- 一旦生成就 immutable（昨天不會再多出新 episode）
- 生成時用 Curator LLM 從該時段同 tag 的所有 episode 摘要中再做一次匯總
- 注意：這裡的匯總**不是摘要的摘要**，而是從多個獨立的一行摘要拼接後做歸納，每個一行摘要本身沒有被壓縮過
- 匯總結果帶 `(N段)` 標記，告訴 Worker 背後有多少個原始 episode

觸發時機：`on_agent_start` → `build_global_index()` 內部檢查。如果發現某個已過去的時間段缺少 digest，呼叫 Curator 生成。每次啟動最多生成 3 個新 digest（控制成本）。

`build_global_index()` 的組裝邏輯：
```python
def build_global_index(self):
    now = datetime.now()
    lines = []

    # 近期（48h 內）：逐條，從 index.json，按 salience 降序
    recent = [ep for ep in index if age(ep) < 48h]
    recent.sort(key=lambda ep: ep.salience, reverse=True)
    for ep in recent:
        lines.append(f"#{ep.id} ({relative_time}) — {ep.summary}")

    # 中期（48h ~ 2 週）：按天匯總，從 digests/daily/
    for day in past_days(2, 14):
        digest = load_daily_digest(day, tag)
        lines.append(f"{day} — {digest.summary} ({digest.episode_count}段)")

    # 遠期（2 週+）：按週/月匯總，從 digests/weekly/
    for week in past_weeks(2, ...):
        digest = load_weekly_digest(week, tag)
        lines.append(f"{week} — {digest.summary} ({digest.episode_count}段)")
```

效果：120 個 episodes 在索引中只佔 ~12 行（~800 tokens），而非 120 行（~3000 tokens）。Worker 需要遠期細節時用 `recall_episode` 取回原文。

## 使用方式

```python
from episode_curator import create_agent

agent = create_agent()
answer = agent.run("你的問題")

# 多輪對話
history = []
for query in queries:
    answer = agent.run(query, history)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
```

## 依賴

```
pip install anthropic
```

環境變數：`ANTHROPIC_API_KEY`

## 測試要點

1. **基本 ReAct loop**：Agent 能正確呼叫工具、解析結果、給出答案
2. **壓縮觸發**：模擬高 token 用量，確認 on_token_usage 觸發壓縮
3. **主題切分**：一段包含多主題的訊息，Curator 能切成多個 episode
4. **Episode 不可變**：存入後重新讀取內容完全一致
5. **全局索引不衰減**：壓縮 5 次後，episode 001 的摘要和第一次寫入時一樣
6. **recall_episode**：按 id 取回完整原文、按關鍵字搜尋、按時間過濾
7. **messages 格式**：壓縮後 ctx.messages 維持 user/assistant 交替，tool_use/tool_result 配對完整
8. **facts 持久化**：跨 session 保留，去重正確
9. **主題延續**：第二次壓縮時 Curator 能看到已有 index，複用 tag，摘要標註「接續 #xxx」，不產出不一致的新 tag
10. **時間衰減索引**：build_global_index 對 48h 內逐條顯示、2 週內按天匯總、更早按週匯總。日/週 digest 只為已過去的時段生成且 immutable
11. **認知加權摘要**：Curator 能產出 salience 分數和 6 維度 dimensions，存入 index 且 recall 時可見
12. **Salience 排序**：build_global_index 同一時間層內高 salience episode 排前面，search_episodes 結果受 salience 加權
13. **動態工具載入/卸載**：is_deferred、load_tools/unload_tools、get_tool_catalog、deferred 工具不在 active set
14. **Tool/Skill 載入 TUI E2E**：deferred 工具在 TUI 中正確載入/卸載，skill catalog 與 tool catalog 共存，agent 能先 load_tools 再使用
15. **長對話記憶 recall rate**：seed 5 個 episode 後，search recall rate ≥ 80%，LLM 能正確 recall ≥ 2/3 主題
16. **MCP + Skill 相容性**：MCP 工具 deferred、load/unload/execute 正常、與 system tools 無衝突、skill catalog 在 load/unload cycle 後穩定
17. **Session restart recall rate**：episodes/facts 跨 session 持久化，新 session 搜尋 recall ≥ 67%，新 agent 注入舊 context，LLM 跨 session recall ≥ 2/3
18. **Close-book context 壓縮**：unload_tools 壓縮 tool_result 內容，保留 metadata，不影響非目標工具，小 result 不壓縮
19. **Re-expand 重新展開**：load_tools 還原先前壓縮的 tool_result，store 清理，close→re-open cycle 完整，LLM E2E 驗證
20. **Streaming 輸出**：on_stream_delta hook、streaming 時 suppress on_thought、stream flag 在 on_token_usage 後重置
21. **Status turn 累計**：turn 跨 run() 累加、cumulative tokens、StatusBar 顯示 turn
22. **Context detail panel**：Ctrl+D/T/M 切換 system/tools/msgs 內容、預設隱藏、切換 category
23. **Stream widget**：stream-output widget 存在、stream_delta 更新 widget、answer 清除 stream
24. **TUI Full Pipeline**：TUI input → agent → load_tools → read → answer 完整鏈路、多工具鏈（load→grep→read）、context detail 在 tool 呼叫後顯示 messages

## Progress

| Phase | 描述 | 狀態 | 測試數 | Commit |
|-------|------|------|--------|--------|
| 0 | 專案骨架 (pyproject.toml, .gitignore, tests/) | DONE | — | `fdd07fb` |
| 1 | react_agent.py — AgentContext, SkillPlugin, SkillPluginManager | DONE | 12 | `e0ad16f` |
| 2 | react_agent.py — ReActAgent + ReAct Loop + OAuth auth | DONE | 4 (真實 API) | `d494cdf` |
| 3 | episode_curator.py — EpisodeStore (磁碟存取層) | DONE | 21 | `cd08c0c` |
| 4 | episode_curator.py — Curator (LLM 主題切分) | DONE | 4 (真實 API) | `f42cc24` |
| 5 | EpisodeCuratorPlugin (壓縮 + recall + facts) | DONE | 10 (真實 API) | `fd8c7dc` |
| 6 | Temporal Resolution Decay (日/週匯總) | DONE | 7 (真實 API) | `c3ea9e6` |
| 7 | create_agent() 工廠 + 整合測試 | DONE | 5 (真實 API) | `6e32052` |
| 8 | E2E Playwright 瀏覽器測試 | DONE | 5 (真實 API + Chromium) | `62454ca` |
| 9 | 收尾 — 全部 68 測試通過 | DONE | 68 total | — |
| 10 | 認知加權摘要 — salience + dimensions | DONE | 8 (3 真實 API) | `d8dbfc9` |
| 11 | CLI TUI 介面 (Textual) | DONE | 60 | — |
| 12 | System Tools (read/write/grep/search/bash/web_search) | DONE | 32 | — |
| 13 | Hook Manager (PreToolUse/PostToolUse/Stop) | DONE | 22 | — |
| 14 | MCP Client (JSON-RPC over stdio) | DONE | 16 | — |
| 15 | Skill Loader (SKILL.md frontmatter) | DONE | 14 | — |
| 16 | Dynamic Tool Loading (Open/Close Book + Re-expand) | DONE | 42 | `0f2750c` |
| 17 | TUI E2E — tool/skill loading + memory recall + MCP compat + session restart + full pipeline | DONE | 29 (9 真實 API) | — |

### Spec 測試覆蓋對照

| # | Spec 測試項目 | 測試檔案 | 狀態 |
|---|-------------|---------|------|
| 1 | 基本 ReAct loop | `test_react_loop.py` | PASS |
| 2 | 壓縮觸發 | `test_plugin.py` | PASS |
| 3 | 主題切分 | `test_curator.py` | PASS |
| 4 | Episode 不可變 | `test_episode_store.py` | PASS |
| 5 | 全局索引不衰減 | `test_episode_store.py` | PASS |
| 6 | recall_episode | `test_plugin.py` + `test_episode_store.py` | PASS |
| 7 | messages 格式 | `test_react_loop.py` + `test_plugin.py` | PASS |
| 8 | facts 持久化 | `test_episode_store.py` | PASS |
| 9 | 主題延續 | `test_curator.py` | PASS |
| 10 | 時間衰減索引 | `test_temporal_decay.py` | PASS |
| 11 | 認知加權摘要 | `test_curator.py` + `test_plugin.py` | PASS |
| 12 | Salience 排序 | `test_episode_store.py` + `test_plugin.py` | PASS |
| 13 | 動態工具載入/卸載 | `test_tool_registry.py` | PASS |
| 14 | Tool/Skill 載入 TUI E2E | `test_tui_e2e.py::TestToolDynamicLoadingTUI` | PASS |
| 15 | 長對話記憶 recall rate | `test_tui_e2e.py::TestLongConversationRecall` | PASS |
| 16 | MCP + Skill 相容性 | `test_tui_e2e.py::TestMCPAndSkillCompatibility` | PASS |
| 17 | Session restart recall rate | `test_tui_e2e.py::TestSessionRestartRecall` | PASS |
| 18 | Close-book context 壓縮 | `test_tool_registry.py::TestCloseBookCompression` | PASS |
| 19 | Re-expand 重新展開 | `test_tool_registry.py::TestReExpandOnLoad` + `test_tui_e2e.py` | PASS |
| 20 | Streaming 輸出 | `test_cli_tui.py::TestStreamingEvents` | PASS |
| 21 | Status turn 累計 | `test_cli_tui.py::TestStatusTurnCounter` | PASS |
| 22 | Context detail panel | `test_cli_tui.py::TestContextDetailPanel` | PASS |
| 23 | Stream widget | `test_cli_tui.py::TestContextDetailPanel` | PASS |
| 24 | TUI Full Pipeline | `test_tui_e2e.py::TestTUIFullPipeline` | PASS |

