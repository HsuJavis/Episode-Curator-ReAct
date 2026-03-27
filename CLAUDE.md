# CLAUDE.md — Episode Curator ReAct Agent

## 專案概述

建立一個帶有智慧上下文管理的 ReAct Agent。核心問題：LLM 的 context window 有限，但對話可能無限長。解法：用第二個 LLM（Curator）在幕後管理上下文，把舊對話按主題存成 episode，Worker 永遠不會爆 context。

技術棧：Python 3.11+、Anthropic SDK (`anthropic`)、無其他外部依賴。

## 架構

兩個檔案，一個插件：

```
react_agent.py          ← 基礎 ReAct Agent + SkillPlugin Hook 系統
episode_curator.py      ← 唯一的插件，解決 context window 問題
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
- `save_episode(id, messages, title, summary, tags)`: 存完整原文 + index 條目（immutable）
- `load_episode(id)`: 讀原文（100% 無損）
- `build_global_index()`: 拼接所有 episode 摘要，按 tag 分組，帶相對時間
- `search_episodes(query, limit, recent_hours)`: 搜尋 title/summary/tags，支援時間過濾
- `add_facts(facts)`: 新增永久事實（去重，最多 50 條）

index.json 的每個條目結構：
```json
{
  "title": "PostgreSQL 資料庫設計",
  "summary": "設計 products 表結構，確認 7 個產品，討論 index 策略",
  "tags": ["database", "postgresql"],
  "message_count": 6,
  "created_at": "2026-03-26T14:30:00"    ← 絕對時間，永不修改
}
```

`build_global_index()` 中的 `format_time()` 在每次呼叫時即時計算相對時間（「3天前」「2小時前」），不修改 created_at。

### Curator

第二個 LLM，唯一職責：把一段 messages 按主題切分，每段寫 title + summary + tags。

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
    "continues_episode": "001"
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
      "continues_episode": "001"
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

messages 格式化時帶編號 `[0] [user] ...`，讓 Curator 能用 `message_indices` 引用。

Curator model 用 Haiku（便宜快速），max_tokens=800（多段需要更多 output）。

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

    # 近期（48h 內）：逐條，從 index.json
    recent = [ep for ep in index if age(ep) < 48h]
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

