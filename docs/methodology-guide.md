# PA_MCP 新手方法论地图（Methodology Guide）

> 本图是 PA_MCP 全部研究资产的统一索引：**11 个策略 × 8 个大牛方法 × 6 个分析方法 × 6 个 LLM 能力**，按新手四步走导航。UI「📚 研究总览」Tab 的 🗺️ 按钮与 MCP 工具 `get_methodology_guide` 均消费同一份编目数据（`src/pa_mcp/research/methodology_guide.py`），口径一致。
>
> ⚠️ 研究参考，非投资建议。仓位纪律（RiskGuard 单票 ≤20%、回撤 10/12/15% 分级）优先于一切信号。

---

## 一、为什么需要这张地图

PA_MCP 有 10 个 UI Tab、92 个 MCP 工具。新手最容易迷失的三个问题：

1. **这些东西是什么关系？** —— 海龟既是"策略"也是"大牛方法"；缠论既看结构也出背驰信号；"因子"既是选股工具也是验证工具。
2. **按什么顺序用？** —— 直接点"一站式分析"可以，但不知道结果怎么来的。
3. **什么时候该用 LLM？** —— 全部按钮都能点，但 AI 分析要花钱、要配置 key，哪些是免费的确定性工具？

地图把全部资产编入 **四类**，按 **四步** 走，每步给出"当前市场状态"下的推荐。

## 二、四类资产总览（31 项）

| 类别 | 是什么 | 数量 | 免费？ |
|---|---|---|---|
| 🎯 **策略（strategy）** | 规则化买卖信号（回测/扫描用） | 11 | ✅ 全免费 |
| 📚 **大牛方法（method）** | 大师体系：欧奈尔/缠中说禅/海龟/利弗莫尔/格雷厄姆/Asness/板块轮动/游资情绪 | 8 | ✅ 全免费 |
| 🔬 **分析方法（analysis）** | 验证信号是否靠谱：事件研究/Walk-Forward/过拟合诊断/因子检验 | 6 | ✅ 全免费 |
| 🔮 **LLM 能力（llm）** | AI 解读/预测/研报/经验沉淀（需配置 `config/llm_config.json`） | 6 | ❌ 需 API key，有 token 成本 |

### 策略（11）

| 策略 | 难度 | 一句话 | 适用市场 | 入口 |
|---|---|---|---|---|
| bollinger_mean_reversion | ⭐ | 跌到布林下轨买入，回到中轨卖出 | 启动期/低迷期 | 策略回测「运行回测」 |
| ma_golden_cross | ⭐ | 短期均线上穿长期均线买入 | 发酵期/启动期 | 同上 |
| oversold_bounce | ⭐ | 跌过头了赌反弹 | 启动期/低迷期 | 同上 |
| platform_breakout | ⭐⭐ | 横盘放量突破买入 | 发酵期/高潮期 | 同上 |
| volume_price_momentum | ⭐⭐ | 放量上涨时跟随 | 发酵期/高潮期 | 同上 |
| turtle | ⭐⭐ | 突破 20 日新高买入，跌破离场 | 发酵期 | 同上 |
| livermore_pivot | ⭐⭐⭐ | 只在趋势确认的关键点进场 | 发酵期 | 同上 |
| first_board_breakout | ⭐⭐⭐ | 首次涨停买入（游资短线） | 发酵期/高潮期 | 同上 |
| dragon_second_wave | ⭐⭐⭐ | 龙虎榜后第二波 | 发酵期/高潮期 | 同上 |
| range_grid | ⭐⭐ | 区间内高抛低吸 | 低迷期 | 同上 |
| roe_pb_value | ⭐⭐ | 高 ROE + 低 PB 长期持有 | 低迷期/冰点期 | 同上 |

> 新手默认：**bollinger_mean_reversion** ⭐ —— 规则最简单、信号最多、无需高频盯盘。

### 大牛方法（8）

| 方法 | 难度 | 一句话 | 适用市场 | UI 入口 | MCP 工具 |
|---|---|---|---|---|---|
| CANSLIM（欧奈尔） | ⭐⭐ | 七要素筛选成长股 | 发酵/启动 | 📡 市场扫描「🧬 CANSLIM」 | `scan_canslim` |
| 缠论（缠中说禅） | ⭐⭐⭐ | 分型→笔→中枢→背驰 | 全状态 | 📊 数据看板「🌀 缠论结构」 | `chan_analysis` |
| 海龟交易 | ⭐⭐ | 突破入场 + ATR 仓位 | 发酵期 | 🛠️ 策略回测「🐢 海龟仓位」 | `turtle_position_size` |
| 利弗莫尔关键点 | ⭐⭐⭐ | 关键点突破 + 站稳 MA60 | 发酵期 | 🛠️ 策略回测 | `get_decision_tree` |
| 格雷厄姆 | ⭐⭐ | 7 条防御标准 + 安全边际 | 低迷/冰点 | 📡 市场扫描「📗 格雷厄姆」 | `graham_screen` |
| 价值×动量（Asness） | ⭐⭐ | 「便宜且走强」四象限 | 发酵/启动 | 📡 市场扫描「⚖️ 价值动量」 | `value_momentum_screen` |
| 板块轮动 | ⭐⭐ | RS 动量排名 + 轮入/轮出 | 发酵/启动/高潮 | 📡 市场扫描「🔄 板块轮动」 | `predict_sector_rotation` |
| 游资情绪周期 | ⭐⭐ | 涨停梯队→四阶段判定 | 全状态 | 📡 市场扫描「🌡️ 情绪周期」 | `sentiment_cycle` |

### 分析方法（6）—— 全部纯确定性、免费

| 方法 | 难度 | 一句话 | UI 入口 | MCP 工具 |
|---|---|---|---|---|
| 信号事件研究 | ⭐ | 信号后 5/10/20 日收益 vs 基准 | 🧪 研究评估「📊 信号事件研究」 | `research_event_study` |
| Walk-Forward | ⭐⭐ | 15 folds 样本外检验 | 🧪 研究评估「运行 Walk-Forward」 | `research_walk_forward` |
| 过拟合诊断 | ⭐⭐ | DSR/PBO 揭穿回测图 | 🧪 研究评估「🎲 过拟合诊断」 | `backtest_overfit_diagnosis` |
| 因子扫描 | ⭐ | IC/分层检验 10 因子 | 📡 市场扫描「🧬 因子批量扫描」 | `factor_scan` |
| 因子中性化 | ⭐⭐⭐ | OLS 残差化剥离风格 | 🧪 研究评估「🧮 因子中性化」 | `factor_neutralize` |
| 全策略对比 | ⭐ | 全部策略同台事件研究 | 📚 研究总览「🏁 全策略对比」 | `strategy_compare` |

### LLM 能力（6）

| 能力 | 一句话 | 成本提示 | UI 入口 | MCP 工具 |
|---|---|---|---|---|
| AI 多维分析 | fast 单次 / deep 5 分析师并行 / debate 再+3 大师辩论 | fast≈8K；deep≈50K；debate≈50K+5 次调用 | 💬 AI 对话 | `agent_analyze_stock` |
| 市场预测 | 1d/5d/20d 方向/概率/区间，落盘可验证 | 单票单周期 1 次调用 | 🔮 市场预测「预测」 | `predict_market` |
| AI 研究报告 | 全部研究结果 LLM 综述 | ≈2-4K tokens/次 | 📚 研究总览「📋 AI 研报」 | `ai_market_report` |
| 经验库 RAG | 分析自动沉淀+回放相似案例 | 自动触发 | （自动） | `agent_experience_search` |
| 长期记忆 | 决策回填→胜率→贝叶斯权重→偏差检测 | 自动触发 | 🔮 市场预测「🧠 记忆状态」 | `agent_memory_status` |
| 预测→仓位 | 概率×命中率×校准→仓位（≤20%） | 1 次调用 | 🔮 市场预测「💼 仓位建议」 | `predict_position_size` |

## 三、四步走关系图

```
数据层（DuckDB / 4 源容灾行情）
        │
        ▼
① 看环境 ── 市场状态 / 游资情绪 / 市场结构 / 板块轮动（LLM 可选）
        │  环境状态作为一切推荐的前提
        ▼
② 选方法 ── 11 策略 × 8 大牛方法（难度 ⭐~⭐⭐⭐，按状态过滤）
        │  选中信号
        ▼
③ 做验证 ── 事件研究 / Walk-Forward / DSR·PBO / 因子中性化（纯确定性、免费）
        │  高置信信号
        ▼
④ 增强解读 ── AI 分析 / 预测 / AI 研报 / 经验库回放（LLM，需 key，有成本）
        │
        ▼
结论 + 仓位纪律（RiskGuard ≤20% 单票 / 回撤 10-12-15% 分级）
        └── 止损纪律优先于一切信号
```

## 四、新手四步流程

### ① 看环境（1 分钟）
**做什么**：判断今天是高潮/发酵/启动/低迷/冰点。
**为什么**：同一只股票，高潮期追高 vs 冰点期抄底，命运完全不同。
**怎么用**：
- UI：🔮 市场预测「🧭 市场诊断 + 策略路由」（LLM 优先、确定性兜底）→ 📡 市场扫描「🌡️ 游资情绪周期」「🧭 情绪×轮动矩阵」
- MCP：`agent_market_diagnosis` → `market_structure` → `sentiment_cycle` → `regime_matrix`

### ② 选方法（2 分钟）
**做什么**：按当前状态选策略/大牛方法（决策地图已按状态过滤好）。
**怎么用**：
- UI：📚 研究总览「🚀 一站式分析」（全流水线一键）→ 📡 市场扫描「🧬 CANSLIM」「📗 格雷厄姆」等
- MCP：`get_methodology_guide`（先读地图）→ `scan_market` → `one_click_analysis` → `strategy_compare`

### ③ 做验证（3 分钟）—— 免费，务必做
**做什么**：验证选中的信号过去靠不靠谱。
**怎么用**：
- UI：🧪 研究评估「📊 信号事件研究」「运行 Walk-Forward」「🎲 回测过拟合诊断」
- MCP：`research_event_study` → `research_walk_forward` → `backtest_overfit_diagnosis`
- **判定标准**：事件研究 `has_edge`（n≥20 且胜率>50% 且超额>0）+ Walk-Forward `is_promotable`（多数 fold 正收益）双达标才算"可信"，否则仅当实验。

### ④ 增强解读（可选，有成本）
**做什么**：让 AI 深研/预测/出研报。
**怎么用**：
- UI：💬 AI 对话（深度分析）→ 🔮 市场预测「预测」「💼 预测→仓位建议」→ 📚 研究总览「📋 AI 研报」
- MCP：`agent_analyze_stock` → `predict_market` → `predict_position_size` → `ai_market_report`
- **未配置 LLM 时**：此步跳过，前三步完全不受影响（系统自动降级确定性模式）。

## 五、MCP/Agent 开发者视角

新工具：**`get_methodology_guide(market_state="")`**（只读，92 工具之一）。

```jsonc
{
  "success": true,
  "data": {
    "market_state": "fermenting",          // 缺省自动检测，unknown=数据未就绪
    "market_state_zh": "发酵期",
    "llm_configured": false,               // LLM 是否可用
    "beginner_default": "bollinger_mean_reversion",
    "steps": [                             // 四步，每步含 recommended 条目
      { "step": 1, "title": "① 看环境", "goal": "...",
        "recommended": [ { "id": "method.sentiment_cycle", "name_zh": "...",
                           "difficulty": 2, "ui_entry": {...}, "mcp_tool": [...] } ] }
    ],
    "routing_consistency": {               // 两套策略路由对齐结果
      "overlap": [...], "guide_only": [...], "routing_only": [...]
    },
    "catalog": [ /* 当前状态可用的全部编目条目 */ ],
    "report": "## 🗺️ 新手决策地图 ..."     // markdown，可直接展示
  }
}
```

**推荐调用序列**（新手场景）：`pa_help`（看全貌）→ `get_methodology_guide`（读地图）→ `agent_market_diagnosis`（环境）→ `one_click_analysis`（全流水线）→ 按地图推荐用 `scan_market`/`scan_canslim` 等 → `research_event_study`（验证）→ `agent_analyze_stock(depth='fast')`（AI 解读）。

**Prompt 示例**：

```
先调用 get_methodology_guide() 获取当前市场状态与推荐方法，
再按地图的步骤依次分析：市场诊断 → 扫描推荐策略 → 事件研究验证 →
（若 llm_configured 为 true）深度分析并给出仓位建议（≤20%）。
所有结论标注数据来源，不编造。
```

## 六、LLM 成本建议

| 档位 | 调用量 | 典型场景 | 建议频率 |
|---|---|---|---|
| fast（`depth='fast'`） | ≈8K tokens/次 | 日常看个股 | 每票每天 1 次 |
| deep（`depth='deep'`） | ≈50K tokens/次 | 决策前深度研究 | 决策点/周 ≤3 票 |
| debate（`depth='debate'`） | 50K+5 次额外调用 | 重大分歧标的 | 罕见（默认关闭） |
| `ai_market_report` | ≈2-4K tokens/次 | 周度综述 | 每周 1 次 |
| `predict_market` | 1 次/票/周期 | 预测 1d/5d/20d | 关注标的 ≤5 只 |

省钱原则：**前三步全部免费**，LLM 只用在"需要解读与综合"的环节；未配置 key 时所有 LLM 入口自动降级确定性模式（预测→统计模式、研报→模板模式、诊断→规则模式），不会报错。

## 七、维护指南

编目数据在 `src/pa_mcp/research/methodology_guide.py` 的 `METHOD_CATALOG`：

- **新增/修改策略**：改 `src/pa_mcp/research/strategy_guide.py` 的 `STRATEGY_GUIDE`（难度/说明/风险/适用市场只维护这一处）；`METHOD_CATALOG` 里若已有该策略条目则无需动。
- **新增 UI 按钮 / MCP 工具 / LLM 能力**：三步走——
  1. `METHOD_CATALOG` 加一条（id 前缀 = 类别，`mcp_tool` 填真实工具名，`ui_entry.tab` 用真实 Tab 名，LLM 类必须带 `llm_cost_hint`）；
  2. `tests/test_methodology_guide.py` 的 `TOOL_SNAPSHOT`/`KNOWN_TABS` 补新名字；
  3. 跑 `python -m pytest tests/test_methodology_guide.py`。
- **两套路由表**：策略速查推荐（`STRATEGY_GUIDE.default_for`）与市场诊断路由（`orchestrator.MARKET_STATE_STRATEGY_ROUTING`）是**两个独立维护的推荐源**，决策地图的 `routing_consistency` 首次把它们对齐展示并标注差异（例如发酵期：诊断路由额外推荐 `macd_divergence_swing`，而策略速查未收录该策略——出现这类差异属正常，地图只是如实呈现）。若希望两者收敛，把策略加进 `STRATEGY_GUIDE` 即可。

## 八、与「小白使用指南」的关系

README 顶部「小白使用指南」讲的是**操作入口**（哪个按钮能直接点）；本文档讲的是**方法论体系**（为什么、按什么顺序、花不花钱）。UI 顶部「🧭 新手三步走」引导（理财操作）与「🗺️ 新手决策地图」（研究方法）互补：前者回答"今天该操作什么"，后者回答"怎么研究、怎么验证"。

---

*本系统为研究工具，不构成投资建议。过往表现不代表未来收益。详见 [DISCLAIMER.md](../DISCLAIMER.md)。*
