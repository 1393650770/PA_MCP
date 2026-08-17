# PA_MCP — A 股量化理财助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-116%2F116-brightgreen.svg)]()

A 股量化研究 + 理财分析一体化系统：**多源数据容灾 · 策略研究闭环 · 组合回测 · 理财 Agent 界面**。

- **Web UI（Gradio）**：数据看板 / AI 对话 / 多股对比 / 研究评估 / 组合构建 / 策略回测 / 持仓管理 / **🔮 市场预测**
- **MCP Server**：81 工具供 Claude 等 Agent 调用（269 项测试全绿）
- **数据管线**：8-phase 全自动调度 + 4 源容灾（熔断/缓存/限流）
- **研究层**：事件研究 / Walk-Forward OOS / 参数优化 / 成本敏感性

> ⚠️ **免责声明**：本系统为研究工具，非投资建议。免费行情可能有延迟。过往表现不代表未来收益。详见 [DISCLAIMER.md](DISCLAIMER.md)。

---

## 🚀 快速启动（Windows）

### 一键启动 Web UI

```bat
双击 start_ui.bat
```

浏览器自动打开 **http://127.0.0.1:7860**。首次启动前需安装依赖：

```powershell
# 安装
pip install -e ".[dev]"

# 配置 LLM（可选，不配则对话走规则分析模式）
copy config\llm_config.example.json config\llm_config.json
# 编辑 config\llm_config.json 填入 API key；或用环境变量 ANTHROPIC_API_KEY
```

### 其他启动方式

```bash
# Web UI
python -m pa_mcp.ui.gradio_app        # 或 pa-mcp-ui

# MCP Server（供 Claude Code / OpenClaw 等 Agent 调用）
python -m pa_mcp.server               # stdio 模式
# 或 start.bat（stdio/http 二选一）

# 数据调度（增量续传；--full 全量重跑）
python -m pa_mcp.data.scheduler
python -m pa_mcp.data.scheduler --full

# 引导种子股票池（AKShare 不可用时）
python scripts/bootstrap_universe.py --limit 100
```

---

## 🧭 小白使用指南（按钮能不能直接用？）

**🟢 直接可用**（点就出结果——缺数据自动补，无需任何配置）：
- 📚 研究总览 Tab：**🚀 一站式分析**（推荐第一个点，全市场到持仓一条龙）
  · 🏁 全策略对比 · 🧬 因子扫描 · 🎯 因子选股 · ⚖️ 价值动量
  · 📗 格雷厄姆 · 🏛️ 市场结构 · 📊 数据体检
- 📡 市场扫描 Tab：🔄 板块轮动预测 · 🧬 CANSLIM · 🌡️ 游资情绪周期
  · 🧭 情绪×轮动矩阵 · 📋 AI 研报
- 🔮 市场预测 Tab：预测（1d/5d/20d）· 🎯 共振 · 🧮 综合信号 · 💼 仓位建议
- 🛡️ 持仓风险面板/图（无持仓时提示添加，不会崩）

**🟡 需先做一件事**（添加自选/持仓后更有意义）：
- 💼 组合管理 Tab：先点「添加持仓」再分析（无持仓时面板会提示）
- 📡 市场扫描 Tab：扫描会自动包含你的持仓（📌 标记）

**🔴 需要 LLM 配置**（不配也能用，但少 AI 解读）：
- 💬 AI 对话 · 📋 AI 研报（LLM 综述段）· 🤖 因子+AI 预测融合
- 配置方法：复制 `config/llm_config.example.json` 为 `llm_config.json` 填 key

> 全部 57 个按钮均经过自动化验证（`scripts/verify_ui.py`），
> 网络/数据缺失时自动降级提示，不会崩溃。

## 🔬 接口验证

所有 MCP 工具与 Agent 接口均有自动化验证脚本（`scripts/verify_interfaces.py`，双模式）：

```bash
# 隔离模式：全部 89 工具（临时种子库 + LLM/网络隔离，不碰真实数据）
venv\Scripts\python.exe scripts/verify_interfaces.py
# 真实 LLM 模式：16 个 LLM 相关工具（真实豆包 API，验证 LLM 全链路）
venv\Scripts\python.exe scripts/verify_interfaces.py --llm
```

**当前状态**：隔离模式 ✅ 85 / ⚠️ 4（纯网络依赖）/ ❌ 0；真实 LLM 模式 ✅ 16/16；Agent 服务 5/5。
验证曾发现并修复 7 个真实 bug（prompt 花括号/持仓 id/自选股约束/调度器构造/事件研究嵌套 asyncio/fallback 参数/logging 混用）。

## 🗺️ 界面总览（10 Tab）

| Tab | 功能 |
|---|---|
| 📊 数据看板 | K线+MA+成交量、主力资金流、实时估值、财报、龙虎榜、缠论结构图、背驰事件研究 |
| 💬 AI 对话 | LLM 或规则工具模式（资金流/体检/回测/对比） |
| 🔀 多股对比 | 归一化走势 + 估值对比表（2-5 只） |
| 📡 市场扫描 | 策略信号扫描 + CANSLIM 成长股 + 板块轮动预测 + 游资情绪周期 + 情绪×轮动矩阵 + 因子选股（+AI 融合）+ 因子组合回测 + 全策略对比 + 格雷厄姆 + 价值动量 + 敏感性 + 背驰组合回测 + AI 研报 |
| 🧪 研究评估 | Walk-Forward OOS + 信号事件研究（含板块基准）+ 回测过拟合诊断（DSR/PBO）+ 因子中性化 |
| 📦 组合构建 | 多票信号 → 共享账本组合回测（真实现金约束 + 止盈止损） |
| 🛠️ 策略回测 | 10 策略 + 沪深300 基准对比 + 海龟 ATR 仓位计算 |
| 💼 组合管理 | 持仓增删查 + AI 分析（含预测与市场综述）+ 策略信号 |
| 🔮 市场预测 | 1d/5d/20d 预测 + 批量对比 + 仓位建议 + 决策树（含指数结构）+ 验证成绩单（命中率/Brier/IC/校准/周期分组） |
| 📚 研究总览 | 共享股票池一键访问全部研究工具（研报/对比/选股/价值动量/格雷厄姆/市场结构/持仓风险/情绪） |

---

## 🏗️ 架构

```
┌─ Web UI (Gradio 10 Tab) ────────────────────────┐
│  看板│对话│对比│扫描│研究│构建│回测│持仓│预测│总览│
├─────────────────────────────────────────────────┤
│  MCP Server (81 tools)                          │
├─────────────────────────────────────────────────┤
│  Agent: LLMPort (Anthropic 官方 SDK / 多供应商)  │
├─────────────────────────────────────────────────┤
│  研究层: 事件研究 + Walk-Forward + 参数优化      │
│  组合层: 共享账本回测 + 止盈止损 + 约束权重      │
│  回测层: 事件驱动 + A股撮合(涨跌停/T+1/整手)     │
├─────────────────────────────────────────────────┤
│  数据层: DataSourceRouter 多源容灾               │
│    akshare/tencent/sina/eastmoney               │
│    熔断 + 两级缓存 + 东财限流 + 北交所920        │
│  8-phase 管线: 日历→股票池→日线→分钟→财报→       │
│    资金流→龙虎榜→指标 (断点续传)                │
└─────────────────────────────────────────────────┘
```

---

## ✨ 核心能力

### 数据（多源容灾）

- **4 源链**：akshare（全市场快照）→ tencent（最快最稳）→ sina（备用）→ eastmoney（限流补充）
- **熔断器**：连续失败 3 次 → 冷却 300s → 半开探测；失败源自动跳过
- **两级缓存**：内存 TTL 5min + DuckDB 持久化，降低免费 API 请求量
- **8-phase 管线**：日线/分钟线（腾讯 mkline 640 根）/财报（AKShare 宽表转置）/资金流/龙虎榜全部真实实现，断点续传 + 覆盖率统计
- **北交所 920 新号段**支持；种子股票池兜底

### 研究（防伪 alpha）

- **信号事件研究**：信号后 5/10/20 日收益 vs 无条件基准 → 预测力判定
- **Walk-Forward OOS**：15 folds 真实数据评估，晋级门槛 = 多数 fold 正收益
- **参数优化**：ParamRange 网格（完整回测目标函数）+ 跨股平均（防单票过拟合）
- **成本敏感性**：1x/1.5x/2x 成本档
- **实测结论**：`bollinger_mean_reversion` 通过多证据检验（3/3 股事件研究达标 + 成本不敏感 + 组合正收益）

### 组合与回测

- **共享账本组合回测**：多票联合执行、真实现金约束、T+1、单票上限、止盈止损
- **事件驱动引擎**：信号延迟一天执行（无未来函数）、A股撮合（涨跌停/整手/费用）
- **9 策略**全部真实信号（P0-6 修复：signal_time 市场时间）

### Agent / 界面

- **LLM Ports & Adapters**：Anthropic 官方 SDK（Messages API）、OpenAI-compatible（doubao/ark 通用回退）、DeepSeek、智谱、通义
- **多 Agent 深度分析**（借鉴 ai-hedge-fund）：5 分析师并行（技术/资金/情绪/基本面/事件）+ 组合经理合成 + RiskGuard 仓位上限
- **Bull/Bear 辩论**（借鉴 TradingAgents）：多头论证 → 空头反驳 → **投资大师团**（格雷厄姆价值/索罗斯反身性/利弗莫尔趋势 3 位并行 + 置信加权合议），`depth='debate'` 可开关
- **长期记忆**：分析自动记录决策 → 收益回填 → 胜率/盈亏比 → 贝叶斯策略权重 → 认知偏差检测（过度自信/处置效应）
- **选股与研究体系**：
  - 🧬 **因子工厂**：10 因子注册表（MA/RSI/MACD/ATR/布林/量比/动量/新高/缠论背驰）→ IC/分层检验 → 多因子选股（+AI 预测融合，权重可调）→ 敏感性分析
  - ⚖️ **价值×动量复合**（Asness 2013）：格雷厄姆评分 × 60 日动量 → 四象限（「便宜且走强」最佳）
  - 📗 **格雷厄姆筛选**：防御性 7 条 + 成长公式内在价值/安全边际
  - 🏁 **全策略对比**：10 策略同台事件研究（有效判定可追溯）
  - 📋 **AI 市场研究报告**：全部研究结果 LLM 综述（总结/关注/风险/思路，不编造）
- **大牛方法库**：
  - 🧬 **CANSLIM 成长股扫描**（欧奈尔《笑傲股市》）：C 当季盈利 / A 年度增长 / N 新高 / S 突破放量 / L 领军股 / M 市场方向（I 机构数据暂缺）——确定性规则，无 LLM 依赖
  - 🌀 **缠论结构分析**（缠中说禅）：K 线包含合并 → 分型 → 笔 → 中枢（3 笔重叠区间）→ 背驰（MACD 面积对比，涨/跌动能衰竭信号）
  - 🐢 **海龟交易策略**（Richard Dennis 海龟实验）：唐奇安 20 日突破入场 + ATR 波动率目标仓位（1 单位 = 账户×1%÷ATR）+ 10 日通道离场参考，自动注册进回测/事件研究/扫描
  - 🎯 **利弗莫尔关键点策略**（《股票大作手回忆录》）：枢轴关键点突破（最小阻力线）+ 站稳 MA60（只在上升趋势做多）+ 放量确认（无量突破视为假突破硬过滤），跌破枢轴低点离场参考
  - 🔄 **板块轮动预测**：东财板块 RS 排名（几何动量）+ 加速检测 + 轮入/轮出信号 + **LLM 预测未来一周强势板块**（无 LLM 动量延续降级），落盘 5 日回填验证 top3 超额收益；**板块领涨股挖掘** + **个股预测自动注入板块强弱上下文**（板块轮动→个股闭环）
  - 💼 **预测→仓位建议**（借鉴 ai-hedge-fund Risk Manager）：预测概率 × 同方向历史命中率 × 概率桶校准 → 建议仓位（≤20% 硬上限），推导链可追溯
  - 🌡️ **游资情绪周期**：涨停梯队（首板/2板/3板/4板+）+ 连板高度 + 晋级率 + 情绪分 + 四阶段判定（启动/发酵/高潮/退潮+冰点）+ 退潮预警（游资龙头战法核心）
- **两阶段分析**：市场诊断（高潮/发酵/启动/低迷/冰点）→ 策略路由 → 决策；JSON 校验失败自动重试
- **🔮 市场预测**（借鉴 PA_Agent 未来走势预期）：周期位置 + 方向概率分布 + 期望收益区间 + 多场景推演；预测落盘可验证（命中率/Brier 概率校准/技能分/**IC 信息系数与 ICIR**（Qlib 标准）/概率校准分桶检测过度自信/**AI vs 统计模式对比**）——预测可检验，非算命
- **经验库（RAG）**（借鉴 PA_Agent 经验库）：每次分析自动沉淀，后续分析自动参考相似历史案例（含事后验证）
- **理财专业 prompts**：估值分析 / 持仓体检 / 财报解读 / 投资备忘录
- **QMT BrokerPort 骨架**：券商确认后接入实盘（风控 ID 不可绕过、订单幂等）

---

## 📁 项目结构

```
config/               # 配置（llm_config.json 已 gitignore）
src/pa_mcp/
  data/               # 数据层（router 多源容灾 / scheduler 8-phase / store）
  backtest/           # 回测（事件驱动引擎 / broker A股撮合 / ledger 账本）
  research/           # 研究（event_study / strategy_eval walk-forward）
  portfolio/          # 组合（backtest 共享账本 / pipeline 信号→组合）
  engine/             # 策略（13 策略注册 / 指标 / 市场状态）
  risk/               # 风控（RiskGuard 纯函数 + 回撤分级）
  agent/              # LLM（llm_port / anthropic / orchestrator / prediction / experience）
  execution/          # 交易（brokers 端口 + QMT 骨架）
  ui/                 # Web 界面（gradio_app）
  server.py           # MCP Server（49 工具）
docs/capability-matrix.md   # 能力矩阵（真实状态）
docs/mcp-tools.md           # MCP 工具全览（97 个，与 UI 功能对等）
docs/methodology-guide.md   # 新手方法论地图（四步走 + 四类资产编目）
docs/openclaw-guide.md      # OpenClaw 新手使用指南（MCP 配置 + 对话模板 + 定时任务）
scripts/              # 工具脚本（bootstrap_universe 等）
tests/                # 146 项测试
```

---

## 🧪 测试

```bash
python -m pytest tests/ -v    # 116 passed
```

## 📜 License

MIT — 详见 [LICENSE](LICENSE)。本系统不提供任何投资建议。
