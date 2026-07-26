# PA_MCP — 全栈 A 股量化 MCP 设计规格书

**日期:** 2026-07-26
**版本:** 1.0
**作者:** mx + 小鹿

---

## 1. 项目概述

### 1.1 目标

构建一个全栈 A 股量化交易 MCP Server，让 Claude 等 AI Agent 能够：
- 查询 A 股实时/历史行情数据
- 执行技术分析、基本面分析、情绪分析
- 运行 40+ 内置战法并输出交易信号
- 进行策略回测和参数优化
- 通过多 Agent 辩论机制做出投资决策
- 提供盯盘预警、每日复盘、持仓管理

### 1.2 核心原则

- **免费开源优先**：数据源用 AKShare + BaoStock + efinance，不依赖付费 API
- **全栈覆盖**：超短线打板、波段趋势、中长线价值三种风格都支持
- **Docker 化部署**：一键 docker-compose up
- **吸取 10+ 开源项目精华**：不重复造轮子，集百家之长

### 1.3 参考项目

| 项目 | 吸收要点 |
|------|---------|
| TradingAgents-astock (2.6k★) | 7 分析师 + 牛熊辩论 + A股约束 |
| aiagents-stock (1.7k★) | 龙虎榜团队 + 主力资金 + 板块轮动 |
| DeepPulse (34★) | 40 战法 + ReAct 引擎 + 长期记忆 + 熔断器 |
| Vibe-Research (1k★) | 牛熊辩论 + 事实呈现（不给买卖结论） |
| cn-financial-mcp (30★) | 42 工具全覆盖设计 |
| china-stock-mcp (44★) | 多源 fallback + 双传输模式 |
| StockAgent (353★) | 分布式微服务架构 |
| EasyQuant (52★) | 双 LLM 交叉审查因子挖掘 |
| QuantsPlaybook (5.7k★) | 100+ 策略库 |
| FinGPT (21k★) | 中文金融情感分析 |

---

## 2. 架构设计

### 2.1 分层架构

```
┌──────────────────────────────────────────────────┐
│  📡 MCP 接口层 (FastMCP + HTTP/SSE + stdio)       │
│  30+ Tools: 数据查询 | 技术分析 | 策略信号 |       │
│  回测 | 盯盘 | 选股 | 复盘 | 持仓管理              │
├──────────────────────────────────────────────────┤
│  🧠 AI Agent 决策层 (LangGraph)                    │
│  多分析师并行 → 牛熊对抗辩论 → 风控评审 → 最终决策  │
│  情绪周期判断 | 游资追踪 | 题材热度 | 政策影响      │
├──────────────────────────────────────────────────┤
│  ⚔️ 战法 + 策略引擎                                │
│  8 大类 40+ 内置战法: 首板/接力/低吸/龙头/         │
│  趋势/波段/价值/网格                               │
│  策略注册 + 参数优化 + 信号输出                     │
├──────────────────────────────────────────────────┤
│  ⏪ 回测引擎 (backtrader 封装)                      │
│  T+1 规则 | 涨跌停限制 | 印花税/佣金 | A股交易日历  │
│  向量化回测 + 蒙特卡洛参数优化                      │
├──────────────────────────────────────────────────┤
│  📊 技术分析引擎 (TA-Lib + pandas-ta)              │
│  150+ 指标 | 60+ K线形态识别 | 自定义因子           │
│  量价关系 | 筹码分布 | 支撑阻力                      │
├──────────────────────────────────────────────────┤
│  💾 数据层 (DuckDB + Redis 缓存)                   │
│  AKShare + BaoStock + efinance 多源聚合            │
│  每日 17:30 自动更新 | 数据质量校验 | 复权处理      │
│  龙虎榜 | 北向资金 | 融资融券 | 大宗交易            │
├──────────────────────────────────────────────────┤
│  📰 信息层                                         │
│  财联社/巨潮公告/互动易/雪球热帖/东方财富新闻       │
│  中文金融情感分析 (FinGPT 中国模型)                 │
│  公告解析 | 题材提取 | 事件驱动                     │
└──────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 组件 | 技术选型 | 理由 |
|------|---------|------|
| MCP 框架 | FastMCP (Python) | 高性能、双传输、社区活跃 |
| 数据源 | AKShare + BaoStock + efinance | 免费、覆盖广、多源 fallback |
| 数据库 | DuckDB | 嵌入式 OLAP、零配置、快 |
| 缓存 | Redis | 实时行情缓存、消息队列 |
| 技术指标 | TA-Lib + pandas-ta | 150+ 指标、K线形态识别 |
| 回测引擎 | backtrader | 事件驱动、A 股适配成熟 |
| Agent 框架 | LangGraph | 有向图编排、checkpoint、人机交互 |
| 情感分析 | FinGPT + bardsai/finance-sentiment-zh-base | 中文金融 NLP |
| 图表 | mplfinance + matplotlib | K 线图、技术指标叠加 |
| 部署 | Docker + docker-compose | 一键部署 |

### 2.3 项目目录结构

```
pa-mcp/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-07-26-pa-mcp-design.md
├── src/
│   └── pa_mcp/
│       ├── __init__.py
│       ├── server.py              # MCP Server 入口
│       ├── config.py              # 配置管理
│       ├── data/                  # 数据层
│       │   ├── __init__.py
│       │   ├── sources/           # 数据源适配器
│       │   │   ├── __init__.py
│       │   │   ├── akshare_adapter.py
│       │   │   ├── baostock_adapter.py
│       │   │   └── efinance_adapter.py
│       │   ├── aggregator.py      # 多源聚合 + fallback
│       │   ├── cache.py           # Redis 缓存
│       │   ├── store.py           # DuckDB 存储
│       │   └── scheduler.py       # 定时更新任务
│       ├── analysis/              # 分析引擎
│       │   ├── __init__.py
│       │   ├── technical.py       # 技术指标 (TA-Lib)
│       │   ├── fundamental.py     # 基本面分析
│       │   ├── sentiment.py       # 情感分析 (FinGPT)
│       │   ├── capital_flow.py    # 资金流向
│       │   └── patterns.py        # K线形态识别
│       ├── strategy/              # 策略引擎
│       │   ├── __init__.py
│       │   ├── base.py            # 策略基类
│       │   ├── registry.py        # 策略注册表
│       │   ├── signals.py         # 信号生成
│       │   └── strategies/        # 内置战法
│       │       ├── __init__.py
│       │       ├── limit_up.py    # 首板/接力/龙头
│       │       ├── trend.py       # 趋势跟踪
│       │       ├── swing.py       # 波段操作
│       │       ├── value.py       # 价值投资
│       │       ├── grid.py        # 网格交易
│       │       └── custom/        # 用户自定义策略
│       ├── backtest/              # 回测引擎
│       │   ├── __init__.py
│       │   ├── engine.py          # Backtrader 封装
│       │   ├── a_share_rules.py   # A股规则 (T+1/涨跌停/税费)
│       │   ├── calendar.py        # A股交易日历
│       │   └── optimization.py    # 参数优化 (蒙特卡洛/网格)
│       ├── agent/                 # AI Agent 决策层
│       │   ├── __init__.py
│       │   ├── graph.py           # LangGraph 状态图
│       │   ├── analysts.py        # 5 个专业分析师
│       │   ├── debate.py          # 牛熊辩论
│       │   ├── risk.py            # 风控评审
│       │   ├── market_state.py    # 市场情绪周期判断
│       │   └── memory.py          # 长期记忆 (决策记录追踪)
│       ├── info/                  # 信息层
│       │   ├── __init__.py
│       │   ├── news.py            # 新闻聚合 (财联社等)
│       │   ├── announcements.py   # 公告解析 (巨潮)
│       │   └── social.py          # 社交媒体 (雪球/互动易)
│       └── tools/                 # MCP 工具定义
│           ├── __init__.py
│           ├── market_data.py     # 行情数据工具
│           ├── screener.py        # 选股工具
│           ├── analysis_tools.py  # 分析工具
│           ├── review.py          # 复盘工具
│           ├── strategy_tools.py  # 策略工具
│           ├── alerts.py          # 盯盘预警工具
│           ├── agent_tools.py     # AI 决策工具
│           └── portfolio.py       # 持仓管理工具
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_analysis/
│   ├── test_strategy/
│   ├── test_backtest/
│   └── test_agent/
├── config/
│   ├── default.yaml               # 默认配置
│   └── strategies/                # 策略配置
├── scripts/
│   ├── init_db.py                 # 初始化数据库
│   ├── update_data.py             # 手动更新数据
│   └── run_backtest.py            # 命令行回测
└── docker/
    ├── Dockerfile
    ├── cron-scheduler/
    │   └── crontab
    └── redis/
        └── redis.conf
```

---

## 3. MCP 工具设计（30+ Tools）

### 3.1 行情数据 (Market Data)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `get_realtime_quote` | symbol, source | 实时五档盘口 |
| `get_kline` | symbol, period(daily/weekly/monthly/1m/5m/30m/60m), start, end, adjust(qfq/hfq/bfq) | K 线 OHLCV 数组 |
| `get_market_overview` | - | 指数/涨跌家数/成交额/北向 |
| `search_stock` | keyword | 匹配的股票列表 |
| `get_stock_info` | symbol | 公司基本信息 |

### 3.2 选股 (Screener)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `scan_limit_up` | date, board(main/chi_next/sci_tech) | 涨停股 + 封板强度/炸板/连板统计 |
| `scan_volume_surge` | ratio, market_cap_min | 放量异动股 |
| `scan_breakout` | period, breakout_type(resistance/platform/high) | 突破形态股 |
| `scan_continuous_limit_up` | min_days | 连板股分析 |
| `scan_hot_sector` | top_n | 当日热点板块 + 领涨股 |

### 3.3 分析 (Analysis)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `analyze_technical` | symbol, indicators[] | 多指标综合分析报告 |
| `analyze_fundamental` | symbol | 财报数据/估值/杜邦分析 |
| `analyze_sentiment` | symbol/daterange | 新闻情感得分/舆情趋势 |
| `analyze_capital_flow` | symbol/category | 主力/散户/北向资金流向 |
| `compare_stocks` | symbols[], dimensions[] | 多股横向对比 |
| `analyze_chart_pattern` | symbol, period | K 线形态识别 (头肩顶/双底/三角形等) |

### 3.4 复盘 (Review)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `review_daily_limit_up` | date | 涨停复盘: 封板时间/炸板率/题材分布/次日溢价统计 |
| `review_dragon_tiger` | date | 龙虎榜: 游资席位识别/买卖力量/知名游资动向 |
| `review_sector_rotation` | days | 板块轮动分析 + 持续性判断 |
| `review_market_sentiment` | date | 市场情绪: 涨跌比/炸板率/连板高度/成交量 |

### 3.5 策略 (Strategy)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `list_strategies` | category | 可用策略列表 + 描述 |
| `run_strategy` | strategy_name, symbol/scope | 策略信号: 买/卖/持有 + 置信度 + 逻辑 |
| `backtest_strategy` | strategy_name, symbol, start, end, capital | 回测报告: 收益/回撤/夏普/胜率/交易记录 |
| `optimize_strategy` | strategy_name, param_ranges, symbol | 最优参数 + 参数敏感性分析 |
| `create_custom_strategy` | name, rules_json | 注册自定义策略 |

### 3.6 盯盘预警 (Alerts)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `watch_price_alert` | symbol, condition(above/below/cross), price | 创建价格预警 |
| `watch_volume_alert` | symbol, volume_ratio | 创建量能预警 |
| `watch_limit_up_alert` | symbol | 涨停预警 |
| `list_alerts` | status(active/triggered) | 预警列表 |
| `remove_alert` | alert_id | 删除预警 |

### 3.7 AI 决策 (Agent)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `agent_analyze_stock` | symbol, dimensions[] | 多分析师综合评估报告 |
| `agent_scan_market` | strategy_preference, risk_level | AI 全市场扫描 + 推荐列表 |
| `agent_debate` | symbol, thesis | 牛熊辩论过程 + 结论 |
| `agent_market_state` | - | 当前市场情绪周期判断 + 仓位建议 |
| `agent_morning_brief` | - | 盘前简报: 隔夜消息/外盘/今日关注 |

### 3.8 持仓管理 (Portfolio)

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `portfolio_summary` | - | 持仓概况 + 盈亏 + 风险指标 |
| `portfolio_risk` | - | VaR/CVaR/最大回撤/相关性矩阵 |
| `portfolio_rebalance` | target_weights | 再平衡建议 |
| `portfolio_add` | symbol, cost, shares, date | 添加持仓 |
| `portfolio_remove` | holding_id | 删除持仓 |

---

## 4. Agent 决策流程

### 4.1 市场状态机

```
市场状态: 高潮期 → 发酵期 → 启动期 → 低迷期 → 冰点期
         ↑________________________________________↓
                    (周期循环)

判断指标:
- 涨停家数/炸板率/连板高度
- 成交量/融资余额变化
- 北向资金方向
- 涨跌比/跌停家数
- 指数位置 (均线乖离率)
```

### 4.2 多分析师并行

```
Input → 并行分析:
         ├── 技术分析师: K 线形态 + 多指标共振 + 量价关系
         ├── 资金分析师: 主力净买 + 龙虎榜席位 + 北向持仓 + 融资融券
         ├── 情绪分析师: 新闻情感 + 社交媒体热度 + 涨停板情绪 + 研报评级
         ├── 基本面分析师: 财报质量 + 估值分位 + 行业对比 + 成长性
         └── 政策分析师: 政策催化 + 监管风险 + 解禁压力 + 行业周期
       → 分析师报告集合
```

### 4.3 牛熊辩论

```
分析师报告 → Bull Agent: 构建看多论据 + 涨幅预估 + 催化剂
           → Bear Agent: 构建看空论据 + 风险点 + 反向逻辑
           → 交叉质询 (Cross-examination): 互相质疑关键假设
           → Moderator: 综合 → 共识点 + 真实分歧 + 待验证事项
```

### 4.4 风控评审

```
辩论结论 → 三方评审:
            ├── 激进评审: 最大仓位 + 追高逻辑
            ├── 保守评审: 安全边际 + 止损位
            └── 中性评审: 概率加权 + 凯利公式
          → 风控主管: 融合 → 最终信号 + 仓位建议 + 止损止盈
```

### 4.5 长期记忆

```
每次决策记录:
  - 标的 + 日期 + 信号 + 置信度 + 逻辑链
  - 定期回顾: 胜率追踪 / 偏差分析 / 策略排名
  - 反馈学习: 胜率低的信号降权 / 高胜率模式加权重
```

---

## 5. 数据流

### 5.1 日常数据更新流程

```
17:30 (收盘后) → Cron 触发
  → 1. AKShare 拉取全市场日线数据
  → 2. BaoStock 拉取财务数据 (增量)
  → 3. efinance 拉取资金流向 + 龙虎榜
  → 4. 数据校验 (行数对齐/字段完整/去重)
  → 5. 写入 DuckDB + 更新 Redis 缓存
  → 6. 技术指标批量计算 (全市场预计算)
  → 7. 情感分析预计算 (当日新闻/公告)
```

### 5.2 实时查询流程

```
MCP Tool Call → Redis 检查缓存 (实时行情 3s TTL)
              → 缓存命中 → 直接返回
              → 缓存未命中 → akshare 实时接口
                           → 失败？→ efinance fallback
                           → 失败？→ 新浪 fallback
              → 写入 Redis → 返回 + 标注数据源
```

### 5.3 Agent 分析流程

```
agent_analyze_stock(symbol="000001") →
  1. 市场状态判断 (从 Redis 读取当日预计算)
  2. 并行拉取: K线 + 财报 + 新闻 + 资金 + 龙虎榜
  3. 5 分析师并行出报告
  4. 牛熊辩论
  5. 风控评审
  6. 输出: 结构化分析报告 + 信号 + 置信度
```

---

## 6. 策略引擎

### 6.1 策略基类

```python
class BaseStrategy:
    name: str
    category: str  # limit_up/trend/swing/value/grid
    description: str
    timeframe: str  # intraday/daily/weekly/monthly
    params: dict    # 可优化参数

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]
    def get_params_space(self) -> dict  # 参数搜索空间
    def validate(self) -> bool
```

### 6.2 八大战法分类

| 类别 | 战法 | 参考来源 |
|------|------|---------|
| 🔥 首板 | 首板突破 / 低位首板 / 新题材首板 | DeepPulse + QuantsPlaybook |
| 🔗 接力 | 二板定龙 / 弱转强 / 分歧转一致 | 游资战法 |
| ⭐ 龙头 | 龙头首阴 / 龙头二波 / 龙回头 | 龙头战法 |
| 📉 低吸 | 缩量回踩 / 均线低吸 / 恐慌低吸 | 低吸战法 |
| 📈 趋势 | 均线多头 / 平台突破 / 杯柄形态 | 趋势跟踪 |
| 🌊 波段 | MACD 金叉 / 布林带 / 波浪理论 | 波段操作 |
| 💎 价值 | 低 PE 高分位 / ROE 因子 / 股息率 | 价值投资 |
| 🕸️ 网格 | 震荡网格 / 趋势网格 / 动态网格 | ash-mcp |

### 6.3 策略信号格式

```json
{
  "symbol": "000001",
  "name": "平安银行",
  "strategy": "平台突破战法",
  "signal": "buy",
  "confidence": 0.78,
  "entry_price": 12.50,
  "stop_loss": 11.80,
  "take_profit": [13.80, 15.00],
  "reasoning": [
    "平台整理 21 天，缩量极致",
    "今日放量突破平台高点 12.45",
    "MACD 零轴上方金叉",
    "所属银行板块近 5 日资金净流入"
  ],
  "risk_factors": [
    "大盘处于低迷期，系统性风险较高",
    "上方 13.80 有密集套牢区"
  ],
  "timestamp": "2026-07-26T14:30:00+08:00"
}
```

---

## 7. 回测引擎

### 7.1 A 股特有规则

- **T+1**: 当日买入次日才能卖出
- **涨跌停**: 主板 ±10%、创业板/科创板 ±20%、北交所 ±30%
- **交易费用**: 佣金 0.025%（最低 5 元）+ 印花税 0.05%（卖出单边）+ 过户费 0.001%
- **最小交易单位**: 100 股（1 手）
- **停牌处理**: 停牌期间跳过，复牌首日特殊处理

### 7.2 回测指标

- 累计收益率 / 年化收益率
- 夏普比率 / 索提诺比率 / 卡玛比率
- 最大回撤 / 回撤持续时间
- 胜率 / 盈亏比 / 平均持仓天数
- Alpha / Beta / 信息比率
- 每月收益分布 / 连续亏损次数

---

## 8. 部署架构

### 8.1 Docker Compose

```yaml
services:
  pa-mcp-server:
    build: .
    ports:
      - "8080:8080"    # HTTP/SSE
    environment:
      - PA_MCP_TRANSPORT=http
      - REDIS_URL=redis://redis:6379
      - DUCKDB_PATH=/data/pa_mcp.duckdb
    volumes:
      - pa_mcp_data:/data
      - ./config:/app/config
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  cron-scheduler:
    build:
      context: .
      dockerfile: docker/cron-scheduler/Dockerfile
    volumes:
      - pa_mcp_data:/data
    depends_on:
      - redis
      - pa-mcp-server

volumes:
  pa_mcp_data:
  redis_data:
```

---

## 9. 开发阶段

### Phase 1: 数据基础 (Week 1)
- 数据源适配器 (AKShare + BaoStock + efinance)
- DuckDB 存储 + 表结构设计
- Redis 缓存层
- 定时更新任务
- 5 个行情数据 MCP Tools

### Phase 2: 分析引擎 (Week 2)
- TA-Lib + pandas-ta 技术指标
- 基本面分析模块
- 情感分析模块 (FinGPT)
- K 线形态识别
- 选股 + 复盘 + 分析 MCP Tools

### Phase 3: 策略 + 回测 (Week 2-3)
- 策略基类 + 注册表
- 8 大类 40+ 战法实现
- backtrader 回测封装 (A 股规则)
- 蒙特卡洛参数优化
- 策略 + 回测 MCP Tools

### Phase 4: Agent 决策 (Week 3)
- LangGraph 状态图
- 5 个专业分析师
- 牛熊辩论 + 风控评审
- 市场状态判断
- 长期记忆系统
- Agent MCP Tools

### Phase 5: 部署 + 交付 (Week 3-4)
- Docker 打包 + docker-compose
- 盯盘预警系统
- 持仓管理
- 完整测试套件
- 文档 + README

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 免费数据源不稳定 | 三源 fallback + 数据质量校验 + 异常告警 |
| LLM 幻觉导致错误决策 | 牛熊辩论 + 风控评审 + 长期记忆追踪胜率 |
| AKShare 接口变动 | 适配器模式隔离 + 版本锁定 + CI 每日构建验证 |
| 回测过拟合 | 蒙特卡洛验证 + 样本外测试 + 参数敏感性分析 |
| 性能瓶颈（全市场扫描） | DuckDB OLAP + Redis 预计算 + 分批处理 |

---

## 11. 成功指标

- ✅ 30+ MCP Tools 全部可调用
- ✅ 40+ 战法可运行并输出信号
- ✅ 回测引擎正确模拟 T+1/涨跌停/税费
- ✅ Agent 分析一只股票 < 60 秒
- ✅ docker-compose up 一键启动
- ✅ 所有数据源 fallback 测试通过
- ✅ 每日自动更新数据成功率 > 95%
