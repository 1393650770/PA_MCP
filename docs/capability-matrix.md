# PA_MCP 能力矩阵

自动生成于 2026-07-30。每项能力附真实入口、数据范围、测试覆盖、已知限制和归属阶段。

状态定义：
- `implemented_verified` — 生产实现并经过测试验证
- `implemented_unverified` — 代码实现但未经系统性测试
- `prototype` — 实验性实现，不建议用于可信研究
- `stub` — 声明但只有占位/空实现
- `planned` — 列入实施计划，尚未编码
- `unavailable` — 当前不做（当前阶段不纳入）

## 数据能力

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| 日K线数据 | prototype | `router.fetch_daily_kline()` → akshare/sina/tencent/eastmoney | 4源按序回退，前复权，无历史公司行为版本 | `tests/test_router.py` | 默认前复权会被未来分红重写; 免费API有3-15s延迟; Sina无成交额字段 |
| 多源容灾(路由+熔断) | implemented_verified | `data/router.py` DataSourceRouter | 顺序回退 + CircuitBreaker(CLOSED/OPEN/HALF_OPEN) | 22项路由/熔断/解析测试 | 无并行取最快; 冷却默认300s; 阈值默认3次 |
| 腾讯行情 | implemented_verified | `sources/tencent_adapter.py` | 日/周/月/分钟K线 + 实时估值快照(qt.gtimg.cn 50+字段: PE/PB/市值/换手/量比/涨跌停价) | 解析+代码映射+实时估值fixture测试; 真实网络验证33根日线 | 日期需YYYY-MM-DD格式; key为qfqday新结构; volume手→股×100; 无全市场快照 |
| 东方财富 | implemented_verified | `sources/eastmoney_adapter.py` | 日/周/月/分钟K线, 无key | 解析+secid映射测试; 真实网络验证 | 有风控会断连(实测); router限流1.2s; volume手→股×100; 北交所secid=0.920xxx |
| 新浪行情 | implemented_verified | `sources/sina_adapter.py` | 日K线(scale=240), volume为股 | 真实网络验证跨源一致(与腾讯误差<0.0001%) | 仅不复权数据; 无成交额字段; 北交所不支持(映射sh) |
| 百度股市通 | unavailable | — | 实测403(需cookie) | — | 暂不采用 |
| mootdx(通达信) | unavailable | — | 实测服务器探测失败(依赖国内IP) | — | 可在国内网络环境启用 |
| 分钟K线 | stub | `akshare_adapter.get_minute_kline()` / tencent / eastmoney | 仅当日/近期，免费API延迟 | 无 | 未接入调度; 无法历史回放 |
| 股票基本信息 | prototype | `router.fetch_realtime_spot_all()` → `stock_basic` | 当前全市场快照(AKShare/Sina) | 无 | 无历史版本; 退市股票缺失; 板块全部标为main |
| 交易日历 | implemented_verified | `scheduler._update_calendar()` → `trade_calendar` | 365天内置日历 | pipeline端到端 | 非交易所官方数据; 历史年份未覆盖 |
| 财务数据 | implemented_verified | `scheduler._update_financials()` (AKShare财务摘要宽表转置) | 每期营收/净利/EPS/净利率等, 最近8期/股 | 真实测试8期入库 | ROE指标名可能缺失; AKShare慢(0.3s/股) |
| 指数日线 | prototype | `akshare_adapter.get_index_daily()` | 上证/深证/沪深300 | 无 | 仅有价格，无总收益; 无历史成分 |
| 财务数据 | stub | `scheduler._update_financials()` | 无 | 无 | 生产代码为占位 |
| 资金流向 | stub | `scheduler._update_capital_flow()` | 无 | 无 | 生产代码为占位 |
| 龙虎榜 | prototype | `akshare_adapter.get_dragon_tiger_detail()` → `dragon_tiger` | 当日龙虎榜明细 | 无 | 数据约18:00后可用 |
| 跨源一致性校验 | prototype | `router.verify_consistency()` | 样本symbol双源close比对 | 无 | 需网络; 供调度器日终可选调用 |
| 公司行为(分红送转) | unavailable | — | — | — | 无数据源 |
| 证券历史状态(ST/停牌/退市) | unavailable | — | — | — | 无数据源 |
| 历史股票池/指数成分 | unavailable | — | — | — | 无数据源 |
| Point-in-time 查询 | planned | — | — | — | 阶段B |
| Dataset snapshot/复现 | planned | — | — | — | 阶段B |

## 策略与信号

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| 策略抽象 (BaseStrategy/Signal) | prototype | `engine/strategies/base.py` | 信号无市场时间，timestamp为创建时间 | 部分 | 需增加signal_time/available_at/snapshot_id |
| 策略注册表 (StrategyRegistry) | implemented_unverified | `engine/strategies/base.py:210-289` | auto_discover存在但server未调用 | 无 | 已修复(server lifespan)但待集成测试 |
| 趋势类策略 | prototype | `engine/strategies/trend.py` | MA交叉/平台突破 | 仅断言返回list | 信号时序待验证 |
| 涨停策略 | prototype | `engine/strategies/limit_up.py` | 统一用9.5%阈值 | 无 | 不区分板块涨跌幅 |
| 波段/反转/事件/网格策略 | prototype | `engine/strategies/swing.py, reversal.py, event_driven.py, grid.py` | 部分策略数据依赖未满足 | 无 | 价值/资金流/事件数据链为stub |
| 信号缓存 (signal_cache) | prototype | `signal_cache` DDL | 表存在但无写入路径 | 无 | precompute_signals配置无对应实现 |
| 市场扫描 (agent_scan_market) | prototype | `server.py:1023-1195` | 空缓存误判为快路径；倒序K线→策略rolling | 无 | 已修复(行数>0检查)但待集成测试 |

## 回测

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| 单票回测 (DuckDBBacktester) | prototype/untrusted | `backtest/engine.py` | 同收盘成交; 硬编码entry/exit; indicator_cache与kline字段不兼容 | 无端到端测试 | 不可用于可信策略评估 |
| A股规则辅助函数 | prototype | `backtest/rules.py` | 板块枚举、费用、整手、涨跌停价格 | 部分(费用/整手) | 未接入撮合; 滑点/涨跌停/停牌/流动性规则未生效 |
| 回测报告 | prototype | `backtest/engine.py:_compute_metrics()` | 声明的alpha/beta/calmar为空; 仅保留最近20笔交易/120净值点 | 无 | PnL与费用不守恒 |
| 事件驱动回测引擎 | planned | — | — | — | 阶段C |
| A股golden cases | planned | — | — | — | 阶段C |
| Prefix invariance测试 | planned | — | — | — | 阶段C |

## 组合与风险

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| RiskGuard | skeleton/unwired | `risk/guard.py` | 未接入回测/组合/订单/MCP | 仅验证新仓缩放 | 总仓位算法可能超限; 持仓与账户未同步 |
| 组合构建 | planned | — | — | — | 阶段E |
| 绩效归因 | planned | — | — | — | 阶段E |
| 多资产优化 | planned | — | — | — | 阶段E |

## 理财分析工具

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| 估值快照 | implemented_verified | `get_valuation_snapshot()` + `TencentAdapter.get_realtime_quote()` | 实时PE/PB/市值/换手/量比/涨跌停价/日内位置 | 真实网络验证 + fixture测试 | 免费数据3-15s延迟 |
| 持仓体检 | implemented_unverified | `agent_portfolio_review()` | 组合持仓 × 实时估值 × 集中度规则 | 无 | 依赖portfolio表有数据 |
| 财报分析 | implemented_unverified | `agent_earnings_analysis()` | 财务表(income/balance/cashflow) 关键指标+评分 | 无 | 依赖财务数据入库 |
| 专业prompts | implemented | `tools/prompts.py` | valuation-analysis / portfolio-health / earnings-interpretation / investment-memo | 9个prompts注册验证 | — |

## LLM/Agent

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| Agent 快速分析 (fast) | prototype | `orchestrator.fast_analyze()`, `agent_analyze_stock()` | 单LLM调用，K线压缩为文本摘要 + 经验库RAG注入 | 无 | 依赖LLM adapter |
| Agent 深度分析 (deep) | implemented_verified | `orchestrator.deep_analyze()` | **真实5分析师并行(技术/资金/情绪/基本面/事件) + 组合经理合成 + RiskGuard仓位上限**，借鉴ai-hedge-fund多agent模式 | `tests/test_orchestrator_two_stage.py` 11项(mock adapter) | 无LLM时降级确定性规则分析(不编造) |
| 两阶段分析 (诊断→路由→决策) | implemented_verified | `orchestrator.market_diagnosis()` / `analyze_with_diagnosis()`, MCP `agent_market_diagnosis`, UI「市场诊断」 | Stage1市场诊断(高潮/发酵/启动/低迷/冰点，LLM或确定性) → 策略路由(5状态→策略清单) → 注入5分析师prompt | 诊断/路由/注入11项测试 | 诊断依赖市场指标数据(涨停数/成交额) |
| JSON校验+重试 | implemented_verified | `orchestrator._chat_json_with_retry()` | 分析师/PM/诊断/预测输出字段校验，失败反馈LLM重试1次（借鉴PA_Agent validation_retry） | retry成功/放弃/error 3项测试 | 重试消耗额外token |
| 市场预测 (LLM未来走势) | implemented_verified | `agent/prediction.py` PredictionService, MCP `predict_market`, UI「市场预测」Tab | 确定性特征抽取(趋势/动量/波动/量能/周期位置) → LLM预测(方向+概率分布+期望收益+关键价位+多场景) → 落盘prediction_log → 到期回填验证 | `tests/test_prediction.py` 14项 + 真实行情冒烟(哈药股份→看涨0.85) | 无LLM降级确定性统计预测；预测可验证不等于保证收益 |
| 预测验证闭环 | implemented_verified | `agent/prediction.py evaluate_predictions()`, MCP `evaluate_predictions`, UI「预测验证成绩单」 | 回填到期预测真实收益：命中率/方向一致率/**Brier概率校准分数**/气候学基准/技能分/期望vs实际相关性 | 命中/未中/震荡阈值/Brier完美校准/Brier技能分/相关性测试 | 需K线数据覆盖预测期；全同类别样本时气候学基准=0技能分无定义 |
| Bull/Bear辩论+投资大师 | implemented_verified | `orchestrator._run_debate()` (TradingAgents风格), MCP `agent_analyze_stock(depth='debate')` | deep基础上：Bull论证(3论点+反驳预案) → Bear反驳(3论点+反驳bull+最大遗漏风险) → 投资大师裁定(方向/分数/仓位/证伪条件/风格)覆盖PM结论 | 裁定覆盖/失败保留PM/debate关闭零额外调用 3项测试 | 额外+3次LLM调用(成本)；默认关闭；RiskGuard 20%硬上限保留 |
| 经验库 (RAG) | implemented_verified | `agent/experience.py` ExperienceService, MCP `agent_experience_search`, 分析后自动落盘 | 每次AI分析自动沉淀(方向/周期/分数/风险) → 按符号/周期/方向检索top-N → 注入后续分析prompt → 事后回填5d/20d收益标记hit/miss（借鉴PA_Agent experience_reader） | `tests/test_experience.py` 5项 | 周期位置来自确定性特征(分析结果未含特征时标unknown) |
| Anthropic provider | broken → fixed | `llm_anthropic.py` (新增) | 官方SDK + Messages API | 无 | 原实现走/chat/completions shim; 新adapter修复但待测试 |
| OpenAI-compatible provider | prototype | `llm_openai_compat.py` (新增) | /chat/completions协议（doubao/ark等通用回退） | 无 | |
| 多provider端口 (LLMPort) | prototype | `llm_port.py` (新增) | 统一接口，支持adapter热切换(register None清除) | 无 | |
| 策略自动发现 | implemented_unwired | `base.py:auto_discover()` | 存在但server未调用→已修复 | 无 | 待server启动集成测试 |
| 盘前简报 (morning_brief) | prototype | `agent_morning_brief()` | 市场状态+涨停+资金+龙虎榜+信号 | 无 | |
| 多股票对比 | prototype | `agent_compare_stocks()` | 技术/资金/事件维度 | 无 | |

## 交易执行

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| Paper Broker | planned | — | — | — | 阶段F |
| OMS | planned | — | — | — | 阶段F |
| 实盘券商adapter | planned | — | — | — | 阶段G |
| 日终对账 | planned | — | — | — | 阶段F |

## 研究与实验

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| 横截面因子研究 | planned | — | — | — | 阶段D |
| Walk-forward/OOS | planned | — | — | — | 阶段D |
| Run Recorder/manifest | planned | — | — | — | 阶段D |
| 事件研究(公告→可交易) | planned | — | — | — | 阶段D |
| 风格基准(benchmark) | planned | — | — | — | 阶段D |
| 成本/容量敏感性 | planned | — | — | — | 阶段D |

## 工程基础

| 能力 | 状态 | 入口 | 测试 | 限制 |
|---|---|---|---|---|
| DuckDBStore (显式列映射) | fixed | `store.py` | 待测试 | 批次A.1修复，INSERT改显式列映射+类型校验 |
| 数据调度器 | fixed | `scheduler.py` | 待测试 | 批次A.2修复：__main__入口+PhaseStatus枚举+stub标记+全量游标 |
| MCP Server (FastMCP) | implemented_unverified | `server.py` | 无 | 76个legacy测试不覆盖核心链路 |
| Web UI (Gradio) | implemented_verified | `ui/gradio_app.py`, `pa-mcp-ui` 启动命令 | **9 Tab**（数据看板/AI对话/多股对比/市场扫描/研究评估/组合构建/策略回测/组合管理/**市场预测🔮**）+ 数据源健康面板 | 对话需 LLM key，无key时规则工具降级 |
| 市场预测UI | implemented_verified | `predict_market_ui()`/`prediction_history_ui()`/`evaluate_predictions_ui()`/`market_diagnosis_ui()` | 预测Tab：方向概率+区间+周期+场景+关键位；市场诊断+策略路由；预测验证成绩单；历史记录 | 真实行情冒烟通过 | 预测落盘prediction_log表 |
| MCP工具数 | implemented_verified | `server.py` | **49个工具**（新增predict_market/prediction_history/evaluate_predictions/agent_market_diagnosis/agent_experience_search） | UI功能与MCP工具对等 | — |
| 批量装载 | implemented_verified | `scheduler` + `scripts/bootstrap_universe.py` | 100只/2222行/19.5s/99%覆盖 + 断点续传0开销 | AKShare全市场快照当前网络不可达时用种子池 |
| 完整调度 | implemented_verified | `python -m pa_mcp.data.scheduler` | 30只种子池端到端113s，8-phase全部成功（日线660/分钟7200/财报240/龙虎榜49） | 需要种子池或AKShare快照 |
| Walk-Forward研究 | implemented_verified | `research/strategy_eval.py` + UI「研究评估」Tab | 15 folds真实OOS评估（长历史分页1000+根），晋级门槛=多数fold正收益 | 实测9策略×3股筛选0达标（真实结论：单票规则策略无稳定alpha） |
| 信号事件研究 | implemented_verified | `research/event_study.py` + UI「研究评估」Tab | 信号后5/10/20日收益 vs 无条件基准，3股×9策略独立评估 | **发现bollinger_mean_reversion 3/3股达标**（10日胜率最高63.6%）|
| 成本敏感性 | implemented_verified | `backtest/engine.py` slippage_bps + FeeSchedule倍数 | bollinger 3股×3成本档：1x→2x收益降<0.1pp（成本不敏感） | 引擎新增滑点参数；高价股预算修正（10%→25%现金兜底1手） |
| 参数优化 | implemented_verified | ParamRange网格 + 事件研究评分 | 5日超额+0.36%→+1.57%（bb_period=10,bb_std=3.0） | 教训：事件研究最优≠回测最优，需在目标函数上直接优化 |
| 参数优化v2 | implemented_verified | 完整回测目标函数 | 000001收益+8.13%→+9.90%（bb_period=15,bb_std=2.5） | 单票调优有过拟合风险（300750下降），需跨股/组合层调优 |
| 信号→组合管线 | implemented_verified | `portfolio/pipeline.py` + UI「组合构建」Tab | 多票信号→约束权重→组合净值合成（7 Tab UI实测） | 高价股被min_notional过滤；组合层面调优是下一步 |
| 组合回测引擎 | implemented_verified | `portfolio/backtest.py` PortfolioBacktestEngine | **多票共享现金/持仓账本**：延迟一天执行+等权分配+T+1+单票10%上限，NAV守恒实测（50万=现金21万+持仓29万） | 简化：每天每票一笔信号、期末不清仓 |
| 止盈止损 | implemented_verified | `portfolio/backtest.py` take_profit_pct/stop_loss_pct | 实测：回撤-25.25%→-17.93%（改善）但收益+0.38%→-6.93%（均值回归上止损割肉+交易成本） | 参数需按策略特性调优，不能默认启用 |
| LLM配置 | verified | `config/llm_config.example.json` | 模型ID已修正（claude-sonnet-5/claude-opus-5有效别名），base_url留空走SDK默认 | 复制为llm_config.json填key，或设ANTHROPIC_API_KEY |
| 跨股参数优化 | implemented_verified | 6股完整回测均值评分 | 跨股最优(15,2.0)均收益+20.80% vs 默认+20.61%（参数空间平坦=稳健） | 单票调优有过拟合风险，跨股平均更稳健 |
| 组合验证 | implemented_verified | 多票等权回测 | 默认参数组合均收益+23.63%（300750达+62.75%） | 10万账户分仓后高价股买不起（300750一手2万） |
| QMT接入 | skeleton | `execution/brokers/base.py` QMTBrokerAdapter | BrokerPort接口+幂等+风控ID校验 | 券商确认后实现xtquant连接 |
| 高价股边界 | implemented_verified | `backtest/engine.py` | 10%现金买不起100股（如茅台1600元）→ 跳过不交易，不报资金不足 | 修复前抛Insufficient cash异常 |
| CI (GitHub Actions) | prototype | `.github/workflows/ci.yml` | 数据健康检查仅验证spot快照数量 | 不验证PIT/复权/字段语义 |
| 配置文件 | prototype | `config/default.yaml`, `config/llm_config.example.json` | 无 | LLM示例model ID可能无效; Anthropic base_url建议留空 |

## 指标数量

实际运行时通过 `StrategyRegistry.list_all()` 和指标函数列表统计；README中不手写固定数字。
