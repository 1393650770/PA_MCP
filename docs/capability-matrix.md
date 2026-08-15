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
| 预测验证闭环 | implemented_verified | `agent/prediction.py evaluate_predictions()`, MCP `evaluate_predictions`, UI「预测验证成绩单」 | 回填到期预测真实收益：命中率/方向一致率/**Brier概率校准分数**/气候学基准/技能分/期望vs实际相关性/**IC信息系数(Spearman, Qlib标准)**/**ICIR滚动稳定性**/**概率校准分桶(过度自信检测)**/**AI vs 统计模式对比** | 命中/未中/震荡阈值/Brier校准/IC-ICIR/校准分桶/模式对比测试 | 需K线数据覆盖预测期；全同类别样本时气候学基准=0技能分无定义；ICIR需≥2个周窗口 |
| Bull/Bear辩论+投资大师团 | implemented_verified | `orchestrator._run_debate()` (TradingAgents风格), MCP `agent_analyze_stock(depth='debate')` | deep基础上：Bull论证(3论点+反驳预案) → Bear反驳(3论点+反驳bull+最大遗漏风险) → **3位大师并行独立判断**（格雷厄姆价值派/索罗斯反身性派/利弗莫尔趋势派）→ 置信加权投票合议覆盖PM结论 | 裁定覆盖/加权投票/大师团全失败保留PM/关闭零额外调用 4项测试 | 额外+5次LLM调用(成本)；默认关闭；RiskGuard 20%硬上限保留 |
| 长期记忆（决策回放） | implemented_verified | `agent/memory.py` LongTermMemory, MCP `agent_memory_status`, UI「长期记忆状态」 | 分析自动记录决策(SQLite) → 5日收益回填 → 胜率/盈亏比 → 贝叶斯策略权重 → 偏差检测(过度自信/处置效应)；db路径已改绝对(PROJECT_ROOT) | 记录/回填/权重/偏差/绝对路径 4项测试 | 回填需≥5天；偏差阈值固定(60天/5次) |
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

## 大牛方法

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| CANSLIM 成长股扫描 | implemented_verified | `research/canslim.py` CanslimScanner, MCP `scan_canslim`, UI「市场扫描」Tab | 欧奈尔七要素：C净利同比≥20% / A 4期均值≥25%或ROE≥17% / N 250日新高 / S突破放量≥1.5 / L池内RS前20% / M市场状态(冰点否决)；I机构数据暂缺不评分 | 要素判定/冰点否决/池扫描排序/空库 4项 | 财务数据需调度器装载；RS 为池内相对排名 |
| 缠论结构分析 | implemented_verified | `engine/indicators/chan.py`, MCP `chan_analysis`, UI「数据看板」Tab | 缠中说禅体系：K线包含合并→分型(顶/底)→笔(交替连接)→中枢(3笔重叠区间)→背驰(MACD面积对比，上涨/下跌衰竭信号)+价位相对中枢位置；**scan_beichi_signals 滑动窗口扫描 + 事件研究验证预测力**（MCP `chan_beichi_event_study`，UI「背驰信号事件研究」） | 合并/分型/笔交替/中枢重叠/端到端/背驰扫描/事件研究 7项 | 简化实现（非完整缠论；背驰用笔级近似）；真实行情冒烟通过 |
| 海龟交易策略 | implemented_verified | `engine/strategies/turtle.py`, MCP `turtle_position_size`, UI「策略回测」Tab | 唐奇安20日突破入场/10日通道离场参考 + ATR波动率目标仓位（1单位=账户×1%÷ATR，≤10%上限）+ 突破强度评分；自动注册进回测/事件研究/市场扫描 | 突破信号/横盘无信号/参数空间/自动注册/时间排序 5项 | 真实行情冒烟（601728→13信号）；A股不做空，仅做多信号 |
| 利弗莫尔关键点策略 | implemented_verified | `engine/strategies/livermore.py`, UI「策略回测」Tab | 枢轴关键点突破（前20日高点，shift(1) 无未来函数）+ 站稳MA60（只在上升趋势做多，不抄底）+ 放量确认（无量突破=假突破硬过滤）+ 跌破枢轴低点离场参考；自动注册进回测/事件研究/扫描 | 突破信号/下跌无信号/自动注册/参数空间/量能过滤 5项 | 趋势确认严格 → 信号偏少；A股不做空 |
| 板块轮动分析+预测 | implemented_verified | `research/sector_rotation.py`, MCP `predict_sector_rotation`/`sector_rotation_status`/`evaluate_sector_predictions`/`sector_leaders`, UI「市场扫描」Tab | 东财板块(BK)行情/资金流 + RS排名(20日几何涨幅) + 动量加速(5日vs20日几何日均差) + 轮入/轮出信号 + LLM预测未来一周强势板块(无LLM动量延续降级) + 落盘sector_prediction 5日回填验证top3超额 + **板块领涨股挖掘(60日RS) + 个股预测板块上下文注入** | 排名/确定性预测/落盘验证/格式/领涨股/上下文注入 8项 | 东财push2接口偶发断连(实测)；板块数据需先装载(load_sector_data)；sector_daily/sector_prediction新表 |
| 预测→仓位建议 | implemented_verified | `agent/prediction.py position_sizing()`, MCP `predict_position_size`, UI「市场预测」Tab | 借鉴ai-hedge-fund Risk Manager：预测概率 × 同方向历史命中率 × 概率桶校准 → 建议仓位（≤20% RiskGuard 硬上限），输出完整推导链 | 历史全中放大/上限约束/down零仓位 3项 | 历史样本少时校准弱；非投资建议 |
| 因子正交化（风格中性化） | implemented_verified | `research/orthogonalize.py` FactorNeutralizer, MCP `factor_neutralize`, UI「研究评估」Tab | 逐日 OLS（收益~市值+板块收益）残差化 → 纯个股 alpha（IR 排序/胜率/累计）；风格 β 诊断（市值/板块/残差波动）；市值用 stock_basic.market_cap 静态快照（与近期收益解耦，缺失回退 close 并注明共线风险） | 隔离 alpha/排序一致性/最小样本/无数据 4项 | 需 ≥5 只股票（自由度）；同板块效果最佳；市值缺失时 close 回退有共线风险 |
| 游资情绪周期 | implemented_verified | `research/sentiment_cycle.py`, MCP `sentiment_cycle`, UI「市场扫描」Tab | 涨停梯队（首板/2板/3板/4板+）+ 连板高度（连续涨停判定） + 晋级率（≥2板÷昨日涨停） + 情绪分（0-100）+ 情绪四阶段（启动/发酵/高潮/退潮+冰点）+ 近5日趋势与退潮预警；落库 sentiment_daily；**sentiment_summary 供市场诊断注入（LLM 上下文 + 确定性冰点降级）** | 发酵/退潮/冰点/高潮/摘要/诊断集成 6项 | 收盘涨停判定（≥9.5% 10cm 近似）；无盘中炸板数据（诚实标注）；数据范围=库内股票池 |

## 研究与实验

| 能力 | 状态 | 入口 | 数据范围 | 测试 | 限制 |
|---|---|---|---|---|---|
| 横截面因子研究 | planned | — | — | — | 阶段D |
| Walk-forward/OOS | planned | — | — | — | 阶段D |
| Run Recorder/manifest | planned | — | — | — | 阶段D |
| 事件研究(公告→可交易) | planned | — | — | — | 阶段D |
| 风格基准(benchmark) | planned | — | — | — | 阶段D |
| 成本/容量敏感性 | planned | — | — | — | 阶段D |

| AI 市场研究报告 | implemented_verified | `research/ai_report.py`, MCP `ai_market_report`, UI「研究总览」Tab | 聚合市场状态/市场结构/情绪矩阵/板块轮动/因子选股/价值动量/持仓风险/预测成绩单 → LLM 综述（总结/关注/风险/思路），无 LLM 模板降级 | 模板降级/mock LLM/LLM 失败/持仓段 4项 | LLM 只解释数据不编造 |
| 市场结构联合分析 | implemented_verified | `research/market_structure.py`, MCP `market_structure`, UI 🏛️ | 上证指数缠论结构（分型/笔/中枢/背驰）× 情绪矩阵 → 联合判断（偏多/偏空/中性）；联动决策树/持仓风险/AI 研报 | 联合判定/完整分析/无数据降级 3项 | 指数需库内数据或网络 |
| 持仓风险面板 | implemented_verified | `research/portfolio_risk.py`, MCP `portfolio_risk_dashboard`, UI 🛡️ | 持仓盈亏 × 批量预测 × 集中度（HHI/行业） × 风险评分（0-100）+ 指数结构方向调整（偏空+10/偏多-5） | 盈亏/集中度/评分/偏空调整 3项 | 预测默认确定性模式控成本 |
| 研究结果导出 | implemented_verified | MCP `export_research_data`, UI 📤 | 四种+导出：选股/预测/持仓/格雷厄姆/组合净值 → CSV 文本（可复制/Excel） | CSV 可解析 2项 | 无 |
| 决策树市场偏向 | implemented_verified | `build_decision_tree(market_bias=)`, MCP `get_decision_tree`, UI 🌳 | 指数偏空 → 看涨预测降级中性+仓位上限 30%；偏多 → 看跌降级+上限≥40% | 偏空/偏多修正 2项 | 需指数数据 |
| 预测周期分组验证 | implemented_verified | `prediction._summary by_horizon`, UI 成绩单 | 1d/5d/20d 命中率/Brier 独立统计（短周期是否更准） | 分组结构/1d 判定 1项 | 需各周期样本 |

| 预测周期分组验证 | implemented_verified | `prediction._summary by_horizon`, UI 成绩单 | 1d/5d/20d 命中率/Brier 独立统计（短周期是否更准） | 分组结构/1d 判定 1项 | 需各周期样本 |
| 校准曲线可视化 | implemented_verified | `ui._build_calibration_figure`, UI 🎯 | 概率桶 vs 实际命中率柱状图 + 完美校准参考线，过度自信红色标记 | 图构建/参考线/红标 1项 | 需 ≥4 条有方向预测 |
| 持仓风险可视化 | implemented_verified | `ui.portfolio_risk_fig`, UI 📊 | 持仓权重饼图（预测方向标注）+ 盈亏条形图 + 风险评分标题 | 空持仓降级冒烟 | 复用风险面板数据 |
| 每日数据自动更新 | implemented_verified | MCP `run_daily_update`, cron `pamcp-data-load` | 调度器 8-phase 全链路 + 数据体检联动（评分<70 告警） | 调度器既有测试覆盖 | 耗时 2 分钟+ |
| 市场结构×预测联动 | implemented_verified | `prediction._market_bias_context`, 预测 prompt | 指数缠论方向注入个股预测 prompt（个股/板块/大盘三层环境） | 注入文本/无数据空串 1项 | 库内指数数据优先 |
| 多周期预测共振 | implemented_verified | `research/resonance.py`, MCP `predict_resonance`, UI 🎯 | 1d/5d/20d 三周期方向一致性：全同向=强共振(100%)/两周期=共振(70%)/分歧=观望；**强共振覆盖决策树单周期方向 + 仓位校准(×1.3/×0.7) + 持仓共振列/减仓告警 + 事件研究验证** | 结构/强趋势/无数据/共振覆盖/仓位校准/持仓字段/事件研究 7项 | 确定性模式控成本 |

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
