# PA_MCP MCP 工具全览

> 自动盘点：共 **97** 个 MCP 工具（与 UI 功能对等，新增工具自动入 pa_help）。研究参考，非投资建议。

## 数据（16）
| 工具 | 说明 |
|---|---|
| `analyze_stock` | 个股综合分析（数据看板对等）：K线+实时估值+资金流+策略信号+缠论。 |
| `analyze_timeframe_alignment` | 多周期对齐分析：日/周/月均线共振与背离。 |
| `calc_vwap` | VWAP 成交量加权均价计算。 |
| `get_data_source_health` | 数据源健康状态：各源熔断/成功/失败统计。 |
| `get_kline` | Get historical K-line (OHLCV) data. |
| `get_major_events` | Get all major events for a stock: block trades, lockup expiry, insider |
| `get_market_overview` | Get current A-share market overview with key metrics. |
| `get_market_sentiment` | Get current market sentiment assessment with position suggestion. |
| `get_realtime_quote` | Get real-time stock quote with 5-level depth. |
| `get_stock_capital_flow` | 个股主力资金流：主力/超大单/大单/中单/小单净流入（东财独有数据）。 |
| `get_stock_info` | Get stock basic info: name, industry, market cap, list date. |
| `get_stock_name` | 股票代码 → 名称（DB优先+内置字典兜底）。 |
| `get_valuation_snapshot` | 专业估值快照：PE/PB/市值/换手率/量比/涨跌停价（腾讯实时数据）。 |
| `review_daily_limit_up` | Daily limit-up review: seal time distribution, break rate, sector clus |
| `review_dragon_tiger` | Daily dragon-tiger board review: seat analysis, famous trader tracking |
| `scan_limit_up` | Scan limit-up stocks with chain ladder analysis (连板梯队) and seal qualit |
| `scan_volume_surge` | Scan stocks with abnormal volume surge. |
| `search_stock` | Search stocks by name or code. |

## 预测（9）
| 工具 | 说明 |
|---|---|
| `evaluate_predictions` | 验证历史预测：回填已到期预测的真实收益，计算命中率/Brier/方向一致率。 |
| `evaluate_sector_predictions` | 板块轮动预测验证：回填已到期预测的 top3 超额收益（vs 全板块平均）。 |
| `predict_future_chart` | 未来 K 线路径预测（LLM 全维分析 → 三情景 OHLC 路径）。 |
| `predict_market_multi` | 多股票批量预测对比（方向/概率/区间/周期并排比较）。 |
| `predict_position_size` | 预测驱动的仓位建议（Risk Manager 思路，借鉴 ai-hedge-fund）。 |
| `predict_sector_rotation` | 板块轮动预测：RS 动量排名 + 资金流 + LLM 解读 → 未来一周强势板块。 |
| `prediction_history` | 查看某股票的历史预测记录与验证结果（方向/概率/实际收益/命中状态）。 |
| `sector_leaders` | 板块领涨股挖掘（板块轮动 → 个股闭环）。 |
| `sector_rotation_status` | 板块轮动当前状态：RS 排名 / 轮入轮出 / 轮动速度（只读分析，不预测）。 |

## Agent/LLM（13）
| 工具 | 说明 |
|---|---|
| `agent_analyze_stock` | AI-powered multi-dimensional stock analysis. |
| `agent_compare_stocks` | Side-by-side comparison of multiple stocks across all dimensions. |
| `agent_debate_picks` | 选股多 Agent 辩论：对候选股票逐一 Bull/Bear 辩论 + 3 位投资大师裁定。 |
| `agent_earnings_analysis` | 财报专业分析：从财务表提取关键指标并生成结构化解读。 |
| `agent_experience_search` | 经验库检索（RAG）：按符号/周期位置/方向检索历史 AI 分析案例。 |
| `agent_market_diagnosis` | 两阶段分析 Stage 1：LLM 市场诊断 + 策略路由。 |
| `agent_market_state` | Get current market regime and position sizing suggestion. |
| `agent_memory_status` | 长期记忆状态：决策记录数量/胜率/盈亏比 + 认知偏差检测。 |
| `agent_morning_brief` | Daily pre-market briefing: overnight news, global markets, today's wat |
| `agent_portfolio_review` | 持仓体检：结合实时行情、估值、风险规则输出专业组合诊断。 |
| `agent_scan_market` | AI-powered full market scan — run all strategies, rank by strength sco |
| `agent_sector_analysis` | Sector/industry rotation analysis — identify leading and lagging secto |
| `portfolio_ai_analysis` | 持仓股 AI 综合分析：真实数据 + 策略信号 + LLM 解读。 |

## 选股/研究（25）
| 工具 | 说明 |
|---|---|
| `backtest_overfit_diagnosis` | 回测过拟合与多重检验诊断（DSR / Harvey-Liu / CSCV-PBO）。 |
| `chan_analysis` | 缠论结构分析（缠中说禅体系）：分型→笔→中枢→背驰。 |
| `chan_beichi_backtest` | 缠论背驰信号组合回测验证。 |
| `chan_beichi_event_study` | 缠论背驰信号事件研究（大牛方法可检验性）。 |
| `evaluate_factor` | 单因子检验（量化标准）：IC + 分层（Q1-Q5）+ 单调性 + 覆盖率。 |
| `factor_library` | 因子库清单（借鉴 factor-skill-factory）：已注册因子列表。 |
| `factor_neutralize` | 因子正交化（风格中性化，借鉴 factor-orthogonalize）。 |
| `factor_portfolio_backtest` | 因子选股组合回测（选股 → 组合闭环）。 |
| `factor_prediction_sensitivity` | 预测权重敏感性分析：AI 预测在选股里该占多大权重。 |
| `factor_scan` | 因子批量扫描：全部注册因子在一只股票上的 IC/分层检验排行。 |
| `factor_stock_selection` | 多因子截面选股（Qlib 风格，可选 AI 预测融合）。 |
| `get_strategy_guide` | 策略速查：按市场状态推荐策略 + 新手难度星级。 |
| `get_strategy_info` | 策略说明与最优策略检测。 |
| `graham_screen` | 格雷厄姆价值筛选（《聪明的投资者》防御性投资标准）。 |
| `list_strategies` | List available trading strategies. |
| `portfolio_backtest` | 组合构建回测：多票共享账本组合（信号→约束权重→联合执行）。 |
| `research_event_study` | 信号事件研究：检验策略信号的预测力（信号后5/10/20日收益 vs 基准）。 |
| `research_event_study_sector` | 板块基准事件研究（风格匹配，学术标准）。 |
| `research_walk_forward` | Walk-Forward OOS 评估：多fold真实数据验证策略可交易性。 |
| `scan_canslim` | CANSLIM 成长股扫描（欧奈尔《笑傲股市》七要素选股法）。 |
| `scan_market` | 市场扫描：板块漏斗（热门+冷门板块成分股）+ 策略买入信号候选。 |
| `strategy_compare` | 全策略事件研究对比（多策略同台检验）。 |
| `turtle_position_size` | 海龟交易仓位计算（ATR 波动率目标）。 |
| `value_momentum_backtest` | 价值×动量组合回测验证（复合选股 → 滚动调仓组合）。 |
| `value_momentum_screen` | 价值 × 动量 复合选股（Asness et al. 2013 学术框架）。 |

## 组合/自选（12）
| 工具 | 说明 |
|---|---|
| `list_alerts` | List configured price/volume alerts. |
| `portfolio_add` | Add a holding to portfolio（可登记止盈止损计划，新手友好）. |
| `portfolio_remove` | Remove a holding from portfolio (按代码删除，与 UI 同一口径). |
| `portfolio_risk_dashboard` | 持仓风险面板：实时盈亏 × 批量预测 × 集中度 × 风险评分。 |
| `portfolio_strategy_signals` | 持仓股当前策略信号：检查每只持仓是否触发买入信号。 |
| `portfolio_summary` | Get portfolio summary with P&ampL and cost basis. |
| `watch_price_alert` | Create a price alert. |
| `watch_volume_alert` | Create a volume surge alert. |
| `watchlist_add` | Add a stock to your watchlist (自选股). |
| `watchlist_overview` | Get real-time overview of all watchlist stocks with key metrics. |
| `watchlist_remove` | Remove a stock from your watchlist. |
| `watchlist_show` | Show all symbols in your watchlist (symbols only, no analysis). |

## 整合/市场（17）
| 工具 | 说明 |
|---|---|
| `ai_market_report` | AI 市场研究报告：确定性研究结果 → LLM 综述。 |
| `consensus_event_study` | 综合信号事件研究：融合投票是否有额外预测力。 |
| `evaluate_methods` | 开源方法评价：量化方法可信度 + 理财方法评估状态 + 持仓×方法评价。 |
| `export_research_data` | 研究结果导出为 CSV（可复制/Excel 导入）。 |
| `get_decision_tree` | 决策树可视化（借鉴 PA_Agent 决策树机制）：逐层闸门推演。 |
| `get_methodology_guide` | 新手决策地图：四步研究路径（看环境→选方法→做验证→增强解读）+ 四类资产编目。 |
| `market_structure` | 市场结构联合分析：上证指数缠论结构 × 情绪×轮动矩阵。 |
| `one_click_analysis` | 🚀 一站式分析：流水线整合报告（市场→选股→个股→持仓）。 |
| `predict_resonance` | 多周期预测共振：1d/5d/20d 三周期方向一致性。 |
| `regime_matrix` | 情绪 × 轮动联合矩阵（Regime Matrix）。 |
| `resonance_event_study` | 共振信号事件研究：三周期共振是否有额外预测力。 |
| `sector_hot_cold` | 今日热门/冷门板块（新浪行业板块实时榜，不依赖东财历史数据）。 |
| `sentiment_cycle` | 游资情绪周期分析（涨停梯队/连板高度/晋级率/阶段判定）。 |
| `signal_consensus` | 综合决策信号：多信号源加权投票融合。 |
| `trading_actions` | 💰 今日操作面板：持仓止盈止损 / 买入候选 / 操作建议（含 LLM 解读）。 |
| `watchlist_consensus` | 自选股综合信号扫描：批量 5 源加权投票 → 强看涨/看跌/分歧清单。 |
| `watchlist_resonance` | 自选股共振扫描：批量三周期共振 → 强共振看涨/看跌/分歧清单。 |

## 系统（3）
| 工具 | 说明 |
|---|---|
| `data_quality_report` | 数据质量体检：表覆盖 + K 线完整性（OHLC 一致性/非正/NaN/缺口）。 |
| `pa_help` | Get a complete guide to PA_MCP — all tools, common workflows, and data |
| `run_daily_update` | 每日数据自动更新：调度器 8-phase 全链路（日历/股票池/日线/ |

*分类共 95 个；其余工具见 `pa_help` 动态清单。*