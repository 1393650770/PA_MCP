# OpenClaw 新手使用指南（PA_MCP × OpenClaw）

> 本指南面向**零基础新手**：通过 OpenClaw（AI 助手）用自然语言驱动 PA_MCP 的 92 个研究工具，不需要懂代码、不需要懂方法论。
>
> ⚠️ 研究参考，非投资建议。

---

## 一、一次配置，永久使用

### 1. 配置 PA_MCP MCP Server

在 OpenClaw 配置中注册 PA_MCP（MCP 客户端通用配置）：

```json
{
  "mcpServers": {
    "pa-mcp": {
      "command": "D:\\Project\\AI\\PA_MCP\\PA_MCP\\venv\\Scripts\\python.exe",
      "args": ["-m", "pa_mcp.server"],
      "env": { "PA_MCP_CONFIG": "D:\\Project\\AI\\PA_MCP\\PA_MCP\\config\\default.yaml" }
    }
  }
}
```

> Windows 注意：`venv\Scripts\python.exe` 用绝对路径；首次启动会自动装载数据（需网络）。

### 2. （推荐）安装定时任务

`config/openclaw_cron.json` 已备好 **10 个全自动任务**（盘前简报 / 收盘复盘 / 持仓体检 / 数据更新 / 板块晨报 / 周日周报…），按文件内 `_usage` 说明合并进 OpenClaw 配置即可。装好后**新手什么都不用做**，每天定时收到推送。

### 3. 验证

对 OpenClaw 说一句："**帮我看看今天的大盘**"——能正常回答即配置成功。

---

## 二、新手怎么问（照着说就行）

PA_MCP 内置了 **新手引导 prompt（`newbie-guide`）**，新手只需说一句话，OpenClaw 会自动完成"读工具清单 → 读决策地图 → 按四步走引导"。

| 你想做什么 | 直接对 OpenClaw 说 |
|---|---|
| 第一次使用，想被带着走 | "我是新手，请用 newbie-guide 引导我使用 PA_MCP" |
| 看今天大盘/市场情绪 | "帮我看看今天的大盘和情绪" |
| 选股票 | "帮我选几只股票" / "用 CANSLIM 帮我扫描成长股" |
| 我的持仓怎么样 | "看看我的持仓有没有风险" / "我的持仓要操作吗" |
| 验证一个信号靠不靠谱 | "布林策略在 000001 上靠不靠谱？帮我验证" |
| 深度分析/预测 | "深度分析一下 300750" / "预测一下平安银行下周走势" |
| 出报告 | "给我出一份今天的市场研究报告" |

**OpenClaw 内部自动执行**（新手无感知）：
1. `pa_help`（了解 92 工具）
2. `get_methodology_guide`（四步决策地图：看环境 → 选方法 → 做验证 → 增强解读）
3. 按你的问题调对应工具（市场诊断/扫描/持仓体检/事件研究/AI 分析）

---

## 三、四步走：新手应该了解的最小方法论

OpenClaw 引导你时遵循的顺序（与 UI「新手决策地图」同一套口径）：

| 步骤 | 回答的问题 | 主要工具 |
|---|---|---|
| ① 看环境 | 现在是牛市还是熊市？情绪/轮动在哪阶段？ | `agent_market_diagnosis` `market_structure` `sentiment_cycle` `regime_matrix` |
| ② 选方法 | 当前状态适合哪些策略/方法？ | `get_methodology_guide` `scan_market` `scan_canslim` `factor_stock_selection` |
| ③ 做验证 | 信号靠不靠谱？ | `research_event_study` `research_walk_forward` `backtest_overfit_diagnosis` |
| ④ 增强解读 | 要不要 AI 深研/预测？ | `agent_analyze_stock` `predict_market` `ai_market_report` |

新手只需要记住：**先看环境，再选方法，先验证再信，最后才谈操作**。

---

## 四、常用对话模板（进阶新手）

```text
今天收盘后帮我：
1. 跑一下数据更新（run_daily_update）
2. 看市场状态和情绪（agent_market_diagnosis + sentiment_cycle）
3. 扫一遍策略信号（scan_market）
4. 检查我的持仓（portfolio_risk_dashboard）
5. 最后出一份简短的明日关注清单
```

```text
用 newbie-guide 教我：
先告诉我现在市场是什么状态，然后推荐 2 个适合当前市场的策略，
再用其中一个扫一下我的自选股，最后用事件研究验证它靠不靠谱。
```

---

## 五、常见问题

**Q：AI 分析工具（预测/深度分析）不可用？**
A：需要配置 LLM（`config/llm_config.json` 填 API key）。未配置时**不影响**前面三步（环境/选股/验证全是免费确定性工具），OpenClaw 会提示"第 ④ 步可跳过"。

**Q：数据是空的/结果很少？**
A：先让 OpenClaw 调 `run_daily_update()` 补数据（约 2 分钟）；日线数据自动走 5 源容灾（akshare/腾讯/新浪/东财/同花顺），无需手动操作。

**Q：结果里有 "研究参考，非投资建议"？**
A：这是系统规范——所有结论均来自统计/研究，不是投资建议。仓位纪律（单票 ≤20%）优先于一切信号。

**Q：OpenClaw 不认 MCP 工具？**
A：确认 MCP server 配置路径正确且 `python -m pa_mcp.server` 能独立启动；OpenClaw 需要重启以加载新注册的 MCP 工具。

---

## 六、给管理员的说明

- 全部工具为**只读研究**（LLM 不持有下单能力，见 `docs/methodology-guide.md` 第八节）
- 定时任务模板维护在 `config/openclaw_cron.json`（修改后需重新导入）
- 工具能力与状态以 `docs/capability-matrix.md` 为准
