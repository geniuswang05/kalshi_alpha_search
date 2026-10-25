# Kalshi Alpha Search — 在预测市场里，钱到底从哪来？

一个对 Kalshi 预测市场的系统性"找 alpha"研究项目：从"用机器学习打败市场"出发，
一路排除到唯一站得住的正期望来源，并做成可前向测试的做市机器人。

核心结论（全部基于 Kalshi 真实历史数据，样本外检验，扣真实手续费）：

| # | 假设 | 数据 | 结论 |
|---|---|---|---|
| 1 | 机器学习预测 NBA 胜负能打败市场 | 7 赛季比赛 + 2898 个 Kalshi 合约真实价格 | ❌ 市场 logloss 0.5685 < 模型 0.6099；加伤病特征 0.5940 仍输；混合回归模型系数≈0 |
| 2 | 调参能救 | 107 组参数 × t 检验 | ❌ 0 组 t>1.65（随机预期 ~5 组） |
| 3 | 同场两队合约套利 | 82,337 个赛前快照 | ❌ 扣费后仅 62 个快照为正，且多为 K 线不同步假象 |
| 4 | 主流联赛被动做市赚价差 | 27 个体育系列、650 万笔成交 | ❌ 已实现价差 0.02–0.14¢/张 < maker 费 0.2–0.4¢/张 |
| 5 | 终局冷门偏差（卖"奇迹彩票"） | 成交样本 + 9,452 个市场固定时钟快照 | ❌ 假发现：成交样本 p=10⁻⁵，但大样本快照两侧完全校准（6.35% vs 6.20%；5.66% vs 6.08%） |
| 6 | 用 Polymarket 当信号交易 Kalshi | 2,351 场 MLB 分钟级对齐 | ❌ 比赛中是 Kalshi 领先（PM 1 分钟内追上 92%，t=213），反向无效 |
| 7 | 收割交易所流动性激励（LIP） | 6,657 个活跃激励计划 + 实时订单簿 + 影子做市前向测试 | ✅ 唯一的正期望来源（交易所补贴，不依赖信息优势）——见下文与 `strategies/` |

## 为什么是 LIP

Kalshi 的流动性激励计划（官方规则见 CFTC 备案与 help.kalshi.com）按"挂单是否帮助填满 Target Size、
离最优价多远"每秒给挂单打分，按份额分配每个市场每天 $10–$1,000 的奖金——**挂单不需要成交也能拿奖励**。
2026 年 8–12 月的激励总额约 $1,340 万。这是唯一一个不要求"比市场更聪明"的收益来源：
它要求的是**资金、风控和工程**——这恰好是散户量化能做到的。

风险与限制（必须读）：
- 奖励份额来自实时订单簿 + 官方计分规则的模拟；对手看不到影子单、不会反应，所以份额是**上限**。
  真实份额只能用真实小额订单测出来。
- 被成交 = 被逆向选择（前向测试里单个油价市场 2 小时亏 $153）。v2 用类别分散、单市场止损、
  避开临近结算来控制。
- Kalshi 保留对"有违计划目的"的参与者取消资格的权利；在几乎无流动性的"死盘"挂极端价格领奖励
  可能被认定为滥用——本项目的策略主动排除价差 > 20¢ 的市场。

## 代码结构

```
config/settings.py                 全局参数
data/
  kalshi_client.py                 Kalshi API v2 客户端（RSA-PSS 签名、V2 下单/撤单）
  kalshi_universe.py               市场/成交/K线 拉取（live + historical 两段自动拼接）
  nba_fetcher.py                   NBA 比赛 + 球员 game log（stats.nba.com）
models/                            v1/v2 预测模型：Elo、滚动效率、缺阵价值、XGBoost + isotonic 校准
backtest/
  run_backtest.py / run_v2.py      假设 1：模型 vs Kalshi 真实价格回测
  param_sweep.py                   假设 2：参数扫描
  maker_taker.py                   假设 4：maker/taker 收益分解 + 已实现价差（市场聚类 t）
  flb_analysis.py / endgame_*.py   假设 5：终局冷门偏差（成交样本 → 完整翻页 → 大样本快照）
scripts/
  sample_trades.py                 27 个体育系列分层抽样成交
  complete_windows.py              补全被 1000 笔上限截断的窗口（修正前视型抽样偏差）
strategies/
  cross_venue.py / xvenue_analysis.py   假设 6：Kalshi × Polymarket 领先-滞后
  lip_scanner.py                   LIP 计分规则实现 + 实时订单簿扫描
  lip_analysis.py                  奖励 vs 成交风险
  shadow_mm.py                     ★ LIP 做市机器人（影子/前向测试；v2 = 分散 + 止损）
  shadow_report.py                 前向测试报告（奖励、库存盯市、markout、回撤）
  endgame_bot.py                   假设 5 的执行机器人（策略已被否定，保留作为工程示例）
```

## 复现

```bash
pip install nba_api xgboost scikit-learn pandas pyarrow scipy requests cryptography
python scripts/sample_trades.py 40            # 成交抽样（~1h）
python scripts/complete_windows.py            # 补全截断窗口
python backtest/maker_taker.py                # 谁在赚钱
python backtest/endgame_snapshots_large.py    # 终局冷门大样本检验
python strategies/cross_venue.py mlb 2026-04-01 && python strategies/xvenue_analysis.py mlb
python strategies/lip_scanner.py && python strategies/lip_analysis.py
python strategies/shadow_mm.py 8 30 -1 v2     # LIP 影子做市 8 小时（不下单、不需要 key）
python strategies/shadow_report.py
```

## 实盘测量：竞争下的真实奖励份额（需要你本人操作）

`strategies/lip_live.py` 用真实小额订单测量影子测试测不到的东西：对手看到我们的挂单后，
会不会加量、抢价，我们的真实份额还剩多少。所有挂单都是 post_only（只做 maker）。

1. **演练**（不下单、不需要 key）：`python strategies/lip_live.py --hours 1`
2. **demo 模拟盘**（假钱，验证下单/撤单/成交机制；demo 没有激励计划）：
   在 demo.kalshi.co 注册并创建 demo API key，然后
   `export KALSHI_KEY_ID=... KALSHI_PRIVATE_KEY_PATH=...` →
   `python strategies/lip_live.py --env demo --live --hours 1`
3. **正式环境小额**（真钱）：换成正式环境的 key，账户里放约 $300，
   `python strategies/lip_live.py --env prod --live --i-understand-real-money --hours 4`
   默认 3 个市场、每侧 100 张、总亏损 $30 即全部撤单退出；在项目根目录 `touch STOP` 可随时全部撤单退出。
4. **核对奖励**：计划周期结束后在 Kalshi 网站 Rewards 页面查看实际到账，
   `python strategies/lip_live_report.py data_cache/lip_live/<run> --paid <到账金额>`
   输出：份额随时间的下降（竞争反应）、对手同价位挂单的放大倍数、30 分钟 markout、实际/估计奖励比。

## 方法论里最有价值的一课：一个 p=10⁻⁵ 的假发现

"比赛最后阶段卖 2–10¢ 的冷门"在成交样本里看起来极强（572 个市场只逆转 11 次 vs 价格隐含 31.5 次，
p=1.5×10⁻⁵，样本外 t=5.4）。排查过程：
1. 发现每窗口"取最近 1000 笔"的抽样有前视偏差（逆转引发巨量成交把逆转前的成交挤出样本）→ 完整翻页补全 → 信号仍在。
2. 改用固定时钟快照（与成交量无关的抽样），把样本扩大到 9,452 个市场 → 冷门两侧都**完全校准**。
3. 同一批市场的快照也显示低逆转率 → 异常来自那批样本自身的结果，不是可交易的定价错误。

教训：成交数据上的显著性检验要警惕"抽样方式依赖未来"；最终判断要交给与成交无关的大样本检验。

## Kalshi API 速查

- Base URL：`https://external-api.kalshi.com/trade-api/v2`；行情、订单簿、成交、K 线无需 key
- 历史截止线：`GET /historical/cutoff`；之前的数据走 `/historical/markets|trades|markets/{t}/candlesticks`
- 下单 V2：`POST /portfolio/events/orders`（side=bid/ask，价格/数量为定点字符串）
- 费率：taker = 0.07×mult×P(1−P)；`quadratic_with_maker_fees` 系列 maker = taker×0.25，`quadratic` 系列 maker 免费
- 激励：`GET /incentive_programs?status=active&type=liquidity`
