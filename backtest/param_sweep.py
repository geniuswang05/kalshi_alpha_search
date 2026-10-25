"""参数扫描：在 (下注时点 × edge阈值 × edge上限 × 价格区间) 网格上找有没有
任何一个区域是正期望的。

方法论要点：
- 固定注额（每笔1张合约），隔离 edge 质量与 Kelly 复利路径
- 每个格子报 t 统计量（per-bet PnL 均值/标准误）——n 小或 t 低的格子是噪音
- 多重比较警告：108 个格子，纯随机也会有 ~5 个格子 t>1.65，
  单个格子显著 ≠ 真 alpha，要看整体模式是否合理
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.diagnose_alpha import join_at
from backtest.run_backtest import train_and_predict
from signals.edge_detector import taker_fee

HOURS = [4, 24, 48]
MIN_EDGES = [0.02, 0.03, 0.05, 0.08]
EDGE_CAPS = [0.10, 0.15, 1.00]
PRICE_BANDS = [(0.02, 0.98), (0.20, 0.80), (0.35, 0.65)]
MAX_SPREAD = 0.05


def sweep():
    test, _ = train_and_predict()

    joined = {}
    for h in HOURS:
        m = join_at(test, h).dropna(subset=["bid", "ask"]).copy()
        m["fee"] = taker_fee(m["ask"])
        m["edge"] = m["p_model"] - m["ask"] - m["fee"]
        m["cost"] = m["ask"] + m["fee"]
        m["pnl_flat"] = m["won"] - m["cost"]        # 每笔买1张的PnL
        m["spread_ok"] = (m["ask"] - m["bid"]) <= MAX_SPREAD
        joined[h] = m

    rows = []
    for h, me, cap, (lo, hi) in product(HOURS, MIN_EDGES, EDGE_CAPS, PRICE_BANDS):
        m = joined[h]
        mask = (m["spread_ok"] & (m["edge"] >= me) & (m["edge"] < cap)
                & (m["ask"] >= lo) & (m["ask"] <= hi))
        sel = m[mask]
        n = len(sel)
        if n < 30:
            continue
        pnl = sel["pnl_flat"]
        roi = pnl.sum() / sel["cost"].sum()
        t = pnl.mean() / (pnl.std() / np.sqrt(n)) if pnl.std() > 0 else np.nan
        rows.append({"hours": h, "min_edge": me, "edge_cap": cap,
                     "band": f"{lo:.2f}-{hi:.2f}", "n": n,
                     "roi": roi, "t_stat": t, "hit": sel["won"].mean(),
                     "avg_edge": sel["edge"].mean()})
    res = pd.DataFrame(rows).sort_values("t_stat", ascending=False)

    print("\n===== 参数扫描结果 =====")
    print(f"有效格子: {len(res)}  |  ROI>0 的格子: {(res['roi'] > 0).sum()}"
          f"  |  t>1.65 的格子: {(res['t_stat'] > 1.65).sum()}"
          f"  （随机预期约 {int(len(res)*0.05)} 个）")
    print("\nt 统计量最高的 10 个格子:")
    print(res.head(10).round(3).to_string(index=False))
    print("\nt 统计量最低的 5 个格子:")
    print(res.tail(5).round(3).to_string(index=False))

    # 按单维度聚合看整体模式
    print("\n按 edge 阈值聚合（全部格子平均 ROI）:")
    print(res.groupby("min_edge")[["roi", "t_stat"]].mean().round(3).to_string())
    print("\n按时点聚合:")
    print(res.groupby("hours")[["roi", "t_stat"]].mean().round(3).to_string())
    print("\n按价格区间聚合:")
    print(res.groupby("band")[["roi", "t_stat"]].mean().round(3).to_string())
    return res


if __name__ == "__main__":
    sweep()
