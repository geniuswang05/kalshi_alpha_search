"""实盘 LIP 测试报告：竞争反应、逆向选择、奖励估计 vs 实际到账。

用法:
  python strategies/lip_live_report.py data_cache/lip_live/<run>            # 只看日志
  python strategies/lip_live_report.py data_cache/lip_live/<run> --paid 12.34
      （计划周期结束后，把 Kalshi 网站 Rewards 页面上这些市场实际到账的金额填进来）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(run: Path, paid: float | None):
    log = pd.read_csv(run / "log.csv", parse_dates=["ts"])
    fills = pd.read_csv(run / "fills.csv", parse_dates=["ts"]) if (run / "fills.csv").exists() else pd.DataFrame()
    hours = (log["ts"].max() - log["ts"].min()).total_seconds() / 3600
    last = log.sort_values("ts").groupby("ticker").tail(1)
    print(f"=== {run.name}：{last['ticker'].nunique()} 个市场，{hours:.1f} 小时 ===")

    # 1) 竞争反应：份额随时间的变化（进场后第一个 15 分钟 vs 之后）
    log["mins"] = log.groupby("ticker")["ts"].transform(lambda s: (s - s.min()).dt.total_seconds() / 60)
    early = log[log["mins"] <= 15].groupby("ticker")["share_real"].mean()
    late = log[log["mins"] > 15].groupby("ticker")["share_real"].mean()
    growth = (log.groupby("ticker").apply(lambda g: g["others_at_our_px"].iloc[-1] / max(g["entry_others"].dropna().iloc[0], 1)
                                          if g["entry_others"].notna().any() else np.nan, include_groups=False))
    react = pd.DataFrame({"share_first15m": early, "share_after": late, "others_size_x": growth})
    print("\n1) 竞争反应（份额下降 / 对手同价位挂单量放大倍数）:")
    print(react.round(3).to_string())
    print(f"   对手越过我们报价的时间占比: {log['others_jumped_ahead'].mean():.1%}")

    # 2) 逆向选择
    if not fills.empty:
        mids = log[["ts", "ticker", "mid"]].dropna().sort_values("ts")
        mo = []
        for f in fills.itertuples(index=False):
            m = mids[(mids["ticker"] == f.ticker) & (mids["ts"] >= f.ts + pd.Timedelta(minutes=30))]
            if m.empty:
                continue
            val = m["mid"].iat[0] if f.side == "yes" else 1 - m["mid"].iat[0]
            mo.append((val - f.px) * f.qty)
        print(f"\n2) 成交 {len(fills)} 笔 / {fills['qty'].sum():,.0f} 张，maker 费 ${fills['fee'].sum():,.2f}，"
              f"30 分钟 markout ${sum(mo):,.2f}")
    else:
        print("\n2) 无成交")

    # 3) 奖励
    est = last["reward_est"].sum()
    inv = last["mtm"].sum()
    print(f"\n3) 奖励估计 ${est:,.2f}，库存盯市 ${inv:,.2f}，净 ${est + inv:,.2f}")
    if paid is not None:
        print(f"   实际到账 ${paid:,.2f} → 实际/估计 = {paid / est:.2f}（<1 说明规则实现或份额估计偏乐观）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--paid", type=float, default=None)
    a = ap.parse_args()
    main(a.run, a.paid)
