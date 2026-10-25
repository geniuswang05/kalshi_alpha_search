"""LIP 收益/风险分析：奖励收益率 vs 成交风险。

对扫描结果里的每个市场，再拉过去 24h 成交，估算我们挂单的日成交量：
  YES 侧（我们的 YES 买单 @ px）：被 taker 卖 YES（taker_side=no）且成交价 <= px 的量，
  减去排在我们前面的同价及更优挂单量（队列）。NO 侧对称。
这是粗略上界估计（假设成交量会穿透到我们的价位），用来区分"拿奖励几乎不会被成交"
和"挂着就会被吃单"的市场。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import get_with_retry


def latest_scan() -> pd.DataFrame:
    files = sorted(DATA_CACHE_DIR.glob("lip_scan_*.parquet"))
    return pd.read_parquet(files[-1])


def trades_24h(client, ticker: str) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    r = get_with_retry(client, "/markets/trades", {
        "ticker": ticker, "limit": 1000,
        "min_ts": int((now - pd.Timedelta(hours=24)).timestamp())})
    t = pd.DataFrame(r.get("trades", []))
    if t.empty:
        return t
    t["yes_price"] = t["yes_price_dollars"].astype(float)
    t["count"] = t["count_fp"].astype(float)
    t["side"] = t.get("taker_outcome_side", t.get("taker_side"))
    return t


def fill_estimate(t: pd.DataFrame, yes_px: float, no_px: float) -> float:
    """我们两侧挂单一天大约会被成交的张数（未扣队列，上界）。"""
    if t.empty:
        return 0.0
    # taker 卖 YES（=买 NO）成交价 <= 我们的 YES 买价 → 会打到我们的 YES 买单
    hit_yes = t[(t["side"] == "no") & (t["yes_price"] <= yes_px + 1e-9)]["count"].sum()
    # taker 买 YES 成交价 >= 1 − 我们的 NO 买价 → 会打到我们的 NO 买单
    hit_no = t[(t["side"] == "yes") & (t["yes_price"] >= 1 - no_px - 1e-9)]["count"].sum()
    return float(hit_yes + hit_no)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    scan = latest_scan()
    scan = scan[scan["share"] > 0]
    client = KalshiClient()

    tickers = scan["ticker"].unique()
    fills = {}
    for i, tk in enumerate(tickers):
        fills[tk] = trades_24h(client, tk)
        if (i + 1) % 200 == 0:
            print(f"  trades {i+1}/{len(tickers)}", flush=True)

    def est(row):
        yes_px = round(row["best_yes_bid"] - 0.01 * row["offset"], 2)
        no_px = round(row["best_no_bid"] - 0.01 * row["offset"], 2)
        return fill_estimate(fills[row["ticker"]], yes_px, no_px)

    scan["fills_24h_upper"] = scan.apply(est, axis=1)
    scan["fill_ratio"] = scan["fills_24h_upper"] / scan["q"]
    scan.to_parquet(DATA_CACHE_DIR / "lip_scan_with_fills.parquet", index=False)

    # 每个市场选"成交风险低"的最优配置：fill_ratio < 0.25 中 roc 最高
    safe = scan[scan["fill_ratio"] < 0.25]
    best = safe.sort_values("roc_per_day", ascending=False).drop_duplicates("ticker")
    print(f"\n可挂单市场 {scan['ticker'].nunique()}，低成交风险配置可用 {best['ticker'].nunique()}")
    print("\n低成交风险配置的日资金收益率分布（奖励/资金/天）:")
    print(best["roc_per_day"].describe(percentiles=[.25, .5, .75, .9]).round(4))
    print("\nTop 25:")
    cols = ["ticker", "series", "reward_per_day", "q", "offset", "share", "usd_per_day",
            "capital", "roc_per_day", "fills_24h_upper", "spread", "qualified_without_us"]
    print(best[cols].head(25).round(3).to_string(index=False))

    # 组合：在资金预算下贪心选市场
    for budget in (1000, 5000, 20000):
        b = best.sort_values("roc_per_day", ascending=False)
        b = b[b["capital"].cumsum() <= budget]
        print(f"\n资金 ${budget:,}: {len(b)} 个市场, 预计奖励 ${b['usd_per_day'].sum():,.0f}/天"
              f" (资金占用 ${b['capital'].sum():,.0f}, 日收益率 {b['usd_per_day'].sum()/max(b['capital'].sum(),1):.2%})")
