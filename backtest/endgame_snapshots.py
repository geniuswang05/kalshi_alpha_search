"""终局冷门校准的无偏检验：固定时钟快照（不受成交量驱动的抽样偏差影响）。

对每个抽样市场，取事前终局窗口 [anchor−45, anchor+30] 分钟的 1 分钟 K 线，每 5 分钟取一个
快照；若 YES 的 mid 落在 (2¢, 10¢]，记录 (隐含概率=mid 和 ask, 实际结果)。
每个市场只算一次（取窗口内第一个落入区间的快照），避免同一结果重复计数。
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.flb_analysis import EXCLUDE_ANCHOR
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import cutoff_ts, market_candles

OUT = DATA_CACHE_DIR / "endgame_candles"
OUT.mkdir(exist_ok=True)


def fetch(client, row, cutoff):
    path = OUT / f"{row.ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    t0 = int((row.anchor - pd.Timedelta(minutes=45)).timestamp())
    t1 = int((row.anchor + pd.Timedelta(minutes=30)).timestamp())
    try:
        k = market_candles(client, row.series, row.ticker, t0, t1, 1,
                           historical=row.close_time < cutoff)
    except Exception:
        k = pd.DataFrame()
    k.to_parquet(path, index=False)
    return k


if __name__ == "__main__":
    meta = pd.read_parquet(DATA_CACHE_DIR / "sample_market_meta.parquet")
    trades_meta = []
    for f in (DATA_CACHE_DIR / "trade_samples" / "_per_market").glob("*.parquet"):
        d = pd.read_parquet(f, columns=["ticker", "series", "result", "close_time"])
        if not d.empty:
            trades_meta.append(d.iloc[0])
    tm = pd.DataFrame(trades_meta)
    m = tm.merge(meta[["ticker", "expected_expiration_time"]], on="ticker")
    m = m[~m["series"].isin(EXCLUDE_ANCHOR)]
    # 联赛偏移：只用前半段市场估计（与 flb_analysis 一致）
    cut = m["close_time"].median()
    h1 = m[m["close_time"] < cut]
    off = ((h1["close_time"] - h1["expected_expiration_time"]).dt.total_seconds() / 60
           ).groupby(h1["series"]).median()
    m["anchor"] = m["expected_expiration_time"] + pd.to_timedelta(m["series"].map(off).fillna(0), unit="m")
    m["half"] = np.where(m["close_time"] < cut, "H1", "H2")

    client = KalshiClient()
    cutoff = cutoff_ts(client)["market_settled_ts"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        candles = list(pool.map(lambda r: fetch(client, r, cutoff), m.itertuples(index=False)))

    rows = []
    for r, k in zip(m.itertuples(index=False), candles):
        if k.empty:
            continue
        k = k.dropna(subset=["bid", "ask"]).sort_values("ts")
        k = k[k["ts"].dt.minute % 5 == 0]
        k["mid"] = (k["bid"] + k["ask"]) / 2
        hit = k[(k["mid"] > 0.02) & (k["mid"] <= 0.10) & (k["ask"] - k["bid"] <= 0.04)]
        if hit.empty:
            continue
        h = hit.iloc[0]
        rows.append({"ticker": r.ticker, "series": r.series, "half": r.half,
                     "mid": h["mid"], "ask": h["ask"], "bid": h["bid"],
                     "won": float(r.result == "yes")})
    s = pd.DataFrame(rows)
    s.to_parquet(DATA_CACHE_DIR / "endgame_snapshot_calib.parquet", index=False)

    def report(d, label):
        n, k = len(d), int(d["won"].sum())
        if n == 0:
            return
        mu = d["mid"].sum()
        z = (k - mu) / np.sqrt((d["mid"] * (1 - d["mid"])).sum())
        p = stats.binomtest(k, n, d["mid"].mean(), alternative="less").pvalue
        # 卖冷门（按 bid 成交 = taker 卖出）每张收益，含 taker 费 0.07p(1-p)
        pnl = d["bid"] - d["won"] - 0.07 * d["bid"] * (1 - d["bid"])
        print(f"{label:22s} n={n:4d} 逆转 {k:3d} ({k/n:.2%}) vs 隐含 {d['mid'].mean():.2%}"
              f" (期望 {mu:.1f})  z={z:5.2f} p={p:.3g} | 按bid卖出每张 {pnl.mean():+.4f}")

    report(s, "全部")
    for h in ("H1", "H2"):
        report(s[s["half"] == h], h)
    for ser, g in s.groupby("series"):
        report(g, ser)
