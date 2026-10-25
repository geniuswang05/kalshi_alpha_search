"""大样本版终局快照校准：每系列最多 N 个已结算市场（2026 年），固定时钟快照，无成交量抽样偏差。

联赛偏移只用每系列结算时间较早的一半市场估计；报告前后两半分别的结果。
用法: python backtest/endgame_snapshots_large.py [每系列市场数，默认 600]
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.endgame_snapshots import OUT, fetch
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import cutoff_ts, market_meta, settled_markets

SERIES = ["KXNBAGAME", "KXWNBAGAME", "KXNFLGAME", "KXNCAAFGAME", "KXNHLGAME", "KXMLBGAME",
          "KXLALIGAGAME", "KXLIGAMXGAME", "KXEPLGAME", "KXUCLGAME", "KXMLSGAME",
          "KXNPBGAME", "KXKBOGAME", "KXVALORANTGAME", "KXCS2GAME", "KXLOLGAME", "KXIPLGAME"]


def build_universe(client, n_per: int) -> pd.DataFrame:
    path = DATA_CACHE_DIR / f"endgame_universe_{n_per}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for s in SERIES:
        mk = settled_markets(client, s)
        mk = mk[mk["result"].isin(["yes", "no"]) & (mk["close_time"] >= "2026-01-01")]
        mk = mk.sample(min(n_per, len(mk)), random_state=11)
        meta = market_meta(client, mk["ticker"].tolist())
        mk = mk.merge(meta[["ticker", "expected_expiration_time"]], on="ticker")
        cut = mk["close_time"].median()
        first = mk[mk["close_time"] < cut]
        off = ((first["close_time"] - first["expected_expiration_time"]).dt.total_seconds() / 60).median()
        mk["anchor"] = mk["expected_expiration_time"] + pd.Timedelta(minutes=float(off))
        mk["half"] = np.where(mk["close_time"] < cut, "H1", "H2")
        frames.append(mk[["ticker", "series", "result", "close_time", "anchor", "half"]])
        print(f"  {s}: {len(mk)} markets, offset {off:.0f} min", flush=True)
    u = pd.concat(frames, ignore_index=True)
    u.to_parquet(path, index=False)
    return u


def snapshot_rows(u: pd.DataFrame, candles: list) -> pd.DataFrame:
    rows = []
    for r, k in zip(u.itertuples(index=False), candles):
        if k is None or k.empty:
            continue
        k = k.dropna(subset=["bid", "ask"]).sort_values("ts")
        k = k[k["ts"].dt.minute % 5 == 0]
        k = k.assign(mid=(k["bid"] + k["ask"]) / 2)
        hit = k[(k["mid"] > 0.02) & (k["mid"] <= 0.10) & (k["ask"] - k["bid"] <= 0.04)]
        if hit.empty:
            continue
        h = hit.iloc[0]
        rows.append({"ticker": r.ticker, "series": r.series, "half": r.half, "mid": h["mid"],
                     "bid": h["bid"], "ask": h["ask"], "won": float(r.result == "yes")})
    return pd.DataFrame(rows)


def report(d: pd.DataFrame, label: str):
    n, k = len(d), int(d["won"].sum())
    if n == 0:
        return
    mu = d["mid"].sum()
    z = (k - mu) / np.sqrt((d["mid"] * (1 - d["mid"])).sum())
    p = stats.binomtest(k, n, d["mid"].mean(), alternative="less").pvalue
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95)
    pnl = d["bid"] - d["won"] - 0.07 * d["bid"] * (1 - d["bid"])
    print(f"{label:16s} n={n:5d} 逆转 {k:4d} ({k/n:.2%}, CI {lo:.1%}-{hi:.1%}) vs 隐含 {d['mid'].mean():.2%}"
          f"  z={z:5.2f} p={p:.3g} | taker按bid卖 每张 {pnl.mean():+.4f}", flush=True)


if __name__ == "__main__":
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    client = KalshiClient()
    u = build_universe(client, n_per)
    cutoff = cutoff_ts(client)["market_settled_ts"]
    print(f"{len(u)} 个市场，拉终局 1 分钟 K 线 ...", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        candles = list(pool.map(lambda r: fetch(client, r, cutoff), u.itertuples(index=False)))
    s = snapshot_rows(u, candles)
    s.to_parquet(DATA_CACHE_DIR / "endgame_snapshot_calib_large.parquet", index=False)
    report(s, "全部")
    for h in ("H1", "H2"):
        report(s[s["half"] == h], h)
    for ser, g in s.groupby("series"):
        report(g, ser)
