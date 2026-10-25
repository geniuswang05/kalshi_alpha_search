"""分层抽样成交明细，用于 maker/taker 收益分析。

每个系列随机抽 N 个已结算市场；每个市场按"距结算小时数"切 9 个窗口，
每个窗口抓最近一页（≤1000笔）成交。按窗口分别分析，避免跨窗口加权问题。

输出: data_cache/trade_samples/{series}.parquet
用法: python scripts/sample_trades.py [每系列市场数，默认40]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import SLEEP, cutoff_ts, get_with_retry, settled_markets

OUT_DIR = DATA_CACHE_DIR / "trade_samples"
OUT_DIR.mkdir(exist_ok=True)

# (起, 止) 小时数，距 close_time
WINDOWS = [(48, 24), (24, 12), (12, 6), (6, 4), (4, 3), (3, 2), (2, 1), (1, 0.5), (0.5, 0)]

SERIES = ["KXNBAGAME", "KXNFLGAME", "KXMLBGAME", "KXNHLGAME", "KXWNBAGAME", "KXNCAAFGAME",
          "KXNCAAMBGAME", "KXEPLGAME", "KXMLSGAME", "KXUCLGAME", "KXLALIGAGAME", "KXLIGAMXGAME",
          "KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXUFCFIGHT", "KXIPLGAME",
          "KXT20MATCH", "KXCS2GAME", "KXLOLGAME", "KXVALORANTGAME", "KXNPBGAME", "KXKBOGAME",
          "KXEUROLEAGUEGAME", "KXTTELITEMATCH", "KXDARTSMATCH", "KXKHLGAME"]


PER_MARKET_DIR = OUT_DIR / "_per_market"
PER_MARKET_DIR.mkdir(exist_ok=True)


def sample_market(client, series, m, trade_cutoff) -> pd.DataFrame:
    cache = PER_MARKET_DIR / f"{m.ticker}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows, close = [], m.close_time
    for lo_h, hi_h in WINDOWS:
        t0 = close - pd.Timedelta(hours=lo_h)
        t1 = close - pd.Timedelta(hours=hi_h)
        if t1 <= m.open_time:
            continue
        t0 = max(t0, m.open_time)
        path = "/historical/trades" if t1 < trade_cutoff else "/markets/trades"
        r = get_with_retry(client, path, {
            "ticker": m.ticker, "limit": 1000,
            "min_ts": int(t0.timestamp()), "max_ts": int(t1.timestamp())})
        for t in r.get("trades", []):
            rows.append({
                "series": series, "ticker": m.ticker, "result": m.result,
                "close_time": close, "window": f"{lo_h}-{hi_h}h",
                "created_time": t["created_time"],
                "yes_price": float(t["yes_price_dollars"]),
                "count": float(t["count_fp"]),
                "taker_side": t.get("taker_outcome_side") or t.get("taker_side"),
                "is_block": bool(t.get("is_block_trade", False)),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df


def sample_series(client, series, n_markets, trade_cutoff, rng, pool):
    out = OUT_DIR / f"{series}.parquet"
    if out.exists():
        return
    mk = settled_markets(client, series)
    mk = mk[mk["result"].isin(["yes", "no"]) & (mk["volume_fp"] > 0)
            & (mk["close_time"] >= "2026-01-01")]
    if mk.empty:
        return
    pick = mk.sample(min(n_markets, len(mk)), random_state=int(rng.integers(1e9)))
    futures = [pool.submit(sample_market, client, series, m, trade_cutoff)
               for m in pick.itertuples(index=False)]
    frames = [f.result() for f in futures]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    df.to_parquet(out, index=False)
    print(f"{series}: {len(pick)} markets, {len(df)} trades", flush=True)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    client = KalshiClient()
    trade_cutoff = cutoff_ts(client)["trades_created_ts"]
    rng = np.random.default_rng(7)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for s in SERIES:
            try:
                sample_series(client, s, n, trade_cutoff, rng, pool)
            except Exception as e:
                print(f"{s}: FAILED {type(e).__name__}: {e}", flush=True)
