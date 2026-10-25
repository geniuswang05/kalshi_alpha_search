"""补全被 1000 笔上限截断的成交窗口（完整翻页）。

为什么必须补全：每个窗口只取"最近 1000 笔"时，一笔成交是否入样取决于它之后的成交量；
冷门逆转会引发巨量成交，把逆转前的冷门成交挤出样本 → 系统性低估冷门胜率（前视型抽样偏差）。
只用未截断窗口也不行（逆转窗口更容易被截断，等于把逆转剔除）。所以对截断窗口完整翻页。

输出: data_cache/trade_samples/_complete/{ticker}__{window}.parquet
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import cutoff_ts, get_with_retry
from scripts.sample_trades import PER_MARKET_DIR, WINDOWS

OUT = DATA_CACHE_DIR / "trade_samples" / "_complete"
OUT.mkdir(exist_ok=True)
TARGET_WINDOWS = {"4-3h", "3-2h", "2-1h", "1-0.5h", "0.5-0h"}
MAX_PAGES = 400
WIN_BOUNDS = {f"{a}-{b}h": (a, b) for a, b in WINDOWS}


def complete(client, ticker, window, meta, trade_cutoff) -> str:
    path = OUT / f"{ticker}__{window}.parquet"
    if path.exists():
        return "cached"
    lo_h, hi_h = WIN_BOUNDS[window]
    close, open_ = meta["close_time"], meta["open_time"]
    t0 = max(close - pd.Timedelta(hours=lo_h), open_)
    t1 = close - pd.Timedelta(hours=hi_h)
    api = "/historical/trades" if t1 < trade_cutoff else "/markets/trades"
    rows, cursor, pages = [], None, 0
    while pages < MAX_PAGES:
        p = {"ticker": ticker, "limit": 1000,
             "min_ts": int(t0.timestamp()), "max_ts": int(t1.timestamp())}
        if cursor:
            p["cursor"] = cursor
        r = get_with_retry(client, api, p)
        for t in r.get("trades", []):
            rows.append({"series": meta["series"], "ticker": ticker, "result": meta["result"],
                         "close_time": close, "window": window,
                         "created_time": t["created_time"],
                         "yes_price": float(t["yes_price_dollars"]),
                         "count": float(t["count_fp"]),
                         "taker_side": t.get("taker_outcome_side") or t.get("taker_side"),
                         "is_block": bool(t.get("is_block_trade", False))})
        cursor = r.get("cursor")
        pages += 1
        if not cursor or not r.get("trades"):
            break
    df = pd.DataFrame(rows)
    df["complete"] = pages < MAX_PAGES
    df.to_parquet(path, index=False)
    return f"{pages}p"


if __name__ == "__main__":
    sat = pd.read_parquet(DATA_CACHE_DIR / "window_saturation.parquet")
    sat = sat[sat["saturated"] & sat["window"].isin(TARGET_WINDOWS)]
    metas = {}
    for f in PER_MARKET_DIR.glob("*.parquet"):
        d = pd.read_parquet(f, columns=["ticker", "series", "result", "close_time"])
        if not d.empty:
            metas[d["ticker"].iat[0]] = d.iloc[0].to_dict()
    mm = pd.read_parquet(DATA_CACHE_DIR / "sample_market_meta.parquet")[["ticker", "open_time"]]
    opens = dict(zip(mm["ticker"], mm["open_time"]))
    client = KalshiClient()
    trade_cutoff = cutoff_ts(client)["trades_created_ts"]
    jobs = [(r.ticker, r.window) for r in sat.itertuples(index=False) if r.ticker in metas]
    print(f"{len(jobs)} 个截断窗口待补全", flush=True)

    def work(job):
        tk, w = job
        meta = dict(metas[tk])
        meta["open_time"] = opens.get(tk, meta["close_time"] - pd.Timedelta(days=30))
        try:
            return complete(client, tk, w, meta, trade_cutoff)
        except Exception as e:
            return f"error {type(e).__name__}"

    done = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for res in pool.map(work, jobs):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print("done", flush=True)
