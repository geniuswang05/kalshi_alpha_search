"""补贴结束事件的对照组：同一时段补贴持续进行的市场（与处理组同频率采集订单簿）。

优先选与处理组同系列的市场；补贴结束时间必须晚于整个采集窗口 + 8 小时。
输出写入 data_cache/lip_event/books_controls.jsonl，市场清单写 controls_active.parquet。
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.kalshi_client import KalshiClient
from data.kalshi_universe import get_with_retry
from strategies.lip_event_study import OUT, POLL_MIN, POST_HOURS, snap
from strategies.lip_scanner import parse_side

N_CTRL = 150

if __name__ == "__main__":
    client = KalshiClient()
    mk = pd.read_parquet(OUT / "markets.parquet")
    stop_at = mk["event_time"].max() + pd.Timedelta(hours=POST_HOURS)
    path = OUT / "controls_active.parquet"
    if path.exists():
        ctrl = pd.read_parquet(path)
    else:
        rows, cur = [], None
        for _ in range(60):
            p = {"status": "active", "type": "liquidity", "limit": 10000}
            if cur:
                p["cursor"] = cur
            r = get_with_retry(client, "/incentive_programs", p)
            rows += r.get("incentive_programs", [])
            cur = r.get("next_cursor") or r.get("cursor")
            if not cur:
                break
        act = pd.DataFrame(rows)
        act["end"] = pd.to_datetime(act["end_date"], utc=True, format="ISO8601")
        ends = act.groupby("market_ticker")["end"].max()
        cand = ends[ends > stop_at + pd.Timedelta(hours=8)].index.difference(mk["ticker"])
        cand = pd.DataFrame({"ticker": cand})
        cand["series"] = cand["ticker"].str.split("-").str[0]
        treat_series = set(mk.loc[mk["group"] == "end", "series"])
        cand["same_series"] = cand["series"].isin(treat_series)
        cand = cand.sample(frac=1.0, random_state=9).sort_values("same_series", ascending=False)
        cand = cand.groupby("series").head(2)
        sel = []
        for r in cand.itertuples(index=False):
            try:
                m = get_with_retry(client, f"/markets/{r.ticker}", {}).get("market", {})
                close = pd.Timestamp(m.get("close_time"))
                if m.get("status") not in ("active", "open") or close < stop_at:
                    continue
                ob = get_with_retry(client, f"/markets/{r.ticker}/orderbook", {"depth": 0}).get("orderbook_fp") or {}
                if not parse_side(ob.get("yes_dollars")) or not parse_side(ob.get("no_dollars")):
                    continue
            except Exception:
                continue
            sel.append({"ticker": r.ticker, "series": r.series, "group": "control_active",
                        "same_series": r.same_series, "close_time": close})
            if len(sel) >= N_CTRL:
                break
        ctrl = pd.DataFrame(sel)
        ctrl.to_parquet(path, index=False)
    print(f"持续补贴对照组 {len(ctrl)} 个（同系列 {int(ctrl['same_series'].sum())} 个），采集到 {stop_at:%m-%d %H:%M}", flush=True)

    out = open(OUT / "books_controls.jsonl", "a")
    with ThreadPoolExecutor(max_workers=3) as pool:
        while pd.Timestamp.now(tz="UTC") < stop_at:
            t0 = time.time()
            now = pd.Timestamp.now(tz="UTC")
            books = list(pool.map(lambda t: snap(client, t), ctrl["ticker"]))
            for t, b in zip(ctrl["ticker"], books):
                if b is not None:
                    out.write(json.dumps({"ts": now.isoformat(), "ticker": t, "book": b}) + "\n")
            out.flush()
            print(f"{now:%m-%d %H:%M} 对照快照 {sum(b is not None for b in books)}/{len(ctrl)}", flush=True)
            time.sleep(max(0, POLL_MIN * 60 - (time.time() - t0)))
