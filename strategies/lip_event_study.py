"""LIP 前瞻事件研究：补贴开始前后，订单簿发生了什么？（只用公开数据，不下单）

样本（处理组，市场已开盘且有双边报价、事件后仍交易 ≥ POST_HOURS 小时）：
  start：未来 [MIN_LEAD, MAX_LEAD] 小时内补贴开始；end：补贴结束且没有其他计划接续。
对照组：同系列里没有任何激励计划的开盘市场（吸收同时段的整体流动性变化）。
每 POLL 分钟抓一次订单簿（每侧最优 40 档），处理组跟到事件后 POST_HOURS 小时。
原始订单簿以 JSON lines 保存，分析在 lip_event_analysis.py 里做。

用法: python strategies/lip_event_study.py [处理组市场数，默认 300]
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import get_with_retry
from strategies.lip_scanner import parse_side

OUT = DATA_CACHE_DIR / "lip_event"
OUT.mkdir(exist_ok=True)
MIN_LEAD, MAX_LEAD = 2.0, 20.0
POST_HOURS = 8.0
POLL_MIN = 10


def _events(client):
    """两类事件：已开盘市场的补贴开始（start）、补贴结束但市场继续交易（end）。"""
    now = pd.Timestamp.now(tz="UTC")

    def pull(status):
        rows, cur = [], None
        for _ in range(60):
            prm = {"status": status, "type": "liquidity", "limit": 10000}
            if cur:
                prm["cursor"] = cur
            r = get_with_retry(client, "/incentive_programs", prm)
            rows += r.get("incentive_programs", [])
            cur = r.get("next_cursor") or r.get("cursor")
            if not cur:
                break
        d = pd.DataFrame(rows)
        d["start"] = pd.to_datetime(d["start_date"], utc=True, format="ISO8601")
        d["end"] = pd.to_datetime(d["end_date"], utc=True, format="ISO8601")
        d["reward"] = d["period_reward"].astype(float) / 10000
        return d

    up = pull("upcoming")
    up.to_parquet(DATA_CACHE_DIR / "incentives_upcoming.parquet", index=False)
    up["event"], up["event_time"] = "start", up["start"]
    act = pull("active")
    act["event"], act["event_time"] = "end", act["end"]
    ev = pd.concat([up, act], ignore_index=True)
    ev["lead_h"] = (ev["event_time"] - now).dt.total_seconds() / 3600
    ev = ev[(ev["lead_h"] >= MIN_LEAD) & (ev["lead_h"] <= MAX_LEAD)]
    # 同一市场若有重叠计划（结束时另一个计划仍在进行），"结束"不是真正的补贴消失 → 剔除
    covering = act.groupby("market_ticker")["end"].max()
    ev = ev[~((ev["event"] == "end") & (ev["market_ticker"].map(covering) > ev["end"] + pd.Timedelta(minutes=1)))]
    ev["series"] = ev["market_ticker"].str.split("-").str[0]
    return ev.sort_values("event_time").drop_duplicates(["market_ticker", "event"])


def pick(client, n_treat: int):
    ev = _events(client)
    ev = ev.sample(frac=1.0, random_state=3).groupby(["series", "event"]).head(3)
    treat = []
    for r in ev.itertuples(index=False):
        try:
            m = get_with_retry(client, f"/markets/{r.market_ticker}", {}).get("market", {})
            if m.get("status") not in ("active", "open"):
                continue
            close = pd.Timestamp(m.get("close_time"))
            if close < r.event_time + pd.Timedelta(hours=POST_HOURS):
                continue
            ob = get_with_retry(client, f"/markets/{r.market_ticker}/orderbook", {"depth": 0}).get("orderbook_fp") or {}
            if not parse_side(ob.get("yes_dollars")) or not parse_side(ob.get("no_dollars")):
                continue
        except Exception:
            continue
        treat.append({"ticker": r.market_ticker, "series": r.series, "group": r.event,
                      "event_time": r.event_time, "reward": r.reward,
                      "target": float(r.target_size_fp or 1000), "df_bps": float(r.discount_factor_bps or 5000),
                      "close_time": close})
        if len(treat) >= n_treat:
            break
    treat = pd.DataFrame(treat)

    # 对照组：处理组系列里、没有任何激励计划的开盘市场
    allp = set(pd.read_parquet(DATA_CACHE_DIR / "incentives_all.parquet")["market_ticker"])
    allp |= set(pd.read_parquet(DATA_CACHE_DIR / "incentives_upcoming.parquet")["market_ticker"])
    ctrl = []
    for s_ in treat["series"].unique():
        try:
            r = get_with_retry(client, "/markets", {"series_ticker": s_, "status": "open", "limit": 200})
        except Exception:
            continue
        k = 0
        for m in r.get("markets", []):
            if m["ticker"] in allp:
                continue
            ctrl.append({"ticker": m["ticker"], "series": s_, "group": "control",
                         "event_time": pd.NaT, "reward": 0.0, "target": 1000.0, "df_bps": 5000.0,
                         "close_time": pd.Timestamp(m.get("close_time"))})
            k += 1
            if k >= 2:
                break
    out = pd.concat([treat, pd.DataFrame(ctrl)], ignore_index=True)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True)
    return out


def snap(client, ticker):
    try:
        ob = get_with_retry(client, f"/markets/{ticker}/orderbook", {"depth": 0})
        b = ob.get("orderbook_fp") or {}
        # 价格升序排列，最优价在末尾；每侧只留最优 40 档（计分只涉及最优价附近）
        return {k: (b.get(k) or [])[-40:] for k in ("yes_dollars", "no_dollars")}
    except Exception:
        return None


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    client = KalshiClient()
    path_m = OUT / "markets.parquet"
    if path_m.exists():
        mk = pd.read_parquet(path_m)
    else:
        mk = pick(client, n)
        mk.to_parquet(path_m, index=False)
    print(f"补贴开始 {int((mk.group=='start').sum())} 个，补贴结束 {int((mk.group=='end').sum())} 个，"
          f"对照 {int((mk.group=='control').sum())} 个", flush=True)
    stop_at = mk["event_time"].max() + pd.Timedelta(hours=POST_HOURS)
    print(f"预计采集到 {stop_at:%Y-%m-%d %H:%M} UTC", flush=True)

    out = open(OUT / "books.jsonl", "a")
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pd.Timestamp.now(tz="UTC") < stop_at:
            t0 = time.time()
            now = pd.Timestamp.now(tz="UTC")
            live = mk[(mk["group"] == "control") | (mk["event_time"] + pd.Timedelta(hours=POST_HOURS) > now)]
            live = live[live["close_time"] > now]
            books = list(pool.map(lambda t: snap(client, t), live["ticker"]))
            for t, b in zip(live["ticker"], books):
                if b is not None:
                    out.write(json.dumps({"ts": now.isoformat(), "ticker": t, "book": b}) + "\n")
            out.flush()
            print(f"{now:%m-%d %H:%M} 快照 {sum(b is not None for b in books)}/{len(live)}", flush=True)
            time.sleep(max(0, POLL_MIN * 60 - (time.time() - t0)))
