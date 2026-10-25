"""LIP 历史事件研究（小时 K 线，大样本）：补贴开始/结束前后价差与成交量的变化，双重差分。

处理组：
  start —— 市场已开盘 ≥24h 后才开始的补贴；
  end   —— 补贴结束后市场仍交易 ≥24h，且没有其他计划接续。
对照组：同系列、从未有过补贴、在同一时段开盘的市场，赋予同一个"伪事件时间"。
指标：每小时收盘的 YES 买卖价差（ask − bid）、是否双边有报价、成交量。
DiD = (处理组 事件后 − 事件前) − (对照组 事件后 − 事件前)，按事件聚类。

用法: python strategies/lip_history_study.py [每系列事件数上限，默认 25]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import cutoff_ts, get_with_retry, market_meta, parse_candles

OUT = DATA_CACHE_DIR / "lip_history"
OUT.mkdir(exist_ok=True)
WIN_H = 24


def build_events(client, per_series: int) -> pd.DataFrame:
    path = OUT / "events.parquet"
    if path.exists():
        return pd.read_parquet(path)
    allp = pd.read_parquet(DATA_CACHE_DIR / "incentives_all.parquet")
    allp["series"] = allp["market_ticker"].str.split("-").str[0]
    now = pd.Timestamp.now(tz="UTC")
    allp = allp[(allp["start"] >= "2026-08-05") & (allp["end"] <= now - pd.Timedelta(days=2))]
    cover_end = allp.groupby("market_ticker")["end"].max()
    first_start = allp.groupby("market_ticker")["start"].min()

    cand = allp.sample(frac=1.0, random_state=5).drop_duplicates("market_ticker")
    cand = cand.groupby("series").head(per_series * 3)
    meta = market_meta(client, cand["market_ticker"].tolist()).set_index("ticker")
    cand = cand[cand["market_ticker"].isin(meta.index)]
    cand["open_time"] = cand["market_ticker"].map(meta["open_time"])
    cand["close_time"] = cand["market_ticker"].map(meta["close_time"])

    ev = []
    for r in cand.itertuples(index=False):
        if r.start == first_start[r.market_ticker] and r.open_time <= r.start - pd.Timedelta(hours=WIN_H) \
                and r.close_time >= r.start + pd.Timedelta(hours=WIN_H):
            ev.append({"ticker": r.market_ticker, "series": r.series, "event": "start", "t": r.start,
                       "reward_per_day": r.reward / max((r.end - r.start).total_seconds() / 86400, 1 / 24)})
        if r.end == cover_end[r.market_ticker] and r.close_time >= r.end + pd.Timedelta(hours=WIN_H) \
                and r.start <= r.end - pd.Timedelta(hours=WIN_H):
            ev.append({"ticker": r.market_ticker, "series": r.series, "event": "end", "t": r.end,
                       "reward_per_day": r.reward / max((r.end - r.start).total_seconds() / 86400, 1 / 24)})
    ev = pd.DataFrame(ev)
    ev = ev.groupby(["series", "event"]).head(per_series).reset_index(drop=True)
    ev["event_id"] = ev.index

    # 对照：同系列、从未补贴、事件窗口内一直开盘
    inc_mk = set(allp["market_ticker"]) | set(pd.read_parquet(DATA_CACHE_DIR / "incentives_all.parquet")["market_ticker"])
    ctrl = []
    for s, g in ev.groupby("series"):
        try:
            mk = []
            for status in ("settled", "open"):
                r = get_with_retry(client, "/markets", {"series_ticker": s, "status": status, "limit": 1000})
                mk += r.get("markets", [])
            r = get_with_retry(client, "/historical/markets", {"series_ticker": s, "limit": 1000})
            mk += r.get("markets", [])
        except Exception:
            continue
        pool = pd.DataFrame([{"ticker": m["ticker"], "open_time": pd.Timestamp(m["open_time"]),
                              "close_time": pd.Timestamp(m["close_time"])} for m in mk
                             if m["ticker"] not in inc_mk and m.get("open_time") and m.get("close_time")])
        if pool.empty:
            continue
        for e in g.itertuples(index=False):
            ok = pool[(pool["open_time"] <= e.t - pd.Timedelta(hours=WIN_H))
                      & (pool["close_time"] >= e.t + pd.Timedelta(hours=WIN_H))]
            for c in ok.sample(min(2, len(ok)), random_state=e.event_id).itertuples(index=False):
                ctrl.append({"ticker": c.ticker, "series": s, "event": e.event, "t": e.t,
                             "reward_per_day": 0.0, "event_id": e.event_id, "group": "control"})
        time.sleep(0.1)
    ev["group"] = "treat"
    out = pd.concat([ev, pd.DataFrame(ctrl)], ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def candles_for(client, row, cutoff) -> pd.DataFrame:
    f = OUT / "candles" / f"{row.ticker}__{int(row.t.timestamp())}.parquet"
    f.parent.mkdir(exist_ok=True)
    if f.exists():
        return pd.read_parquet(f)
    t0, t1 = int((row.t - pd.Timedelta(hours=WIN_H)).timestamp()), int((row.t + pd.Timedelta(hours=WIN_H)).timestamp())
    try:
        r = get_with_retry(client, f"/historical/markets/{row.ticker}/candlesticks",
                           {"start_ts": t0, "end_ts": t1, "period_interval": 60})
        k = parse_candles(r.get("candlesticks", []))
        if k.empty:
            raise ValueError
    except Exception:
        try:
            r = get_with_retry(client, f"/series/{row.series}/markets/{row.ticker}/candlesticks",
                               {"start_ts": t0, "end_ts": t1, "period_interval": 60})
            k = parse_candles(r.get("candlesticks", []))
        except Exception:
            k = pd.DataFrame()
    k.to_parquet(f, index=False)
    return k


def analyze(ev: pd.DataFrame, candles: list):
    rows = []
    for r, k in zip(ev.itertuples(index=False), candles):
        if k is None or k.empty:
            continue
        k = k.copy()
        k["tau"] = ((k["ts"] - r.t).dt.total_seconds() / 3600).round()
        k["two_sided"] = k["bid"].notna() & k["ask"].notna() & (k["bid"] > 0) & (k["ask"] < 1)
        k["spread"] = np.where(k["two_sided"], k["ask"] - k["bid"], np.nan)
        for post, g in ((0, k[(k["tau"] < 0) & (k["tau"] >= -WIN_H)]), (1, k[(k["tau"] > 0) & (k["tau"] <= WIN_H)])):
            if g.empty:
                continue
            rows.append({"event_id": r.event_id, "event": r.event, "group": r.group, "series": r.series,
                         "post": post, "spread": g["spread"].mean(), "two_sided": g["two_sided"].mean(),
                         "volume": g["volume"].sum(), "reward_per_day": r.reward_per_day})
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "panel.parquet", index=False)

    def did(metric, event):
        x = d[d["event"] == event].pivot_table(index=["event_id", "group"], columns="post", values=metric).dropna()
        x["chg"] = x[1] - x[0]
        x = x["chg"].unstack("group").dropna(subset=["treat"])
        treat = x["treat"]
        ctrl = x["control"].dropna() if "control" in x else pd.Series(dtype=float)
        paired = x.dropna()
        dd = paired["treat"] - paired["control"] if len(paired) else pd.Series(dtype=float)
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 2 else np.nan
        return {"n_treat": len(treat), "treat_chg": treat.mean(), "ctrl_chg": ctrl.mean(),
                "n_pairs": len(dd), "DiD": dd.mean(), "t": t}

    print("\n== 双重差分（事件后 24h − 事件前 24h；处理组 − 对照组）==")
    for event in ("start", "end"):
        for metric, lab in (("spread", "价差（美元）"), ("two_sided", "双边有报价的小时占比"), ("volume", "成交量（张）")):
            r = did(metric, event)
            print(f"  {event:5s} {lab:16s} 处理组变化 {r['treat_chg']:+.4f}  对照组变化 {r['ctrl_chg']:+.4f}  "
                  f"DiD {r['DiD']:+.4f}  t={r['t']:.2f}  (n_treat={r['n_treat']}, 配对={r['n_pairs']})")


if __name__ == "__main__":
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    client = KalshiClient()
    ev = build_events(client, per)
    print(f"事件：start {int(((ev.group=='treat')&(ev.event=='start')).sum())}，end {int(((ev.group=='treat')&(ev.event=='end')).sum())}，"
          f"对照 {int((ev.group=='control').sum())}", flush=True)
    cutoff = cutoff_ts(client)["market_settled_ts"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        candles = list(pool.map(lambda r: candles_for(client, r, cutoff), ev.itertuples(index=False)))
    analyze(ev, candles)
