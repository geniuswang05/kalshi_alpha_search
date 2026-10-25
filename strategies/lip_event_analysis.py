"""LIP 前瞻事件研究分析：补贴结束（及开始）前后订单簿的变化。

每个快照计算（按官方计分规则）：
  spread          YES 买卖价差
  depth_best      两侧最优价上的挂单量之和
  depth_3c        两侧最优价 3¢ 以内的挂单量之和
  qual_score      两侧合格挂单总分（达到 Target Size 所纳入的价位，按 DF 折扣）——竞争强度
  entrant_share   一个新挂单者在两侧最优价各挂 Target Size 时能分到的份额
事件时间 τ = 快照时间 − 补贴结束（或开始）时间，单位小时。
对照：同一时刻"补贴持续进行"的市场的平均值（按 10 分钟时间桶匹配），处理组指标先除以自身事件前均值。

用法: python strategies/lip_event_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategies.lip_event_study import OUT
from strategies.lip_scanner import parse_side, side_scores


def metrics(book: dict, target: float, df: float) -> dict | None:
    y, n = parse_side(book.get("yes_dollars")), parse_side(book.get("no_dollars"))
    if not y or not n:
        return {"two_sided": 0}
    by, bn = y[0][0], n[0][0]
    d3 = sum(s for p, s in y if p >= by - 0.03 + 1e-9) + sum(s for p, s in n if p >= bn - 0.03 + 1e-9)
    ok_y, _, ty = side_scores(y, target, df)
    ok_n, _, tn = side_scores(n, target, df)
    ok_y2, oy, ty2 = side_scores(y, target, df, (by, target))
    ok_n2, on, tn2 = side_scores(n, target, df, (bn, target))
    share = 0.5 * (oy / ty2 + on / tn2) if ok_y2 and ok_n2 and ty2 > 0 and tn2 > 0 else np.nan
    return {"two_sided": 1, "spread": round(1 - by - bn, 4), "depth_best": y[0][1] + n[0][1], "depth_3c": d3,
            "qual_score": (ty if ok_y else 0) + (tn if ok_n else 0), "entrant_share": share}


def load():
    mk = pd.read_parquet(OUT / "markets.parquet")
    ctrl_path = OUT / "controls_active.parquet"
    if ctrl_path.exists():
        c = pd.read_parquet(ctrl_path)
        c["event_time"], c["target"], c["df_bps"] = pd.NaT, 1000.0, 5000.0
        mk = pd.concat([mk, c[["ticker", "series", "group", "event_time", "target", "df_bps", "close_time"]]],
                       ignore_index=True)
    info = mk.drop_duplicates("ticker").set_index("ticker")
    rows = []
    for fname in ("books.jsonl", "books_controls.jsonl"):
        f = OUT / fname
        if not f.exists():
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r["ticker"]
            if t not in info.index:
                continue
            m = metrics(r["book"], float(info.at[t, "target"]), float(info.at[t, "df_bps"]) / 10000)
            rows.append({"ts": pd.Timestamp(r["ts"]), "ticker": t, **m})
    d = pd.DataFrame(rows)
    d = d.merge(info[["group", "event_time", "series"]], left_on="ticker", right_index=True)
    d["event_time"] = pd.to_datetime(d["event_time"], utc=True, errors="coerce")
    d["tau"] = (d["ts"] - d["event_time"]).dt.total_seconds() / 3600
    d["bucket"] = d["ts"].dt.floor("10min")
    return d


METRICS = ["spread", "depth_best", "depth_3c", "qual_score", "entrant_share", "two_sided"]


def event_study(d: pd.DataFrame, event: str):
    tr = d[d["group"] == event].copy()
    if tr.empty:
        print(f"\n[{event}] 无数据")
        return
    ctrl = d[d["group"].isin(["control_active", "control"])]
    cmean = ctrl.groupby("bucket")[METRICS].mean()
    pre = tr[(tr["tau"] < 0) & (tr["tau"] >= -6)]
    have_post = tr[tr["tau"] > 0]["ticker"].unique()
    print(f"\n===== 补贴{'结束' if event == 'end' else '开始'}：{tr['ticker'].nunique()} 个市场，"
          f"已过事件时点的 {len(have_post)} 个 =====")
    bins = [-6, -3, -1, 0, 1, 3, 6, 8]
    tr["tbin"] = pd.cut(tr["tau"], bins)
    prof = tr.groupby("tbin", observed=True)[METRICS].median()
    print("事件时间剖面（中位数）:")
    print(prof.round(4).to_string())

    # 每个市场：事件后(0,8h] − 事件前[-6,0) 的变化，减去对照组同时段的变化
    out = []
    for t, g in tr.groupby("ticker"):
        a, b = g[(g["tau"] >= -6) & (g["tau"] < 0)], g[(g["tau"] > 0) & (g["tau"] <= 8)]
        if len(a) < 3 or len(b) < 3:
            continue
        row = {"ticker": t}
        for k in METRICS:
            ca = cmean.reindex(a["bucket"])[k].mean() if len(cmean) else np.nan
            cb = cmean.reindex(b["bucket"])[k].mean() if len(cmean) else np.nan
            row[f"{k}_chg"] = b[k].mean() - a[k].mean()
            row[f"{k}_did"] = (b[k].mean() - a[k].mean()) - (cb - ca)
            row[f"{k}_pre"] = a[k].mean()
        out.append(row)
    x = pd.DataFrame(out)
    if x.empty:
        print("  事件后数据还不够（需要事件后 ≥3 个快照）")
        return x
    print(f"\n双重差分（{len(x)} 个市场；事件后 8h − 事件前 6h，减去持续补贴对照组同期变化）:")
    for k in METRICS:
        v = x[f"{k}_did"].dropna()
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
        pre = x[f"{k}_pre"].mean()
        rel = v.mean() / pre if pre else np.nan
        print(f"  {k:14s} 事件前均值 {pre:10.4f}  DiD {v.mean():+10.4f}（相对 {rel:+.1%}） t={t:5.2f}  n={len(v)}")
    return x


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    d = load()
    print(f"快照 {len(d):,} 个，市场 {d['ticker'].nunique()} 个，时间 {d['ts'].min():%m-%d %H:%M} ~ {d['ts'].max():%m-%d %H:%M}")
    res_end = event_study(d, "end")
    res_start = event_study(d, "start")
    if res_end is not None and len(res_end):
        res_end.to_parquet(OUT / "did_end.parquet", index=False)
