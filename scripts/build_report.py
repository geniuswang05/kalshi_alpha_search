"""从研究数据生成项目报告页（report/kalshi_edge_atlas.html）。

图表数据全部来自 data_cache 下的分析产出；影子做市前向测试的曲线在测试进行中也可重建。
用法: python scripts/build_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR, PROJECT_ROOT

LANG = "en" if "--lang=en" in sys.argv or "en" in sys.argv[1:] else "zh"
OUT = PROJECT_ROOT / "report" / ("kalshi_edge_atlas_en.html" if LANG == "en" else "kalshi_edge_atlas.html")
OUT.parent.mkdir(exist_ok=True)
T = {
    "zh": {"trade": "成交时刻 · 572 个市场", "YES": "随机时刻 · 落后方 YES", "NO": "随机时刻 · 领先方 NO",
           "mk": "个市场",
           "runs": {"_off-1_v2": "v2 改善价格", "_off0_v2": "v2 加入最优价",
                    "_off-1": "v1 改善价格", "_off0": "v1 加入最优价"},
           "mm": ["主流联赛 · 赛前", "主流联赛 · 比赛中", "长尾赛事 · 赛前", "长尾赛事 · 比赛中"]},
    "en": {"trade": "At taker trades · 572 markets", "YES": "Clock snapshots · trailing side",
           "NO": "Clock snapshots · leading side", "mk": "markets",
           "runs": {"_off-1_v2": "v2 · improve 1 tick", "_off0_v2": "v2 · join best",
                    "_off-1": "v1 · improve 1 tick", "_off0": "v1 · join best"},
           "mm": ["Major leagues · pre-game", "Major leagues · in-game",
                  "Long-tail · pre-game", "Long-tail · in-game"]},
}[LANG]


def ci(k, n):
    lo, hi = stats.binomtest(int(k), int(n)).proportion_ci(0.95)
    return lo, hi


def reversal_data():
    snap = pd.read_parquet(DATA_CACHE_DIR / "endgame_snapshot_bothsides.parquet")
    rows = [  # 成交样本（完整翻页数据，backtest/flb_analysis.py 的事前终局窗口）
        {"label": T["trade"], "n": 572, "k": 11, "implied": 0.0552},
    ]
    for side in ("YES", "NO"):
        d = snap[snap["side"] == side]
        rows.append({"label": f"{T[side]} · {len(d):,} {T['mk']}", "n": len(d), "k": int(d["rev"].sum()),
                     "implied": float(d["p"].mean())})
    for r in rows:
        r["realized"] = r["k"] / r["n"]
        r["lo"], r["hi"] = ci(r["k"], r["n"])
    return rows


def shadow_runs():
    root = DATA_CACHE_DIR / "shadow_mm"
    from strategies.shadow_report import markouts
    names = T["runs"]
    runs = []
    for d in sorted(root.iterdir()):
        if not (d / "log.csv").exists():
            continue
        label = next((v for k, v in names.items() if d.name.endswith(k)), d.name)
        log = pd.read_csv(d / "log.csv", parse_dates=["ts"])
        log["t"] = log["ts"].dt.floor("5min")
        last = log.sort_values("ts").groupby(["t", "ticker"]).tail(1)
        agg = last.groupby("t")[["reward", "mtm"]].sum()
        agg["net"] = agg["reward"] + agg["mtm"]
        t0 = agg.index.min()
        hours = [(t - t0).total_seconds() / 3600 for t in agg.index]
        cands = pd.read_csv(d / "candidates.csv")
        fills = pd.read_csv(d / "fills.csv") if (d / "fills.csv").exists() else pd.DataFrame()
        fin = agg.iloc[-1]
        mo = markouts(log.drop(columns=["t"]), pd.read_csv(d / "fills.csv", parse_dates=["ts"])) \
            if (d / "fills.csv").exists() else pd.DataFrame()
        runs.append({
            "label": label, "id": d.name, "hours": hours,
            "net": agg["net"].round(2).tolist(), "reward": agg["reward"].round(2).tolist(),
            "mtm": agg["mtm"].round(2).tolist(),
            "final": {"hours": round(hours[-1], 1), "reward": round(float(fin["reward"]), 2),
                      "mtm": round(float(fin["mtm"]), 2), "net": round(float(fin["net"]), 2),
                      "capital": round(float(cands["capital"].sum())),
                      "markets": int(cands["ticker"].nunique()),
                      "fills": int(len(fills)),
                      "contracts": round(float(fills["qty"].sum())) if len(fills) else 0,
                      "mo30": round(float(mo["mo30"].sum()), 2) if len(mo) else 0.0,
                      "categories": int(cands["category"].nunique()) if "category" in cands else None},
        })
    order = [names["_off-1_v2"], names["_off0_v2"], names["_off-1"], names["_off0"]]
    runs.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 9)
    return runs


DATA = {
    "reversal": reversal_data(),
    "mm": [  # backtest/maker_taker.py rs_table：10 秒已实现价差 vs maker 费（¢/张）
        {"label": T["mm"][0], "rs": 0.072, "fee": 0.349},
        {"label": T["mm"][1], "rs": 0.015, "fee": 0.191},
        {"label": T["mm"][2], "rs": -0.036, "fee": 0.0},
        {"label": T["mm"][3], "rs": 0.116, "fee": 0.0},
    ],
    "leadlag": {  # strategies/xvenue_analysis.py，2,351 场 MLB，比赛中
        "h": [1, 3, 5, 10],
        "pm_follows": [0.921, 0.962, 0.972, 0.989],
        "kalshi_follows": [-0.012, -0.023, -0.031, -0.049],
    },
    "shadow": shadow_runs(),
    "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
}

TEMPLATE = Path(__file__).with_name(
    "report_template_en.html" if LANG == "en" else "report_template.html").read_text()
html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(DATA, ensure_ascii=False))
OUT.write_text(html)
print(f"-> {OUT}  ({len(html)/1024:.0f} KB)")
for r in DATA["shadow"]:
    print("  ", r["label"], r["final"])
