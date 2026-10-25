"""谁在 Kalshi 赚钱？maker vs taker 已实现收益分解。

每笔成交：taker 买入方向 X（taker_side）价格 p_X，maker 是对手方。
  taker 每张收益 = 1{X赢} − p_X − taker_fee
  maker 每张收益 = p_X − 1{X赢} − maker_fee
费率：taker = 0.07×mult×p(1−p)；maker = 0.25×taker（quadratic_with_maker_fees），
     'quadratic' 系列 maker 免费。

统计要点：同一市场所有成交共享结算结果（强相关），有效样本量 = 市场数。
所以先按市场聚合（张数加权），再在市场之间算均值和 t 统计量（市场聚类）。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR

SAMPLE_DIR = DATA_CACHE_DIR / "trade_samples"
FEES = json.load(open(DATA_CACHE_DIR / "series_fees.json"))

PRICE_BINS = [0, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0]
PHASES = {"48-24h": "pre>6h", "24-12h": "pre>6h", "12-6h": "pre>6h",
          "6-4h": "6-3h", "4-3h": "6-3h",
          "3-2h": "last3h", "2-1h": "last3h", "1-0.5h": "last3h", "0.5-0h": "last3h"}


COMPLETE_DIR = SAMPLE_DIR / "_complete"


def load_trades(complete_only_windows: set | None = None) -> pd.DataFrame:
    """抽样成交 + 用完整翻页数据替换被截断的窗口（见 scripts/complete_windows.py）。

    complete_only_windows: 若给定，这些窗口里仍被截断（未补全）的 (市场, 窗口) 整体丢弃，
    保证分析用到的终局窗口全部是完整数据。
    """
    frames = [pd.read_parquet(f) for f in sorted(SAMPLE_DIR.glob("*.parquet"))]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    comp_files = list(COMPLETE_DIR.glob("*.parquet")) if COMPLETE_DIR.exists() else []
    if comp_files:
        comp = pd.concat([pd.read_parquet(f) for f in comp_files], ignore_index=True)
        comp = comp[comp.get("complete", True) == True].drop(columns=["complete"], errors="ignore")
        key = set(zip(comp["ticker"], comp["window"]))
        df = df[~pd.Series(list(zip(df["ticker"], df["window"])), index=df.index).isin(key)]
        df = pd.concat([df, comp[df.columns]], ignore_index=True)
    else:
        key = set()
    if complete_only_windows:
        sat = pd.read_parquet(DATA_CACHE_DIR / "window_saturation.parquet")
        sat = sat[sat["saturated"] & sat["window"].isin(complete_only_windows)]
        bad = set(zip(sat["ticker"], sat["window"])) - key
        df = df[~pd.Series(list(zip(df["ticker"], df["window"])), index=df.index).isin(bad)]
    df = df[~df["is_block"] & df["taker_side"].isin(["yes", "no"])].copy()

    fee_type = df["series"].map(lambda s: FEES.get(s, ["quadratic", 1])[0])
    mult = df["series"].map(lambda s: float(FEES.get(s, ["quadratic", 1])[1] or 1))
    taker_yes = df["taker_side"] == "yes"
    df["p_taker"] = np.where(taker_yes, df["yes_price"], 1 - df["yes_price"])
    yes_won = (df["result"] == "yes").astype(float)
    df["taker_won"] = np.where(taker_yes, yes_won, 1 - yes_won)

    pq = df["yes_price"] * (1 - df["yes_price"])
    df["taker_fee"] = 0.07 * mult * pq
    df["maker_fee"] = np.where(fee_type == "quadratic_with_maker_fees", 0.25 * 0.07 * mult * pq, 0.0)
    df["taker_ret"] = df["taker_won"] - df["p_taker"] - df["taker_fee"]
    df["maker_ret"] = df["p_taker"] - df["taker_won"] - df["maker_fee"]
    df["maker_ret_gross"] = df["p_taker"] - df["taker_won"]
    df["phase"] = df["window"].map(PHASES)
    df["price_bin"] = pd.cut(df["p_taker"], PRICE_BINS, include_lowest=True)
    return df


def clustered(df: pd.DataFrame, col: str, by: list[str]) -> pd.DataFrame:
    """按 by 分组；组内先按市场张数加权聚合，再跨市场算均值/t。"""
    def agg(g):
        per_mkt = g.groupby("ticker").apply(
            lambda x: pd.Series({"r": np.average(x[col], weights=x["count"]),
                                 "w": x["count"].sum()}), include_groups=False)
        n = len(per_mkt)
        mean = np.average(per_mkt["r"], weights=per_mkt["w"]) if n else np.nan
        # 跨市场的简单均值 t（等权，稳健，不让单个巨量市场主导显著性）
        eq_mean = per_mkt["r"].mean()
        se = per_mkt["r"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        return pd.Series({"markets": n, "trades": len(g), "contracts": g["count"].sum(),
                          "vw_ret": mean, "eq_ret": eq_mean,
                          "t": eq_mean / se if se and se > 0 else np.nan})
    return df.groupby(by, observed=True).apply(agg, include_groups=False)


HORIZONS_S = [10, 60, 300, 1800]


def realized_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """maker 已实现价差：成交后 τ 秒的市场价相对成交价的盈亏（YES 计价，美元/张）。

    RS(τ) = s × (p − m(τ))，s=+1 表示 taker 买 YES（maker 卖出），m(τ) = (t+τ, t+τ+60s]
    内成交的张数加权均价（双边成交平均掉 bid/ask 跳动，近似中间价）。
    RS(10s) ≈ 半个价差；RS(10s) − RS(τ) = τ 内的逆向选择损失。
    """
    out = []
    df = df.copy()
    df["t"] = pd.to_datetime(df["created_time"], utc=True, format="ISO8601").astype("int64") / 1e9
    df["s"] = np.where(df["taker_side"] == "yes", 1.0, -1.0)
    for (ticker, window), g in df.groupby(["ticker", "window"], sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        p = g["yes_price"].to_numpy()
        w = g["count"].to_numpy()
        cum_w = np.concatenate([[0.0], np.cumsum(w)])
        cum_pw = np.concatenate([[0.0], np.cumsum(p * w)])
        res = {"ticker": ticker, "window": window}
        rec = g[["series", "ticker", "window", "count", "maker_fee", "yes_price", "s"]].copy()
        for h in HORIZONS_S:
            lo = np.searchsorted(t, t + h, side="right")
            hi = np.searchsorted(t, t + h + 60, side="right")
            ww = cum_w[hi] - cum_w[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.where(ww > 0, (cum_pw[hi] - cum_pw[lo]) / ww, np.nan)
            rec[f"rs_{h}"] = rec["s"] * (p - m)
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def rs_table(rs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """按 by 分组的张数加权已实现价差（扣 maker 费前/后），市场聚类 t。"""
    rows = []
    for key, g in rs.groupby(by, observed=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["markets"] = g["ticker"].nunique()
        row["maker_fee"] = np.average(g["maker_fee"], weights=g["count"])
        for h in HORIZONS_S:
            col = f"rs_{h}"
            v = g.dropna(subset=[col])
            if v.empty:
                continue
            per_mkt = v.groupby("ticker").apply(
                lambda x: np.average(x[col], weights=x["count"]), include_groups=False)
            row[f"RS{h}s"] = np.average(v[col], weights=v["count"])
            se = per_mkt.std(ddof=1) / np.sqrt(len(per_mkt)) if len(per_mkt) > 1 else np.nan
            row[f"t{h}"] = per_mkt.mean() / se if se and se > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df = load_trades()
    print(f"{len(df):,} 笔成交，{df['ticker'].nunique()} 个市场，{df['series'].nunique()} 个系列\n")

    print("== 按系列：maker 每张净收益（美元/张；正=maker赚）==")
    s = clustered(df, "maker_ret", ["series"]).sort_values("t", ascending=False)
    print(s.round(4).to_string())

    print("\n== 按阶段（距结算）：maker 净收益 ==")
    print(clustered(df, "maker_ret", ["phase"]).round(4).to_string())

    print("\n== 按 taker 买入价格（多头-冷门偏差检验）：taker 净收益 ==")
    print(clustered(df, "taker_ret", ["price_bin"]).round(4).to_string())
