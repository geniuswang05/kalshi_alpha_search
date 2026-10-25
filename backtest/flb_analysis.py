"""多头-冷门偏差（favorite-longshot bias）的正式检验 —— 体育市场。

两个可执行方向：
A. maker 卖冷门：taker 买入价格 ≤ x 的冷门合约时，我们是挂单对手方（被动卖出冷门）。
   每张收益 = p − 1{冷门赢} − maker_fee
B. taker 买热门：主动吃 ask 买价格 ≥ 1−x 的热门。每张收益 = 1{热门赢} − p − taker_fee

分组：价位 × 阶段（赛前 >3h / 最后 3h）× 联赛类型（主流=收 maker 费 / 长尾=不收）。
样本外：按市场结算时间中位数切成前后两半，两半分别算——偏差稳定存在才算数。
t 统计量按市场聚类（同一市场的所有成交共享结算结果）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.maker_taker import FEES, load_trades

LONGSHOT_BINS = [0, 0.03, 0.06, 0.10, 0.15, 0.20]


def cluster_stats(g: pd.DataFrame, col: str) -> pd.Series:
    per = g.groupby("ticker").apply(
        lambda x: np.average(x[col], weights=x["count"]), include_groups=False)
    n = len(per)
    se = per.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return pd.Series({"markets": n, "contracts": g["count"].sum(),
                      "ret": np.average(g[col], weights=g["count"]),
                      "eq_ret": per.mean(), "t": per.mean() / se if se and se > 0 else np.nan})


def prep(complete_only_windows: set | None = None):
    df = load_trades(complete_only_windows)
    df["group"] = df["series"].map(
        lambda s: "major(maker-fee)" if FEES.get(s, ["quadratic"])[0] != "quadratic" else "longtail(no-maker-fee)")
    df["stage"] = np.where(df["phase"] == "last3h", "last3h", "pre>3h")
    close = df.groupby("ticker")["close_time"].first()
    cut = close.median()
    df["half"] = np.where(df["ticker"].map(close) < cut, "H1", "H2")
    # 冷门侧价格：taker 买入的那一侧若 ≤0.5 即为冷门
    df["ls_price"] = np.minimum(df["p_taker"], 1 - df["p_taker"])
    df["taker_bought_longshot"] = df["p_taker"] <= 0.5
    df["ls_bin"] = pd.cut(df["ls_price"], LONGSHOT_BINS, include_lowest=True)
    return df, cut


EXCLUDE_ANCHOR = {"KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXUFCFIGHT"}


def endgame_exante(df: pd.DataFrame, cut, pre_min=45, post_min=30, lo=0.02, hi=0.10):
    """事前可执行版本：终局窗口锚定在 expected_expiration_time（建市场时公布）+ 联赛偏移，
    偏移量只用 H1 市场估计（close − expected_expiration 的中位数）。"""
    from config.settings import DATA_CACHE_DIR
    meta = pd.read_parquet(DATA_CACHE_DIR / "sample_market_meta.parquet")[
        ["ticker", "expected_expiration_time"]]
    d = df.merge(meta, on="ticker", how="inner")
    d = d[~d["series"].isin(EXCLUDE_ANCHOR)]
    mk = d.groupby("ticker").agg(series=("series", "first"), close=("close_time", "first"),
                                 ee=("expected_expiration_time", "first"))
    h1 = mk[mk["close"] < cut]
    offset = ((h1["close"] - h1["ee"]).dt.total_seconds() / 60).groupby(h1["series"]).median()
    d["anchor"] = d["expected_expiration_time"] + pd.to_timedelta(
        d["series"].map(offset).fillna(0), unit="m")
    ts = pd.to_datetime(d["created_time"], utc=True, format="ISO8601")
    rel = (ts - d["anchor"]).dt.total_seconds() / 60
    sel = d[(rel >= -pre_min) & (rel <= post_min) & d["taker_bought_longshot"]
            & (d["ls_price"] > lo) & (d["ls_price"] <= hi)]
    return sel, offset


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df, cut = prep()

    print("== 事前可执行版：终局窗口锚定 expected_expiration_time（偏移量仅用 H1 估计）==")
    sel, offset = endgame_exante(df, cut)
    print("   联赛偏移(分钟):", offset.round(0).to_dict())
    print(sel.groupby("half").apply(lambda g: cluster_stats(g, "maker_ret"),
                                    include_groups=False).round(4).to_string())
    print(sel.groupby("series").apply(lambda g: cluster_stats(g, "maker_ret"),
                                      include_groups=False).sort_values("t", ascending=False).round(4).to_string())
    print()
    print(f"{len(df):,} 笔成交, {df['ticker'].nunique()} 个市场, 切分点 {cut:%Y-%m-%d}\n")

    a = df[df["taker_bought_longshot"]]
    print("== A. maker 卖冷门（taker 买冷门时做对手方）每张净收益 ==")
    tab = a.groupby(["ls_bin", "stage"], observed=True).apply(
        lambda g: cluster_stats(g, "maker_ret"), include_groups=False)
    print(tab.round(4).to_string())

    print("\n   按联赛类型（冷门价 ≤ 0.10）:")
    a10 = a[a["ls_price"] <= 0.10]
    print(a10.groupby(["group", "stage"]).apply(
        lambda g: cluster_stats(g, "maker_ret"), include_groups=False).round(4).to_string())

    print("\n   样本外稳定性（冷门价 ≤ 0.10）:")
    print(a10.groupby(["half", "stage"]).apply(
        lambda g: cluster_stats(g, "maker_ret"), include_groups=False).round(4).to_string())

    b = df[~df["taker_bought_longshot"]]
    print("\n== B. taker 买热门 每张净收益（含 taker 费）==")
    print(b.groupby(["ls_bin", "stage"], observed=True).apply(
        lambda g: cluster_stats(g, "taker_ret"), include_groups=False).round(4).to_string())
    b10 = b[b["ls_price"] <= 0.10]
    print("\n   样本外稳定性（热门价 ≥ 0.90）:")
    print(b10.groupby(["half", "stage"]).apply(
        lambda g: cluster_stats(g, "taker_ret"), include_groups=False).round(4).to_string())

    print("\n== 按系列：maker 卖冷门（≤0.10，全阶段）==")
    print(a10.groupby("series").apply(
        lambda g: cluster_stats(g, "maker_ret"), include_groups=False)
          .sort_values("t", ascending=False).round(4).to_string())
