"""终局冷门流动性提供策略（"卖奇迹彩票"）—— 组合级样本外回测。

规则（全部事前可知）：
- 终局窗口 = expected_expiration_time + 联赛偏移 的前 45 分钟 ~ 后 30 分钟（偏移只用 H1 估计）
- 在窗口内挂单卖出价格 2-10¢ 的冷门（即挂 NO 买单 @ 90-98¢），持有到结算
- 联赛白名单：只用 H1 选（H1 内市场聚类 t > T_SELECT 且均值 > 0）
- 成交假设：拿到窗口内每笔 taker 冷门买单的 PARTICIPATION 比例，每市场上限 CAP 张

评估在 H2（样本外）上做。抽样市场按"该系列同期全部已结算市场数 / 抽样数"放大，
估算全市场容量（每天能做多少钱）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.flb_analysis import cluster_stats, endgame_exante, prep
from config.settings import DATA_CACHE_DIR

T_SELECT = 2.0
PARTICIPATION = 0.10
CAP = 2000


def market_pnl(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tk, g in sel.sort_values("created_time").groupby("ticker"):
        fill = g["count"] * PARTICIPATION
        cum = fill.cumsum()
        keep = (cum - fill) < CAP
        fill = np.minimum(fill[keep], CAP - (cum[keep] - fill[keep]))
        gg = g[keep]
        rows.append({
            "ticker": tk, "series": gg["series"].iat[0], "half": gg["half"].iat[0],
            "date": gg["close_time"].iat[0].tz_convert("America/New_York").date(),
            "contracts": fill.sum(),
            "pnl": (fill * gg["maker_ret"]).sum(),
            "capital": (fill * (1 - gg["p_taker"])).sum(),
            "longshot_won": float(gg["taker_won"].iat[0]),
        })
    return pd.DataFrame(rows)


def scale_factors(markets: pd.DataFrame) -> pd.Series:
    """每个抽样市场代表该系列多少个真实市场（同一时间段内）。"""
    out = {}
    for s, g in markets.groupby("series"):
        f = DATA_CACHE_DIR / "markets" / f"{s}.parquet"
        if not f.exists():
            continue
        allm = pd.read_parquet(f)
        allm = allm[allm["result"].isin(["yes", "no"]) & (allm["close_time"] >= "2026-01-01")]
        out[s] = len(allm) / max(len(g), 1)
    return pd.Series(out)


def summarize(m: pd.DataFrame, label: str):
    d = m.groupby("date")["pnl"].sum()
    eq = d.cumsum()
    dd = (eq - eq.cummax()).min()
    sharpe = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan
    print(f"\n[{label}] 市场 {len(m)}，交易日 {len(d)}")
    print(f"   PnL ${m['pnl'].sum():,.0f}，资金周转 ${m['capital'].sum():,.0f}，"
          f"每次周转收益 {m['pnl'].sum()/m['capital'].sum():.2%}")
    print(f"   冷门逆转率 {m['longshot_won'].mean():.2%}；亏损市场 {(m['pnl']<0).sum()} 个"
          f"（最差 ${m['pnl'].min():,.0f}）")
    print(f"   日均 ${d.mean():,.1f}，日波动 ${d.std():,.1f}，年化夏普 {sharpe:.2f}，"
          f"最大回撤 ${dd:,.0f}，亏损日占比 {(d<0).mean():.1%}")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df, cut = prep(complete_only_windows={"4-3h", "3-2h", "2-1h", "1-0.5h", "0.5-0h"})
    sel, offset = endgame_exante(df, cut)

    h1 = sel[sel["half"] == "H1"]
    stats = h1.groupby("series").apply(lambda g: cluster_stats(g, "maker_ret"),
                                       include_groups=False)
    white = stats[(stats["t"] > T_SELECT) & (stats["eq_ret"] > 0) & (stats["markets"] >= 8)].index
    print("H1 选出的联赛白名单:", sorted(white))

    m = market_pnl(sel[sel["series"].isin(white)])
    summarize(m[m["half"] == "H1"], "H1 样本内（选联赛用的数据）")
    oos = m[m["half"] == "H2"]
    summarize(oos, "H2 样本外")

    print("\n   H2 按联赛:")
    print(oos.groupby("series").agg(markets=("pnl", "size"), pnl=("pnl", "sum"),
                                    roi=("pnl", "sum"), cap=("capital", "sum"),
                                    reversal=("longshot_won", "mean"))
          .assign(roi=lambda x: x["roi"] / x["cap"]).drop(columns="cap")
          .sort_values("pnl", ascending=False).round(3).to_string())

    sf = scale_factors(m)
    oos = oos.assign(w=oos["series"].map(sf))
    days = oos["date"].nunique()
    print(f"\n容量估算（按全部已结算市场放大，H2 期间 {days} 天）:")
    print(f"   全市场 PnL ≈ ${(oos['pnl']*oos['w']).sum():,.0f}，"
          f"日均 ≈ ${(oos['pnl']*oos['w']).sum()/days:,.0f}；"
          f"日均资金周转 ≈ ${(oos['capital']*oos['w']).sum()/days:,.0f}")
    m.to_parquet(DATA_CACHE_DIR / "endgame_market_pnl.parquet", index=False)
