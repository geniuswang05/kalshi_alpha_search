"""跨平台领先-滞后与可执行交易检验（Kalshi 执行，Polymarket 只当信号）。

检验1 领先-滞后：gap(t) = pm_p(t) − k_mid(t)。回归 Δk_mid(t→t+h) = β·gap(t)，
      β>0 且显著 ⇒ Kalshi 向 Polymarket 收敛（PM 领先）。按比赛聚类 t。
检验2 交易：gap > θ 时在 t+1 分钟按 Kalshi ask 买 YES；gap < −θ 时买 NO（= 1 − bid）。
      退出：持有 h 分钟后按对侧价平仓（taker 双边费），或持有到结算（只付开仓费）。
      每场比赛同一时间最多一笔持仓。θ、h 只在前半段（按日期）选，后半段样本外检验。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR

XV = DATA_CACHE_DIR / "xvenue"
FEES = json.load(open(DATA_CACHE_DIR / "series_fees.json"))
SERIES_OF = {"mlb": "KXMLBGAME", "nfl": "KXNFLGAME", "nhl": "KXNHLGAME", "wnba": "KXWNBAGAME"}


def load(league: str) -> pd.DataFrame:
    frames = []
    for f in sorted((XV / league).glob("*.parquet")):
        d = pd.read_parquet(f)
        if d.empty or "pm_p" not in d:
            continue
        d["game"] = f.stem
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["k_bid", "k_ask", "pm_p"])
    df = df[(df["k_ask"] > df["k_bid"]) & (df["k_bid"] > 0) & (df["k_ask"] < 1)]
    df["k_mid"] = (df["k_bid"] + df["k_ask"]) / 2
    df["gap"] = df["pm_p"] - df["k_mid"]
    df["in_game"] = df["ts"] >= df["start_time"]
    df["won"] = (df["result"] == "yes").astype(float)
    df = df.sort_values(["game", "ts"]).reset_index(drop=True)
    mult = float(FEES.get(SERIES_OF[league], ["quadratic", 1])[1] or 1)
    df["fee_mult"] = mult
    return df


def lead_lag(df: pd.DataFrame, horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    rows = []
    for phase, g0 in df.groupby("in_game"):
        for h in horizons:
            g = g0.copy()
            g["fwd"] = g.groupby("game")["k_mid"].shift(-h) - g["k_mid"]
            g["pm_fwd"] = g.groupby("game")["pm_p"].shift(-h) - g["pm_p"]
            g = g.dropna(subset=["fwd", "pm_fwd"])
            g = g[g["gap"].abs() < 0.3]
            # 每场比赛单独回归（过原点），再跨比赛平均 β（聚类稳健）
            def beta(x, y):
                d = (x * x).sum()
                # 整场比赛 gap 几乎恒为 0 的比赛没有识别力，跳过（否则 β 数值爆炸）
                return (x * y).sum() / d if d > 1e-3 else np.nan
            per = g.groupby("game").apply(
                lambda x: pd.Series({"bk": beta(x["gap"], x["fwd"]),
                                     "bp": beta(x["gap"], -x["pm_fwd"])}),
                include_groups=False).dropna()
            n = len(per)
            rows.append({"phase": "in-game" if phase else "pre-game", "h_min": h, "games": n,
                         "beta_kalshi": per["bk"].mean(),
                         "t_kalshi": per["bk"].mean() / (per["bk"].std() / np.sqrt(n)),
                         "beta_pm": per["bp"].mean(),
                         "t_pm": per["bp"].mean() / (per["bp"].std() / np.sqrt(n))})
    return pd.DataFrame(rows)


def taker_fee(p, mult):
    return 0.07 * mult * p * (1 - p)


def simulate(df: pd.DataFrame, theta: float, hold: int | None, phase: str) -> pd.DataFrame:
    """hold=None → 持有到结算。返回每笔交易。"""
    trades = []
    sub = df[df["in_game"] == (phase == "in-game")]
    for game, g in sub.groupby("game", sort=False):
        g = g.reset_index(drop=True)
        n, i = len(g), 0
        gap, bid, ask = g["gap"].to_numpy(), g["k_bid"].to_numpy(), g["k_ask"].to_numpy()
        mult, won = g["fee_mult"].iat[0], g["won"].iat[0]
        while i < n - 1:
            if abs(gap[i]) <= theta or abs(gap[i]) > 0.3:
                i += 1
                continue
            j = i + 1                                   # 1 分钟延迟成交
            side = 1 if gap[i] > 0 else -1
            entry = ask[j] if side == 1 else 1 - bid[j]
            if hold is None:
                payoff = won if side == 1 else 1 - won
                pnl = payoff - entry - taker_fee(entry, mult)
                trades.append({"game": game, "ts": g["ts"].iat[j], "side": side,
                               "entry": entry, "pnl": pnl})
                break                                   # 持有到结算：每场一笔
            k = min(j + hold, n - 1)
            exit_ = bid[k] if side == 1 else 1 - ask[k]
            pnl = exit_ - entry - taker_fee(entry, mult) - taker_fee(exit_, mult)
            trades.append({"game": game, "ts": g["ts"].iat[j], "side": side,
                           "entry": entry, "pnl": pnl})
            i = k + 1
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    per = tr.groupby("game")["pnl"].sum()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else np.nan
    return {"trades": len(tr), "games": len(per), "pnl_per_trade": tr["pnl"].mean(),
            "hit": (tr["pnl"] > 0).mean(), "t_game": t}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    league = sys.argv[1] if len(sys.argv) > 1 else "mlb"
    df = load(league)
    print(f"{league}: {df['game'].nunique()} 场比赛, {len(df):,} 个分钟点")
    print("Kalshi 中位价差:", df.groupby("in_game").apply(
        lambda g: (g["k_ask"] - g["k_bid"]).median(), include_groups=False).to_dict())
    print("|gap| 中位数:", df.groupby("in_game")["gap"].apply(lambda s: s.abs().median()).to_dict())

    print("\n== 检验1：领先-滞后（β_kalshi>0 ⇒ Kalshi 向 PM 收敛；β_pm>0 ⇒ PM 向 Kalshi 收敛）==")
    print(lead_lag(df).round(3).to_string(index=False))

    dates = df.groupby("game")["ts"].min().sort_values()
    split = dates.iloc[len(dates) // 2]
    first = df[df["game"].isin(dates[dates < split].index)]
    second = df[df["game"].isin(dates[dates >= split].index)]

    print(f"\n== 检验2：交易模拟（前半段选参数，样本外 = {split:%Y-%m-%d} 之后）==")
    grid = []
    for phase in ("pre-game", "in-game"):
        for theta in (0.02, 0.03, 0.05, 0.08):
            for hold in (1, 3, 5, 10, 30, None):
                s = summarize(simulate(first, theta, hold, phase))
                s.update({"phase": phase, "theta": theta, "hold": hold or "settle"})
                grid.append(s)
    grid = pd.DataFrame(grid)
    print("前半段（in-sample）t 最高的 8 组:")
    print(grid.sort_values("t_game", ascending=False).head(8).round(4).to_string(index=False))

    best = grid[grid["trades"] >= 50].sort_values("t_game", ascending=False).head(3)
    print("\n样本外验证:")
    for b in best.itertuples(index=False):
        hold = None if b.hold == "settle" else int(b.hold)
        s = summarize(simulate(second, b.theta, hold, b.phase))
        print(f"   {b.phase} θ={b.theta} hold={b.hold}: {s}")
