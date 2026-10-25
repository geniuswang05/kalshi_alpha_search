"""诊断实验：模型到底有没有 alpha，以及在哪个时点可能有。

实验1（增量信息检验）：logistic 回归 用 logit(市场价) 和 logit(模型概率)
一起预测赛果。如果模型系数不显著/接近0，说明模型信息完全被市场包含。

实验2（时点敏感性）：在开赛前 4h / 24h / 48h 三个时点分别取市场价，
看市场 logloss 随时间的变化（越早越"软"则策略应该早下注），
并在每个时点重新模拟策略 PnL。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR, INITIAL_BANKROLL, TEST_SEASON
from backtest.market_snapshots import build_snapshots
from backtest.run_backtest import simulate, train_and_predict


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def join_at(test: pd.DataFrame, hours: float) -> pd.DataFrame:
    snaps = build_snapshots(hours).dropna(subset=["bid", "ask"])
    snaps["date_str"] = snaps["game_date"].dt.strftime("%Y-%m-%d")
    t = test.copy()
    t["date_str"] = t["GAME_DATE"].dt.strftime("%Y-%m-%d")
    m = snaps.merge(
        t[["date_str", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
           "p_home", "HOME_WIN", "GAME_ID"]],
        left_on=["date_str", "home", "away"],
        right_on=["date_str", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"])
    m["p_model"] = np.where(m["bet_on"] == m["home"], m["p_home"], 1 - m["p_home"])
    m["won"] = (m["result"] == "yes").astype(int)
    return m


if __name__ == "__main__":
    test, _ = train_and_predict()

    print("\n===== 实验1：模型在市场价之外的增量信息 =====")
    m4 = join_at(test, 4)
    home = m4[m4["bet_on"] == m4["home"]].dropna(subset=["mid"])
    X = np.column_stack([logit(home["mid"]), logit(home["p_model"])])
    y = home["won"].values
    lr = LogisticRegression(C=1e6).fit(X, y)
    p_blend = lr.predict_proba(X)[:, 1]
    print(f"   系数: 市场 {lr.coef_[0][0]:.3f}  模型 {lr.coef_[0][1]:.3f}")
    print(f"   logloss: 纯市场 {log_loss(y, np.clip(home['mid'], .01, .99)):.4f}"
          f"  纯模型 {log_loss(y, np.clip(home['p_model'], .01, .99)):.4f}"
          f"  混合 {log_loss(y, p_blend):.4f}")
    print("   （模型系数≈0 或混合不优于纯市场 ⇒ 模型无增量信息）")

    print("\n===== 实验2：不同时点的市场效率 =====")
    for hours in (4, 24, 48):
        m = join_at(test, hours)
        h = m[m["bet_on"] == m["home"]].dropna(subset=["mid"])
        yh = h["won"].values
        ll_mkt = log_loss(yh, np.clip(h["mid"], .01, .99))
        spread = (h["ask"] - h["bid"]).median()
        bets = simulate(m)
        pnl = bets["pnl"].sum() if not bets.empty else 0.0
        n = len(bets)
        print(f"   T-{hours:>2}h: n={len(h):4d}  市场logloss={ll_mkt:.4f}  "
              f"中位价差={spread:.3f}  策略交易={n:3d}  PnL=${pnl:,.0f}")
    print("   （若更早时点市场 logloss 更差=更软，早下注才有机会）")
