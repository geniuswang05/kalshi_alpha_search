"""端到端回测：XGBoost 训练 → 校准 → 对 2025-26 赛季用 Kalshi 真实价格模拟交易。

流程：
1. 训练 WinProbModel（2019-24 训练，2024-25 校准，2025-26 纯样本外）
2. 模型 vs 市场：logloss/Brier/准确率对比 —— 能不能打过市场是第一道关
3. 模拟交易：每场比赛两个合约（主/客），edge 达标就按分数 Kelly 买入，
   按 ask 成交、扣 taker 费，结算按 Kalshi 官方 result
4. 输出：PnL、ROI、Sharpe、回撤、edge 分箱实现检验，明细存 parquet

用法: python backtest/run_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_CACHE_DIR, INITIAL_BANKROLL, MIN_EDGE, TEST_SEASON)
from backtest.market_snapshots import build_snapshots
from execution.position_sizing import kelly_stake
from models.xgb_predictor import WinProbModel, reliability_table
from signals.edge_detector import compute_edge, taker_fee, tradeable
from sklearn.metrics import brier_score_loss, log_loss


def train_and_predict():
    table = pd.read_parquet(DATA_CACHE_DIR / "training_table.parquet")
    model = WinProbModel()
    info = model.fit(table)
    print("== 模型训练 ==")
    for k, v in info.items():
        print(f"   {k}: {v}")

    test = table[table["SEASON"] == TEST_SEASON].copy()
    test["p_home"] = model.predict_proba(test)
    print(f"\n   测试赛季 {TEST_SEASON}: {len(test)} 场")
    print("   Top特征:", model.feature_importance(8).round(1).to_dict())
    return test, model


def join_market(test: pd.DataFrame) -> pd.DataFrame:
    """预测 + 市场快照，按 日期+主客队 精确匹配；每场两行（主/客合约）。"""
    snaps = build_snapshots().dropna(subset=["bid", "ask"])
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
    # 结算一致性：Kalshi result 应与 game log 的 HOME_WIN 一致
    home_rows = m[m["bet_on"] == m["home"]]
    mismatch = (home_rows["won"] != home_rows["HOME_WIN"]).mean()
    print(f"\n== 数据对齐 ==\n   匹配合约 {len(m)} 个（{m['GAME_ID'].nunique()} 场比赛）"
          f"，结算与game log不一致率: {mismatch:.4f}")
    return m


def market_vs_model(m: pd.DataFrame):
    """第一道关：模型概率是否比市场价格更接近真相。"""
    home = m[m["bet_on"] == m["home"]].dropna(subset=["mid"])
    y = home["won"].values
    p_mkt = home["mid"].clip(0.01, 0.99).values
    p_mod = home["p_model"].clip(0.01, 0.99).values
    print("\n== 模型 vs 市场（主队合约，n=%d）==" % len(home))
    print(f"   logloss:  市场 {log_loss(y, p_mkt):.4f}  模型 {log_loss(y, p_mod):.4f}")
    print(f"   brier:    市场 {brier_score_loss(y, p_mkt):.4f}  模型 {brier_score_loss(y, p_mod):.4f}")
    print(f"   准确率:   市场 {((p_mkt > .5) == y).mean():.4f}  模型 {((p_mod > .5) == y).mean():.4f}")
    print("\n   模型校准表（测试赛季）:")
    print(reliability_table(y, p_mod).to_string())


def simulate(m: pd.DataFrame) -> pd.DataFrame:
    """按时间顺序逐日模拟下注（复利）。"""
    m = m.sort_values(["game_date", "ticker"]).reset_index(drop=True)
    bankroll = INITIAL_BANKROLL
    bets = []
    for row in m.itertuples(index=False):
        if not tradeable(row.p_model, row.bid, row.ask):
            continue
        fee = taker_fee(row.ask)
        stake = kelly_stake(row.p_model, row.ask, bankroll, fee)
        if stake < 1.0:
            continue
        contracts = stake / (row.ask + fee)
        pnl = contracts * ((1.0 if row.won else 0.0) - row.ask - fee)
        bankroll += pnl
        bets.append({
            "game_date": row.game_date, "ticker": row.ticker,
            "p_model": row.p_model, "ask": row.ask,
            "edge": compute_edge(row.p_model, row.ask),
            "stake": stake, "contracts": contracts,
            "won": row.won, "pnl": pnl, "bankroll": bankroll,
        })
    return pd.DataFrame(bets)


def report(bets: pd.DataFrame):
    print("\n== 回测结果 ==")
    if bets.empty:
        print("   没有产生任何交易（edge 全部低于阈值 %.2f）" % MIN_EDGE)
        return
    total_pnl = bets["pnl"].sum()
    total_staked = bets["stake"].sum()
    daily = bets.groupby("game_date")["pnl"].sum()
    equity = INITIAL_BANKROLL + daily.cumsum()
    drawdown = (equity - equity.cummax())
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan

    print(f"   交易数: {len(bets)}  |  交易日: {daily.shape[0]}")
    print(f"   总PnL: ${total_pnl:,.0f}  (初始资金 ${INITIAL_BANKROLL:,.0f})")
    print(f"   总投入: ${total_staked:,.0f}  |  ROI(按投入): {total_pnl/total_staked*100:.2f}%")
    print(f"   期末资金: ${equity.iloc[-1]:,.0f}  ({(equity.iloc[-1]/INITIAL_BANKROLL-1)*100:+.1f}%)")
    print(f"   命中率: {bets['won'].mean():.3f}  |  平均下注价格: {bets['ask'].mean():.3f}")
    print(f"   日Sharpe(年化): {sharpe:.2f}  |  最大回撤: ${drawdown.min():,.0f} "
          f"({drawdown.min()/INITIAL_BANKROLL*100:.1f}%)")

    print("\n   Edge 分箱实现检验（预测edge vs 实际收益率——应该单调递增）:")
    bets["edge_bin"] = pd.cut(bets["edge"], [0.03, 0.05, 0.08, 0.12, 1.0])
    tab = bets.groupby("edge_bin", observed=True).agg(
        n=("pnl", "size"), avg_edge=("edge", "mean"),
        realized_roi=("pnl", lambda s: s.sum()),
        staked=("stake", "sum"), hit=("won", "mean"))
    tab["realized_roi"] = (tab["realized_roi"] / tab["staked"]).round(3)
    print(tab.drop(columns="staked").round(3).to_string())

    out = DATA_CACHE_DIR / "backtest_bets.parquet"
    bets.drop(columns=["edge_bin"]).to_parquet(out, index=False)
    print(f"\n   明细 -> {out}")


if __name__ == "__main__":
    test, model = train_and_predict()
    m = join_market(test)
    market_vs_model(m)
    bets = simulate(m)
    report(bets)
