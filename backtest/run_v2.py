"""v2 回测：加入缺阵价值特征后，模型 vs 市场 + 策略模拟。

两个版本对照（量化前视偏差）：
- aggressive: 全部缺阵特征（含当场实际缺阵——T-4h 下注有轻微前视）
- conservative: 只用"上一场就已缺阵"特征（赛前铁定可知，零前视）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR, TEST_SEASON
from backtest.run_backtest import join_market, simulate
from models.xgb_predictor import WinProbModel

AGGRESSIVE_ONLY = [f"{side}_{c}" for side in ("HOME", "AWAY", "DIFF")
                   for c in ("MISS_VAL", "MISS_MIN", "MISS_N", "STAR_OUT")]


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def run_variant(table: pd.DataFrame, name: str):
    print(f"\n{'='*24} {name} {'='*24}")
    model = WinProbModel()
    info = model.fit(table)
    print(f"   校准季: acc {info['cal_accuracy']:.4f}, "
          f"logloss {info['cal_logloss_calibrated']:.4f}")

    test = table[table["SEASON"] == TEST_SEASON].copy()
    test["p_home"] = model.predict_proba(test)
    m = join_market(test)

    home = m[m["bet_on"] == m["home"]].dropna(subset=["mid"])
    y = home["won"].values
    p_mkt = home["mid"].clip(0.01, 0.99).values
    p_mod = home["p_model"].clip(0.01, 0.99).values
    X = np.column_stack([logit(p_mkt), logit(p_mod)])
    lr = LogisticRegression(C=1e6).fit(X, y)
    p_blend = lr.predict_proba(X)[:, 1]

    print(f"   样本外 logloss: 市场 {log_loss(y, p_mkt):.4f} | 模型 {log_loss(y, p_mod):.4f}"
          f" | 混合 {log_loss(y, p_blend):.4f}")
    print(f"   brier:          市场 {brier_score_loss(y, p_mkt):.4f} | 模型 {brier_score_loss(y, p_mod):.4f}")
    print(f"   混合系数: 市场 {lr.coef_[0][0]:.3f} | 模型 {lr.coef_[0][1]:.3f}"
          f"   ← 模型系数>0 才说明有增量信息")

    imp = model.feature_importance(10)
    avail_in_top = [f for f in imp.index if "MISS" in f or "STAR" in f or "ROSTER" in f]
    print(f"   Top10特征中的缺阵特征: {avail_in_top or '无'}")

    bets = simulate(m)
    if bets.empty:
        print("   策略: 无交易")
    else:
        pnl, staked = bets["pnl"].sum(), bets["stake"].sum()
        print(f"   策略: {len(bets)} 笔, PnL ${pnl:,.0f}, ROI {pnl/staked*100:+.2f}%, "
              f"命中 {bets['won'].mean():.3f}")
    return {"name": name, "model_ll": log_loss(y, p_mod),
            "blend_ll": log_loss(y, p_blend), "coef_model": lr.coef_[0][1]}


if __name__ == "__main__":
    table = pd.read_parquet(DATA_CACHE_DIR / "training_table.parquet")

    cons_table = table.drop(columns=AGGRESSIVE_ONLY)
    results = [
        run_variant(table, "v2-aggressive（含当场缺阵，轻微前视）"),
        run_variant(cons_table, "v2-conservative（仅赛前可知缺阵，零前视）"),
    ]
    print("\n===== 对照汇总（市场 logloss 基准 ≈ 0.5685）=====")
    for r in results:
        print(f"   {r['name']}: 模型 {r['model_ll']:.4f}, 混合 {r['blend_ll']:.4f}, "
              f"模型混合系数 {r['coef_model']:.3f}")
