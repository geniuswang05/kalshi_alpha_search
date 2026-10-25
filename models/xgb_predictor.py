"""XGBoost 主胜概率模型 + 等张校准。

时间切分（绝不随机切分，防泄漏）：
  训练:  2019-20 .. 2023-24
  校准:  2024-25   （早停验证 + 拟合 isotonic 校准器）
  测试:  2025-26   （walk-forward 样本外，对应 Kalshi 真实价格回测）

校准是本系统的命门：我们拿概率去和市场价格比大小，概率偏 3 个点，
edge 计算就整体偏 3 个点，会系统性亏钱。所以模型输出必须过 isotonic 校准。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CAL_SEASON, TEST_SEASON
from models.feature_engineering import feature_columns

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,            # 浅树：6千样本、特征相关性高，深了必过拟合
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 20,
    "reg_lambda": 2.0,
    "seed": 42,
}
NUM_ROUNDS = 2000
EARLY_STOP = 100


class WinProbModel:
    def __init__(self):
        self.booster: xgb.Booster | None = None
        self.calibrator: IsotonicRegression | None = None
        self.feature_cols: list[str] = []

    def fit(self, table: pd.DataFrame) -> dict:
        """table: build_training_table 的输出（含全部赛季）。返回验证指标。"""
        self.feature_cols = feature_columns(table)
        seasons = sorted(table["SEASON"].unique())
        train_seasons = [s for s in seasons if s < CAL_SEASON]
        assert CAL_SEASON in seasons and TEST_SEASON in seasons

        tr = table[table["SEASON"].isin(train_seasons)]
        ca = table[table["SEASON"] == CAL_SEASON]

        dtr = xgb.DMatrix(tr[self.feature_cols], label=tr["HOME_WIN"])
        dca = xgb.DMatrix(ca[self.feature_cols], label=ca["HOME_WIN"])
        self.booster = xgb.train(
            XGB_PARAMS, dtr, NUM_ROUNDS,
            evals=[(dtr, "train"), (dca, "cal")],
            early_stopping_rounds=EARLY_STOP, verbose_eval=False)

        # 在校准赛季上拟合 isotonic（该赛季对模型是样本外的，校准无泄漏）
        raw_cal = self.booster.predict(
            dca, iteration_range=(0, self.booster.best_iteration + 1))
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        self.calibrator.fit(raw_cal, ca["HOME_WIN"])

        cal_pred = self.calibrator.predict(raw_cal)
        return {
            "train_seasons": train_seasons,
            "best_iteration": self.booster.best_iteration,
            "cal_season": CAL_SEASON,
            "cal_accuracy": float(((raw_cal > 0.5) == ca["HOME_WIN"]).mean()),
            "cal_logloss_raw": float(log_loss(ca["HOME_WIN"], raw_cal)),
            "cal_logloss_calibrated": float(log_loss(ca["HOME_WIN"], cal_pred)),
            "cal_brier_raw": float(brier_score_loss(ca["HOME_WIN"], raw_cal)),
            "cal_brier_calibrated": float(brier_score_loss(ca["HOME_WIN"], cal_pred)),
        }

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        """返回校准后的主胜概率。"""
        d = xgb.DMatrix(table[self.feature_cols])
        raw = self.booster.predict(
            d, iteration_range=(0, self.booster.best_iteration + 1))
        return self.calibrator.predict(raw)

    def feature_importance(self, top_n: int = 15) -> pd.Series:
        imp = self.booster.get_score(importance_type="gain")
        return pd.Series(imp).sort_values(ascending=False).head(top_n)


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """校准诊断表：每个概率桶里，预测均值 vs 实际胜率应该接近。"""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        n=("y", "size"), pred_mean=("p", "mean"), actual=("y", "mean"))
    out["gap"] = out["pred_mean"] - out["actual"]
    return out.round(3)
