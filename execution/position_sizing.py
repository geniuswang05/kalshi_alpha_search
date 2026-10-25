"""仓位管理：分数 Kelly。

二元合约买 YES @ price，赢赔付 $1：
  赢的净赔率 b = (1 − price) / price
  满 Kelly 仓位比例 f* = (p×b − (1−p)) / b = (p − price) / (1 − price)
满 Kelly 对概率估计误差极其敏感（模型概率高估 2 个点，满 Kelly 可能翻倍下注），
所以只下 KELLY_FRACTION 倍，并给单笔上限兜底。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import KELLY_FRACTION, MAX_BET_FRACTION


def kelly_stake(model_prob: float, price: float, bankroll: float,
                fee_per_contract: float = 0.0) -> float:
    """返回下注金额（美元，含手续费预算）。无 edge 返回 0。"""
    eff_cost = price + fee_per_contract          # 每张实际成本
    if model_prob <= eff_cost or eff_cost >= 1.0:
        return 0.0
    f_full = (model_prob - eff_cost) / (1.0 - eff_cost)
    f = min(f_full * KELLY_FRACTION, MAX_BET_FRACTION)
    return bankroll * f
