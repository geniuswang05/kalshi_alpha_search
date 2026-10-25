"""Edge 检测：模型概率 vs Kalshi 市场价格，扣手续费。

Kalshi 手续费（KXNBAGAME 是 quadratic_with_maker_fees，multiplier=1）：
  taker fee = 0.07 × P × (1−P) 美元/张（P 为成交价）
  50¢ 附近最贵（1.75¢/张），价格越极端越便宜。
回测按 taker 成交（吃 ask），保守假设。

买 YES @ ask 的每张期望值：
  EV = p_model × $1 − ask − fee(ask)
  edge = p_model − ask − fee(ask)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import FEE_RATE, MAX_SPREAD, MIN_EDGE


def taker_fee(price: float) -> float:
    """每张合约的 taker 手续费（美元）。"""
    return FEE_RATE * price * (1.0 - price)


def compute_edge(model_prob: float, ask: float) -> float:
    """买 YES 吃 ask 的净 edge（美元/张，也就是概率差扣费）。"""
    return model_prob - ask - taker_fee(ask)


def tradeable(model_prob: float, bid: float, ask: float,
              min_edge: float = MIN_EDGE, max_spread: float = MAX_SPREAD) -> bool:
    """过滤：edge 达标 + 价差可接受 + 价格不在无意义的极端。"""
    if not (0.02 <= ask <= 0.98):
        return False
    if ask - bid > max_spread:
        return False
    return compute_edge(model_prob, ask) >= min_edge
