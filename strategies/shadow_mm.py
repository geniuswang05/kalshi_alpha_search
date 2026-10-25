"""影子做市前向测试（paper trading，不下真实订单）：LIP 奖励 + 价差收入 − 逆向选择。

每 POLL 秒，对每个候选市场：
1. 拉真实订单簿（别人的挂单），算出我们的报价：两侧在最优买价基础上改善 IMPROVE 个 tick
   （不能交叉），每侧挂 Q 张（剩余量 = Q − 已成交，单边库存超限就撤该侧）
2. 按官方 LIP 规则（strategies/lip_scanner.side_scores）计算我们的份额，按时间累计奖励
3. 拉上次轮询以来的真实成交：价格打到/穿过我们报价的 taker 成交 → 记为我们成交（按我们的价）
4. 库存按中间价盯市，扣 maker 手续费

已知偏差：真实挂单会改变别人的行为（taker 看到更好的价格可能多成交）——成交估计偏乐观（少）；
份额按轮询时点的订单簿近似 30 秒内不变。

输出：data_cache/shadow_mm/{run_id}/log.csv（每次轮询每个市场一行）+ fills.csv
用法：python strategies/shadow_mm.py [小时数] [市场数]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import get_with_retry
from strategies.lip_scanner import TICK, parse_side, side_scores

POLL = 30
IMPROVE = 1
MAX_INV_MULT = 1.0       # 单边净库存上限 = 1 × Q
FEES = json.load(open(DATA_CACHE_DIR / "series_fees.json"))


def maker_fee(series: str, price: float) -> float:
    ft, mult = FEES.get(series, ["quadratic", 1])
    if ft == "quadratic_with_maker_fees":
        return 0.25 * 0.07 * float(mult or 1) * price * (1 - price)
    if ft == "quadratic_with_combo_maker_fees":
        return 0.5 * 0.07 * float(mult or 1) * price * (1 - price)
    return 0.0


class MarketState:
    def __init__(self, ticker, series, reward_per_day, target, df):
        self.ticker, self.series = ticker, series
        self.reward_per_sec = reward_per_day / 86400.0
        self.target, self.df = target, df
        self.q = target
        self.yes_pos = 0.0      # 持有 YES 张数
        self.no_pos = 0.0       # 持有 NO 张数
        self.cash = 0.0
        self.reward = 0.0
        self.fees = 0.0
        self.last_trade_ts = int(time.time())
        self.quotes = None      # (yes_px, yes_sz, no_px, no_sz)
        self.mid = None
        self.stopped = False    # v2 单市场止损触发后不再报价（库存继续盯市）

    def net(self):
        return self.yes_pos - self.no_pos

    def mtm(self):
        if self.mid is None:
            return self.cash
        return self.cash + self.yes_pos * self.mid + self.no_pos * (1 - self.mid)


def make_quotes(st: MarketState, yes_levels, no_levels):
    if not yes_levels or not no_levels:
        return None
    by, bn = yes_levels[0][0], no_levels[0][0]
    yes_px, no_px = round(by + IMPROVE * TICK, 2), round(bn + IMPROVE * TICK, 2)
    if yes_px < TICK or no_px < TICK:
        return None
    # 改善后会交叉或越过对手价：退回加入最优价
    if yes_px + no_px > 1 - TICK + 1e-9:
        yes_px, no_px = by, bn
    lim = MAX_INV_MULT * st.q
    yes_sz = st.q if st.net() < lim else 0.0      # 净多 YES 太多就不再买 YES
    no_sz = st.q if -st.net() < lim else 0.0
    return yes_px, yes_sz, no_px, no_sz


def step(client, st: MarketState, dt: float, fills_out: list):
    ob = get_with_retry(client, f"/markets/{st.ticker}/orderbook", {"depth": 0})
    book = ob.get("orderbook_fp") or {}
    yes_lv, no_lv = parse_side(book.get("yes_dollars")), parse_side(book.get("no_dollars"))

    # 1) 上一个区间的奖励：用上一次的报价 + 现在的订单簿近似
    share = 0.0
    if st.quotes is not None:
        yp, ys, np_, ns = st.quotes
        ok_y, oy, ty = side_scores(yes_lv, st.target, st.df, (yp, ys) if ys > 0 else None)
        ok_n, on, tn = side_scores(no_lv, st.target, st.df, (np_, ns) if ns > 0 else None)
        if ok_y and ok_n and ty > 0 and tn > 0:
            share = 0.5 * (oy / ty + on / tn)
        st.reward += share * st.reward_per_sec * dt

    # 2) 上一个区间的成交：真实 taker 成交打到我们上一次的报价
    now = int(time.time())
    if st.quotes is not None:
        yp, ys, np_, ns = st.quotes
        r = get_with_retry(client, "/markets/trades",
                           {"ticker": st.ticker, "limit": 1000, "min_ts": st.last_trade_ts})
        rem_y, rem_n = ys, ns
        for t in sorted(r.get("trades", []), key=lambda x: x["created_time"]):
            price = float(t["yes_price_dollars"])
            cnt = float(t["count_fp"])
            side = t.get("taker_outcome_side") or t.get("taker_side")
            if side == "no" and rem_y > 0 and price <= yp + 1e-9:
                f = min(cnt, rem_y); rem_y -= f
                st.yes_pos += f; st.cash -= f * yp
                fee = f * maker_fee(st.series, yp); st.fees += fee; st.cash -= fee
                fills_out.append({"ticker": st.ticker, "ts": t["created_time"], "side": "yes",
                                  "px": yp, "qty": f})
            elif side == "yes" and rem_n > 0 and price >= 1 - np_ - 1e-9:
                f = min(cnt, rem_n); rem_n -= f
                st.no_pos += f; st.cash -= f * np_
                fee = f * maker_fee(st.series, np_); st.fees += fee; st.cash -= fee
                fills_out.append({"ticker": st.ticker, "ts": t["created_time"], "side": "no",
                                  "px": np_, "qty": f})
    st.last_trade_ts = now

    # YES+NO 成对持有 = 锁定 $1，直接对冲掉（等价于赎回）
    pair = min(st.yes_pos, st.no_pos)
    if pair > 0:
        st.yes_pos -= pair; st.no_pos -= pair; st.cash += pair

    if yes_lv and no_lv:
        st.mid = (yes_lv[0][0] + (1 - no_lv[0][0])) / 2
    if STOP_LOSS and st.mtm() < -max(10.0, 2.0 * st.reward):
        st.stopped = True
    st.quotes = None if st.stopped else make_quotes(st, yes_lv, no_lv)
    return share


STOP_LOSS = False
PER_SERIES_CAP, PER_CATEGORY_CAP, MIN_HOURS_TO_CLOSE = 2, 6, 6


def pick_candidates_v2(n: int, offset: int, hours: float) -> pd.DataFrame:
    """v2：按官方类别分散（每类 ≤6、每系列 ≤2），剔除测试结束后 6 小时内就结算的市场。"""
    from data.kalshi_universe import market_meta
    s = pick_candidates(10_000, offset, hours, per_series=10_000)
    cat = pd.read_parquet(DATA_CACHE_DIR / "lip_series_meta.parquet").set_index("series")["category"]
    s["category"] = s["series"].map(cat).fillna("Other")
    meta = market_meta(KalshiClient(), s["ticker"].tolist()).set_index("ticker")["close_time"]
    min_close = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=hours + MIN_HOURS_TO_CLOSE)
    s = s[s["ticker"].map(meta) > min_close]
    s = s.groupby("series").head(PER_SERIES_CAP).groupby("category").head(PER_CATEGORY_CAP)
    return s.head(n)


def pick_candidates(n: int, offset: int, hours: float, per_series: int = 3) -> pd.DataFrame:
    """从修正规则后的评估（lip_eval_v2）里选：正常市场（价差≤20¢）、低成交风险、
    激励计划在测试结束后仍有效；每系列最多 3 个。"""
    s = pd.read_parquet(DATA_CACHE_DIR / "lip_eval_v2.parquet")
    inc = pd.read_parquet(DATA_CACHE_DIR / "incentives_active.parquet")
    ends = inc.groupby("market_ticker")["end"].max()
    deadline = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=hours)
    s = s[(~s["dead"]) & (s["offset"] == offset) & (s["q"] == s["target"])
          & (s["share_v2"] > 0) & (s["fill_ratio"] < 0.25)]
    s = s[s["ticker"].map(ends) > deadline]
    s = s.sort_values("roc_v2", ascending=False).groupby("series").head(per_series)
    return s.head(n)


def run(hours: float, n_markets: int, offset: int, v2: bool = False):
    global IMPROVE, STOP_LOSS
    IMPROVE = -offset
    STOP_LOSS = v2
    cands = (pick_candidates_v2 if v2 else pick_candidates)(n_markets, offset, hours)
    run_id = (pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M") + f"_off{offset}"
              + ("_v2" if v2 else ""))
    out = DATA_CACHE_DIR / "shadow_mm" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cands.to_csv(out / "candidates.csv", index=False)
    client = KalshiClient()
    states = [MarketState(r.ticker, r.series, r.reward_per_day, r.target, r.df)
              for r in cands.itertuples(index=False)]
    print(f"run {run_id}: {len(states)} markets, {hours}h", flush=True)

    log_path, fills_path = out / "log.csv", out / "fills.csv"
    t_end, last = time.time() + hours * 3600, time.time()
    first = True
    while time.time() < t_end:
        t0 = time.time()
        dt = t0 - last
        last = t0
        fills, rows = [], []
        for st in states:
            try:
                share = step(client, st, dt, fills)
            except Exception as e:
                share = np.nan
                print(f"  {st.ticker}: {type(e).__name__}", flush=True)
            rows.append({"ts": pd.Timestamp.now(tz="UTC"), "ticker": st.ticker, "series": st.series,
                         "share": share, "reward": st.reward, "fees": st.fees, "cash": st.cash,
                         "yes_pos": st.yes_pos, "no_pos": st.no_pos, "mid": st.mid,
                         "mtm": st.mtm(), "quotes": json.dumps(st.quotes)})
        pd.DataFrame(rows).to_csv(log_path, mode="a", header=first, index=False)
        if fills:
            pd.DataFrame(fills).to_csv(fills_path, mode="a",
                                       header=not fills_path.exists(), index=False)
        first = False
        tot_r = sum(s.reward for s in states)
        tot_m = sum(s.mtm() for s in states)
        print(f"{pd.Timestamp.now(tz='UTC'):%H:%M} reward ${tot_r:,.2f}  inventory MTM ${tot_m:,.2f}"
              f"  net ${tot_r + tot_m:,.2f}", flush=True)
        time.sleep(max(0, POLL - (time.time() - t0)))


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # 0=加入最优价, -1=改善1 tick
    v2 = len(sys.argv) > 4 and sys.argv[4] == "v2"             # 分散 + 止损 + 避开临近结算
    run(hours, n, offset, v2)
