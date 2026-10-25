"""终局冷门卖出机器人（"Miracle Seller"）：在比赛最后阶段挂单卖出 2-10¢ 的冷门 YES。

依据：backtest/flb_analysis.py + backtest/endgame_strategy.py —— 终局窗口里冷门的价格隐含
胜率 ~5.4%，实际只赢 ~1.8%（562 个窗口，精确二项 p=1.5e-5；H2 样本外 p=0.002）。

规则（全部事前可知）：
- 白名单联赛；终局窗口 = expected_expiration_time + 联赛偏移 的 [−45, +30] 分钟
- 若冷门 YES 的 ask 在 (2¢, 10¢]：挂 YES 卖单（= NO 买单），价格 = 当前最优 ask（排队）
- 每市场最多 MAX_PER_MARKET 张，全部持仓最坏情况亏损 ≤ MAX_TOTAL_RISK 美元；持有到结算

模式：
- paper（默认）：读真实订单簿/成交，模拟成交（taker 买 YES 且价格 ≥ 我们的 ask），结算后记 PnL。
  不下任何订单，不需要 API key。
- live：需要 KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_PATH，且显式传 --live。会真实下单——
  先在 demo 环境跑通，并自行确认风险上限。

用法: python strategies/endgame_bot.py [--hours 12] [--live]
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import get_with_retry
from strategies.lip_scanner import parse_side

# 全样本事前检验 t>3 的联赛 → 偏移（分钟，close − expected_expiration 的 H1 中位数）
LEAGUE_OFFSET_MIN = {
    "KXNBAGAME": -15, "KXWNBAGAME": -25, "KXNFLGAME": 10, "KXNCAAFGAME": 20,
    "KXLALIGAGAME": -57, "KXLIGAMXGAME": -50, "KXNPBGAME": -52, "KXIPLGAME": 57,
    "KXVALORANTGAME": -110, "KXCS2GAME": -58, "KXT20MATCH": 81,
}
WINDOW_PRE, WINDOW_POST = 45, 30
LO, HI = 0.02, 0.10
MAX_PER_MARKET = 500
MAX_TOTAL_RISK = 2000.0       # 所有持仓若冷门全部逆转的最大亏损（美元）
POLL = 20
FEES = json.load(open(DATA_CACHE_DIR / "series_fees.json"))
OUT = DATA_CACHE_DIR / "endgame_bot"
OUT.mkdir(exist_ok=True)


def maker_fee(series: str, p: float) -> float:
    ft, mult = FEES.get(series, ["quadratic", 1])
    if ft == "quadratic_with_maker_fees":
        return 0.25 * 0.07 * float(mult or 1) * p * (1 - p)
    return 0.0


class Position:
    def __init__(self, ticker, series):
        self.ticker, self.series = ticker, series
        self.short_yes = 0.0      # 卖出的 YES 张数（= 持有 NO）
        self.premium = 0.0        # 收到的权利金（卖出价 × 张数）
        self.fees = 0.0
        self.order_id = None
        self.quote = None
        self.queue_ahead = 0.0    # 排在我们前面的同价挂单量（纸面模式按价格-时间优先消耗）
        self.last_ts = int(time.time())
        self.settled = None

    def risk(self):
        return self.short_yes - self.premium     # 冷门逆转时的亏损


def discover(client) -> pd.DataFrame:
    rows = []
    for s in LEAGUE_OFFSET_MIN:
        r = get_with_retry(client, "/markets", {"series_ticker": s, "status": "open", "limit": 1000})
        for m in r.get("markets", []):
            ee = pd.Timestamp(m.get("expected_expiration_time")) if m.get("expected_expiration_time") else None
            if ee is None:
                continue
            anchor = ee + pd.Timedelta(minutes=LEAGUE_OFFSET_MIN[s])
            rows.append({"ticker": m["ticker"], "series": s, "anchor": anchor})
        time.sleep(0.1)
    return pd.DataFrame(rows)


def in_window(anchor, now) -> bool:
    return anchor - pd.Timedelta(minutes=WINDOW_PRE) <= now <= anchor + pd.Timedelta(minutes=WINDOW_POST)


def run(hours: float, live: bool):
    client = KalshiClient()
    if live and client._private_key is None:
        raise SystemExit("--live 需要 KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_PATH")
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M") + ("_live" if live else "_paper")
    log_path = OUT / f"{run_id}.csv"
    pos: dict[str, Position] = {}
    t_end = time.time() + hours * 3600
    universe, last_disc = pd.DataFrame(), 0.0
    print(f"run {run_id} ({'LIVE' if live else 'paper'}), {hours}h", flush=True)

    while time.time() < t_end:
        now = pd.Timestamp.now(tz="UTC")
        if time.time() - last_disc > 900:
            universe, last_disc = discover(client), time.time()
            print(f"{now:%H:%M} 发现 {len(universe)} 个白名单市场；"
                  f"窗口内 {sum(in_window(a, now) for a in universe['anchor']) if len(universe) else 0}", flush=True)
        total_risk = sum(p.risk() for p in pos.values() if p.settled is None)
        events = []
        for m in universe.itertuples(index=False) if len(universe) else []:
            p = pos.get(m.ticker)
            active = in_window(m.anchor, now)
            if not active and (p is None or p.quote is None):
                continue
            try:
                ob = get_with_retry(client, f"/markets/{m.ticker}/orderbook", {"depth": 5})
            except Exception:
                continue
            book = ob.get("orderbook_fp") or {}
            no_bids = parse_side(book.get("no_dollars"))
            if p is None:
                p = pos[m.ticker] = Position(m.ticker, m.series)

            # 1) 结算上一个区间的模拟成交（paper）
            if p.quote is not None and not live:
                r = get_with_retry(client, "/markets/trades",
                                   {"ticker": m.ticker, "limit": 1000, "min_ts": p.last_ts})
                remaining = MAX_PER_MARKET - p.short_yes
                for t in sorted(r.get("trades", []), key=lambda x: x["created_time"]):
                    side = t.get("taker_outcome_side") or t.get("taker_side")
                    px = float(t["yes_price_dollars"])
                    if side == "yes" and px >= p.quote - 1e-9 and remaining > 0:
                        cnt = float(t["count_fp"])
                        eat = min(cnt, p.queue_ahead)
                        p.queue_ahead -= eat
                        cnt -= eat
                        if cnt <= 0:
                            continue
                        q = min(cnt, remaining)
                        remaining -= q
                        p.short_yes += q
                        p.premium += q * p.quote
                        p.fees += q * maker_fee(p.series, p.quote)
                        events.append(f"FILL {m.ticker} 卖出 {q:.0f} 张 YES @ {p.quote:.2f}")
            p.last_ts = int(time.time())

            # 2) 更新报价
            best_yes_ask = round(1 - no_bids[0][0], 2) if no_bids else None
            want = (active and best_yes_ask is not None and LO < best_yes_ask <= HI
                    and p.short_yes < MAX_PER_MARKET
                    and total_risk + (MAX_PER_MARKET - p.short_yes) * (1 - best_yes_ask) <= MAX_TOTAL_RISK)
            new_quote = best_yes_ask if want else None
            if live and new_quote != p.quote:
                if p.order_id:
                    client.cancel_order(p.order_id)
                    p.order_id = None
                if new_quote is not None:
                    resp = client.create_order(m.ticker, side="ask",
                                               count=MAX_PER_MARKET - p.short_yes, price=new_quote,
                                               client_order_id=str(uuid.uuid4()))
                    p.order_id = (resp.get("order") or resp).get("order_id")
            if new_quote != p.quote:
                events.append(f"QUOTE {m.ticker} {p.quote} -> {new_quote}")
                # 新报价排在该价位全部现有挂单之后
                lvl = [sz for px_, sz in no_bids if abs((1 - px_) - (new_quote or -1)) < 1e-9]
                p.queue_ahead = lvl[0] if lvl else 0.0
            p.quote = new_quote

        # 3) 已结束窗口的持仓：查结算
        for p in pos.values():
            if p.settled is None and p.short_yes > 0 and p.quote is None:
                try:
                    mk = get_with_retry(client, f"/markets/{p.ticker}", {}).get("market", {})
                except Exception:
                    continue
                if mk.get("result") in ("yes", "no"):
                    lost = p.short_yes if mk["result"] == "yes" else 0.0
                    p.settled = p.premium - lost - p.fees
                    events.append(f"SETTLE {p.ticker} result={mk['result']} PnL ${p.settled:,.2f}")

        if events:
            for e in events:
                print(f"{now:%H:%M:%S} {e}", flush=True)
        rows = [{"ts": now, "ticker": p.ticker, "series": p.series, "short_yes": p.short_yes,
                 "premium": p.premium, "fees": p.fees, "quote": p.quote, "settled_pnl": p.settled}
                for p in pos.values()]
        if rows:
            pd.DataFrame(rows).to_csv(log_path, index=False)
        time.sleep(POLL)

    done = [p.settled for p in pos.values() if p.settled is not None]
    print(f"结束：已结算 {len(done)} 个市场，PnL ${sum(done):,.2f}；"
          f"未结算持仓 {sum(1 for p in pos.values() if p.settled is None and p.short_yes > 0)} 个", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    run(a.hours, a.live)
