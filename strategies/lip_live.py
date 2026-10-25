"""LIP 实盘做市 + 竞争反应测量（真实订单）。

要测的东西：影子测试里对手看不到我们，所以奖励份额是上限。真实挂单后，对手会加量、改价，
这里记录我们进场后每个轮询的
  - 真实份额 share_real：订单簿（含我们的真单）按官方规则计分，我们的占比
  - 对手反应：同价位对手挂单量、对手最优价是否越过我们
  - 成交、库存、maker 费、按中间价盯市
奖励的真实到账要等计划周期结束后在 Kalshi 网站 Rewards 页面核对（API 没有个人奖励接口），
程序会定期记录账户余额作为对照。

三种模式：
  python strategies/lip_live.py                      # 演练：不下单、不需要 key（默认）
  python strategies/lip_live.py --env demo --live    # demo 模拟盘真实下单（假钱，只验证交易机制，demo 无激励计划）
  python strategies/lip_live.py --env prod --live --i-understand-real-money   # 正式环境真钱

安全设置（默认很保守，可调）：每侧 --size 张、最多 --markets 个市场、资金上限 --max-capital、
单市场亏损 --max-loss-market 撤单、总亏损 --max-loss-total 全部撤单并退出；
在项目根目录创建 STOP 文件 → 立即全部撤单并退出；任何异常退出也会全部撤单。
所有挂单都是 post_only（只做 maker，不会主动吃单）。
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR, PROJECT_ROOT
from data.kalshi_client import BASE_URL, DEMO_URL, KalshiClient
from data.kalshi_universe import get_with_retry
from strategies.lip_scanner import TICK, evaluate_market, parse_side, side_scores

OUT_ROOT = DATA_CACHE_DIR / "lip_live"
STOP_FILE = PROJECT_ROOT / "STOP"
FEES = json.load(open(DATA_CACHE_DIR / "series_fees.json")) if (DATA_CACHE_DIR / "series_fees.json").exists() else {}


def maker_fee(series: str, p: float) -> float:
    ft, mult = FEES.get(series, ["quadratic", 1])
    if ft == "quadratic_with_maker_fees":
        return 0.25 * 0.07 * float(mult or 1) * p * (1 - p)
    if ft == "quadratic_with_combo_maker_fees":
        return 0.5 * 0.07 * float(mult or 1) * p * (1 - p)
    return 0.0


# ---------------------------------------------------------------- 选市场

def active_incentives(client) -> pd.DataFrame:
    rows, cursor = [], None
    for _ in range(50):
        p = {"status": "active", "type": "liquidity", "limit": 10000}
        if cursor:
            p["cursor"] = cursor
        r = get_with_retry(client, "/incentive_programs", p)
        rows += r.get("incentive_programs", [])
        cursor = r.get("next_cursor") or r.get("cursor")
        if not cursor:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start_date"], utc=True, format="ISO8601")
    df["end"] = pd.to_datetime(df["end_date"], utc=True, format="ISO8601")
    now = pd.Timestamp.now(tz="UTC")
    df = df[(df["start"] <= now) & (df["end"] > now)].copy()
    days = ((df["end"] - df["start"]).dt.total_seconds() / 86400).clip(lower=1 / 24)
    df["reward_per_day"] = df["period_reward"].astype(float) / 10000 / days
    df["target"] = pd.to_numeric(df["target_size_fp"], errors="coerce").fillna(1000)
    df["df"] = pd.to_numeric(df["discount_factor_bps"], errors="coerce").fillna(5000) / 10000
    df["series"] = df["market_ticker"].str.split("-").str[0]
    return df.groupby("market_ticker").agg(series=("series", "first"), reward_per_day=("reward_per_day", "sum"),
                                           target=("target", "max"), df=("df", "min"), end=("end", "min")).reset_index()


def select_markets(data_client, env: str, n: int, size: float, hours: float, max_capital: float) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    if env == "demo":
        # demo 没有激励计划：只挑有双边报价、价差≥2¢ 的市场验证下单机制
        r = get_with_retry(data_client, "/markets", {"status": "open", "limit": 200})
        rows = []
        for m in r.get("markets", []):
            ob = get_with_retry(data_client, f"/markets/{m['ticker']}/orderbook", {"depth": 0})
            book = ob.get("orderbook_fp") or {}
            y, nb = parse_side(book.get("yes_dollars")), parse_side(book.get("no_dollars"))
            if y and nb and 0.02 <= 1 - y[0][0] - nb[0][0] <= 0.20:
                rows.append({"market_ticker": m["ticker"], "series": m["ticker"].split("-")[0],
                             "reward_per_day": 0.0, "target": size * 2, "df": 0.5, "score": 0.0,
                             "capital": size})
            if len(rows) >= n:
                break
        return pd.DataFrame(rows)

    inc = active_incentives(data_client)
    inc = inc[inc["end"] > now + pd.Timedelta(hours=hours)]
    cat_path = DATA_CACHE_DIR / "lip_series_meta.parquet"
    cats = pd.read_parquet(cat_path).set_index("series")["category"] if cat_path.exists() else pd.Series(dtype=str)
    inc["category"] = inc["series"].map(cats).fillna("Other")
    inc = inc.sort_values("reward_per_day", ascending=False).groupby("series").head(1).head(300)
    rows = []
    for m in inc.itertuples(index=False):
        try:
            mk = get_with_retry(data_client, f"/markets/{m.market_ticker}", {}).get("market", {})
            close = pd.Timestamp(mk.get("close_time"))
            if close < now + pd.Timedelta(hours=hours + 6):
                continue
            ob = get_with_retry(data_client, f"/markets/{m.market_ticker}/orderbook", {"depth": 0})
        except Exception:
            continue
        best = None
        for off in (-1, 0):
            ev = evaluate_market(ob, m.target, m.df, size, off)
            if ev is None or ev["spread"] > 0.20 or ev["share"] <= 0:
                continue
            score = ev["share"] * m.reward_per_day / max(ev["capital"], 1)
            if best is None or score > best["score"]:
                best = {"market_ticker": m.market_ticker, "series": m.series, "category": m.category,
                        "reward_per_day": m.reward_per_day, "target": m.target, "df": m.df,
                        "score": score, "capital": ev["capital"], "est_share_shadow": ev["share"]}
        if best:
            rows.append(best)
        time.sleep(0.1)
    sel = pd.DataFrame(rows)
    if sel.empty:
        return sel
    sel = sel.sort_values("score", ascending=False).groupby("category").head(2)
    sel = sel[sel["capital"].cumsum() <= max_capital]
    return sel.head(n)


# ---------------------------------------------------------------- 状态

class Mkt:
    def __init__(self, r):
        self.ticker, self.series = r.market_ticker, r.series
        self.reward_per_sec = r.reward_per_day / 86400
        self.target, self.df = r.target, r.df
        self.yes_pos = self.no_pos = 0.0
        self.cash = self.fees = self.reward_est = 0.0
        self.mid = None
        self.stopped = False
        self.orders = {"yes": None, "no": None}   # {"id", "price", "remaining"}；no 侧价格为 NO 价
        self.last_fill_ts = int(time.time())
        self.entry_others = None                  # 进场时对手在我们价位的挂单量（反应基准）

    def net(self):
        return self.yes_pos - self.no_pos

    def mtm(self):
        if self.mid is None:
            return self.cash
        return self.cash + self.yes_pos * self.mid + self.no_pos * (1 - self.mid)


def strip_ours(levels, price, qty):
    """订单簿一侧减去我们自己的挂单量，得到对手的订单簿。"""
    out = []
    for p, s in levels:
        if price is not None and abs(p - price) < 1e-9:
            s = max(0.0, s - qty)
        if s > 0:
            out.append((p, s))
    return out


def desired_quotes(m: Mkt, oy, on, size):
    if not oy or not on:
        return None, None
    by, bn = oy[0][0], on[0][0]
    spread = round(1 - by - bn, 2)
    if spread >= 0.03:
        yp, npx = round(by + TICK, 2), round(bn + TICK, 2)
    else:
        yp, npx = by, bn
    if yp + npx > 1 - TICK + 1e-9 or yp < TICK or npx < TICK:
        return None, None
    lim = size
    ysz = size if m.net() < lim else 0.0
    nsz = size if -m.net() < lim else 0.0
    return (yp, ysz) if ysz > 0 else None, (npx, nsz) if nsz > 0 else None


# ---------------------------------------------------------------- 主循环

def run(a):
    live = a.live
    if a.env == "prod" and live and not a.i_understand_real_money:
        raise SystemExit("正式环境真钱下单需要同时传 --live --i-understand-real-money")
    base = DEMO_URL if a.env == "demo" else BASE_URL
    trade_client = KalshiClient(base_url=base)
    if live and trade_client._private_key is None:
        raise SystemExit("--live 需要环境变量 KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_PATH（demo 与正式环境的 key 不通用）")
    data_client = trade_client if (live or a.env == "demo") else KalshiClient()

    sel = select_markets(data_client, a.env, a.markets, a.size, a.hours, a.max_capital)
    if sel.empty:
        raise SystemExit("没有符合条件的市场")
    mode = ("LIVE-" + a.env.upper()) if live else "DRY-RUN"
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M") + f"_{mode.lower()}"
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "markets.csv", index=False)
    print(f"[{mode}] run {run_id}: {len(sel)} 个市场，每侧 {a.size:.0f} 张，资金上限 ${a.max_capital:,.0f}", flush=True)
    print(sel[["market_ticker", "reward_per_day", "capital"]].to_string(index=False), flush=True)

    mk = [Mkt(r) for r in sel.itertuples(index=False)]
    log_rows, act_rows, fill_rows = [], [], []
    t_end, last = time.time() + a.hours * 3600, time.time()
    last_bal = 0.0

    def act(kind, m, side, price=None, qty=None, resp=None):
        act_rows.append({"ts": pd.Timestamp.now(tz="UTC"), "kind": kind, "ticker": m.ticker,
                         "side": side, "price": price, "qty": qty, "resp": json.dumps(resp)[:300] if resp else None})

    def place(m, side, price, qty):
        """side: yes = YES 买单；no = NO 买单（= YES 卖单 @ 1−price）。"""
        if not live:
            m.orders[side] = {"id": f"dry-{uuid.uuid4().hex[:8]}", "price": price, "remaining": qty}
            act("place(dry)", m, side, price, qty)
            return
        book_side, yes_px = ("bid", price) if side == "yes" else ("ask", round(1 - price, 2))
        resp = trade_client.create_order(m.ticker, book_side, qty, yes_px,
                                         client_order_id=str(uuid.uuid4()), post_only=True)
        m.orders[side] = {"id": resp.get("order_id"), "price": price,
                          "remaining": float(resp.get("remaining_count", qty))}
        act("place", m, side, price, qty, resp)

    def cancel(m, side):
        o = m.orders[side]
        if o is None:
            return
        if live:
            try:
                resp = trade_client.cancel_order(o["id"])
                act("cancel", m, side, o["price"], o["remaining"], resp)
            except Exception as e:
                act("cancel_error", m, side, o["price"], o["remaining"], {"err": str(e)})
        else:
            act("cancel(dry)", m, side, o["price"], o["remaining"])
        m.orders[side] = None

    def sync_fills(m):
        if live:
            r = trade_client.get_fills(ticker=m.ticker, min_ts=m.last_fill_ts)
            fills = r.get("fills", [])
            for f in fills:
                qty = float(f.get("count_fp") or 0)
                side = f.get("outcome_side") or f.get("side")
                px = float(f["yes_price_dollars"]) if side == "yes" else float(f["no_price_dollars"])
                fee = float(f.get("fee_cost") or 0)
                if side == "yes":
                    m.yes_pos += qty
                else:
                    m.no_pos += qty
                m.cash -= qty * px + fee
                m.fees += fee
                o = m.orders.get(side)
                if o:
                    o["remaining"] = max(0.0, o["remaining"] - qty)
                fill_rows.append({"ts": f.get("created_time"), "ticker": m.ticker, "side": side, "px": px, "qty": qty, "fee": fee})
        else:
            r = get_with_retry(data_client, "/markets/trades", {"ticker": m.ticker, "limit": 1000, "min_ts": m.last_fill_ts})
            for t in sorted(r.get("trades", []), key=lambda x: x["created_time"]):
                px_yes = float(t["yes_price_dollars"]); cnt = float(t["count_fp"])
                ts_side = t.get("taker_outcome_side") or t.get("taker_side")
                for side, hit in (("yes", ts_side == "no"), ("no", ts_side == "yes")):
                    o = m.orders[side]
                    if not hit or not o or o["remaining"] <= 0:
                        continue
                    through = px_yes <= o["price"] + 1e-9 if side == "yes" else px_yes >= 1 - o["price"] - 1e-9
                    if not through:
                        continue
                    q = min(cnt, o["remaining"]); o["remaining"] -= q
                    fee = q * maker_fee(m.series, o["price"])
                    if side == "yes":
                        m.yes_pos += q
                    else:
                        m.no_pos += q
                    m.cash -= q * o["price"] + fee; m.fees += fee
                    fill_rows.append({"ts": t["created_time"], "ticker": m.ticker, "side": side, "px": o["price"], "qty": q, "fee": fee})
        m.last_fill_ts = int(time.time())
        pair = min(m.yes_pos, m.no_pos)
        if pair > 0:
            m.yes_pos -= pair; m.no_pos -= pair; m.cash += pair

    try:
        while time.time() < t_end:
            if STOP_FILE.exists():
                print("检测到 STOP 文件，全部撤单退出", flush=True)
                break
            t0 = time.time(); dt = t0 - last; last = t0
            for m in mk:
                try:
                    ob = get_with_retry(data_client, f"/markets/{m.ticker}/orderbook", {"depth": 0})
                    book = ob.get("orderbook_fp") or {}
                    yl, nl = parse_side(book.get("yes_dollars")), parse_side(book.get("no_dollars"))
                    sync_fills(m)
                    oyo, ono = m.orders["yes"], m.orders["no"]
                    # 对手的订单簿（实盘里盘口包含我们自己的单）
                    oy = strip_ours(yl, oyo["price"] if oyo and live else None, oyo["remaining"] if oyo else 0)
                    on = strip_ours(nl, ono["price"] if ono and live else None, ono["remaining"] if ono else 0)
                    if oy and on:
                        m.mid = (oy[0][0] + 1 - on[0][0]) / 2

                    # 真实份额（我们的单 + 对手当前的单，按官方规则）
                    ok_y, sy, ty = side_scores(oy, m.target, m.df, (oyo["price"], oyo["remaining"]) if oyo and oyo["remaining"] > 0 else None)
                    ok_n, sn, tn = side_scores(on, m.target, m.df, (ono["price"], ono["remaining"]) if ono and ono["remaining"] > 0 else None)
                    share = 0.5 * (sy / ty + sn / tn) if ok_y and ok_n and ty > 0 and tn > 0 else 0.0
                    m.reward_est += share * m.reward_per_sec * dt

                    # 对手反应：在我们价位上对手的挂单量、对手最优价是否越过我们
                    others_at_ours = sum(s for p, s in oy if oyo and abs(p - oyo["price"]) < 1e-9) + \
                        sum(s for p, s in on if ono and abs(p - ono["price"]) < 1e-9)
                    if m.entry_others is None and (oyo or ono):
                        m.entry_others = others_at_ours
                    jumped = bool((oyo and oy and oy[0][0] > oyo["price"] + 1e-9) or (ono and on and on[0][0] > ono["price"] + 1e-9))

                    # 风控
                    if m.mtm() < -a.max_loss_market and not m.stopped:
                        m.stopped = True
                        print(f"  {m.ticker}: 单市场亏损超限，撤单", flush=True)
                    want_y, want_n = (None, None) if m.stopped else desired_quotes(m, oy, on, a.size)
                    for side, want in (("yes", want_y), ("no", want_n)):
                        cur = m.orders[side]
                        if want is None:
                            cancel(m, side)
                        elif cur is None or abs(cur["price"] - want[0]) > 1e-9 or cur["remaining"] <= 0:
                            cancel(m, side)
                            place(m, side, want[0], want[1])

                    log_rows.append({"ts": pd.Timestamp.now(tz="UTC"), "ticker": m.ticker, "share_real": share,
                                     "reward_est": m.reward_est, "others_at_our_px": others_at_ours,
                                     "entry_others": m.entry_others, "others_jumped_ahead": jumped,
                                     "yes_q": json.dumps(m.orders["yes"]), "no_q": json.dumps(m.orders["no"]),
                                     "yes_pos": m.yes_pos, "no_pos": m.no_pos, "cash": m.cash, "fees": m.fees,
                                     "mid": m.mid, "mtm": m.mtm(), "stopped": m.stopped})
                except Exception as e:
                    print(f"  {m.ticker}: {type(e).__name__} {str(e)[:120]}", flush=True)

            total = sum(x.mtm() + x.reward_est for x in mk)
            if sum(x.mtm() for x in mk) < -a.max_loss_total:
                print(f"总亏损超限（库存 ${sum(x.mtm() for x in mk):,.2f}），全部撤单退出", flush=True)
                break
            if live and time.time() - last_bal > 600:
                try:
                    b = trade_client.get_balance()
                    act("balance", mk[0], None, resp=b)
                except Exception:
                    pass
                last_bal = time.time()
            pd.DataFrame(log_rows).to_csv(out / "log.csv", mode="a", header=not (out / "log.csv").exists(), index=False)
            pd.DataFrame(act_rows).to_csv(out / "actions.csv", mode="a", header=not (out / "actions.csv").exists(), index=False)
            if fill_rows:
                pd.DataFrame(fill_rows).to_csv(out / "fills.csv", mode="a", header=not (out / "fills.csv").exists(), index=False)
            log_rows, act_rows, fill_rows = [], [], []
            print(f"{pd.Timestamp.now(tz='UTC'):%H:%M:%S} 奖励估计 ${sum(x.reward_est for x in mk):,.2f}  "
                  f"库存 ${sum(x.mtm() for x in mk):,.2f}  净 ${total:,.2f}", flush=True)
            time.sleep(max(0, a.poll - (time.time() - t0)))
    finally:
        if live:
            try:
                resp = trade_client.cancel_all()
                print("已全部撤单", resp, flush=True)
            except Exception as e:
                print("全部撤单失败，请立刻在网页上手动撤单！", e, flush=True)
        else:
            for m in mk:
                for side in ("yes", "no"):
                    cancel(m, side)
        pd.DataFrame(act_rows).to_csv(out / "actions.csv", mode="a", header=not (out / "actions.csv").exists(), index=False)
        print(f"日志 -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=["prod", "demo"], default="prod")
    ap.add_argument("--live", action="store_true", help="真实下单（默认演练，不下单）")
    ap.add_argument("--i-understand-real-money", action="store_true")
    ap.add_argument("--markets", type=int, default=3)
    ap.add_argument("--size", type=float, default=100, help="每侧挂单张数")
    ap.add_argument("--max-capital", type=float, default=300)
    ap.add_argument("--max-loss-market", type=float, default=10)
    ap.add_argument("--max-loss-total", type=float, default=30)
    ap.add_argument("--hours", type=float, default=4)
    ap.add_argument("--poll", type=float, default=15)
    run(ap.parse_args())
