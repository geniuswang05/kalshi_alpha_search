"""流动性激励（LIP）收益扫描器：用实时订单簿 + 官方计分规则，估算挂单能分到多少奖励。

官方规则（help.kalshi.com/en/articles/13823851）：
- 每秒快照；YES 侧、NO 侧分别计分（订单簿里 yes_dollars=YES买单，no_dollars=NO买单）
- 每侧从最优价往下累加，只有"帮助达到 Target Size"的挂单计分（按价格优先、同价时间优先）
- 参考价 = 累计挂单量首次达到 Target/5 的价位；参考价及更优的挂单乘数 1.0，
  更差的乘 DF^(差几个tick)
- 你在该侧的份额 = 你的分数 / 该侧总分；快照分 = YES份额 + NO份额（全体合计 2.0）
- 任一侧不足 Target Size → 该快照作废（所有人都拿不到）
- 奖励 = 你的总分/全体总分 × 时段奖金 × 有效快照比例

我们的挂单方式：在两侧各挂 Q 张，价位 = 最优买价 − offset 个 tick（offset=0 即加入最优价，
排在同价位队尾）。资金占用 ≈ Q×yes价 + Q×no价。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import SLEEP, get_with_retry

TICK = 0.01


def parse_side(levels) -> list[tuple[float, float]]:
    """订单簿一侧 → [(price, size)]，按价格从优到劣（高到低）。"""
    out = [(float(p), float(s)) for p, s in (levels or []) if float(s) > 0]
    return sorted(out, key=lambda x: -x[0])


def side_scores(levels, target: float, df: float, ours: tuple[float, float] | None = None):
    """计算一侧的计分。levels 不含我们的单；ours=(price, size) 插在同价位队尾。

    返回 (qualifies, our_score, total_score)。
    """
    queue = []
    for p, s in levels:
        queue.append((p, s, False))
    if ours is not None and ours[1] > 0:
        queue.append((ours[0], ours[1], True))
    # 价格优先；同价位我们排最后（新挂单时间最晚）
    queue.sort(key=lambda x: (-x[0], x[2]))

    total_size = sum(s for _, s, _ in queue)
    if total_size < target:
        return False, 0.0, 0.0

    # CFTC 备案规则（2026-02 更新）：按价位整体纳入——"add all bids at the current bid price"，
    # 直到累计 >= target 为止；同价位内不按时间截断
    qualifying, cum, cur_px = [], 0.0, None
    for p, s, mine in queue:
        if cum >= target and p != cur_px:
            break
        qualifying.append((p, s, mine))
        cum += s
        cur_px = p

    # 参考价：累计达到 target/5 的价位
    cum, ref = 0.0, qualifying[-1][0]
    for p, s, _ in qualifying:
        cum += s
        if cum >= target / 5:
            ref = p
            break

    our, tot = 0.0, 0.0
    for p, s, mine in qualifying:
        ticks = max(0, int(round((ref - p) / TICK)))
        sc = s * (df ** ticks)
        tot += sc
        if mine:
            our += sc
    return True, our, tot


def evaluate_market(ob: dict, target: float, df: float, q: float, offset_ticks: int):
    """给定订单簿，估算我们两侧各挂 q 张、离最优价 offset 个 tick 时的份额与资金。"""
    book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
    yes = parse_side(book.get("yes_dollars"))
    no = parse_side(book.get("no_dollars"))
    if not yes or not no:
        return None
    yes_px = round(yes[0][0] - offset_ticks * TICK, 2)
    no_px = round(no[0][0] - offset_ticks * TICK, 2)
    if yes_px < TICK or no_px < TICK:
        return None
    if yes_px + no_px > 1 - TICK + 1e-9:   # 改善价格后两侧不能交叉（否则自成交）
        return None

    base_y, _, tot_y0 = side_scores(yes, target, df)
    base_n, _, tot_n0 = side_scores(no, target, df)
    ok_y, our_y, tot_y = side_scores(yes, target, df, (yes_px, q))
    ok_n, our_n, tot_n = side_scores(no, target, df, (no_px, q))
    if not (ok_y and ok_n):
        share = 0.0
    else:
        share = 0.5 * (our_y / tot_y + our_n / tot_n)
    return {
        "best_yes_bid": yes[0][0], "best_no_bid": no[0][0],
        "spread": round(1 - yes[0][0] - no[0][0], 2),
        "depth_yes": sum(s for _, s in yes), "depth_no": sum(s for _, s in no),
        "qualified_without_us": bool(base_y and base_n),
        "share": share,
        "capital": q * yes_px + q * no_px,
    }


def scan(max_markets: int = 900, per_series_cap: int = 12, seed: int = 0):
    inc = pd.read_parquet(DATA_CACHE_DIR / "incentives_active.parquet")
    now = pd.Timestamp.now(tz="UTC")
    inc = inc[(inc["start"] <= now) & (inc["end"] > now)]
    # 同一市场可能有多个重叠时段：每日奖励相加
    per_mkt = inc.groupby("market_ticker").agg(
        series=("series", "first"), reward_per_day=("reward_per_day", "sum"),
        target=("target", "max"), df_bps=("df_bps", "min"), end=("end", "min")).reset_index()
    per_mkt = per_mkt.sample(frac=1.0, random_state=seed).groupby("series").head(per_series_cap)
    per_mkt = per_mkt.sample(min(max_markets, len(per_mkt)), random_state=seed)

    client = KalshiClient()
    rows, books = [], {}
    for i, m in enumerate(per_mkt.itertuples(index=False)):
        try:
            ob = get_with_retry(client, f"/markets/{m.market_ticker}/orderbook", {"depth": 0})
        except Exception:
            continue
        books[m.market_ticker] = ob
        df = (m.df_bps / 10000.0) if pd.notna(m.df_bps) else 0.5
        target = m.target if pd.notna(m.target) else 1000.0
        for q_frac in (0.2, 0.5, 1.0):
            for off in (-1, 0, 1, 2):
                r = evaluate_market(ob, target, df, q_frac * target, off)
                if r is None:
                    continue
                r.update({"ticker": m.market_ticker, "series": m.series,
                          "reward_per_day": m.reward_per_day, "target": target,
                          "df": df, "q": q_frac * target, "offset": off,
                          "snap_ts": pd.Timestamp.now(tz="UTC")})
                r["usd_per_day"] = r["share"] * m.reward_per_day
                r["roc_per_day"] = r["usd_per_day"] / r["capital"] if r["capital"] > 0 else np.nan
                rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(per_mkt)}", flush=True)
        time.sleep(SLEEP)
    out = pd.DataFrame(rows)
    path = DATA_CACHE_DIR / f"lip_scan_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M}.parquet"
    out.to_parquet(path, index=False)
    import json
    json.dump(books, open(str(path).replace(".parquet", "_books.json"), "w"))
    print(f"-> {path} ({out['ticker'].nunique()} markets)")
    return out


if __name__ == "__main__":
    scan(max_markets=1500, per_series_cap=25, seed=1)
