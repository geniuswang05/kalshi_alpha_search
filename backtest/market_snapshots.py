"""把 Kalshi 历史K线变成"每场比赛的赛前市场快照"。

- 解析 ticker: KXNBAGAME-{YYMONDD}{客队}{主队}-{结算方}
- 快照时点: close_time − SNAPSHOT_HOURS_BEFORE_CLOSE 小时之前的最后一根K线
  （close_time ≈ 比赛结束后几分钟结算，NBA 比赛约 2.2-2.7h，留 4h 稳妥在开球前）
- 每场比赛输出两行（主队合约 + 客队合约），带 bid/ask/mid 和结算结果
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR, SNAPSHOT_HOURS_BEFORE_CLOSE

TICKER_PAT = re.compile(r"^KXNBAGAME-(\d{2}[A-Z]{3}\d{2})([A-Z]+)-([A-Z]+)$")


def parse_tickers(markets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in markets.itertuples(index=False):
        m = TICKER_PAT.match(row.ticker)
        if not m:
            continue
        pair, side = m.group(2), m.group(3)
        if pair.endswith(side):
            away, home = pair[: -len(side)], side
        elif pair.startswith(side):
            away, home = side, pair[len(side):]
        else:
            continue
        rows.append({
            "ticker": row.ticker,
            "game_date": pd.to_datetime(m.group(1), format="%y%b%d"),
            "away": away, "home": home,
            "bet_on": side,                      # 该合约赌哪队赢
            "close_time": pd.Timestamp(row.close_time),
            "result": row.result,                # 'yes'/'no'（Kalshi 官方结算）
        })
    return pd.DataFrame(rows)


def build_snapshots(hours_before_close: float = SNAPSHOT_HOURS_BEFORE_CLOSE) -> pd.DataFrame:
    markets = pd.read_parquet(DATA_CACHE_DIR / "kalshi_nba_markets.parquet")
    candles = pd.read_parquet(DATA_CACHE_DIR / "kalshi_nba_candles.parquet")
    parsed = parse_tickers(markets)

    cutoff = (parsed.set_index("ticker")["close_time"]
              - pd.Timedelta(hours=hours_before_close))
    candles = candles.merge(cutoff.rename("cutoff"), left_on="ticker", right_index=True)
    pre = candles[candles["ts"] <= candles["cutoff"]]

    # 每个合约取 cutoff 前最后一根K线
    idx = pre.groupby("ticker")["ts"].idxmax()
    snap = pre.loc[idx, ["ticker", "ts", "price_close",
                         "yes_bid_close", "yes_ask_close", "volume"]].rename(
        columns={"ts": "snap_ts", "price_close": "last_trade",
                 "yes_bid_close": "bid", "yes_ask_close": "ask",
                 "volume": "snap_volume"})

    out = parsed.merge(snap, on="ticker", how="left")
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    # 快照距结算的小时数（诊断用：应该 >= SNAPSHOT_HOURS_BEFORE_CLOSE）
    out["hours_before_close"] = (
        (out["close_time"] - out["snap_ts"]).dt.total_seconds() / 3600)
    return out


if __name__ == "__main__":
    snaps = build_snapshots()
    ok = snaps.dropna(subset=["bid", "ask"])
    print(f"合约 {len(snaps)}，有快照的 {len(ok)}")
    print("快照距结算小时数分布:")
    print(ok["hours_before_close"].describe().round(1))
    print("价差分布(美元):")
    print((ok["ask"] - ok["bid"]).describe().round(3))
