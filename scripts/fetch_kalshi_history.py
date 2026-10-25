"""抓取 Kalshi 上赛季 NBA 单场合约（KXNBAGAME）的历史价格。

两步：
1. 分页拉 /historical/markets 的全部已结算合约元数据
   -> data_cache/kalshi_nba_markets.parquet
2. 逐合约拉 /historical/markets/{ticker}/candlesticks（小时级）
   -> data_cache/kalshi_candles/{ticker}.parquet（断点续传：已存在的跳过）
   最后合并 -> data_cache/kalshi_nba_candles.parquet

约 2900 个合约、每个 1 次请求，限速 + 重试，全程约 15-30 分钟。
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient

SERIES = "KXNBAGAME"
CANDLE_DIR = DATA_CACHE_DIR / "kalshi_candles"
CANDLE_DIR.mkdir(exist_ok=True)
SLEEP = 0.15          # 请求间隔（秒）
PERIOD_MINUTES = 60   # 小时级K线


def fetch_markets(client: KalshiClient) -> pd.DataFrame:
    out_file = DATA_CACHE_DIR / "kalshi_nba_markets.parquet"
    if out_file.exists():
        df = pd.read_parquet(out_file)
        print(f"[1/3] 合约元数据已缓存: {len(df)} 个")
        return df

    print("[1/3] 拉取合约元数据 ...")
    rows, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = client._get("/historical/markets", params=params)
        ms = r.get("markets", [])
        rows.extend(ms)
        cursor = r.get("cursor")
        print(f"      累计 {len(rows)} 个")
        if not cursor or not ms:
            break
        time.sleep(SLEEP)

    df = pd.json_normalize(rows)
    for col in ("open_time", "close_time", "expiration_time", "expected_expiration_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True, format="ISO8601")
    df.to_parquet(out_file, index=False)
    print(f"      -> {out_file} ({len(df)} 个合约)")
    return df


def _get_with_retry(client: KalshiClient, path: str, params: dict, max_tries: int = 5) -> dict:
    for attempt in range(max_tries):
        try:
            return client._get(path, params=params)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 429 or code >= 500:
                wait = 2.0 * (attempt + 1)
                print(f"      HTTP {code}，{wait:.0f}s 后重试 ...")
                time.sleep(wait)
                continue
            raise
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"重试 {max_tries} 次仍失败: {path}")


def fetch_candles(client: KalshiClient, markets: pd.DataFrame):
    print(f"[2/3] 拉取 {len(markets)} 个合约的K线（已存在的跳过）...")
    done = skipped = empty = 0
    t0 = time.time()

    for row in markets.itertuples(index=False):
        ticker = row.ticker
        out = CANDLE_DIR / f"{ticker}.parquet"
        if out.exists():
            skipped += 1
            continue

        start_ts = int(pd.Timestamp(row.open_time).timestamp())
        end_ts = int(pd.Timestamp(row.close_time).timestamp()) + 3600
        r = _get_with_retry(
            client, f"/historical/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": PERIOD_MINUTES})
        candles = r.get("candlesticks", [])

        rows = []
        for c in candles:
            price = c.get("price") or {}
            bid = c.get("yes_bid") or {}
            ask = c.get("yes_ask") or {}
            rows.append({
                "ticker": ticker,
                "end_period_ts": c.get("end_period_ts"),
                "price_open": price.get("open"), "price_high": price.get("high"),
                "price_low": price.get("low"), "price_close": price.get("close"),
                "price_mean": price.get("mean"),
                "yes_bid_close": bid.get("close"), "yes_ask_close": ask.get("close"),
                "volume": c.get("volume"), "open_interest": c.get("open_interest"),
            })
        if rows:
            pd.DataFrame(rows).to_parquet(out, index=False)
        else:
            # 空的也落一个标记文件，避免重跑时反复请求
            pd.DataFrame(columns=["ticker", "end_period_ts"]).to_parquet(out, index=False)
            empty += 1
        done += 1
        if done % 100 == 0:
            rate = done / (time.time() - t0)
            eta = (len(markets) - skipped - done) / rate / 60
            print(f"      {done} 个完成（跳过 {skipped}，空 {empty}），ETA {eta:.0f} 分钟")
        time.sleep(SLEEP)

    print(f"      完成: 新抓 {done}，跳过 {skipped}，无数据 {empty}")


def combine() -> pd.DataFrame:
    print("[3/3] 合并 ...")
    files = sorted(CANDLE_DIR.glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["end_period_ts"])

    num_cols = [c for c in df.columns if c not in ("ticker",)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)

    out = DATA_CACHE_DIR / "kalshi_nba_candles.parquet"
    df.to_parquet(out, index=False)
    print(f"      -> {out}: {len(df)} 根K线, {df['ticker'].nunique()} 个合约")
    return df


if __name__ == "__main__":
    client = KalshiClient()
    markets = fetch_markets(client)
    fetch_candles(client, markets)
    df = combine()
    print("\n===== 概览 =====")
    print("时间范围:", df["ts"].min(), "~", df["ts"].max())
    print("每合约K线数分布:")
    print(df.groupby("ticker").size().describe().round(1))
