"""Kalshi 市场全集工具：枚举系列、拉已结算市场（live + historical 两段）、拉成交明细。

历史数据分两段存放（分界线见 GET /historical/cutoff）：
- 截止线之前结算的市场/成交 → /historical/markets、/historical/trades
- 截止线之后的 → /markets?status=settled、/markets/trades
本模块把两段拼起来，调用方不用关心分界线。
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient

TRADES_DIR = DATA_CACHE_DIR / "trades"
TRADES_DIR.mkdir(exist_ok=True)
MARKETS_DIR = DATA_CACHE_DIR / "markets"
MARKETS_DIR.mkdir(exist_ok=True)
SLEEP = 0.12


def get_with_retry(client: KalshiClient, path: str, params: dict, max_tries: int = 6) -> dict:
    for attempt in range(max_tries):
        try:
            return client._get(path, params=params)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 429 or code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed after {max_tries} tries: {path} {params}")


def _paginate(client, path, params, key, max_pages=200):
    rows, cursor = [], None
    for _ in range(max_pages):
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        r = get_with_retry(client, path, p)
        batch = r.get(key, [])
        rows.extend(batch)
        cursor = r.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(SLEEP)
    return rows


def settled_markets(client: KalshiClient, series: str, refresh: bool = False) -> pd.DataFrame:
    """某系列全部已结算市场（historical + live 两段合并），带缓存。"""
    cache = MARKETS_DIR / f"{series}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    hist = _paginate(client, "/historical/markets",
                     {"series_ticker": series, "limit": 1000}, "markets")
    live = _paginate(client, "/markets",
                     {"series_ticker": series, "status": "settled", "limit": 1000}, "markets")
    df = pd.json_normalize(hist + live)
    if df.empty:
        return df
    df = df.drop_duplicates("ticker")
    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601", errors="coerce")
    for col in ("volume_fp", "open_interest_fp"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["series"] = series
    keep = [c for c in ("ticker", "event_ticker", "series", "title", "yes_sub_title",
                        "open_time", "close_time", "result", "volume_fp",
                        "open_interest_fp") if c in df.columns]
    df = df[keep]
    df.to_parquet(cache, index=False)
    return df


def market_trades(client: KalshiClient, ticker: str, settled_before_cutoff: bool,
                  max_pages: int = 300) -> pd.DataFrame:
    """某市场全部成交（每页 1000），带缓存。"""
    cache = TRADES_DIR / f"{ticker}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    path = "/historical/trades" if settled_before_cutoff else "/markets/trades"
    rows = _paginate(client, path, {"ticker": ticker, "limit": 1000}, "trades",
                     max_pages=max_pages)
    if not rows and settled_before_cutoff:
        # 截止线附近的市场，成交可能跨两段
        rows = _paginate(client, "/markets/trades", {"ticker": ticker, "limit": 1000},
                         "trades", max_pages=max_pages)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["ticker", "created_time", "yes_price_dollars",
                                   "count_fp", "taker_outcome_side", "is_block_trade"])
    else:
        df = df[["ticker", "created_time", "yes_price_dollars", "count_fp",
                 "taker_outcome_side", "is_block_trade"]]
    df.to_parquet(cache, index=False)
    return df


def _num(d: dict | None, key: str):
    """K线字段兼容：历史端点用 'close'，实时端点用 'close_dollars'。"""
    if not d:
        return None
    v = d.get(key, d.get(f"{key}_dollars"))
    return float(v) if v not in (None, "") else None


def parse_candles(candles: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candles:
        rows.append({
            "ts": pd.to_datetime(c["end_period_ts"], unit="s", utc=True),
            "bid": _num(c.get("yes_bid"), "close"),
            "ask": _num(c.get("yes_ask"), "close"),
            "last": _num(c.get("price"), "close"),
            "volume": float(c.get("volume", c.get("volume_fp")) or 0),
        })
    return pd.DataFrame(rows)


def market_candles(client: KalshiClient, series: str, ticker: str, start_ts: int,
                   end_ts: int, period: int, historical: bool) -> pd.DataFrame:
    path = (f"/historical/markets/{ticker}/candlesticks" if historical
            else f"/series/{series}/markets/{ticker}/candlesticks")
    r = get_with_retry(client, path, {"start_ts": start_ts, "end_ts": end_ts,
                                      "period_interval": period})
    return parse_candles(r.get("candlesticks", []))


def cutoff_ts(client: KalshiClient) -> dict:
    r = client._get("/historical/cutoff")
    return {k: pd.Timestamp(v) for k, v in r.items()}


def market_meta(client: KalshiClient, tickers: list[str]) -> pd.DataFrame:
    """批量拉市场元数据（含事前可知的 expected_expiration_time），historical + live 两段。"""
    rows = []
    tickers = list(dict.fromkeys(tickers))
    for path in ("/historical/markets", "/markets"):
        missing = [t for t in tickers if t not in {r["ticker"] for r in rows}]
        for i in range(0, len(missing), 40):      # ticker 很长时 100 个会让 URL 超长
            chunk = missing[i:i + 40]
            try:
                r = get_with_retry(client, path, {"tickers": ",".join(chunk), "limit": 1000})
            except Exception:
                continue
            rows.extend(r.get("markets", []))
            time.sleep(SLEEP)
    df = pd.json_normalize(rows).drop_duplicates("ticker")
    for c in ("expected_expiration_time", "occurrence_datetime", "open_time", "close_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601", errors="coerce")
    keep = [c for c in ("ticker", "expected_expiration_time", "occurrence_datetime",
                        "open_time", "close_time") if c in df.columns]
    return df[keep]
