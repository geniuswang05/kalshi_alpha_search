"""跨平台（Kalshi × Polymarket）同场比赛的分钟级价格对齐。

匹配规则：Polymarket 事件 slug 形如 mlb-mil-lad-2026-10-15（outcomes 顺序 = slug 里球队顺序），
Kalshi ticker 形如 KXMLBGAME-26OCT152000MILLAD-MIL。按 日期 + 两队代码（无序）匹配，
对代码不一致的球队做映射。只用 Kalshi 一侧合约：选 slug 第一支球队对应的 Kalshi 市场，
Polymarket 取 outcomes[0] 的 token —— 两者押的是同一支球队。

输出：data_cache/xvenue/{league}/{kalshi_ticker}.parquet，每分钟一行：
  ts, k_bid, k_ask, k_last, k_vol, pm_p, result, start_time
"""

import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR
from data.kalshi_client import KalshiClient
from data.kalshi_universe import cutoff_ts, market_candles, settled_markets

OUT = DATA_CACHE_DIR / "xvenue"
OUT.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (research script)"

LEAGUES = {
    # league: (kalshi series, polymarket series_slug)
    "mlb": ("KXMLBGAME", "mlb"),
    "nfl": ("KXNFLGAME", "nfl"),
    "nhl": ("KXNHLGAME", "nhl"),
    "wnba": ("KXWNBAGAME", "wnba"),
}
# Polymarket 代码 → Kalshi 代码（只列不一致的）
CODE_MAP = {
    "mlb": {"ari": "AZ", "az": "AZ", "chw": "CWS", "was": "WSH", "oak": "ATH", "ath": "ATH",
            "kcr": "KC", "sdp": "SD", "sfg": "SF", "tbr": "TB"},
    "nfl": {"la": "LA", "lar": "LA", "wsh": "WAS", "jac": "JAX"},
    "nhl": {"lak": "LA", "njd": "NJ", "sjs": "SJ", "tbl": "TB", "vgk": "VGK", "mon": "MTL"},
    "wnba": {},
}
KALSHI_TICKER = re.compile(r"^[A-Z0-9]+-(\d{2}[A-Z]{3}\d{2})(\d{4})?([A-Z]+)-([A-Z]+)$")


def curl_json(url: str, retries: int = 4):
    for i in range(retries):
        p = subprocess.run(["curl", "-s", "-A", UA, "--max-time", "40", url],
                           capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip().startswith(("{", "[")):
            return json.loads(p.stdout)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"curl failed: {url}")


GAME_SLUG = re.compile(r"^[a-z]+-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}$")


def pm_events(series_slug: str, since: str, until: str | None = None) -> pd.DataFrame:
    """按事件"创建日期"逐天分页（Gamma 的 offset 上限约 2000）。只保留单场胜负事件。"""
    rows = []
    day = pd.Timestamp(since, tz="UTC") - pd.Timedelta(days=10)   # 事件一般提前几天创建
    end = pd.Timestamp(until, tz="UTC") if until else pd.Timestamp.now(tz="UTC")
    while day < end:
        nxt = day + pd.Timedelta(days=1)
        offset = 0
        while offset < 2000:
            url = (f"https://gamma-api.polymarket.com/events?series_slug={series_slug}"
                   f"&closed=true&limit=100&offset={offset}"
                   f"&start_date_min={day:%Y-%m-%dT%H:%M:%SZ}&start_date_max={nxt:%Y-%m-%dT%H:%M:%SZ}")
            batch = curl_json(url)
            for _ in range(3):
                if isinstance(batch, list):
                    break
                time.sleep(3)          # 限流/错误时返回的是 dict
                batch = curl_json(url)
            if not isinstance(batch, list) or not batch:
                break
            for e in batch:
                if not GAME_SLUG.match(e.get("slug", "")) or not e.get("startTime"):
                    continue
                m = (e.get("markets") or [None])[0]
                if not m:
                    continue
                try:
                    toks = json.loads(m["clobTokenIds"])
                    outs = json.loads(m["outcomes"])
                except Exception:
                    continue
                rows.append({"slug": e["slug"], "start": pd.Timestamp(e["startTime"]),
                             "token0": toks[0], "outcome0": outs[0],
                             "pm_volume": float(e.get("volume") or 0)})
            if len(batch) < 100:
                break
            offset += 100
        day = nxt
        time.sleep(0.1)
    df = pd.DataFrame(rows).drop_duplicates("slug")
    return df[df["start"] >= pd.Timestamp(since, tz="UTC")]


def match_games(league: str, since: str) -> pd.DataFrame:
    k_series, pm_slug = LEAGUES[league]
    client = KalshiClient()
    km = settled_markets(client, k_series)
    km = km[km["result"].isin(["yes", "no"]) & (km["close_time"] >= pd.Timestamp(since, tz="UTC"))]
    parsed = []
    for t in km.itertuples(index=False):
        m = KALSHI_TICKER.match(t.ticker)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), format="%y%b%d").date()
        pair, team = m.group(3), m.group(4)
        other = pair[:-len(team)] if pair.endswith(team) else pair[len(team):]
        parsed.append({"ticker": t.ticker, "date": date, "team": team, "other": other,
                       "close_time": t.close_time, "result": t.result})
    kdf = pd.DataFrame(parsed)

    pdf = pm_events(pm_slug, since)
    cmap = CODE_MAP.get(league, {})
    out = []
    for e in pdf.itertuples(index=False):
        parts = e.slug.split("-")
        if len(parts) < 6:
            continue
        a, b = parts[1], parts[2]
        date = pd.Timestamp("-".join(parts[3:6])).date()
        ka, kb = cmap.get(a, a.upper()), cmap.get(b, b.upper())
        cand = kdf[(kdf["date"] == date) & (kdf["team"] == ka) & (kdf["other"] == kb)]
        if len(cand) != 1:
            continue
        c = cand.iloc[0]
        out.append({"league": league, "series": k_series, "kalshi_ticker": c.ticker,
                    "close_time": c.close_time, "result": c.result, "slug": e.slug,
                    "start": e.start, "token0": e.token0, "outcome0": e.outcome0,
                    "pm_volume": e.pm_volume})
    return pd.DataFrame(out)


def fetch_game(row, trade_cutoff) -> str:
    d = OUT / row.league
    d.mkdir(exist_ok=True)
    path = d / f"{row.kalshi_ticker}.parquet"
    if path.exists():
        return "cached"
    client = KalshiClient()
    t0 = int((row.start - pd.Timedelta(hours=2)).timestamp())
    t1 = int(row.close_time.timestamp())
    if t1 - t0 < 3600:          # 改期比赛：PM 开球时间晚于 Kalshi 结算
        pd.DataFrame().to_parquet(path)
        return "rescheduled"
    k = market_candles(client, row.series, row.kalshi_ticker, t0, t1, 1,
                       historical=row.close_time < trade_cutoff)
    if k.empty:
        pd.DataFrame().to_parquet(path)
        return "no kalshi"
    h = curl_json(f"https://clob.polymarket.com/prices-history?market={row.token0}"
                  f"&startTs={t0}&endTs={t1}&fidelity=1").get("history", [])
    if not h:
        pd.DataFrame().to_parquet(path)
        return "no pm"
    pm = pd.DataFrame(h).rename(columns={"p": "pm_p"})
    pm["ts"] = pd.to_datetime(pm["t"], unit="s", utc=True)
    k = k.rename(columns={"bid": "k_bid", "ask": "k_ask", "last": "k_last", "volume": "k_vol"})
    m = pd.merge_asof(k.sort_values("ts"), pm.sort_values("ts")[["ts", "pm_p"]],
                      on="ts", direction="backward")
    m["result"] = row.result
    m["start_time"] = row.start
    m["pm_volume"] = row.pm_volume
    m.to_parquet(path, index=False)
    return "ok"


if __name__ == "__main__":
    leagues = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mlb"]
    since = sys.argv[2] if len(sys.argv) > 2 else "2026-04-01"
    trade_cutoff = cutoff_ts(KalshiClient())["market_settled_ts"]
    for lg in leagues:
        mpath = OUT / f"matches_{lg}.parquet"
        if mpath.exists():
            games = pd.read_parquet(mpath)
        else:
            games = match_games(lg, since)
            games.to_parquet(mpath, index=False)
        print(f"{lg}: matched {len(games)} games", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            def safe(r):
                try:
                    return fetch_game(r, trade_cutoff)
                except Exception as e:
                    return f"error {type(e).__name__}"
            res = list(pool.map(safe, games.itertuples(index=False)))
        print(f"{lg}: " + str(pd.Series(res).value_counts().to_dict()), flush=True)
