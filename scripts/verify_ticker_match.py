"""验证 Kalshi ticker 与比赛表的匹配率（日期 + 主客队约定 + 缩写差异）。"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR

mk = pd.read_parquet(DATA_CACHE_DIR / "kalshi_nba_markets.parquet")
games = pd.read_parquet(DATA_CACHE_DIR / "training_table.parquet")

PAT = re.compile(r"^KXNBAGAME-(\d{2}[A-Z]{3}\d{2})([A-Z]+)-([A-Z]+)$")

rows = []
for t in mk["ticker"]:
    m = PAT.match(t)
    if not m:
        rows.append({"ticker": t, "date": pd.NaT, "pair": None, "side": None})
        continue
    date = pd.to_datetime(m.group(1), format="%y%b%d")
    pair, side = m.group(2), m.group(3)
    rows.append({"ticker": t, "date": date, "pair": pair, "side": side})
parsed = pd.DataFrame(rows)
print("解析失败数:", parsed["date"].isna().sum())

# 从 pair 里剥出另一队
def split_pair(pair: str, side: str):
    if pair.startswith(side):
        return side, pair[len(side):]      # side 在前
    if pair.endswith(side):
        return pair[: -len(side)], side    # side 在后
    return None, None

parsed[["team1", "team2"]] = parsed.apply(
    lambda r: pd.Series(split_pair(r["pair"], r["side"]) if r["pair"] else (None, None)),
    axis=1)
bad = parsed[parsed["team1"].isna()]
print("pair 无法拆分数:", len(bad))
if len(bad):
    print(bad.head())

# 比赛表：2025-26 赛季（Kalshi 数据的主要覆盖期）
g = games[games["SEASON"] == "2025-26"].copy()
g["date_str"] = g["GAME_DATE"].dt.strftime("%Y-%m-%d")
print(f"\n比赛表 2025-26: {len(g)} 场")
print("比赛表缩写:", sorted(set(g["HOME_TEAM_ABBREVIATION"]) | set(g["AWAY_TEAM_ABBREVIATION"])))

kalshi_codes = set(parsed["team1"].dropna()) | set(parsed["team2"].dropna())
nba_codes = set(g["HOME_TEAM_ABBREVIATION"]) | set(g["AWAY_TEAM_ABBREVIATION"])
print("Kalshi 缩写:", sorted(kalshi_codes))
print("Kalshi 有而 NBA 表没有:", sorted(kalshi_codes - nba_codes))
print("NBA 表有而 Kalshi 没有:", sorted(nba_codes - kalshi_codes))

# 主客队顺序约定验证：team1/team2 谁是主队？
# 用比赛表做双向 join 统计
parsed["date_str"] = parsed["date"].dt.strftime("%Y-%m-%d")
merged_t1home = parsed.merge(
    g, left_on=["date_str", "team1", "team2"],
    right_on=["date_str", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"])
merged_t2home = parsed.merge(
    g, left_on=["date_str", "team2", "team1"],
    right_on=["date_str", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"])
print(f"\n按 team1=主队 匹配上: {len(merged_t1home)}")
print(f"按 team2=主队 匹配上: {len(merged_t2home)}")

# 总匹配率（不管顺序）
both = pd.concat([merged_t1home, merged_t2home])
matched_tickers = both["ticker"].nunique()
season_tickers = parsed[(parsed["date"] >= "2025-10-01") & (parsed["date"] <= "2026-04-15")]
print(f"\n2025-26 常规赛窗口内 Kalshi ticker 数: {len(season_tickers)}")
print(f"与比赛表匹配上的: {matched_tickers} "
      f"({matched_tickers / max(len(season_tickers), 1) * 100:.1f}%)")

# 未匹配样本
unmatched = season_tickers[~season_tickers["ticker"].isin(both["ticker"])]
print(f"未匹配数: {len(unmatched)}")
if len(unmatched):
    print(unmatched[["ticker", "date_str", "team1", "team2"]].head(10).to_string(index=False))
