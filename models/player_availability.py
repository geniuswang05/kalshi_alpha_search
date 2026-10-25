"""球员缺阵价值特征（v2 的核心新增信息）。

思路：
1. 球员身价 = 近15场 Game Score（Hollinger GmSc）滚动均值（含最近一场，
   这是赛前可知的——他上一场打完，数据就有了）
2. 预期轮换名单 = 该队最近3场出场过的所有球员（自动处理交易/轮换变动）
3. 缺阵 = 预期名单里今天没上场的人；缺阵价值 = 这些人的身价之和

两个版本（量化前视偏差用）：
- 激进版 (MISS_*)：按当场实际未出场算。开球时公开，但我们 T-4h 下注，
  轻微前视（当日突发轮休在 T-4h 可能未公布）
- 保守版 (MISS_CONS_*)：只算"上一场就已经缺阵"的球员——赛前铁定可知，零前视

输出粒度：每 (GAME_ID, TEAM_ID) 一行。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROSTER_LOOKBACK = 3     # 预期名单 = 最近N场出场过的人
RATING_WINDOW = 15      # 身价滚动窗口
RATING_MIN_GAMES = 5    # 不足N场用默认身价
DEFAULT_RATING = 3.0    # 边缘球员默认身价（GmSc量级）
STAR_RATING = 12.0      # 星级门槛（全明星级 GmSc≈15+，主力≈10-14）


def game_score(df: pd.DataFrame) -> pd.Series:
    return (df["PTS"] + 0.4 * df["FGM"] - 0.7 * df["FGA"]
            - 0.4 * (df["FTA"] - df["FTM"]) + 0.7 * df["OREB"] + 0.3 * df["DREB"]
            + df["STL"] + 0.7 * df["AST"] + 0.7 * df["BLK"]
            - 0.4 * df["PF"] - df["TOV"])


def build_availability_features(player_logs: pd.DataFrame) -> pd.DataFrame:
    logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).copy()
    logs["GMSC"] = game_score(logs)

    # 球员身价：近15场GmSc滚动均值（含当场之前的最近一场；对"今天"而言无前视）
    grp = logs.groupby("PLAYER_ID", sort=False)
    logs["RATING"] = grp["GMSC"].transform(
        lambda s: s.rolling(RATING_WINDOW, min_periods=RATING_MIN_GAMES).mean())
    logs["RATING"] = logs["RATING"].fillna(DEFAULT_RATING).clip(lower=0.0)
    logs["AVG_MIN"] = grp["MIN"].transform(
        lambda s: s.rolling(RATING_WINDOW, min_periods=RATING_MIN_GAMES).mean()
    ).fillna(10.0)

    # 每队比赛序号
    team_games = (logs[["TEAM_ID", "GAME_ID", "GAME_DATE"]]
                  .drop_duplicates()
                  .sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]))
    team_games["SEQ"] = team_games.groupby("TEAM_ID").cumcount()
    logs = logs.merge(team_games, on=["TEAM_ID", "GAME_ID", "GAME_DATE"])

    # 出场记录（带该场后的身价，供向后replicate）
    appear = logs[["TEAM_ID", "SEQ", "PLAYER_ID", "RATING", "AVG_MIN"]].copy()

    # 预期名单：把每次出场向后复制到之后 1..N 场
    reps = []
    for k in range(1, ROSTER_LOOKBACK + 1):
        r = appear.copy()
        r["SEQ"] = r["SEQ"] + k
        r["SRC_SEQ"] = r["SEQ"] - k
        reps.append(r)
    roster = pd.concat(reps, ignore_index=True)
    # 同一球员可能由多场出场记录生成——保留最近一次（身价最新）
    roster = (roster.sort_values("SRC_SEQ")
              .drop_duplicates(["TEAM_ID", "SEQ", "PLAYER_ID"], keep="last"))

    # 今天是否出场
    played_today = appear[["TEAM_ID", "SEQ", "PLAYER_ID"]].assign(PLAYED=1)
    roster = roster.merge(played_today, on=["TEAM_ID", "SEQ", "PLAYER_ID"], how="left")
    roster["PLAYED"] = roster["PLAYED"].fillna(0).astype(int)
    # 上一场是否出场（保守版：上一场就缺阵的人，赛前铁定可知）
    played_prev = appear[["TEAM_ID", "SEQ", "PLAYER_ID"]].assign(PLAYED_PREV=1)
    played_prev["SEQ"] = played_prev["SEQ"] + 1
    roster = roster.merge(played_prev, on=["TEAM_ID", "SEQ", "PLAYER_ID"], how="left")
    roster["PLAYED_PREV"] = roster["PLAYED_PREV"].fillna(0).astype(int)

    # 只看预期名单里身价前8的人（缺个第12人不重要）
    roster["RANK"] = roster.groupby(["TEAM_ID", "SEQ"])["RATING"].rank(
        ascending=False, method="first")
    top8 = roster[roster["RANK"] <= 8].copy()

    top8["MISS_VAL"] = np.where(top8["PLAYED"] == 0, top8["RATING"], 0.0)
    top8["MISS_MIN"] = np.where(top8["PLAYED"] == 0, top8["AVG_MIN"], 0.0)
    top8["MISS_N"] = (top8["PLAYED"] == 0).astype(int)
    top8["STAR_OUT"] = ((top8["PLAYED"] == 0)
                        & (top8["RATING"] >= STAR_RATING)).astype(int)
    cons = (top8["PLAYED_PREV"] == 0)
    top8["MISS_CONS_VAL"] = np.where(cons, top8["RATING"], 0.0)
    top8["MISS_CONS_N"] = cons.astype(int)

    feats = top8.groupby(["TEAM_ID", "SEQ"]).agg(
        MISS_VAL=("MISS_VAL", "sum"), MISS_MIN=("MISS_MIN", "sum"),
        MISS_N=("MISS_N", "sum"), STAR_OUT=("STAR_OUT", "max"),
        MISS_CONS_VAL=("MISS_CONS_VAL", "sum"), MISS_CONS_N=("MISS_CONS_N", "sum"),
        ROSTER_VAL=("RATING", "sum"),   # 预期名单前8总身价（球队强度的独立信号）
    ).reset_index()

    out = feats.merge(team_games, on=["TEAM_ID", "SEQ"])
    return out.drop(columns=["SEQ", "GAME_DATE"])


AVAIL_COLS = ["MISS_VAL", "MISS_MIN", "MISS_N", "STAR_OUT",
              "MISS_CONS_VAL", "MISS_CONS_N", "ROSTER_VAL"]


def merge_into_table(table: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """把缺阵特征并进训练表（HOME_/AWAY_ 前缀 + DIFF）。"""
    home = avail.rename(columns={"TEAM_ID": "HOME_TEAM_ID",
                                 **{c: f"HOME_{c}" for c in AVAIL_COLS}})
    away = avail.rename(columns={"TEAM_ID": "AWAY_TEAM_ID",
                                 **{c: f"AWAY_{c}" for c in AVAIL_COLS}})
    out = table.merge(home, on=["GAME_ID", "HOME_TEAM_ID"], how="left")
    out = out.merge(away, on=["GAME_ID", "AWAY_TEAM_ID"], how="left")
    for c in AVAIL_COLS:
        out[f"HOME_{c}"] = out[f"HOME_{c}"].fillna(0.0)
        out[f"AWAY_{c}"] = out[f"AWAY_{c}"].fillna(0.0)
        out[f"DIFF_{c}"] = out[f"HOME_{c}"] - out[f"AWAY_{c}"]
    return out
