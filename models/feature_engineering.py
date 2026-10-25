"""NBA 特征工程。

输入：nba_fetcher.to_games() 输出的每场一行宽表（按时间排序）。
输出：训练表 —— 每行是一场比赛"开赛前"可知的特征 + HOME_WIN 标签。

防未来函数是这里的第一原则：
- 所有滚动统计都 shift(1)，只用该场之前的比赛；
- Elo 用的是赛前评分（先记录、再用赛果更新）；
- 赛季初样本不足的比赛整行丢弃（MIN_GAMES_FOR_FEATURES）。

特征分四组：
1. Elo：赛前 Elo 差（含跨赛季回归、带分差乘数的更新，538 风格）
2. 效率四因子：滚动 ORTG/DRTG/NET/PACE/eFG%/TOV率/OREB率/罚球率
3. 状态：近 N 场胜率、近 N 场净胜分
4. 日程：休息天数、是否背靠背、近 7 天比赛场数
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    ELO_HOME_ADV,
    ELO_INITIAL,
    ELO_K,
    ELO_MEAN,
    ELO_SEASON_CARRYOVER,
    MIN_GAMES_FOR_FEATURES,
    ROLLING_WINDOWS,
)


# ======================================================================
# 1. Elo
# ======================================================================

def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """逐场计算赛前 Elo。返回 games + HOME_ELO_PRE / AWAY_ELO_PRE / ELO_DIFF / ELO_HOME_WIN_PROB。

    更新规则（FiveThirtyEight NBA Elo）：
    - 期望胜率 E = 1 / (1 + 10^(-(elo_diff + home_adv)/400))
    - 分差乘数 mov_mult = ((mov+3)^0.8) / (7.5 + 0.006*elo_diff_winner)
      （防止强队大胜刷分的自相关问题）
    - 跨赛季：elo = carry*elo + (1-carry)*1505
    """
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    elo: dict[int, float] = {}
    prev_season: dict[int, str] = {}

    home_pre = np.empty(len(games))
    away_pre = np.empty(len(games))

    for i, row in enumerate(games.itertuples(index=False)):
        h, a, season = row.HOME_TEAM_ID, row.AWAY_TEAM_ID, row.SEASON
        for team in (h, a):
            if team not in elo:
                elo[team] = ELO_INITIAL
                prev_season[team] = season
            elif prev_season[team] != season:
                elo[team] = ELO_SEASON_CARRYOVER * elo[team] + (1 - ELO_SEASON_CARRYOVER) * ELO_MEAN
                prev_season[team] = season

        home_pre[i], away_pre[i] = elo[h], elo[a]

        # 用赛果更新（更新只影响之后的比赛，不影响本行的赛前值）
        diff = elo[h] + ELO_HOME_ADV - elo[a]
        exp_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        home_win = row.HOME_WIN
        mov = abs(row.HOME_PTS - row.AWAY_PTS)
        elo_diff_winner = diff if home_win else -diff
        mov_mult = ((mov + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)
        shift = ELO_K * mov_mult * (home_win - exp_home)
        elo[h] += shift
        elo[a] -= shift

    out = games.copy()
    out["HOME_ELO_PRE"] = home_pre
    out["AWAY_ELO_PRE"] = away_pre
    out["ELO_DIFF"] = out["HOME_ELO_PRE"] - out["AWAY_ELO_PRE"]
    out["ELO_HOME_WIN_PROB"] = 1.0 / (
        1.0 + 10.0 ** (-(out["ELO_DIFF"] + ELO_HOME_ADV) / 400.0))
    return out


# ======================================================================
# 2-4. 球队滚动特征（在"每队每场"长表上算，再 merge 回宽表）
# ======================================================================

def _team_long_table(games: pd.DataFrame) -> pd.DataFrame:
    """把宽表拆回长表：每队每场一行，带上对手数据用于算防守效率。"""
    def one_side(side: str, opp: str) -> pd.DataFrame:
        df = pd.DataFrame({
            "GAME_ID": games["GAME_ID"],
            "GAME_DATE": games["GAME_DATE"],
            "SEASON": games["SEASON"],
            "TEAM_ID": games[f"{side}_TEAM_ID"],
            "IS_HOME": 1 if side == "HOME" else 0,
            "PTS": games[f"{side}_PTS"],
            "OPP_PTS": games[f"{opp}_PTS"],
            "FGA": games[f"{side}_FGA"],
            "FGM": games[f"{side}_FGM"],
            "FG3M": games[f"{side}_FG3M"],
            "FTA": games[f"{side}_FTA"],
            "OREB": games[f"{side}_OREB"],
            "DREB": games[f"{side}_DREB"],
            "TOV": games[f"{side}_TOV"],
            "OPP_FGA": games[f"{opp}_FGA"],
            "OPP_FTA": games[f"{opp}_FTA"],
            "OPP_OREB": games[f"{opp}_OREB"],
            "OPP_DREB": games[f"{opp}_DREB"],
            "OPP_TOV": games[f"{opp}_TOV"],
            "WIN": games["HOME_WIN"] if side == "HOME" else 1 - games["HOME_WIN"],
        })
        return df

    long = pd.concat([one_side("HOME", "AWAY"), one_side("AWAY", "HOME")], ignore_index=True)
    return long.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _per_game_efficiency(long: pd.DataFrame) -> pd.DataFrame:
    """单场效率指标（之后再做滚动平均）。回合数用标准估计式。"""
    df = long.copy()
    df["POSS"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    df["OPP_POSS"] = df["OPP_FGA"] - df["OPP_OREB"] + df["OPP_TOV"] + 0.44 * df["OPP_FTA"]
    pace_poss = (df["POSS"] + df["OPP_POSS"]) / 2.0

    df["ORTG"] = 100.0 * df["PTS"] / pace_poss          # 每百回合得分
    df["DRTG"] = 100.0 * df["OPP_PTS"] / pace_poss      # 每百回合失分
    df["NET_RTG"] = df["ORTG"] - df["DRTG"]
    df["PACE"] = pace_poss
    df["EFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["TOV_RATE"] = df["TOV"] / df["POSS"]
    df["OREB_RATE"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"])
    df["FT_RATE"] = df["FTA"] / df["FGA"]
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    return df


ROLLING_STATS = ["ORTG", "DRTG", "NET_RTG", "PACE", "EFG_PCT",
                 "TOV_RATE", "OREB_RATE", "FT_RATE", "WIN", "MARGIN"]


def _rolling_team_features(long: pd.DataFrame) -> pd.DataFrame:
    """对每支球队算 shift(1) 后的滚动均值 —— 只用该场之前的信息。"""
    df = _per_game_efficiency(long)
    grp = df.groupby("TEAM_ID", sort=False)

    feats = df[["GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON"]].copy()
    for w in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            # transform 保持索引对齐，shift(1) 保证只用该场之前的比赛
            feats[f"{stat}_L{w}"] = grp[stat].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=max(3, w // 2)).mean())

    # 日程特征
    feats["REST_DAYS"] = grp["GAME_DATE"].diff().dt.days.clip(upper=10)
    feats["IS_B2B"] = (feats["REST_DAYS"] == 1).astype(int)

    def _games_last7(dates: pd.Series) -> np.ndarray:
        s = pd.Series(1.0, index=pd.DatetimeIndex(dates))
        return s.rolling("7D").sum().shift(1).to_numpy()

    feats["GAMES_LAST7"] = grp["GAME_DATE"].transform(_games_last7)

    # 该队本赛季已打场数（用于过滤赛季初样本）
    feats["SEASON_GAME_NUM"] = (
        df.groupby(["TEAM_ID", "SEASON"], sort=False).cumcount() + 1)
    return feats


# ======================================================================
# 汇总入口
# ======================================================================

def build_training_table(games: pd.DataFrame) -> pd.DataFrame:
    """games 宽表 -> 训练表（特征全部是赛前可知的）。"""
    games = compute_elo(games)
    long = _team_long_table(games)
    team_feats = _rolling_team_features(long)

    base_cols = ["GAME_ID", "GAME_DATE", "SEASON",
                 "HOME_TEAM_ID", "HOME_TEAM_ABBREVIATION",
                 "AWAY_TEAM_ID", "AWAY_TEAM_ABBREVIATION",
                 "HOME_ELO_PRE", "AWAY_ELO_PRE", "ELO_DIFF", "ELO_HOME_WIN_PROB",
                 "HOME_WIN"]
    table = games[base_cols].copy()

    feat_cols = [c for c in team_feats.columns
                 if c not in ("GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON")]
    home_feats = team_feats.rename(
        columns={"TEAM_ID": "HOME_TEAM_ID", **{c: f"HOME_{c}" for c in feat_cols}})
    away_feats = team_feats.rename(
        columns={"TEAM_ID": "AWAY_TEAM_ID", **{c: f"AWAY_{c}" for c in feat_cols}})

    table = table.merge(
        home_feats.drop(columns=["GAME_DATE", "SEASON"]),
        on=["GAME_ID", "HOME_TEAM_ID"], how="left", validate="1:1")
    table = table.merge(
        away_feats.drop(columns=["GAME_DATE", "SEASON"]),
        on=["GAME_ID", "AWAY_TEAM_ID"], how="left", validate="1:1")

    # 差值特征：树模型能自己学交互，但显式给差值能明显帮它省分裂
    for w in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            table[f"DIFF_{stat}_L{w}"] = (
                table[f"HOME_{stat}_L{w}"] - table[f"AWAY_{stat}_L{w}"])
    table["DIFF_REST_DAYS"] = table["HOME_REST_DAYS"] - table["AWAY_REST_DAYS"]

    # 过滤：双方赛季内场次都达标 + 无缺失
    ok = ((table["HOME_SEASON_GAME_NUM"] > MIN_GAMES_FOR_FEATURES)
          & (table["AWAY_SEASON_GAME_NUM"] > MIN_GAMES_FOR_FEATURES))
    table = table[ok].dropna().reset_index(drop=True)
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    """给模型用的特征列（排除 ID/标签/元信息）。"""
    exclude = {"GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ID", "AWAY_TEAM_ID",
               "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION", "HOME_WIN",
               "HOME_SEASON_GAME_NUM", "AWAY_SEASON_GAME_NUM"}
    return [c for c in table.columns if c not in exclude]
