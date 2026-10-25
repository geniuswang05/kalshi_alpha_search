"""NBA 历史数据获取。

数据源：stats.nba.com（通过 nba_api 库），免费、无需 key。
拉取每支球队每场比赛的 box score（LeagueGameLog），本地缓存成 parquet，
再把"每场比赛两行（主队一行、客队一行）"合并成"每场一行"的宽表。

用法：
    from data.nba_fetcher import NBADataFetcher
    fetcher = NBADataFetcher()
    team_logs = fetcher.fetch_team_game_logs()   # 每队每场一行
    games = fetcher.to_games(team_logs)          # 每场比赛一行（HOME_* / AWAY_*）
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_CACHE_DIR,
    REQUEST_SLEEP,
    REQUEST_TIMEOUT,
    SEASONS,
    SEASON_TYPE,
)

# 每队每场 box score 里我们保留的统计列
STAT_COLS = [
    "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]


class NBADataFetcher:
    def __init__(self, cache_dir: Path = DATA_CACHE_DIR):
        self.cache_dir = Path(cache_dir)

    # ---------- 拉取 ----------

    def fetch_team_game_logs(self, seasons=SEASONS, force_refresh=False) -> pd.DataFrame:
        """拉取多个赛季的球队 game log，每队每场一行。带本地缓存。"""
        frames = []
        for season in seasons:
            cache_file = self.cache_dir / f"team_gamelog_{season.replace('-', '_')}.parquet"
            if cache_file.exists() and not force_refresh:
                frames.append(pd.read_parquet(cache_file))
                continue
            df = self._fetch_one_season(season)
            df.to_parquet(cache_file, index=False)
            frames.append(df)
            time.sleep(REQUEST_SLEEP)
        out = pd.concat(frames, ignore_index=True)
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
        return out.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    def _fetch_one_season(self, season: str) -> pd.DataFrame:
        from nba_api.stats.endpoints import leaguegamelog

        print(f"  fetching {season} from stats.nba.com ...")
        resp = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=SEASON_TYPE,
            player_or_team_abbreviation="T",
            timeout=REQUEST_TIMEOUT,
        )
        df = resp.get_data_frames()[0]
        df["SEASON"] = season
        keep = ["SEASON", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
                "MATCHUP", "WL"] + STAT_COLS
        return df[keep].copy()

    # ---------- 球员级 game log（伤病/缺阵特征用）----------

    PLAYER_STAT_COLS = ["MIN", "PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB",
                        "AST", "STL", "BLK", "TOV", "PF"]

    def fetch_player_game_logs(self, seasons=SEASONS, force_refresh=False) -> pd.DataFrame:
        """每球员每场一行（只含上场了的球员——缺阵就是不出现在这里）。"""
        from nba_api.stats.endpoints import leaguegamelog

        frames = []
        for season in seasons:
            cache_file = self.cache_dir / f"player_gamelog_{season.replace('-', '_')}.parquet"
            if cache_file.exists() and not force_refresh:
                frames.append(pd.read_parquet(cache_file))
                continue
            print(f"  fetching player logs {season} ...")
            resp = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=SEASON_TYPE,
                player_or_team_abbreviation="P", timeout=REQUEST_TIMEOUT)
            df = resp.get_data_frames()[0]
            df["SEASON"] = season
            keep = ["SEASON", "GAME_ID", "GAME_DATE", "TEAM_ID",
                    "PLAYER_ID", "PLAYER_NAME"] + self.PLAYER_STAT_COLS
            df = df[keep].copy()
            df.to_parquet(cache_file, index=False)
            frames.append(df)
            time.sleep(REQUEST_SLEEP)
        out = pd.concat(frames, ignore_index=True)
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
        return out.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # ---------- 变形 ----------

    @staticmethod
    def to_games(team_logs: pd.DataFrame) -> pd.DataFrame:
        """把每队每场一行合并成每场比赛一行。

        MATCHUP 里 "LAL vs. BOS" 表示 LAL 主场，"LAL @ BOS" 表示 LAL 客场。
        输出列：HOME_* / AWAY_* 前缀的统计 + HOME_WIN 标签。
        """
        logs = team_logs.copy()
        logs["IS_HOME"] = logs["MATCHUP"].str.contains(" vs. ", regex=False)

        # 中立场地比赛（墨西哥城/巴黎赛、NBA杯决赛）两队 MATCHUP 都是 "@"，
        # 无法区分主客场且本就没有主场优势，直接剔除
        home_count = logs.groupby("GAME_ID")["IS_HOME"].transform("sum")
        neutral = logs.loc[home_count != 1, "GAME_ID"].unique()
        if len(neutral) > 0:
            print(f"  剔除 {len(neutral)} 场中立场地比赛: {list(neutral)}")
            logs = logs[~logs["GAME_ID"].isin(neutral)]

        home = logs[logs["IS_HOME"]].copy()
        away = logs[~logs["IS_HOME"]].copy()

        keep = ["SEASON", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "WL"] + STAT_COLS
        home = home[keep].add_prefix("HOME_").rename(
            columns={"HOME_SEASON": "SEASON", "HOME_GAME_ID": "GAME_ID", "HOME_GAME_DATE": "GAME_DATE"})
        away = away[keep].add_prefix("AWAY_").rename(columns={"AWAY_GAME_ID": "GAME_ID"})
        away = away.drop(columns=["AWAY_SEASON", "AWAY_GAME_DATE"])

        games = home.merge(away, on="GAME_ID", how="inner", validate="1:1")
        games["HOME_WIN"] = (games["HOME_WL"] == "W").astype(int)
        games = games.drop(columns=["HOME_WL", "AWAY_WL"])
        return games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


if __name__ == "__main__":
    fetcher = NBADataFetcher()
    team_logs = fetcher.fetch_team_game_logs()
    games = fetcher.to_games(team_logs)
    print(f"team_logs: {team_logs.shape}, games: {games.shape}")
    print(games[["SEASON", "GAME_DATE", "HOME_TEAM_ABBREVIATION",
                 "AWAY_TEAM_ABBREVIATION", "HOME_PTS", "AWAY_PTS", "HOME_WIN"]].tail())
