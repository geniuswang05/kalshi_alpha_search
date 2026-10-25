"""端到端构建训练数据集：拉取 NBA 历史数据 → 特征工程 → 保存训练表。

用法：
    python scripts/build_dataset.py
产出：
    data_cache/training_table.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_CACHE_DIR
from data.nba_fetcher import NBADataFetcher
from models.feature_engineering import build_training_table, feature_columns
from models.player_availability import build_availability_features, merge_into_table


def main():
    print("[1/3] 拉取 NBA 历史 game log ...")
    fetcher = NBADataFetcher()
    team_logs = fetcher.fetch_team_game_logs()
    games = fetcher.to_games(team_logs)
    print(f"      {games['SEASON'].nunique()} 个赛季，共 {len(games)} 场比赛")

    print("[2/3] 特征工程（Elo + 滚动效率 + 日程 + 缺阵价值）...")
    table = build_training_table(games)
    player_logs = fetcher.fetch_player_game_logs()
    avail = build_availability_features(player_logs)
    table = merge_into_table(table, avail)
    feats = feature_columns(table)
    print(f"      训练表 {table.shape[0]} 行 × {len(feats)} 个特征")

    print("[3/3] 保存 ...")
    out = DATA_CACHE_DIR / "training_table.parquet"
    table.to_parquet(out, index=False)
    print(f"      -> {out}")

    # 快速健全性检查
    print("\n===== 健全性检查 =====")
    print(f"主队胜率（基准）: {table['HOME_WIN'].mean():.3f}  (联盟长期约 0.55-0.58)")
    elo_acc = ((table['ELO_HOME_WIN_PROB'] > 0.5) == (table['HOME_WIN'] == 1)).mean()
    print(f"纯 Elo 预测准确率: {elo_acc:.3f}  (538 Elo 基准约 0.65-0.67)")
    print("\n特征列样例:", feats[:8], "...")
    print("\n最近 5 场:")
    cols = ["GAME_DATE", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
            "ELO_HOME_WIN_PROB", "HOME_WIN"]
    print(table[cols].tail().to_string(index=False))


if __name__ == "__main__":
    main()
