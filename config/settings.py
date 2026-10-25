"""全局配置：数据、特征、Elo 参数集中在这里改。"""

from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(exist_ok=True)

# ---------- 数据获取 ----------
# nba_api 的赛季格式是 "2023-24" 这种
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
           "2025-26"]  # 2025-26 是回测赛季（有 Kalshi 真实价格）
SEASON_TYPE = "Regular Season"  # 可选 "Playoffs"，训练时建议先只用常规赛
REQUEST_TIMEOUT = 60            # stats.nba.com 有时很慢
REQUEST_SLEEP = 1.0             # 每次请求间隔秒数，防止被限流

# ---------- 特征工程 ----------
ROLLING_WINDOWS = [5, 10, 20]   # 滚动统计窗口（场数）
MIN_GAMES_FOR_FEATURES = 10     # 赛季初样本不足的比赛直接丢弃，不进训练集

# ---------- 建模 ----------
TEST_SEASON = "2025-26"         # 回测赛季（有 Kalshi 真实价格）
CAL_SEASON = "2024-25"          # 校准/早停用的验证赛季（训练用它之前的所有赛季）

# ---------- 策略 ----------
SNAPSHOT_HOURS_BEFORE_CLOSE = 4  # 赛前价格快照 = close_time 前 4 小时的最后一根K线
MIN_EDGE = 0.03                  # 最小edge阈值（扣完手续费后）
MAX_SPREAD = 0.05                # bid-ask 价差超过 5¢ 的不交易（流动性差）
KELLY_FRACTION = 0.25            # 分数Kelly
MAX_BET_FRACTION = 0.05          # 单笔最多押总资金的 5%
INITIAL_BANKROLL = 10000.0       # 回测初始资金（美元）
FEE_RATE = 0.07                  # Kalshi taker费系数: fee = 0.07 * P * (1-P) 每张

# ---------- Elo 参数（FiveThirtyEight NBA Elo 风格）----------
ELO_INITIAL = 1500.0
ELO_K = 20.0                    # 更新步长
ELO_HOME_ADV = 100.0            # 主场优势（Elo 分）
ELO_SEASON_CARRYOVER = 0.75     # 跨赛季向均值回归：new = carry*old + (1-carry)*1505
ELO_MEAN = 1505.0
