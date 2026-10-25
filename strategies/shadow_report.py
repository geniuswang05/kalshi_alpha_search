"""影子做市（LIP）前向测试报告。

读 data_cache/shadow_mm/{run}/log.csv + fills.csv：
- 总账：奖励（上限估计）、maker 费、库存盯市盈亏、净收益
- 逆向选择：每笔成交按成交后 5 / 30 分钟的中间价计算盈亏（markout）
- 按系列拆分；按 30 分钟切片的净收益序列（波动、最大回撤）
用法: python strategies/shadow_report.py [run_dir ...]（默认全部 run）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_CACHE_DIR

ROOT = DATA_CACHE_DIR / "shadow_mm"


def load_run(run: Path):
    log = pd.read_csv(run / "log.csv", parse_dates=["ts"])
    fills = pd.read_csv(run / "fills.csv", parse_dates=["ts"]) if (run / "fills.csv").exists() else pd.DataFrame()
    return log, fills


def markouts(log: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return fills
    out = []
    mids = log[["ts", "ticker", "mid"]].dropna().sort_values("ts")
    for f in fills.itertuples(index=False):
        m = mids[mids["ticker"] == f.ticker]
        row = {"ticker": f.ticker, "side": f.side, "qty": f.qty, "px": f.px}
        for h in (5, 30):
            later = m[m["ts"] >= f.ts + pd.Timedelta(minutes=h)]
            if later.empty:
                row[f"mo{h}"] = np.nan
                continue
            mid = later["mid"].iat[0]
            val = mid if f.side == "yes" else 1 - mid         # 持有 YES 或 NO 的价值
            row[f"mo{h}"] = (val - f.px) * f.qty
        out.append(row)
    return pd.DataFrame(out)


def report(run: Path):
    log, fills = load_run(run)
    hours = (log["ts"].max() - log["ts"].min()).total_seconds() / 3600
    last = log.sort_values("ts").groupby("ticker").tail(1)
    reward, fees, mtm = last["reward"].sum(), last["fees"].sum(), last["mtm"].sum()
    cands = pd.read_csv(run / "candidates.csv")
    capital = cands["capital"].sum()
    print(f"\n=== {run.name}：{last['ticker'].nunique()} 个市场，已运行 {hours:.1f} 小时，名义资金 ${capital:,.0f} ===")
    print(f"   奖励（份额上限估计） ${reward:,.2f}  →  ${reward / max(hours, 1e-9) * 24:,.0f}/天")
    print(f"   库存盯市（含已扣 maker 费 ${fees:,.2f}） ${mtm:,.2f}")
    print(f"   净收益 ${reward + mtm:,.2f}  →  ${(reward + mtm) / max(hours, 1e-9) * 24:,.0f}/天，"
          f"日资金收益率 {(reward + mtm) / max(hours, 1e-9) * 24 / capital:.2%}")

    if not fills.empty:
        mo = markouts(log, fills)
        print(f"   成交 {len(fills)} 笔 / {fills['qty'].sum():,.0f} 张；"
              f"5分钟 markout ${mo['mo5'].sum():,.2f}，30分钟 markout ${mo['mo30'].sum():,.2f}"
              f"（负数 = 被逆向选择）")

    tot = log.groupby("ts")[["reward", "mtm"]].sum().sort_index()
    tot["net"] = tot["reward"] + tot["mtm"]
    sl = tot["net"].resample("30min").last().dropna().diff().dropna()
    if len(sl) > 2:
        eq = tot["net"].resample("30min").last().dropna()
        dd = (eq - eq.cummax()).min()
        print(f"   30分钟净收益：均值 ${sl.mean():,.2f}，标准差 ${sl.std():,.2f}，亏损时段 {(sl < 0).mean():.0%}，"
              f"最大回撤 ${dd:,.2f}")

    by = last.groupby("series").agg(markets=("ticker", "size"), reward=("reward", "sum"),
                                    mtm=("mtm", "sum"))
    by["net"] = by["reward"] + by["mtm"]
    print("   按系列（净收益排序）:")
    print(by.sort_values("net", ascending=False).round(2).to_string().replace("\n", "\n      "))
    share = log.groupby("ticker")["share"].mean()
    print(f"   平均份额（时间平均）: 中位 {share.median():.2f}，四分位 {share.quantile(.25):.2f}-{share.quantile(.75):.2f}")


def settled_value(run: Path) -> dict:
    """库存按结算结果计价（已结算市场），未结算的仍用最后的中间价。"""
    from data.kalshi_client import KalshiClient
    from data.kalshi_universe import get_with_retry
    log = pd.read_csv(run / "log.csv", parse_dates=["ts"])
    last = log.sort_values("ts").groupby("ticker").tail(1)
    client = KalshiClient()
    tot = {"reward": 0.0, "value": 0.0, "settled_markets": 0, "open_markets": 0}
    for r in last.itertuples(index=False):
        res = None
        try:
            mk = get_with_retry(client, f"/markets/{r.ticker}", {}).get("market", {})
            res = mk.get("result") if mk.get("result") in ("yes", "no") else None
        except Exception:
            pass
        if res is None:
            try:
                mk = get_with_retry(client, f"/historical/markets/{r.ticker}", {}).get("market", {})
                res = mk.get("result") if mk.get("result") in ("yes", "no") else None
            except Exception:
                pass
        tot["reward"] += r.reward
        if res:
            tot["value"] += r.cash + r.yes_pos * (res == "yes") + r.no_pos * (res == "no")
            tot["settled_markets"] += 1
        else:
            tot["value"] += r.mtm
            tot["open_markets"] += 1
    tot["net"] = tot["reward"] + tot["value"]
    return tot


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    runs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.iterdir() if p.is_dir())
    for r in runs:
        report(r)
