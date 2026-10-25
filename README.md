# Kalshi Alpha Search

**Where does money on a prediction market actually come from?**

A systematic search for trading edges on [Kalshi](https://kalshi.com), a CFTC-regulated prediction market. The project starts from "use machine learning to beat the market" and works through seven hypotheses on real Kalshi data. Six fail. The one that survives is not a forecast at all: it is the exchange's own liquidity subsidy. The project turns that subsidy into a market-making bot and studies it with a public-data event study.

Every result below uses real Kalshi prices, is tested out of sample, and is net of Kalshi's actual fee schedule.

| # | Hypothesis | Data | Result |
|---|---|---|---|
| 1 | Machine learning beats the NBA closing price | 7 seasons of games, 2,898 Kalshi contracts | ❌ Log loss: market 0.5685, model 0.6099. With injury features 0.5940, still worse. In a blended regression the model's coefficient is ≈ 0 |
| 2 | Parameter tuning rescues the model | 107 parameter cells with t-tests | ❌ 0 cells with t > 1.65 (≈ 5 expected by chance) |
| 3 | Arbitrage between the two teams' contracts in one game | 82,337 pre-game snapshots | ❌ Only 62 snapshots profitable after both legs' fees, mostly unsynchronized candles |
| 4 | Passive market making in major leagues | 27 sports series, 6.5M trades | ❌ Realized spread 0.02–0.14¢ per contract < maker fee 0.2–0.4¢ |
| 5 | End-game longshot bias ("selling miracle tickets") | Trade sample + fixed-clock snapshots of 9,452 markets | ❌ False discovery: p ≈ 10⁻⁵ in the trade sample, but both sides perfectly calibrated at scale (6.35% vs 6.20%; 5.66% vs 6.08%) |
| 6 | Trading Kalshi on Polymarket signals | 2,351 MLB games aligned by the minute | ❌ Kalshi leads in-game: Polymarket closes 92% of the gap within one minute (t = 213); the reverse signal does not pay |
| 7 | Harvesting Kalshi's Liquidity Incentive Program (LIP) | 6,657 active programs, live order books, four 8-hour shadow tests | ✅ The only positive-expectancy source, and it requires no information edge. Thin once adverse selection is counted |

## Why the liquidity subsidy

Kalshi's Liquidity Incentive Program photographs each eligible order book once per second. Each side is scored by walking down from the best bid and admitting whole price levels until resting size reaches the market's Target Size. Orders below the reference price are discounted by a Discount Factor per tick. Each participant receives their share of the qualifying score multiplied by the period's prize, $10–$1,000 per market per day. **Resting orders are paid whether or not they trade.** Programs running from August to December 2026 total about $13.4M.

This is the one source of return that does not require being smarter than the market. It requires capital, risk control and engineering, which a retail quant can actually supply. The rules are implemented from Kalshi's February 11, 2026 CFTC filing and its help-center article.

### Shadow forward test (no orders sent)

Every 30 seconds the bot read the real order book and real trades for 30 markets. It scored its hypothetical quotes under the program rules and treated any real taker trade at or through those quotes as a fill. Inventory was marked to the mid.

| Run | Selection | Reward (upper bound) | Inventory | Net | 30-min markout |
|---|---|---|---|---|---|
| v1 · improve 1 tick | rank by reward per $ | $1,048 | +$486 | +$1,534 | +$281 |
| v1 · join best | rank by reward per $ | $566 | −$26 | +$540 | +$34 |
| v2 · improve 1 tick | diversified + stops | $281 | −$207 | +$74 | −$200 |
| v2 · join best | diversified + stops | $243 | −$183 | +$60 | −$103 |

The v1 runs were concentrated in daily state gas-price contracts, so their inventory gain is an unhedged directional bet, not a property of the strategy. The diversified v2 runs are the better guide: adverse selection consumes most of the reward, leaving about 0.2% of capital over eight hours.

Limitations:
- **Reward share is an upper bound.** Shadow quotes are invisible to competitors, who would add size against real quotes.
- **Fills mean adverse selection.** In one v1 run a single gas market lost $153 in two hours. v2 adds category diversification, a per-market stop and avoidance of markets near settlement.
- **Abuse risk.** Kalshi can revoke participants whose activity is "inconsistent with the purpose of the program". The strategy excludes near-empty books (spread > 20¢) where resting at extreme prices could be judged abusive.

## Event study: what does the subsidy actually buy? (in progress)

Measuring the competitive reward share with real orders requires a U.S. Kalshi account, so this part answers the same question from public data alone. Two difference-in-differences designs:

- **Forward, order-book level** (`strategies/lip_event_study.py`): about 300 markets whose subsidy ends while trading continues, snapshotted every 10 minutes from before the end to 8 hours after. Controls are about 150 markets whose subsidy continues over the same hours. Metrics: spread, depth at the best price, total qualifying score, and the share a new entrant quoting Target Size at the best price would receive.
- **Historical, candle level** (`strategies/lip_history_study.py`): 12,109 subsidy-end and 2,411 subsidy-start events with never-subsidized controls from the same series and time window.

Results: _pending_.

## Repository layout

```
config/settings.py                    Global parameters
data/
  kalshi_client.py                    Kalshi API v2 client (RSA-PSS signing, V2 order entry, demo/prod)
  kalshi_universe.py                  Markets, trades and candles across live + historical endpoints
  nba_fetcher.py                      NBA team and player game logs (stats.nba.com)
models/                               Elo, rolling efficiency, player availability, XGBoost + isotonic calibration
backtest/
  run_backtest.py, run_v2.py          H1: model vs real Kalshi prices
  param_sweep.py                      H2: parameter sweep
  maker_taker.py                      H4: maker/taker returns and realized spread (market-clustered t)
  flb_analysis.py, endgame_*.py       H5: end-game longshot bias (trade sample → full pagination → large snapshot test)
scripts/
  sample_trades.py                    Stratified trade sampling across 27 sports series
  complete_windows.py                 Re-fetches windows truncated at 1,000 trades (fixes look-ahead sampling bias)
  build_report.py                     Builds the HTML reports in report/ from the analysis outputs
strategies/
  cross_venue.py, xvenue_analysis.py  H6: Kalshi × Polymarket lead-lag
  lip_scanner.py                      LIP scoring rules + live order-book scan
  lip_analysis.py                     Reward vs fill risk
  shadow_mm.py, shadow_report.py      LIP shadow market maker and forward-test report
  lip_live.py, lip_live_report.py     LIP bot with real orders (dry-run / demo / prod) and competition-response report
  lip_event_*.py, lip_history_study.py  Event study of subsidy start/end
  endgame_bot.py                      Execution bot for H5 (strategy rejected; kept as an engineering example)
report/                               Generated English and Chinese reports
```

## Reproducing

```bash
pip install -r requirements.txt
python scripts/sample_trades.py 40            # trade sampling (~1h)
python scripts/complete_windows.py            # complete truncated windows
python backtest/maker_taker.py                # who makes money
python backtest/endgame_snapshots_large.py    # large-sample longshot test
python strategies/cross_venue.py mlb 2026-04-01 && python strategies/xvenue_analysis.py mlb
python strategies/lip_scanner.py && python strategies/lip_analysis.py
python strategies/shadow_mm.py 8 30 -1 v2     # 8-hour LIP shadow test (no orders, no API key)
python strategies/shadow_report.py
python strategies/lip_event_study.py 300      # event study collection (~1 day)
python strategies/lip_event_analysis.py
```

Market data, order books, trades and candles need no API key. Raw data is cached under `data_cache/`, which is not committed except for `series_fees.json`.

## Real-order test (requires a U.S. Kalshi account)

`strategies/lip_live.py` measures what a shadow test cannot: whether competitors add size or step ahead once they see real quotes, and what share remains. All orders are post-only.

1. **Dry run** (no orders, no key): `python strategies/lip_live.py --hours 1`
2. **Demo** (fake money; validates order handling; the demo has no incentive programs): create a demo API key at demo.kalshi.co, set `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`, then `python strategies/lip_live.py --env demo --live --hours 1`
3. **Production, small size** (real money): `python strategies/lip_live.py --env prod --live --i-understand-real-money --hours 4`. Defaults: 3 markets, 100 contracts per side, all orders cancelled and the bot exits at a $30 total loss. `touch STOP` in the project root cancels everything and exits.
4. **Check rewards** after the program period ends on Kalshi's Rewards page: `python strategies/lip_live_report.py data_cache/lip_live/<run> --paid <amount>`

## The most useful lesson: a false discovery at p ≈ 10⁻⁵

Selling 2–10¢ longshots in the final minutes of games looked exceptionally strong in the trade sample. Longshots reversed in 11 of 572 markets where their prices implied 31.5, p = 1.5 × 10⁻⁵, out-of-sample t = 5.4. The investigation:

1. Each window kept only its latest 1,000 trades. A comeback's burst of volume pushes the earlier longshot trades out of the sample, which is a look-ahead sampling bias. All truncated windows were re-fetched in full. The signal survived.
2. Fixed-clock snapshots, a sampling scheme independent of volume, across 9,452 markets showed longshots on both sides **perfectly calibrated**.
3. Snapshots of the same markets reproduced the low reversal rate, so the anomaly came from those markets' outcomes, not from a tradeable mispricing.

Lesson: significance tests on trade data must be checked for sampling that depends on the future, and the final verdict belongs to a large test whose sampling is independent of trading.

## Kalshi API quick reference

- Base URL: `https://external-api.kalshi.com/trade-api/v2` (demo: `https://external-api.demo.kalshi.co/trade-api/v2`)
- Historical boundary: `GET /historical/cutoff`; older data lives under `/historical/markets`, `/historical/trades` and `/historical/markets/{ticker}/candlesticks`
- Order entry V2: `POST /portfolio/events/orders` (side = bid/ask on the YES book; price and count as fixed-point strings)
- Fees: taker = 0.07 × multiplier × P(1−P) per contract; maker = 25% of taker on `quadratic_with_maker_fees` series, free on `quadratic` series
- Incentives: `GET /incentive_programs?status=active&type=liquidity`

Nothing in this repository constitutes investment advice.
