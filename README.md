# windtunnel

A research framework for systematic trading strategies, built to give **honest**
out-of-sample results. It is a learning project. Read `CLAUDE.md` for the rules it
follows.

> Status: Stage 5 (evaluation and reports) is complete. Paper and live trading come in Stage 6.

## Setup

```bash
# needs Python 3.11 and uv (https://docs.astral.sh/uv/)
uv sync                 # installs the exact pinned versions from uv.lock
uv run pytest           # the test suite needs no network
```

## Getting data

All data comes from public endpoints, and no API keys are needed.

```bash
# the whole default universe: BTC/USDT and ETH/USDT (1h, 1d) plus SPY/QQQ/GLD/TLT (1d)
uv run windtunnel fetch --all --start 2017-01-01

# one symbol; switch exchange if Binance is blocked where you live
uv run windtunnel fetch --source ccxt --exchange kraken --symbol BTC/USDT --timeframe 1d

# re-print the data-quality report for something already cached
uv run windtunnel validate --source ccxt --symbol BTC/USDT --timeframe 1d
```

Data is cached under `data/cache/` (gitignored) as parquet, with a `.meta.json`
sidecar. Running `fetch` again only downloads new bars.

### Reading the validation report

| Field | Meaning | What to do |
|---|---|---|
| gaps | missing bars (crypto: any; ETFs: > 4 calendar days) | Exchange outages happen. Check whether a gap overlaps your test period. |
| missing bus. days | weekdays with no ETF bar | About 9 per year are exchange holidays, which is normal. |
| outliers | returns > 10 robust σ from a trailing median | Look at them. Crypto crashes are real. A 3× spike that reverts next bar is probably a bad print. |
| OHLC inconsistent | high < max(open, close) or low > min(open, close) | This should be 0. If it isn't, the source is buggy. |
| revised bars | cached bars the source later changed | Normal for dividend-adjusted ETFs. Suspicious for crypto. |
| survivorship note | why these symbols flatter results | Always read this before believing a backtest. |

### Conventions

- Every timestamp is UTC. Each bar is indexed by its **open** time.
- A signal at bar *t* may use data up to the close of *t*. It trades at the open of *t+1*.
- ETF prices are split- and dividend-adjusted. Daily ETF bars are labelled 00:00 UTC of
  the trading date.
- Only completed bars are ever stored. The bar that is still forming is dropped.

## Running a backtest

```bash
# try the pipeline without any downloads (random-walk data, so expect "no edge")
uv run windtunnel backtest --synthetic 3000 --strategy ma_trend

# on real cached data (run `fetch` first)
uv run windtunnel backtest --source ccxt --symbol BTC/USDT --timeframe 1d --strategy ma_trend
uv run windtunnel backtest --source yfinance --symbol SPY --strategy ts_momentum --sizer vol
```

Strategies: `ma_trend`, `ts_momentum`, `zscore_mr`, `buy_and_hold`. Useful flags:

| flag | default | meaning |
|---|---|---|
| `--sizer fixed\|vol` | fixed | 100% per unit of signal, or volatility targeting (`--vol-target 0.2`) |
| `--fee-bps`, `--slippage-bps` | 10, 2 | costs per trade (10 bps = 0.1%). Use your exchange's real fee. |
| `--train-days`, `--test-days` | 730, 180 | walk-forward windows |
| `--prior-trials` | 0 | how many other configurations you already tried on this data. **Be honest.** |
| `--long-short` | off | allow shorts (research only; the spot/live setup is long/flat) |

Each run prints a verdict and a results table, and writes `reports/<market>_<strategy>/report.html`
(open it in a browser) plus `report.md`.

### Reading the report

1. **Verdict** (top box): plain-English conclusions. If it says the strategy did not beat
   buy-and-hold after costs out of sample, believe it.
2. **Equity and drawdown**: out-of-sample only, after costs, next to buy & hold (orange) and
   vol-targeted buy & hold (green). Log scale, so equal vertical distances are equal % moves.
3. **Results table**: every figure is net of costs. The last column reruns everything with
   **costs doubled**. If the edge vanishes there, it was never robust.
4. **Robustness checks**:
   - *Bootstrap CI*: if the interval includes 0, the Sharpe can't be told apart from luck.
   - *Deflated Sharpe*: corrects for having tried N configurations. You want > 0.95.
   - *Shuffled-data sanity*: the same pipeline on data with its time order destroyed. It
     should be about 0. A clearly positive value points to a bug.
5. **Parameter heatmap** (in-sample): a lone bright cell surrounded by red is overfitting.
   Broad plateaus are more believable.
6. **Walk-forward folds**: which parameters each train window picked, and how they did on the
   next unseen window. Parameters that jump around from fold to fold mean the "optimum" is noise.

Rules of thumb for this project: a strategy is only interesting if it beats **both**
benchmarks out of sample after costs, its Sharpe CI excludes 0, the deflated Sharpe
exceeds 0.95, **and** it still holds up with costs doubled. Expect most runs to fail
these tests. That's the honest result, not a bug.
