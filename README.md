# windtunnel

A research framework for systematic trading strategies, built to give **honest**
out-of-sample results. It is a learning project. Read `CLAUDE.md` for the rules it
follows.

> Status: Stage 3 (backtest engine) is complete. Strategies, evaluation and paper/live
> trading come in later stages.

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
