# windtunnel

A research framework for systematic trading strategies, built to give **honest**
out-of-sample results. It is a learning project. Read `CLAUDE.md` for the rules it
follows.

## How to use it: the whole workflow

1. **Set up** on your own computer (once).
2. **Download data** for the markets you care about.
3. **Backtest** a strategy and read the report honestly. Most will fail, and that's normal.
4. If one survives every check, **paper trade** it on live prices for a few months.
5. **Compare** paper results with what the backtest predicted for the same period.
6. Only then, and only if you choose to: **shadow mode**, then **live** with the minimum cap.

Each step has a section below. Skipping steps is how people lose money.

## 1. Setup

You need Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, in a terminal:

```bash
git clone https://github.com/Jbek925/windtunnel.git
cd windtunnel
git checkout claude/trading-framework-plan-vw2181   # until this is merged into main
uv sync                 # installs Python deps at the exact pinned versions (uv.lock)
uv run pytest           # ~180 tests, no network needed; all should pass
uv run windtunnel --help
```

## 2. Getting data

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

## 3. Running a backtest

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

## 4. Paper trading

The paper trader runs a strategy on **live public prices** and simulates fills with the
**same cost model and the same fill function** as the backtest. It needs no API keys.

1. Pick a strategy and **fixed parameters**. For example, use the parameters most walk-forward
   folds chose in your report. Don't pick the single best cell of the heatmap.
2. Copy and edit the config:
   ```bash
   cp configs/paper_example.toml configs/my_paper.toml
   # edit: exchange, symbol, strategy, params, fee_bps (use YOUR exchange's real fee)
   ```
3. Try one step by hand:
   ```bash
   uv run windtunnel paper run --once --config configs/my_paper.toml
   uv run windtunnel paper status --config configs/my_paper.toml
   ```
4. Run it continuously (Ctrl+C stops it cleanly):
   ```bash
   uv run windtunnel paper run --config configs/my_paper.toml
   ```

What it does each time a bar closes: mark to market, compute the signal, apply the risk
limits, trade at the next bar's open, then save everything to SQLite (`data/paper/trader.sqlite`)
in one transaction. It's safe to stop and restart at any time. It carries on where it left
off and never double-trades.

**Risk controls** (in the `[risk]` section of the config):

| control | default | what happens |
|---|---|---|
| `max_weight` | 1.0 | position never above 100% of equity (no leverage, ever) |
| `max_daily_loss` | 5% | go flat for the rest of the UTC day |
| `max_drawdown` | 25% | **kill switch**: go flat and stay flat until you run `paper run --reset-kill-switch` |
| `stale_after_bars` | 1.5 | no new bar for 1.5 bar-lengths: stop trading and alert |
| `max_bar_move` | 25% | a single-bar move that big might be bad data, so don't trade on it |

### Running it unattended

On a cheap Linux server (or a Raspberry Pi), either:

- **systemd** (recommended; restarts on crashes): see the instructions at the top of
  `deploy/windtunnel-paper.service`.
- **cron**: run `paper run --once` just after each bar closes. See `deploy/crontab.example`.

Logs go to `logs/trader.log` (rotated). For phone alerts, put
`WINDTUNNEL_ALERT_WEBHOOK=https://ntfy.sh/<a-long-random-topic>` in `~/.config/windtunnel.env`
and install the ntfy app. Warnings, errors and kill-switch events will be pushed to you.

## 5. Comparing paper with the backtest

```bash
uv run windtunnel paper compare --config configs/my_paper.toml
```

This re-runs the backtest over the exact bars the paper trader saw and prints both equity
curves, rebased to 1.0, plus the gap. In paper mode the gap should be about 0. A test proves
they match to 10 decimal places. If it isn't, check `paper status` for risk events or missed
bars. In live mode, the gap is the real-world cost of slippage, partial fills and missed orders.
That gap is the most honest number this project produces.

**How long to paper trade?** At least 3 months, and preferably 6+ for a daily strategy.
Treat a few months as a check that everything *works*. It says almost nothing about
whether the strategy *makes money*, because even a real edge is invisible over that short a sample.

## 6. Going live (real money): read all of this first

**Default answer: don't, yet.** Live trading exists because you asked for it, but nothing in
the backtests so far is evidence of an edge. If you do go ahead, treat the money as tuition
you can afford to lose entirely, and follow this checklist in order:

1. **Choose an exchange that legally serves your country** and check its **spot** fees at your
   tier. Put the real taker fee in `fee_bps` and re-run the backtest with it.
2. **Create an API key** on the exchange website with:
   - permissions: **query funds + create/cancel orders only**
   - **withdrawals DISABLED** (critical: then a stolen key can't empty your account)
   - **IP whitelist** set to your server's IP, if the exchange supports it
3. **Put the key in environment variables, never in a file in this repo**:
   ```bash
   # in ~/.config/windtunnel.env (chmod 600), used by the systemd unit:
   WINDTUNNEL_API_KEY=...
   WINDTUNNEL_API_SECRET=...
   # WINDTUNNEL_API_PASSWORD=...   # only some exchanges (e.g. OKX)
   ```
   Never paste keys into a chat, an issue, a commit or a config file. The config loader
   rejects anything that looks like a key.
4. **Shadow mode first (at least a week).** Make a separate config with `mode = "shadow"` and
   a **different `db_path`**. It uses your real balances and logs every order it *would*
   send, but sends nothing. Check the orders in `paper status` look sane.
5. **Live with the minimum cap.** Set `mode = "live"` and a new `db_path`. Keep
   `max_live_notional = 50` (the trader can never hold more than 50 USDT of BTC, whatever
   is in the account). Then all three opt-ins must be present:
   ```bash
   export WINDTUNNEL_LIVE=I_UNDERSTAND_REAL_MONEY
   uv run windtunnel paper run --live --config configs/my_live.toml
   ```
   If any of the three is missing, it prints what's missing and runs as the **paper** trader.
6. **Check on it**: `paper status` and `paper compare` weekly. At startup the live trader
   **halts** if your balance doesn't match its records or there are orders it didn't place.
   So don't trade that pair by hand in the same account.

What the live broker guarantees (all tested against a fake exchange):
- **limit orders only**, priced within ±0.2% of the reference, cancelled if unfilled after 10 min
- **spot only, long/flat only**, no leverage, no withdrawals, no transfers
- position value **never above `max_live_notional`**, with a final hard check before every order
- the same bar is **never ordered twice**, even across crashes (deterministic order ids)

What it can't protect you from: the exchange going bust or being hacked, your key leaking
from your own machine, extreme gaps (a limit order may simply not fill during a crash),
and, most likely of all, the strategy having no edge.

## Project layout

```
src/windtunnel/
  data/        schema, ccxt + yfinance sources, validation, parquet cache, synthetic data
  backtest/    cost model, sizing, engine (plan_fill), lookahead checker, walk-forward
  strategies/  ma_trend, ts_momentum, zscore_mr, buy_and_hold (+ benchmarks)
  evaluation/  metrics, robustness (bootstrap CI, deflated Sharpe), evaluate(), report
  paper/       store (SQLite), feed, risk, broker, runner, alerts
  live/        gate (triple opt-in, credentials), ccxt_broker (the only real-order code)
  config.py    trader config (TOML)
  cli.py       `windtunnel` command
configs/       example trader config
deploy/        systemd unit, crontab example
tests/         offline tests; nothing touches the network or a real account
```
