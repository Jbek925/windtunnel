# windtunnel: project rules for Claude

A Python research framework for systematic trading strategies. It is a **learning project** for a first-year aeronautical engineering student, who knows Python and maths well but is new to quant trading. **The priority is honest results, not impressive-looking ones.**

## Hard rules (never break these)

1. **Live trading is off by default.** Real orders need all three of:
   - `mode = "live"` in the config file,
   - the env var `WINDTUNNEL_LIVE=I_UNDERSTAND_REAL_MONEY`,
   - the `--live` CLI flag.

   If any one is missing, the system runs as the paper trader. Never weaken this gate, add a bypass, or default any of them to on.
2. **API keys only come from environment variables.** They are never written to files, config, SQLite, logs, reports, test fixtures or git. `.env` is gitignored. Never ask the user to paste keys into chat. Tell the user to create **trade-only keys with withdrawals disabled**, IP-whitelisted where the exchange supports it.
3. **Hard capital cap.** `max_live_notional` (default 50 in quote currency) is enforced inside the live broker. No order may push exposure above it, whatever the strategy or sizer says. Only the user changes the cap, by editing their config. Never raise it in code or in examples.
4. **Spot only, long/flat only.** No leverage, margin, perps, shorting, withdrawals or transfers. Live code may only call these private ccxt methods: `fetch_balance`, `create_order` (limit only), `fetch_order`, `cancel_order`, `fetch_open_orders`. Public calls are `fetch_ticker`, `fetch_ohlcv` and `load_markets`. A test enforces this allow-list, and only `src/windtunnel/live/` may touch credentials.
5. **Never overstate results.** Always show every result net of costs and next to the buy-and-hold benchmark. If a strategy does not beat buy-and-hold after costs out-of-sample, say so plainly. Report confidence intervals and multiple-testing caveats. A backtest is not evidence of future profit.

## Research conventions

- **Time:** everything is UTC. Each bar is indexed by its **open** time (the ccxt convention). A signal at bar *t* may use data up to and including the **close** of bar *t*.
- **Fills** happen at the **open of bar t+1**. The engine applies the only shift (in `backtest/engine.py`), and strategies never shift their own signal. The overnight/weekend gap (close t → open t+1) accrues to the *old* position.
- The same `CostModel` and timing contract are used in backtest, paper and live.
- **Parameters are chosen on train data only** (walk-forward). Every parameter combination tried counts toward the trial count used in the deflated Sharpe ratio.
- Keep strategy parameter counts small (≤ 2 tunable parameters). Every extra parameter is an overfitting risk.
- Data validation flags problems (gaps, duplicates, outliers) but never silently "fixes" them.
- Survivorship: BTC/ETH and SPY/QQQ/GLD/TLT were chosen with hindsight, so results do not generalise to "crypto" or "ETFs".

## Engineering conventions

- src layout (`src/windtunnel/`), Python 3.11, type hints everywhere, docstrings on public functions and classes.
- `ruff check`, `ruff format --check` and `mypy --strict` (on `src`) must all be clean.
- **Tests never use the network.** They use `windtunnel.data.synthetic` or fake exchange objects. Tests that genuinely need the network are marked `@pytest.mark.network` and skipped by default.
- Dependencies are pinned exactly in `pyproject.toml` and managed with uv (`uv.lock` is committed).
- Minimal dependencies: no backtesting frameworks. The mechanics should stay visible.

## Commands

```bash
uv sync                                   # install
uv run pytest                             # tests (no network)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run windtunnel --help                  # CLI
```

## Process

Work stage by stage (data → engine → strategies → evaluation → paper/live). At the end of each stage:
1. Run the tests and linters.
2. Commit with a clear message and push to the working branch.
3. Summarise what was built and flag weaknesses and assumptions.
4. **Wait for the user's approval before continuing.**

Note: the cloud dev container cannot reach exchanges or Yahoo, so the user runs real-data commands locally.
