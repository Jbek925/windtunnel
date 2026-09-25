"""Weekly digest: a plain-English summary of the paper trader, sent to Telegram.

Written for someone with no finance background. It answers four questions:

1. Did the pretend account gain or lose money this week?
2. Would simply buying and holding have done better? (That's the bar to beat.)
3. What did the strategy actually do (trades, fees, current position)?
4. Is everything working, or does something need attention?

Telegram setup (one-off, about 3 minutes):

* In Telegram, message ``@BotFather``, send ``/newbot``, and copy the **token** it gives you.
* Send any message to your new bot, then open
  ``https://api.telegram.org/bot<TOKEN>/getUpdates`` in a browser and copy ``"chat":{"id": ...}``.
* Put both in environment variables (never in the repo)::

      WINDTUNNEL_TELEGRAM_TOKEN=123456:ABC...
      WINDTUNNEL_TELEGRAM_CHAT_ID=987654321
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request

import pandas as pd

from windtunnel.config import TraderConfig
from windtunnel.paper.feed import is_stale
from windtunnel.paper.store import Store

log = logging.getLogger("windtunnel.digest")

TOKEN_ENV = "WINDTUNNEL_TELEGRAM_TOKEN"
CHAT_ENV = "WINDTUNNEL_TELEGRAM_CHAT_ID"

SAME_BAND = 0.001  # within ±0.1 percentage points counts as "about the same"


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def _money(x: float, quote: str) -> str:
    return f"{x:,.2f} {quote}"


def _compare(strategy: float, holding: float) -> str:
    if abs(strategy - holding) <= SAME_BAND:
        return "about the SAME as simply holding"
    return "BETTER than simply holding" if strategy > holding else "WORSE than simply holding"


def build_digest(
    store: Store, cfg: TraderConfig, now: pd.Timestamp | None = None, days: int = 7
) -> str:
    """Return the digest text for the ``days`` up to ``now``. It is plain text, no markup."""
    now = now or pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=days)
    quote = cfg.symbol.split("/")[1]
    coin = cfg.symbol.split("/")[0]
    header = f"📊 windtunnel weekly report ({start:%d %b} – {now:%d %b %Y})"

    eq = store.table("equity")
    if eq.empty:
        return (
            f"{header}\n\n⚠️ No results recorded yet. Is the paper trader running?\n"
            "Try: uv run windtunnel paper status --config <your config>"
        )
    eq["bar_ts"] = pd.to_datetime(eq["bar_ts"], utc=True)
    eq = eq.sort_values("bar_ts").reset_index(drop=True)
    before = eq[eq["bar_ts"] <= start]
    week = eq[eq["bar_ts"] > start]
    first = before.iloc[-1] if len(before) else eq.iloc[0]
    last = eq.iloc[-1]

    lines = [header, ""]
    lines.append(f"Practice account: pretend money, {cfg.strategy} strategy on {cfg.symbol}")
    if quote in ("USDT", "USDC"):
        lines.append(f"({quote} is a crypto 'dollar': 1 {quote} ≈ 1 US dollar.)")
    if week.empty:
        lines.append("• No new results this week (see Health below).")
    else:
        strat = last["equity"] / first["equity"] - 1
        hold = last["close"] / first["close"] - 1
        lines += [
            f"• This week: {_money(first['equity'], quote)} → {_money(last['equity'], quote)} "
            f"({_pct(strat)})",
            f"• If you had just bought {coin} and held it: {_pct(hold)}",
            f"➡️ The strategy did {_compare(strat, hold)} this week.",
        ]

    fills = store.table("fills")
    if not fills.empty:
        fills["bar_ts"] = pd.to_datetime(fills["bar_ts"], utc=True)
        fills = fills[fills["bar_ts"] > start]
    n_trades = len(fills)
    fees = float((fills["fee"] + fills["slippage"]).sum()) if n_trades else 0.0
    invested = last["units"] * last["close"] / last["equity"] if last["equity"] > 0 else 0.0
    lines += [
        "",
        f"What it did: {n_trades} trade(s), about {_money(fees, quote)} in fees and costs.",
        f"Right now it holds {invested:.0%} in {coin} and {1 - invested:.0%} in cash.",
    ]

    total_days = (last["bar_ts"] - eq.iloc[0]["bar_ts"]).days
    if total_days >= 1:
        strat_all = last["equity"] / eq.iloc[0]["equity"] - 1
        hold_all = last["close"] / eq.iloc[0]["close"] - 1
        lines.append(
            f"Since the start ({total_days} days): strategy {_pct(strat_all)} vs holding "
            f"{_pct(hold_all)}, so {_compare(strat_all, hold_all).lower()}."
        )

    lines += ["", *_health(store, cfg, now, start, last["bar_ts"])]
    lines += [
        "",
        "Reminder: one week is like one coin flip. It tells you almost nothing. Only months "
        "of results, compared with simply holding, mean anything. All money here is pretend.",
    ]
    return "\n".join(lines)


def _health(
    store: Store,
    cfg: TraderConfig,
    now: pd.Timestamp,
    start: pd.Timestamp,
    last_bar: pd.Timestamp,
) -> list[str]:
    problems: list[str] = []
    if is_stale(last_bar, now, cfg.timeframe, cfg.risk.stale_after_bars):
        problems.append(f"No new prices since {last_bar:%d %b %H:%M} UTC. Is it still running?")
    if store.get("kill_switch", False):
        problems.append(
            "The safety stop (kill switch) is ON: losses hit the limit, so it stopped trading. "
            "It stays off until you reset it by hand."
        )
    ev = store.table("events")
    if not ev.empty:
        ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="ISO8601")
        bad = ev[(ev["ts"] > start) & ev["level"].isin(["WARNING", "ERROR", "CRITICAL"])]
        if len(bad):
            problems.append(f"{len(bad)} warning(s) this week, most recent:")
            problems += [f"   – {m}" for m in bad["message"].tail(3)]
    if not problems:
        return [f"Health: ✅ all good (latest price {last_bar:%d %b %H:%M} UTC, no errors)."]
    return ["Health: ⚠️ needs a look", *[f"• {p}" for p in problems]]


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None) -> None:
    """Send ``text`` to a Telegram chat. The token comes from the environment by default.

    Errors are re-raised with the token removed, so it never ends up in logs or tracebacks.
    """
    token = token or os.environ.get(TOKEN_ENV)
    chat_id = chat_id or os.environ.get(CHAT_ENV)
    if not token or not chat_id:
        raise RuntimeError(f"set {TOKEN_ENV} and {CHAT_ENV} in the environment (see digest.py)")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=20) as resp:
            reply = json.loads(resp.read().decode())
    except Exception as exc:
        raise RuntimeError(f"Telegram send failed: {str(exc).replace(token, '***')}") from None
    if not reply.get("ok"):
        raise RuntimeError(f"Telegram refused the message: {reply.get('description')}")
