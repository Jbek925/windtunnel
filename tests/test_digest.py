import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.test_paper import _cfg, _drive, _runner
from windtunnel.data.synthetic import gbm
from windtunnel.paper.digest import build_digest, send_telegram
from windtunnel.paper.feed import ReplayFeed
from windtunnel.paper.store import Store


def _run(tmp_path: Path, n: int = 150) -> tuple[Store, Any, pd.Timestamp]:
    bars = gbm(n, seed=21)
    cfg = _cfg(tmp_path)
    feed = ReplayFeed(bars, "1d", start=60)
    runner = _runner(cfg, feed)
    _drive(runner, feed, until=n)
    now = bars.index[-1] + pd.Timedelta(hours=1)  # just after the last bar closed
    return runner.store, cfg, now


def test_digest_reads_plainly_and_compares_with_holding(tmp_path: Path) -> None:
    store, cfg, now = _run(tmp_path)
    text = build_digest(store, cfg, now=now)
    assert "pretend money" in text
    assert "If you had just bought BTC and held it" in text
    assert "than simply holding this week" in text
    assert "Health: ✅" in text
    assert "one coin flip" in text
    for jargon in ("Sharpe", "drawdown", "bps", "alpha", "CAGR"):
        assert jargon not in text


def test_digest_numbers_match_the_database(tmp_path: Path) -> None:
    store, cfg, now = _run(tmp_path)
    eq = store.table("equity")
    eq["bar_ts"] = pd.to_datetime(eq["bar_ts"], utc=True)
    start = now - pd.Timedelta(days=7)
    first = eq[eq["bar_ts"] <= start].iloc[-1]
    last = eq.iloc[-1]
    text = build_digest(store, cfg, now=now)
    assert f"{last['equity'] / first['equity'] - 1:+.1%}" in text
    assert f"{last['close'] / first['close'] - 1:+.1%}" in text


def test_digest_flags_problems(tmp_path: Path) -> None:
    store, cfg, now = _run(tmp_path)
    store.set("kill_switch", True)
    store.event("error", "exchange timed out", "ERROR")
    text = build_digest(store, cfg, now=now + pd.Timedelta(days=3))  # prices stopped 3 days ago
    assert "Health: ⚠️ needs a look" in text
    assert "No new prices since" in text
    assert "kill switch" in text and "exchange timed out" in text


def test_digest_with_no_data(tmp_path: Path) -> None:
    store = Store(tmp_path / "empty.sqlite")
    text = build_digest(store, _cfg(tmp_path), now=pd.Timestamp("2026-09-25", tz="UTC"))
    assert "No results recorded yet" in text


def test_send_telegram_posts_message(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    class Resp:
        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: object) -> None: ...

        def read(self) -> bytes:
            return json.dumps({"ok": True}).encode()

    def fake_urlopen(req: Any, timeout: float) -> Resp:
        sent["url"], sent["data"] = req.full_url, req.data.decode()
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    send_telegram("hello", token="123:SECRETTOKEN", chat_id="42")
    assert sent["url"] == "https://api.telegram.org/bot123:SECRETTOKEN/sendMessage"
    assert "chat_id=42" in sent["data"] and "text=hello" in sent["data"]


def test_send_telegram_never_leaks_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req: Any, timeout: float) -> None:
        raise OSError(f"cannot reach {req.full_url}")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError) as err:
        send_telegram("hi", token="123:SECRETTOKEN", chat_id="42")
    assert "SECRETTOKEN" not in str(err.value)


def test_send_telegram_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WINDTUNNEL_TELEGRAM_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="WINDTUNNEL_TELEGRAM_TOKEN"):
        send_telegram("hi")


def test_cli_digest_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from windtunnel.cli import main

    _store, cfg, _ = _run(tmp_path)
    conf = tmp_path / "c.toml"
    conf.write_text(f'db_path = "{cfg.db_path}"\n')
    assert main(["digest", "--config", str(conf), "--dry-run"]) == 0
    assert "windtunnel weekly report" in capsys.readouterr().out
