import pandas as pd
import pytest

from tests.helpers import PeekNextClose, SameBarReturn, SmaTrend
from windtunnel.backtest.lookahead import LookaheadError, check_causal, check_no_lookahead
from windtunnel.data.synthetic import gbm


@pytest.fixture
def bars() -> pd.DataFrame:
    return gbm(600, seed=9)


def test_honest_strategies_pass(bars: pd.DataFrame) -> None:
    check_no_lookahead(SmaTrend(lookback=20), bars)
    check_no_lookahead(SameBarReturn(), bars)


def test_deliberate_lookahead_is_caught(bars: pd.DataFrame) -> None:
    """The check must catch a strategy that peeks at the next close."""
    with pytest.raises(LookaheadError, match="future data"):
        check_no_lookahead(PeekNextClose(), bars)


@pytest.mark.parametrize(
    "leaky",
    [
        lambda b: b["close"].rolling(21, center=True).mean(),  # centred window
        lambda b: b["close"] / b["close"].max(),  # full-sample normalisation
        lambda b: b["close"].where(b.index.dayofweek == 0).bfill(),  # back-fill
        lambda b: (b["close"] - b["close"].mean()) / b["close"].std(),  # full-sample z-score
    ],
    ids=["centred", "global-max", "bfill", "global-zscore"],
)
def test_subtle_leaks_are_caught(bars: pd.DataFrame, leaky) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(LookaheadError):
        check_causal(leaky, bars)


def test_nan_warmup_is_not_a_false_positive(bars: pd.DataFrame) -> None:
    check_causal(lambda b: b["close"].rolling(200).mean(), bars, min_history=10)


def test_too_short_input_rejected() -> None:
    with pytest.raises(ValueError):
        check_causal(lambda b: b["close"], gbm(30, seed=0))
