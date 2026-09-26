import pandas as pd
from dataclasses import replace

from src.config import MomentumConfig
from src.rebalance import generate_rebalance_dates


def _config(freq: str) -> MomentumConfig:
    return MomentumConfig(rebalance_freq=freq, backtest_start="2020-01-01", backtest_end="2021-12-31")


def _trading_days() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", "2021-12-31")


def test_monthly_gives_twelve_dates_per_year():
    dates = generate_rebalance_dates(_trading_days(), _config("M"))
    per_year = pd.Series(dates).dt.year.value_counts()
    assert per_year[2020] == 12
    assert per_year[2021] == 12


def test_quarterly_gives_four_dates_per_year():
    dates = generate_rebalance_dates(_trading_days(), _config("Q"))
    per_year = pd.Series(dates).dt.year.value_counts()
    assert per_year[2020] == 4
    assert per_year[2021] == 4


def test_semiannual_gives_two_dates_per_year_in_june_and_december():
    dates = generate_rebalance_dates(_trading_days(), _config("S"))
    per_year = pd.Series(dates).dt.year.value_counts()
    assert per_year[2020] == 2
    assert per_year[2021] == 2
    months = sorted(pd.Series(dates).dt.month.unique())
    assert months == [6, 12]


def test_dates_are_last_trading_day_of_period():
    dates = generate_rebalance_dates(_trading_days(), _config("Q"))
    # 2020-03-31 é terça-feira útil -> deve ser exatamente essa data
    assert pd.Timestamp("2020-03-31") in dates


def test_dates_respect_backtest_window():
    cfg = replace(_config("Q"), backtest_start="2021-01-01", backtest_end="2021-12-31")
    dates = generate_rebalance_dates(_trading_days(), cfg)
    assert dates.min() >= pd.Timestamp("2021-01-01")
    assert dates.max() <= pd.Timestamp("2021-12-31")
