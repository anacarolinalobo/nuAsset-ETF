import dataclasses

import numpy as np
import pandas as pd

from src.universe import eligible_universe


def test_requires_minimum_history(synthetic_returns, small_config):
    early_date = synthetic_returns.index[10]  # menos que min_history_days
    universe = eligible_universe(synthetic_returns, early_date, small_config)
    assert len(universe) == 0


def test_excludes_delisted_ticker_after_last_trade(synthetic_returns, small_config):
    late_date = synthetic_returns.index[synthetic_returns.index >= "2020-09-01"][0]
    universe = eligible_universe(synthetic_returns, late_date, small_config)
    assert "T9" not in universe


def test_includes_active_liquid_tickers(synthetic_returns, small_config):
    date = synthetic_returns.index[synthetic_returns.index >= "2020-04-15"][0]
    universe = eligible_universe(synthetic_returns, date, small_config)
    assert "T0" in universe
    assert len(universe) > 0


def test_volume_filter_is_noop_without_min_adtv(synthetic_returns, small_config, synthetic_volume):
    date = synthetic_returns.index[synthetic_returns.index >= "2020-04-15"][0]
    without_volume = eligible_universe(synthetic_returns, date, small_config)
    with_volume = eligible_universe(synthetic_returns, date, small_config, volume_wide=synthetic_volume)
    assert list(with_volume) == list(without_volume)


def test_volume_filter_excludes_illiquid_ticker_by_adtv(synthetic_returns, small_config, synthetic_volume):
    config = dataclasses.replace(small_config, min_adtv_brl=10_000.0, volume_window_days=21)
    date = synthetic_returns.index[synthetic_returns.index >= "2020-04-15"][0]

    universe = eligible_universe(synthetic_returns, date, config, volume_wide=synthetic_volume)

    assert "T1" not in universe
    assert "T0" in universe
