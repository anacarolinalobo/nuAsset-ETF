import numpy as np
import pandas as pd

from src.config import MomentumConfig
from src.universe import eligible_universe


def _config(**overrides):
    base = dict(
        min_history_days=10,
        liquidity_window_days=10,
        min_active_ratio=0.5,
        adtv_min_percentile=0.30,
        market_cap_min_percentile=0.20,
    )
    base.update(overrides)
    return MomentumConfig(**base)


def _returns_fixture():
    dates = pd.bdate_range("2020-01-01", periods=30)
    return pd.DataFrame(
        {t: np.full(30, 0.001) for t in ["A", "B", "C", "D", "E"]}, index=dates
    )


def test_low_adtv_ticker_excluded_before_momentum_ranking():
    returns = _returns_fixture()
    date = returns.index[-1]

    adtv = pd.DataFrame(
        {"A": [1_000_000.0] * 30, "B": [900_000.0] * 30, "C": [800_000.0] * 30,
         "D": [700_000.0] * 30, "E": [1_000.0] * 30},  # E é claramente ilíquida
        index=returns.index,
    )

    cfg = _config(adtv_min_percentile=0.30, market_cap_min_percentile=0.0)
    eligible = eligible_universe(returns, date, cfg, adtv_wide=adtv)
    assert "E" not in eligible


def test_small_market_cap_ticker_excluded():
    returns = _returns_fixture()
    date = returns.index[-1]

    mcap = pd.DataFrame(
        {"A": [5e9] * 30, "B": [4e9] * 30, "C": [3e9] * 30,
         "D": [2e9] * 30, "E": [1e6] * 30},  # E é micro-cap
        index=returns.index,
    )

    cfg = _config(adtv_min_percentile=0.0, market_cap_min_percentile=0.20)
    eligible = eligible_universe(returns, date, cfg, market_cap_wide=mcap)
    assert "E" not in eligible


def test_missing_external_data_does_not_penalize_ticker():
    returns = _returns_fixture()
    date = returns.index[-1]

    # ADTV só cobre A-D; E fica sem dado externo e não deve ser excluída
    # só por isso (cai no proxy de histórico, que ela passa).
    adtv = pd.DataFrame(
        {"A": [1_000_000.0] * 30, "B": [900_000.0] * 30, "C": [800_000.0] * 30, "D": [700_000.0] * 30},
        index=returns.index,
    )

    cfg = _config(adtv_min_percentile=0.30)
    eligible = eligible_universe(returns, date, cfg, adtv_wide=adtv)
    assert "E" in eligible


def test_backward_compatible_without_external_data():
    returns = _returns_fixture()
    date = returns.index[-1]
    cfg = _config()
    eligible = eligible_universe(returns, date, cfg)
    assert set(eligible) == {"A", "B", "C", "D", "E"}


def test_absolute_floor_filters_below_threshold():
    returns = _returns_fixture()
    date = returns.index[-1]

    adtv = pd.DataFrame(
        {"A": [1_000_000.0] * 30, "B": [900_000.0] * 30, "C": [800_000.0] * 30,
         "D": [700_000.0] * 30, "E": [50_000.0] * 30},
        index=returns.index,
    )

    cfg = _config(adtv_min_percentile=0.0, min_adtv_reais=100_000.0)
    eligible = eligible_universe(returns, date, cfg, adtv_wide=adtv)
    assert "E" not in eligible
    assert "A" in eligible
