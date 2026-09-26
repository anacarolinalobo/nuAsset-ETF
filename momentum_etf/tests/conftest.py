import numpy as np
import pandas as pd
import pytest

from src.config import MomentumConfig


@pytest.fixture
def small_config() -> MomentumConfig:
    return MomentumConfig(
        lookback_days=60,
        skip_days=5,
        vol_window_days=60,
        min_history_days=65,
        liquidity_window_days=30,
        min_active_ratio=0.8,
        entry_percentile=0.70,
        hold_percentile=0.50,
        min_names=3,
        max_names=6,
        weight_cap=0.40,
        rebalance_freq="ME",
        execution_lag_days=1,
        backtest_start="2020-04-01",
        backtest_end="2020-12-31",
        transaction_cost_bps=20.0,
    )


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    tickers = [f"T{i}" for i in range(10)]

    data = {}
    for i, t in enumerate(tickers):
        drift = 0.001 if i < 5 else -0.0005  # metade com tendência de alta clara
        data[t] = rng.normal(drift, 0.01, len(dates))

    df = pd.DataFrame(data, index=dates)

    # T9 "deslista" no meio do ano — vira NaN a partir de julho
    df.loc[df.index >= "2020-07-01", "T9"] = np.nan
    return df


@pytest.fixture
def synthetic_cdi(synthetic_returns) -> pd.Series:
    return pd.Series(0.0003, index=synthetic_returns.index)


@pytest.fixture
def synthetic_volume(synthetic_returns) -> pd.DataFrame:
    """ADTV (R$) por ticker: T0 bem líquido, T1 abaixo de qualquer corte razoável."""
    volume = pd.DataFrame(1_000_000.0, index=synthetic_returns.index, columns=synthetic_returns.columns)
    volume["T1"] = 100.0
    return volume
