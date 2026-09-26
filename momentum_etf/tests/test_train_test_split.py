import numpy as np
import pandas as pd
import pytest

from src.config import MomentumConfig
from src.train_test_split import (
    TrainTestResult,
    grid_search,
    run_train_test_split,
    split_train_test,
)


@pytest.fixture
def base_config() -> MomentumConfig:
    return MomentumConfig(
        lookback_days=60, skip_days=5, vol_window_days=60,
        min_history_days=65, liquidity_window_days=30, min_active_ratio=0.8,
        entry_percentile=0.70, hold_percentile=0.60, min_names=3, max_names=6,
        weight_cap=0.40, rebalance_freq="Q", execution_lag_days=1,
        backtest_start="2020-06-01", backtest_end="2023-12-31",
        transaction_cost_bps=20.0,
    )


@pytest.fixture
def long_returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", "2023-12-31")
    tickers = [f"T{i}" for i in range(10)]

    data = {}
    for i, t in enumerate(tickers):
        drift = 0.001 if i < 5 else -0.0005
        data[t] = rng.normal(drift, 0.01, len(dates))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def long_cdi(long_returns) -> pd.Series:
    return pd.Series(0.0003, index=long_returns.index)


def test_split_train_test_gives_roughly_equal_halves(long_returns, base_config):
    train_start, train_end, test_start, test_end = split_train_test(long_returns, base_config)

    trading_days = long_returns.index[
        (long_returns.index >= pd.Timestamp(base_config.backtest_start))
        & (long_returns.index <= pd.Timestamp(base_config.backtest_end))
    ]
    n_train = ((trading_days >= train_start) & (trading_days <= train_end)).sum()
    n_test = ((trading_days >= test_start) & (trading_days <= test_end)).sum()

    assert abs(n_train - n_test) <= 1
    assert train_end < test_start


def test_split_raises_on_too_short_period(long_returns):
    cfg = MomentumConfig(backtest_start="2023-12-28", backtest_end="2023-12-29")
    with pytest.raises(ValueError):
        split_train_test(long_returns, cfg)


def test_grid_search_picks_a_valid_config_from_grid(long_returns, long_cdi, base_config):
    train_start, train_end, _, _ = split_train_test(long_returns, base_config)
    grid = {"lookback_days": [40, 60], "entry_percentile": [0.60, 0.70]}

    best_config, best_score, grid_results = grid_search(
        long_returns, long_cdi, base_config, grid, train_start, train_end
    )

    assert best_config.lookback_days in grid["lookback_days"]
    assert best_config.entry_percentile in grid["entry_percentile"]
    assert len(grid_results) == 4
    assert grid_results["objetivo"].iloc[0] == best_score
    # ordenado do melhor para o pior
    assert (grid_results["objetivo"].diff().dropna() <= 1e-9).all()


def test_grid_search_derives_hold_percentile_from_entry(long_returns, long_cdi, base_config):
    train_start, train_end, _, _ = split_train_test(long_returns, base_config)
    grid = {"entry_percentile": [0.80]}

    best_config, _, _ = grid_search(long_returns, long_cdi, base_config, grid, train_start, train_end)

    # gap original era 0.70 - 0.60 = 0.10 -> hold deveria ser 0.80 - 0.10 = 0.70
    assert abs(best_config.hold_percentile - 0.70) < 1e-9


def test_run_train_test_split_end_to_end(long_returns, long_cdi, base_config):
    grid = {"lookback_days": [40, 60], "rebalance_freq": ["Q"]}

    result = run_train_test_split(long_returns, long_cdi, long_cdi, base_config, grid)

    assert isinstance(result, TrainTestResult)
    assert result.train_end < result.test_start
    assert not result.optimized_test_metrics.empty
    assert not result.baseline_test_metrics.empty
    assert result.overfit_gap == result.best_train_score - result.optimized_test_score


def test_invalid_objective_raises(long_returns, long_cdi, base_config):
    train_start, train_end, _, _ = split_train_test(long_returns, base_config)
    with pytest.raises(ValueError):
        grid_search(
            long_returns, long_cdi, base_config, {"lookback_days": [60]},
            train_start, train_end, objective="nao_existe",
        )
