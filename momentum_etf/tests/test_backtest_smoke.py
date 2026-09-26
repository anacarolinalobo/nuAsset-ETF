import numpy as np
import pandas as pd

from src.backtest import run_backtest


def test_backtest_runs_end_to_end(synthetic_returns, synthetic_cdi, small_config):
    result = run_backtest(synthetic_returns, synthetic_cdi, small_config)

    assert len(result.returns_net) > 0
    assert result.returns_net.notna().all()
    assert np.isfinite(result.returns_net.to_numpy()).all()
    assert len(result.weights_history) >= 1


def test_weights_sum_close_to_one_at_each_rebalance(synthetic_returns, synthetic_cdi, small_config):
    result = run_backtest(synthetic_returns, synthetic_cdi, small_config)
    for date, weights in result.weights_history.items():
        assert abs(weights.sum() - 1.0) < 1e-6


def test_net_returns_below_gross_when_turnover_positive(synthetic_returns, synthetic_cdi, small_config):
    from src.rebalance import effective_date, generate_rebalance_dates

    result = run_backtest(synthetic_returns, synthetic_cdi, small_config)
    calc_dates = generate_rebalance_dates(synthetic_returns.index, small_config)
    effective_days = {
        effective_date(d, synthetic_returns.index, small_config) for d in calc_dates
    }
    # nos dias sem custo (fora da data efetiva de rebalance), bruto == líquido
    non_rebalance_days = [d for d in result.returns_net.index if d not in effective_days]
    if non_rebalance_days:
        diffs = (
            result.returns_gross[non_rebalance_days] - result.returns_net[non_rebalance_days]
        )
        assert (diffs.abs() < 1e-9).all()


def test_delisted_ticker_removed_from_holdings(synthetic_returns, synthetic_cdi, small_config):
    result = run_backtest(synthetic_returns, synthetic_cdi, small_config)
    delisted_tickers = {d["ticker"] for d in result.delistings_log}
    if delisted_tickers:
        last_weights = result.weights_history[max(result.weights_history.keys())]
        assert not delisted_tickers & set(last_weights.index)
