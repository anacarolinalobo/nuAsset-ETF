import numpy as np
import pandas as pd

from src import metrics


def test_max_drawdown_known_series():
    returns = pd.Series([0.10, -0.20, 0.05, 0.30])
    dd = metrics.max_drawdown(returns)
    # curva: 1.10, 0.88, 0.924, 1.2012 -> pico 1.10, vale 0.88 -> dd = -0.20
    assert abs(dd - (-0.20)) < 1e-9


def test_total_return_compounding():
    returns = pd.Series([0.10, 0.10])
    assert abs(metrics.total_return(returns) - 0.21) < 1e-9


def test_sharpe_zero_vol_returns_nan():
    returns = pd.Series([0.001] * 10)
    risk_free = pd.Series([0.001] * 10, index=returns.index)
    assert np.isnan(metrics.sharpe_ratio(returns, risk_free))


def test_active_share_identical_portfolios_is_zero():
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert metrics.active_share(w, w) == 0.0


def test_active_share_disjoint_portfolios_is_one():
    p = pd.Series({"A": 1.0})
    b = pd.Series({"B": 1.0})
    assert abs(metrics.active_share(p, b) - 1.0) < 1e-9
