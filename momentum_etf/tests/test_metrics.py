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


def test_hit_rate_vs_benchmark_works_regardless_of_pandas_freq_alias():
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    returns = pd.Series(0.001, index=dates)
    benchmark = pd.Series(0.0005, index=dates)

    # deve funcionar com o default "ME" na versão de pandas instalada aqui...
    assert metrics.hit_rate_vs_benchmark(returns, benchmark) == 1.0


def test_compatible_resample_freq_falls_back_when_alias_unsupported(monkeypatch):
    # ...e também simulando uma versão de pandas antiga, que só aceita "M"
    # (não "ME") -- exatamente o traceback visto em produção com pandas < 2.2.
    real_to_offset = pd.tseries.frequencies.to_offset

    def fake_old_pandas_to_offset(freq):
        if freq == "ME":
            raise ValueError("Invalid frequency: ME")
        if freq == "M":
            return real_to_offset("ME")  # "M" tinha o mesmo significado antes da 2.2
        return real_to_offset(freq)

    monkeypatch.setattr(pd.tseries.frequencies, "to_offset", fake_old_pandas_to_offset)
    assert metrics._compatible_resample_freq("ME") == "M"
    assert metrics._compatible_resample_freq("M") == "M"
