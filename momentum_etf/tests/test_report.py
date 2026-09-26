import numpy as np
import pandas as pd

from src.report import _sanitize_for_plot, build_dashboard


def test_sanitize_replaces_inf_with_nan():
    series = pd.Series([1.0, np.inf, 2.0, -np.inf, 3.0])
    cleaned = _sanitize_for_plot(series, "teste")
    assert cleaned.replace([np.inf, -np.inf], np.nan).equals(cleaned)
    assert cleaned.isna().sum() == 2
    assert list(cleaned.dropna()) == [1.0, 2.0, 3.0]


def test_sanitize_is_noop_when_all_finite():
    series = pd.Series([1.0, 2.0, 3.0])
    cleaned = _sanitize_for_plot(series, "teste")
    pd.testing.assert_series_equal(cleaned, series.astype(float))


def test_build_dashboard_does_not_crash_with_inf_in_benchmark(tmp_path):
    """Regressão: um inf no benchmark (não limpo por acoes_retornos.csv)
    não pode mais derrubar a geração do dashboard inteiro."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    port_returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, len(dates)), index=dates)
    bench_returns = port_returns.copy()
    bench_returns.iloc[150] = np.inf  # dado corrompido não capturado a montante

    turnover = pd.Series([0.2, 0.15], index=[dates[50], dates[200]])
    n_holdings = pd.Series([40, 42], index=[dates[50], dates[200]])
    risk_free = pd.Series(0.0003, index=dates)
    summary = pd.Series({"Retorno acumulado": 0.1, "Sharpe": 1.2})

    out_path = tmp_path / "dashboard.html"
    build_dashboard(
        out_path, port_returns, port_returns, bench_returns, turnover, n_holdings, risk_free, summary
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
