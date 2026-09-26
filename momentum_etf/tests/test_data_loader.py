import numpy as np
import pandas as pd

from src.data_loader import benchmarks_to_returns, clean_returns, flag_suspicious_returns


def test_benchmarks_to_returns_converts_level_to_pct_change():
    dates = pd.bdate_range("2020-01-01", periods=4)
    # nível tipo Ibovespa: sobe 1%, sobe 2%, cai 1%
    levels = pd.DataFrame({"ibovespa": [100000.0, 101000.0, 103020.0, 101989.8]}, index=dates)

    returns = benchmarks_to_returns(levels)

    assert pd.isna(returns.iloc[0, 0])
    assert abs(returns.iloc[1, 0] - 0.01) < 1e-6
    assert abs(returns.iloc[2, 0] - 0.02) < 1e-6
    assert abs(returns.iloc[3, 0] - (-0.01)) < 1e-6


def test_benchmarks_to_returns_scale_invariant_to_arbitrary_base():
    dates = pd.bdate_range("2020-01-01", periods=3)
    levels_a = pd.DataFrame({"x": [10.0, 11.0, 9.9]}, index=dates)
    levels_b = pd.DataFrame({"x": [1000.0, 1100.0, 990.0]}, index=dates)

    pd.testing.assert_frame_equal(benchmarks_to_returns(levels_a), benchmarks_to_returns(levels_b))


def test_flags_extreme_return():
    dates = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.DataFrame({"A": [0.01, 0.01, 1.50, 0.01, 0.01]}, index=dates)

    flags, summary = flag_suspicious_returns(returns, max_abs_daily_return=1.00)

    assert len(flags) == 1
    assert flags.iloc[0]["tipo"] == "retorno_extremo"
    assert summary.set_index("tipo").loc["retorno_extremo", "n_ocorrencias"] == 1


def test_flags_unadjusted_split_reversal_pair():
    dates = pd.bdate_range("2020-01-01", periods=4)
    # dia 2: +100% (erro de cotação/split não ajustado); dia 3: reversão
    # matemática implícita -r/(1+r) = -1/2 = -50%
    returns = pd.DataFrame({"A": [0.01, 1.00, -0.50, 0.01]}, index=dates)

    flags, summary = flag_suspicious_returns(returns, max_abs_daily_return=100.0)

    tipos = set(flags["tipo"])
    assert "possivel_split_nao_ajustado" in tipos
    assert len(flags) == 2  # os dois dias do par


def test_clean_returns_sets_flagged_points_to_nan():
    dates = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.DataFrame({"A": [0.01, 0.01, 1.50, 0.01, 0.01]}, index=dates)

    flags, _ = flag_suspicious_returns(returns, max_abs_daily_return=1.00)
    cleaned = clean_returns(returns, flags)

    assert pd.isna(cleaned.loc[dates[2], "A"])
    # o resto da série permanece intacto
    assert cleaned.loc[dates[0], "A"] == 0.01


def test_clean_returns_noop_when_no_flags():
    dates = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.01, 0.01, 0.01]}, index=dates)
    empty_flags = pd.DataFrame(columns=["date", "ticker", "retorno", "tipo"])

    cleaned = clean_returns(returns, empty_flags)
    pd.testing.assert_frame_equal(cleaned, returns)


def test_cleaning_removes_overflow_risk_in_cumulative_product():
    dates = pd.bdate_range("2020-01-01", periods=5)
    # retorno absurdo (erro de cotação) que estouraria cumprod se mantido
    returns = pd.DataFrame({"A": [0.01, 500.0, 0.01, 0.01, 0.01]}, index=dates)

    flags, _ = flag_suspicious_returns(returns, max_abs_daily_return=1.00)
    cleaned = clean_returns(returns, flags)

    cumulative = (1 + cleaned["A"].fillna(0)).cumprod()
    assert np.isfinite(cumulative).all()
