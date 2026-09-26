import numpy as np
import pandas as pd

from src.signal import compute_signal_on_date
from src.universe import eligible_universe


def test_signal_ranks_uptrend_above_downtrend(synthetic_returns, small_config):
    date = synthetic_returns.index[synthetic_returns.index >= "2020-04-15"][0]
    eligible = eligible_universe(synthetic_returns, date, small_config)
    signal = compute_signal_on_date(synthetic_returns, date, eligible, small_config)

    uptrend = [t for t in signal.index if t.startswith("T") and int(t[1:]) < 5]
    downtrend = [t for t in signal.index if t.startswith("T") and int(t[1:]) >= 5]

    assert signal[uptrend].mean() > signal[downtrend].mean()


def test_signal_has_no_lookahead(synthetic_returns, small_config):
    date = synthetic_returns.index[synthetic_returns.index >= "2020-04-15"][0]
    eligible = eligible_universe(synthetic_returns, date, small_config)
    signal_before = compute_signal_on_date(synthetic_returns, date, eligible, small_config)

    mutated = synthetic_returns.copy()
    mutated.loc[mutated.index > date, :] = 999.0  # corrompe só o futuro

    signal_after = compute_signal_on_date(mutated, date, eligible, small_config)

    pd.testing.assert_series_equal(
        signal_before.sort_index(), signal_after.sort_index(), check_exact=False
    )


def test_zscore_is_standardized(synthetic_returns, small_config):
    date = synthetic_returns.index[synthetic_returns.index >= "2020-04-15"][0]
    eligible = eligible_universe(synthetic_returns, date, small_config)
    signal = compute_signal_on_date(synthetic_returns, date, eligible, small_config)

    assert abs(signal.mean()) < 1e-8
    assert abs(signal.std(ddof=0) - 1.0) < 1e-8
