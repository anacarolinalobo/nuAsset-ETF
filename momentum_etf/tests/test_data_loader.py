import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_benchmarks


def test_levels_are_converted_to_daily_returns(tmp_path):
    path = tmp_path / "benchmarks_diarios.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "cdi": [3.0, 3.003, 3.006003],
            "ibovespa": [100000.0, 101000.0, np.nan],
        }
    ).to_csv(path, index=False)
    df = load_benchmarks(path)
    assert np.isnan(df["cdi"].iloc[0])
    assert df["cdi"].iloc[1] == pytest.approx(0.001)
    assert df["ibovespa"].iloc[1] == pytest.approx(0.01)


def test_returns_are_kept_as_is(tmp_path):
    path = tmp_path / "benchmarks_diarios.csv"
    pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "cdi": [0.0004, 0.0004]}).to_csv(path, index=False)
    df = load_benchmarks(path)
    assert df["cdi"].tolist() == [0.0004, 0.0004]
