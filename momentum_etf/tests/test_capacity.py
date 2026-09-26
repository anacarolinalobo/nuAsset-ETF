import numpy as np
import pandas as pd

from src.capacity import estimate_capacity


def test_capacity_formula_basic():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    adtv = pd.Series({"A": 1_000_000.0, "B": 2_000_000.0})

    result = estimate_capacity(weights, adtv, max_participation_rate=0.10, days_to_build_position=5)

    # capacidade_A = 1_000_000 * 0.10 * 5 / 0.5 = 1_000_000
    assert abs(result.capacity_per_name["A"] - 1_000_000.0) < 1e-6
    # capacidade_B = 2_000_000 * 0.10 * 5 / 0.5 = 2_000_000
    assert abs(result.capacity_per_name["B"] - 2_000_000.0) < 1e-6


def test_product_capacity_is_minimum_across_names():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    adtv = pd.Series({"A": 1_000_000.0, "B": 5_000_000.0})

    result = estimate_capacity(weights, adtv)

    assert result.product_capacity == result.capacity_per_name.min()
    assert result.product_capacity < result.capacity_per_name.max()


def test_single_zero_adtv_does_not_zero_whole_product():
    """Regressão: um papel sem cobertura de ADTV não pode derrubar a
    capacidade do produto inteiro para zero -- deve ser excluído do
    cálculo do gargalo e reportado à parte."""
    weights = pd.Series({"A": 0.5, "B": 0.5})
    adtv = pd.Series({"A": 1_000_000.0, "B": 0.0})

    result = estimate_capacity(weights, adtv)

    assert result.product_capacity > 0
    assert "B" in result.uncovered_tickers.index
    assert "B" not in result.capacity_per_name.index
    assert abs(result.uncovered_weight - 0.5) < 1e-9


def test_missing_adtv_treated_same_as_zero():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    adtv = pd.Series({"A": 1_000_000.0})  # B nem aparece na série de ADTV

    result = estimate_capacity(weights, adtv)

    assert "B" in result.uncovered_tickers.index
    assert result.uncovered_weight == 0.5


def test_all_names_uncovered_gives_nan_product_capacity():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    adtv = pd.Series({"A": 0.0, "B": 0.0})

    result = estimate_capacity(weights, adtv)

    assert np.isnan(result.product_capacity)
    assert result.capacity_per_name.empty
    assert abs(result.uncovered_weight - 1.0) < 1e-9


def test_zero_weight_positions_are_ignored():
    weights = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    adtv = pd.Series({"A": 1_000_000.0, "B": 1_000_000.0, "C": 0.0})

    result = estimate_capacity(weights, adtv)

    assert "C" not in result.capacity_per_name.index
    assert "C" not in result.uncovered_tickers.index


def test_higher_participation_rate_increases_capacity():
    weights = pd.Series({"A": 1.0})
    adtv = pd.Series({"A": 1_000_000.0})

    low = estimate_capacity(weights, adtv, max_participation_rate=0.05)
    high = estimate_capacity(weights, adtv, max_participation_rate=0.20)

    assert high.product_capacity > low.product_capacity
