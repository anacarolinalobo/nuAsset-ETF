import pandas as pd

from src.sensitivity import train_test_grid_search


def test_train_test_grid_search_splits_and_returns_expected_shape(
    synthetic_returns, synthetic_cdi, small_config
):
    grid = {"lookback_days": [40, 60], "entry_percentile": [0.6, 0.7]}

    result = train_test_grid_search(synthetic_returns, synthetic_cdi, small_config, grid)

    n_combos = len(grid["lookback_days"]) * len(grid["entry_percentile"])
    assert len(result["grid_results"]) == n_combos
    assert set(result["best_params"].keys()) == set(grid.keys())

    train_start, train_end = result["train_period"]
    test_start, test_end = result["test_period"]
    assert train_end < test_start
    assert train_start < train_end
    assert test_start < test_end

    for key in (
        "sharpe_treino_otimizado",
        "sharpe_teste_otimizado",
        "sharpe_teste_default",
        "gap_treino_teste",
    ):
        assert isinstance(result[key], float)
    assert isinstance(result["otimizado_bate_default_no_teste"], (bool,))


def test_semester_rebalance_produces_roughly_half_the_quarterly_dates(
    synthetic_returns, small_config
):
    from dataclasses import replace

    from src.rebalance import generate_rebalance_dates

    quarterly_cfg = replace(small_config, rebalance_freq="QE")
    semester_cfg = replace(small_config, rebalance_freq="S")

    quarterly_dates = generate_rebalance_dates(synthetic_returns.index, quarterly_cfg)
    semester_dates = generate_rebalance_dates(synthetic_returns.index, semester_cfg)

    assert len(semester_dates) < len(quarterly_dates)
    assert len(semester_dates) >= 1
