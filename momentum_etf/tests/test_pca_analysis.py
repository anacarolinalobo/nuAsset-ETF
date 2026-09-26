import numpy as np
import pandas as pd
import pytest

from src.pca_analysis import (
    compute_asset_class_pca,
    compute_universe_pca,
    dominant_component_for_asset,
    pc1_loading_vs_momentum_score,
)


def _dates(n=100):
    return pd.bdate_range("2022-01-01", periods=n)


def test_perfectly_correlated_assets_pc1_explains_all_variance():
    dates = _dates(100)
    rng = np.random.default_rng(0)
    common_factor = rng.normal(0, 0.01, len(dates))
    # A, B, C são a mesma série (correlação 1.0) -> um único fator de risco
    returns = pd.DataFrame({"A": common_factor, "B": common_factor, "C": common_factor}, index=dates)

    result = compute_universe_pca(returns, ["A", "B", "C"], dates[-1], window_days=100, n_components=3)

    assert result.explained_variance_ratio.iloc[0] > 0.99


def test_independent_assets_variance_spreads_across_components():
    dates = _dates(200)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        {t: rng.normal(0, 0.01, len(dates)) for t in ["A", "B", "C", "D"]}, index=dates
    )

    result = compute_universe_pca(returns, ["A", "B", "C", "D"], dates[-1], window_days=200, n_components=4)

    # com ativos independentes, PC1 não deveria dominar quase tudo
    assert result.explained_variance_ratio.iloc[0] < 0.60


def test_explained_variance_ratio_sums_to_at_most_one_and_is_decreasing():
    dates = _dates(150)
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(
        {t: rng.normal(0, 0.01 + 0.005 * i, len(dates)) for i, t in enumerate(["A", "B", "C", "D", "E"])},
        index=dates,
    )

    result = compute_universe_pca(returns, ["A", "B", "C", "D", "E"], dates[-1], window_days=150, n_components=5)

    ratios = result.explained_variance_ratio
    assert ratios.sum() <= 1.0 + 1e-9
    assert (ratios.diff().dropna() <= 1e-9).all()  # não-crescente


def test_incomplete_history_assets_are_dropped_not_imputed():
    dates = _dates(50)
    rng = np.random.default_rng(3)
    data = {t: rng.normal(0, 0.01, len(dates)) for t in ["A", "B"]}
    data["C"] = rng.normal(0, 0.01, len(dates))
    data["C"][:10] = np.nan  # histórico incompleto
    returns = pd.DataFrame(data, index=dates)

    result = compute_universe_pca(returns, ["A", "B", "C"], dates[-1], window_days=50, n_components=2)

    assert "C" in result.dropped_assets
    assert "C" not in result.loadings.index


def test_pc1_sign_convention_mean_loading_positive():
    dates = _dates(100)
    rng = np.random.default_rng(4)
    common = rng.normal(0, 0.01, len(dates))
    returns = pd.DataFrame({"A": common, "B": common * 0.9, "C": common * 1.1}, index=dates)

    result = compute_universe_pca(returns, ["A", "B", "C"], dates[-1], window_days=100, n_components=1)

    assert result.loadings["PC1"].mean() > 0


def test_asset_class_pca_includes_momentum_index_column():
    dates = _dates(100)
    rng = np.random.default_rng(5)
    benchmarks = pd.DataFrame(
        {t: rng.normal(0, 0.005, len(dates)) for t in ["cdi", "ibovespa", "ifix"]}, index=dates
    )
    index_returns = pd.Series(rng.normal(0.0005, 0.012, len(dates)), index=dates)

    result = compute_asset_class_pca(benchmarks, index_returns, dates[-1], window_days=100, n_components=4)

    assert "Índice Momentum" in result.loadings.index


def test_pc1_loading_vs_momentum_score_aligns_by_asset():
    loadings = pd.DataFrame({"PC1": [0.5, 0.3, 0.1]}, index=["A", "B", "C"])
    scores = pd.Series({"A": 1.5, "B": -0.2, "D": 2.0})  # D não está nas loadings

    merged = pc1_loading_vs_momentum_score(loadings, scores)

    assert set(merged.index) == {"A", "B"}
    assert merged.loc["A", "carga_pc1"] == 0.5
    assert merged.loc["A", "score_momentum"] == 1.5


def test_dominant_component_for_asset_returns_correct_pc():
    loadings = pd.DataFrame(
        {"PC1": [0.9, 0.1, 0.2], "PC2": [0.1, 0.8, -0.1]}, index=["A", "B", "C"]
    )
    dominant_pc, ranked = dominant_component_for_asset(loadings, "B")
    assert dominant_pc == "PC2"
    assert ranked.index[0] == "B"  # maior carga em PC2 é a própria B


def test_raises_when_too_few_observations():
    dates = _dates(1)
    returns = pd.DataFrame({"A": [0.01], "B": [0.01]}, index=dates)
    with pytest.raises(ValueError):
        compute_universe_pca(returns, ["A", "B"], dates[-1], window_days=1, n_components=5)
