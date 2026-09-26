import numpy as np
import pandas as pd

from src.market_factors import (
    classify_factor_regimes,
    factor_exposure,
    fit_market_factors,
    momentum_performance_by_regime,
    standardize_returns,
)


def test_standardize_returns_has_zero_mean_unit_std_per_ticker(synthetic_returns):
    standardized = standardize_returns(synthetic_returns)
    means = standardized.mean(skipna=True)
    stds = standardized.std(skipna=True, ddof=1)

    assert (means.abs() < 1e-8).all()
    assert (stds.sub(1.0).abs() < 1e-8).all()


def test_fit_market_factors_shapes_and_explained_variance(synthetic_returns):
    factors = fit_market_factors(synthetic_returns, n_components=3)

    assert list(factors.loadings.index) == ["PC1", "PC2", "PC3"]
    assert set(factors.loadings.columns) == set(synthetic_returns.columns)
    assert list(factors.factor_scores.columns) == ["PC1", "PC2", "PC3"]
    assert len(factors.factor_scores) == len(synthetic_returns)

    # variância explicada é decrescente e cada fração está em [0, 1]
    ratios = factors.explained_variance_ratio.to_numpy()
    assert (ratios >= 0).all() and (ratios <= 1).all()
    assert (np.diff(ratios) <= 1e-9).all()

    # convenção de sinal: maioria das cargas do PC1 é positiva
    assert factors.pct_tickers_positive_pc1 >= 0.5


def test_classify_factor_regimes_produces_ordered_buckets(synthetic_returns):
    factors = fit_market_factors(synthetic_returns, n_components=2)
    regimes = classify_factor_regimes(factors.factor_scores, n_buckets=3)

    assert set(regimes.columns) == {"PC1", "PC2"}
    for col in regimes.columns:
        assert list(regimes[col].cat.categories) == ["baixo", "médio", "alto"]
        counts = regimes[col].value_counts()
        # tercis devem ser razoavelmente balanceados
        assert counts.min() > 0
        assert counts.max() / counts.min() < 3


def test_momentum_performance_by_regime_covers_all_regime_days(
    synthetic_returns, synthetic_cdi
):
    factors = fit_market_factors(synthetic_returns, n_components=2)
    regimes = classify_factor_regimes(factors.factor_scores, n_buckets=3)

    momentum_returns = pd.Series(
        np.random.default_rng(1).normal(0.0005, 0.01, len(synthetic_returns)),
        index=synthetic_returns.index,
    )

    perf = momentum_performance_by_regime(momentum_returns, regimes, synthetic_cdi)

    assert set(perf["fator"].unique()) == {"PC1", "PC2"}
    for factor in ("PC1", "PC2"):
        n_dias_total = perf.loc[perf["fator"] == factor, "n_dias"].sum()
        assert n_dias_total == len(momentum_returns)


def test_factor_exposure_returns_betas_and_bounded_r_squared(synthetic_returns):
    factors = fit_market_factors(synthetic_returns, n_components=2)
    # retorno sintético fortemente ligado ao PC1, para checar que a regressão captura isso
    momentum_returns = 0.5 * factors.factor_scores["PC1"] + pd.Series(
        np.random.default_rng(2).normal(0, 0.0001, len(synthetic_returns)),
        index=factors.factor_scores.index,
    )

    exposure = factor_exposure(momentum_returns, factors.factor_scores)

    assert "beta_PC1" in exposure.index
    assert "beta_PC2" in exposure.index
    assert 0.0 <= exposure["r_quadrado"] <= 1.0
    assert exposure["r_quadrado"] > 0.9  # relação quase determinística por construção
