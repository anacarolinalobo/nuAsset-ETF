"""Métricas de desempenho do índice.

Cobre o mínimo pedido no case (retorno acumulado/anualizado, volatilidade,
Sharpe, máximo drawdown, turnover anualizado, tracking error, beta, active
share, atribuição) e acrescenta métricas próprias mais relevantes para o
que um índice de momentum long-only se propõe a entregar:

  - Sortino e Calmar: momentum é conhecido por ter drawdowns assimétricos
    ("momentum crashes" em reversões bruscas de mercado — Daniel &
    Moskowitz, 2016); volatilidade simétrica (Sharpe) subestima esse risco.
  - Hit rate mensal vs. benchmark: relevante para o discurso comercial do
    produto ("com que frequência bate o Ibovespa"), não só a média.
  - Up/down capture: mostra SE e ONDE o fator entrega o prêmio esperado —
    idealmente captura mais alta que baixa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    return (1 + returns.fillna(0)).prod() - 1


def annualized_return(returns: pd.Series) -> float:
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return np.nan
    return (1 + total_return(returns)) ** (1 / n_years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series) -> float:
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    denom = excess.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return (excess.mean() / denom) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: pd.Series, risk_free: pd.Series) -> float:
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    downside = excess[excess < 0]
    denom = downside.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return (excess.mean() / denom) * np.sqrt(TRADING_DAYS_PER_YEAR)


def cumulative_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    curve = cumulative_curve(returns)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = cumulative_curve(returns)
    running_max = curve.cummax()
    return curve / running_max - 1


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns) / abs(mdd)


def turnover_annualized(turnover_history: pd.Series, rebalances_per_year: float) -> float:
    return turnover_history.mean() * rebalances_per_year


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    diff = (returns - benchmark_returns.reindex(returns.index)).dropna()
    return diff.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat(
        [returns, benchmark_returns.reindex(returns.index)], axis=1
    ).dropna()
    if len(aligned) < 2:
        return np.nan
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var(ddof=1)
    if var == 0:
        return np.nan
    return cov / var


def active_share(
    portfolio_weights: pd.Series, benchmark_weights: pd.Series
) -> float:
    all_tickers = portfolio_weights.index.union(benchmark_weights.index)
    p = portfolio_weights.reindex(all_tickers, fill_value=0.0)
    b = benchmark_weights.reindex(all_tickers, fill_value=0.0)
    return 0.5 * (p - b).abs().sum()


def _compatible_resample_freq(freq: str) -> str:
    """Escolhe o alias de frequência de resample aceito pela versão do pandas instalada.

    O pandas trocou "M"/"Q"/"Y" por "ME"/"QE"/"YE" no `resample` a partir da
    2.2 e removeu de vez os antigos na 3.0 — mas quem ainda estiver em uma
    versão anterior à 2.2 não reconhece os novos aliases. Testa o alias
    pedido e cai para a variante alternativa se a versão instalada não
    aceitar, para o mesmo código rodar em qualquer uma delas.
    """
    alternates = {"ME": "M", "M": "ME", "QE": "Q", "Q": "QE", "YE": "Y", "Y": "YE"}
    try:
        pd.tseries.frequencies.to_offset(freq)
        return freq
    except ValueError:
        alt = alternates.get(freq)
        if alt is None:
            raise
        return alt


def hit_rate_vs_benchmark(returns: pd.Series, benchmark_returns: pd.Series, freq: str = "ME") -> float:
    freq = _compatible_resample_freq(freq)
    port_m = (1 + returns).resample(freq).prod() - 1
    bench_m = (1 + benchmark_returns.reindex(returns.index).fillna(0.0)).resample(freq).prod() - 1
    aligned = pd.concat([port_m, bench_m], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return (aligned.iloc[:, 0] > aligned.iloc[:, 1]).mean()


def up_down_capture(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    bench = benchmark_returns.reindex(returns.index).fillna(0.0)
    up_mask = bench > 0
    down_mask = bench < 0

    up_capture = (
        returns[up_mask].mean() / bench[up_mask].mean()
        if up_mask.any() and bench[up_mask].mean() != 0
        else np.nan
    )
    down_capture = (
        returns[down_mask].mean() / bench[down_mask].mean()
        if down_mask.any() and bench[down_mask].mean() != 0
        else np.nan
    )
    return up_capture, down_capture


def summary_table(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: pd.Series,
    turnover_history: pd.Series,
    rebalances_per_year: float,
    latest_weights: pd.Series | None = None,
    latest_benchmark_weights: pd.Series | None = None,
) -> pd.Series:
    up_cap, down_cap = up_down_capture(returns, benchmark_returns)
    data = {
        "Retorno acumulado": total_return(returns),
        "Retorno anualizado": annualized_return(returns),
        "Volatilidade anualizada": annualized_vol(returns),
        "Sharpe": sharpe_ratio(returns, risk_free),
        "Sortino": sortino_ratio(returns, risk_free),
        "Máximo drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
        "Turnover anualizado": turnover_annualized(turnover_history, rebalances_per_year),
        "Tracking error": tracking_error(returns, benchmark_returns),
        "Beta vs Ibovespa": beta(returns, benchmark_returns),
        "Hit rate mensal vs Ibovespa": hit_rate_vs_benchmark(returns, benchmark_returns),
        "Up-capture": up_cap,
        "Down-capture": down_cap,
    }
    if latest_weights is not None and latest_benchmark_weights is not None:
        data["Active share (última carteira)"] = active_share(
            latest_weights, latest_benchmark_weights
        )
    return pd.Series(data)
