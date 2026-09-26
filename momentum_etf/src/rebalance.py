"""Calendário de rebalanceamento e defasagem de execução.

Frequência trimestral: equilíbrio entre responsividade ao sinal (momentum
decai — sinal calculado há 6+ meses perde poder preditivo) e turnover
(rebalancear mais que isso, ex. mensal, eleva custo sem ganho líquido
relevante para um fator que se move em janelas de meses, não semanas).

A defasagem de execução (`execution_lag_days`) simula o tempo operacional
entre o cálculo dos novos pesos (fechamento da data de rebalance) e o
momento em que a carteira real passa a refleti-los — os pesos "antigos"
continuam valendo durante esse intervalo.
"""

from __future__ import annotations

import pandas as pd

from src.config import MomentumConfig


_PERIOD_FREQ_ALIASES = {"ME": "M", "QE": "Q", "YE": "Y", "AE": "Y"}


def _period_ends(trading_days: pd.DatetimeIndex, period_freq: str) -> pd.DatetimeIndex:
    period_freq = _PERIOD_FREQ_ALIASES.get(period_freq, period_freq)
    series = pd.Series(trading_days, index=trading_days)
    period_ends = series.groupby(trading_days.to_period(period_freq)).max()
    return pd.DatetimeIndex(period_ends.values).sort_values()


def generate_rebalance_dates(
    trading_days: pd.DatetimeIndex, config: MomentumConfig
) -> pd.DatetimeIndex:
    """Última data de pregão de cada período (`config.rebalance_freq`).

    Aceita "M" (mensal), "Q" (trimestral) ou "S" (semestral). "S" não é um
    alias nativo de `Period` do pandas (que não lida bem com múltiplos
    como "2Q" para essa finalidade de agrupamento), então é construído a
    partir dos fins de trimestre, mantendo apenas o 2º e o 4º de cada ano
    (fim do 1º e do 2º semestre).
    """
    if config.rebalance_freq == "S":
        quarterly = _period_ends(trading_days, "Q")
        dates = quarterly[1::2]
    else:
        dates = _period_ends(trading_days, config.rebalance_freq)

    start = pd.Timestamp(config.backtest_start)
    end = pd.Timestamp(config.backtest_end)
    return dates[(dates >= start) & (dates <= end)]


def effective_date(
    rebalance_date: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
    config: MomentumConfig,
) -> pd.Timestamp:
    """Primeira data de pregão em que os novos pesos passam a valer."""
    future_days = trading_days[trading_days > rebalance_date]
    lag = config.execution_lag_days
    if len(future_days) == 0:
        return rebalance_date
    idx = min(lag, len(future_days)) - 1
    return future_days[idx]
