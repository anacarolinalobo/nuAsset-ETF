"""Sinal de momentum: momentum 12-1 ajustado a risco, em z-score cross-sectional.

Definição (por ticker, na data `date`):
    ret_acum   = retorno acumulado entre (date - lookback_days) e (date - skip_days)
    vol_diaria = desvio-padrão dos retornos diários na mesma janela
    score_bruto = ret_acum / max(vol_diaria, min_daily_vol)   se risk_adjust
                = ret_acum                                     caso contrário
    score = z-score de score_bruto DENTRO do universo elegível naquela data

O "skip" do último mês (pular de t-252 a t-21 em vez de ir até t) evita
capturar reversão de curtíssimo prazo / microestrutura, um efeito bem
documentado na literatura de momentum (Jegadeesh & Titman, 1993) e distinto
do efeito de tendência que o índice quer capturar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MomentumConfig


def _cumulative_return(returns: pd.Series) -> float:
    if returns.isna().all():
        return np.nan
    return (1.0 + returns.fillna(0.0)).prod() - 1.0


def compute_signal_on_date(
    returns_wide: pd.DataFrame,
    date: pd.Timestamp,
    eligible: pd.Index,
    config: MomentumConfig,
) -> pd.Series:
    """Calcula o score de momentum (z-score) para os tickers elegíveis em `date`.

    Usa apenas `returns_wide.loc[:date]` — nenhuma informação posterior a
    `date` entra no cálculo (garantia anti look-ahead).
    """
    history = returns_wide.loc[:date, eligible]
    window_start_idx = -config.lookback_days
    window_end_idx = -config.skip_days if config.skip_days > 0 else None

    signal_window = history.iloc[window_start_idx:window_end_idx]
    if signal_window.empty:
        return pd.Series(dtype=float)

    cum_return = signal_window.apply(_cumulative_return, axis=0)

    if config.risk_adjust:
        vol_window = history.tail(config.vol_window_days)
        daily_vol = vol_window.std(axis=0, ddof=1).clip(lower=config.min_daily_vol)
        raw_score = cum_return / daily_vol
    else:
        raw_score = cum_return

    raw_score = raw_score.dropna()
    if raw_score.empty or raw_score.std(ddof=0) == 0:
        return pd.Series(0.0, index=raw_score.index)

    z_score = (raw_score - raw_score.mean()) / raw_score.std(ddof=0)
    return z_score


def compute_signal_history(
    returns_wide: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    eligible_by_date: dict[pd.Timestamp, pd.Index],
    config: MomentumConfig,
) -> dict[pd.Timestamp, pd.Series]:
    """Calcula o score de momentum em cada data de rebalanceamento."""
    signals = {}
    for date in rebalance_dates:
        eligible = eligible_by_date[date]
        signals[date] = compute_signal_on_date(returns_wide, date, eligible, config)
    return signals
