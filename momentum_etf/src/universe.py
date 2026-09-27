"""Universo elegível: quais ações podem compor o índice em cada data.

Regra central para evitar look-ahead bias: a elegibilidade em uma data `t`
usa exclusivamente informação disponível ATÉ `t` (nunca dados futuros —
nem mesmo "sei que essa ação vai ser deslistada semana que vem").

Sobrevivência: como `returns_wide` já contém ações que saíram de negociação
(mantidas na base até o último dia negociado, conforme o briefing), o
universo em cada data é calculado diretamente sobre os dados disponíveis
até aquela data — sem excluir a priori quem depois vai sumir da base. Isso
é o que evita survivorship bias: a ação só some do universo quando o
histórico dela efetivamente acaba, não antes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MomentumConfig
from src.market_data import MarketData


def eligible_universe(
    returns_wide: pd.DataFrame,
    date: pd.Timestamp,
    config: MomentumConfig,
    market_data: MarketData | None = None,
) -> pd.Index:
    """Retorna os tickers elegíveis na data `date`.

    Critérios:
      1. Histórico mínimo de `min_history_days` pregões até `date` (para o
         sinal de momentum de `lookback_days` estar plenamente formado).
      2. Ainda "viva": teve pelo menos um retorno não-nulo nos últimos
         `liquidity_window_days` pregões até `date` (uma ação deslistada
         não terá mais observações depois do seu último pregão).
      3. Liquidez proxy: fração de pregões com retorno observado (não-NaN)
         na janela de `liquidity_window_days` >= `min_active_ratio`. Sem
         dado de volume na base fornecida, presença de retorno diário é o
         proxy disponível quando `market_data` não é passado.

    Com `market_data` (volume B3 + market cap CVM), o critério 3 passa a
    usar dado real de negociação:
      3a. Fração de pregões com volume > 0 na janela de
          `liquidity_window_days` >= `min_active_ratio`.
      3b. ADTV (mediana do volume financeiro diário em
          `adtv_window_days`) >= `min_adtv_brl`.
      4.  Market cap mais recente conhecido até `date` >=
          `min_market_cap_brl` (papéis sem market cap passam, a menos que
          `require_market_cap`).
    """
    history = returns_wide.loc[:date]
    if len(history) < config.min_history_days:
        return pd.Index([], name="ticker")

    window = history.tail(config.liquidity_window_days)
    active_ratio = window.notna().mean(axis=0)

    has_full_history = history.notna().sum(axis=0) >= config.min_history_days
    is_liquid = active_ratio >= config.min_active_ratio
    is_alive = window.notna().any(axis=0)

    if market_data is not None:
        is_liquid = _passes_market_filters(market_data, date, config, active_ratio.index)

    mask = has_full_history & is_liquid & is_alive
    return mask[mask].index


def _passes_market_filters(
    market_data: MarketData,
    date: pd.Timestamp,
    config: MomentumConfig,
    tickers: pd.Index,
) -> pd.Series:
    traded = market_data.traded_ratio(date, config.liquidity_window_days).reindex(tickers)
    adtv = market_data.adtv(date, config.adtv_window_days).reindex(tickers)
    mcap = market_data.market_cap_on(date).reindex(tickers)

    trades_often = traded.fillna(0.0) >= config.min_active_ratio
    enough_adtv = adtv.fillna(0.0) >= config.min_adtv_brl
    if config.require_market_cap:
        big_enough = mcap.fillna(0.0) >= config.min_market_cap_brl
    else:
        big_enough = mcap.isna() | (mcap >= config.min_market_cap_brl)

    return trades_often & enough_adtv & big_enough
