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
from src.volume_loader import adtv_on_date


def eligible_universe(
    returns_wide: pd.DataFrame,
    date: pd.Timestamp,
    config: MomentumConfig,
    volume_wide: pd.DataFrame | None = None,
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
         proxy disponível — limitação documentada no README.
      4. Liquidez por volume real (opcional): se `volume_wide` (ADTV por
         ticker, ver src/volume_loader.py) for passado e
         `config.min_adtv_brl` estiver definido, exige ADTV (na janela de
         `volume_window_days` pregões até `date`) >= `min_adtv_brl`. Sem
         `volume_wide`, esse critério é ignorado e o comportamento é
         idêntico ao anterior (só o proxy por retorno).
    """
    history = returns_wide.loc[:date]
    if len(history) < config.min_history_days:
        return pd.Index([], name="ticker")

    window = history.tail(config.liquidity_window_days)
    active_ratio = window.notna().mean(axis=0)

    has_full_history = history.notna().sum(axis=0) >= config.min_history_days
    is_liquid = active_ratio >= config.min_active_ratio
    is_alive = window.notna().any(axis=0)

    mask = has_full_history & is_liquid & is_alive

    if volume_wide is not None and config.min_adtv_brl is not None:
        adtv = adtv_on_date(volume_wide, date, config.volume_window_days)
        adtv = adtv.reindex(mask.index).fillna(0.0)
        mask = mask & (adtv >= config.min_adtv_brl)

    return mask[mask].index
