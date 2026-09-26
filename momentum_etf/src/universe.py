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

A elegibilidade é decidida em DOIS ESTÁGIOS, nesta ordem — o filtro de
liquidez/tamanho roda ANTES do sinal de momentum ser sequer calculado
(`signal.py` só recebe o universo já filtrado aqui), para que "é
negociável em tamanho razoável" seja uma pergunta independente de "teve
bom momentum recente":

  1. Liquidez e tamanho (`_liquidity_size_filter`): quando ADTV real
     (COTAHIST) e/ou valor de mercado (FRE) estão disponíveis, exclui o
     terço menos líquido e o quinto menor em valor de mercado da seção
     transversal do dia (cortes configuráveis em `MomentumConfig`). Sem
     esses dados externos, cai no proxy de presença de retorno (ver
     abaixo) — o pipeline permanece funcional mesmo sem COTAHIST/FRE.
  2. Histórico mínimo e proxy de atividade (`_history_filter`): garante
     que sobrou histórico suficiente para calcular o sinal de 12 meses e
     que o papel não está tecnicamente "morto" (sem nenhum retorno
     recente) mesmo que tenha passado no filtro de liquidez externo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MomentumConfig


def eligible_universe(
    returns_wide: pd.DataFrame,
    date: pd.Timestamp,
    config: MomentumConfig,
    adtv_wide: pd.DataFrame | None = None,
    market_cap_wide: pd.DataFrame | None = None,
) -> pd.Index:
    """Retorna os tickers elegíveis na data `date`.

    `adtv_wide` e `market_cap_wide` são opcionais (date x ticker,
    construídos em `src/cotahist.py` e `src/market_cap.py`). Quando
    ausentes, o filtro de liquidez/tamanho é pulado e a elegibilidade
    depende só do proxy de histórico — comportamento idêntico ao da
    versão anterior deste módulo, preservado para quando os dados
    externos (COTAHIST/FRE) não estiverem disponíveis.
    """
    history = returns_wide.loc[:date]
    if len(history) < config.min_history_days:
        return pd.Index([], name="ticker")

    candidates = returns_wide.columns
    candidates = _liquidity_size_filter(candidates, date, config, adtv_wide, market_cap_wide)
    candidates = _history_filter(history[candidates], config)

    return candidates


def _liquidity_size_filter(
    candidates: pd.Index,
    date: pd.Timestamp,
    config: MomentumConfig,
    adtv_wide: pd.DataFrame | None,
    market_cap_wide: pd.DataFrame | None,
) -> pd.Index:
    result = pd.Index(candidates)

    if adtv_wide is not None and date in adtv_wide.index:
        adtv_today = adtv_wide.loc[date].reindex(result)
        result = _apply_percentile_and_floor(
            result, adtv_today, config.adtv_min_percentile, config.min_adtv_reais
        )

    if market_cap_wide is not None and date in market_cap_wide.index:
        mcap_today = market_cap_wide.loc[date].reindex(result)
        result = _apply_percentile_and_floor(
            result, mcap_today, config.market_cap_min_percentile, config.min_market_cap_reais
        )

    return result


def _apply_percentile_and_floor(
    candidates: pd.Index,
    values: pd.Series,
    min_percentile: float | None,
    min_absolute: float | None,
) -> pd.Index:
    valid = values.dropna()
    if valid.empty:
        # Nenhum dado externo para nenhum candidato nesta data — não
        # exclui ninguém aqui; o filtro de histórico/proxy decide sozinho.
        return candidates

    mask = pd.Series(True, index=valid.index)
    if min_percentile is not None and min_percentile > 0:
        cutoff = valid.quantile(min_percentile)
        mask &= valid >= cutoff
    if min_absolute is not None:
        mask &= valid >= min_absolute

    passed = set(mask[mask].index)
    # Tickers sem dado externo (ex.: mapeamento ticker->CNPJ não
    # encontrado) não são penalizados por este estágio — permanecem
    # candidatos e serão avaliados só pelo filtro de histórico.
    missing = set(candidates) - set(valid.index)
    return pd.Index(sorted(passed | missing))


def _history_filter(history: pd.DataFrame, config: MomentumConfig) -> pd.Index:
    window = history.tail(config.liquidity_window_days)
    active_ratio = window.notna().mean(axis=0)

    has_full_history = history.notna().sum(axis=0) >= config.min_history_days
    is_liquid = active_ratio >= config.min_active_ratio
    is_alive = window.notna().any(axis=0)

    mask = has_full_history & is_liquid & is_alive
    return mask[mask].index
