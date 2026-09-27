"""Critério de ponderação: peso proporcional ao score de momentum (com teto).

Alternativas descartadas (documentadas para a discussão do case):
  - Peso igual (equal-weight): mais simples e menos concentrado, mas dilui
    o tilt de momentum — um papel com sinal fraco pesa o mesmo que um com
    sinal forte, o que vai contra o próprio racional do fator.
  - Peso por valor de mercado: replicaria a distorção de cap-weight que o
    índice de momentum busca evitar, e a base fornecida não traz free
    float / valor de mercado diretamente (só retornos e composição do
    Ibovespa, que poderia servir de proxy grosseiro, mas não foi usada
    para não importar viés de outro índice para dentro do sinal).

Score-weighted foi escolhido por concentrar peso onde o sinal é mais forte
(maior expected exposure ao fator) mantendo um teto por ativo para not
depender demais de um único nome de sinal extremo — tipicamente os menos
líquidos, para os quais o score tende a ser mais ruidoso.

Com dados de market cap e volume (src/market_data.py) há duas extensões
opcionais, controladas em `MomentumConfig`:
  - `weighting_scheme="score_sqrt_mcap"`: peso ∝ score x sqrt(market cap).
    A raiz quadrada inclina a carteira para nomes maiores (mais baratos de
    negociar) sem replicar o cap-weight descartado acima.
  - `target_aum_brl`: teto de peso por papel derivado do ADTV, para que a
    carteira seja implementável no tamanho-alvo do produto. O peso que não
    couber em nenhum papel fica em caixa (CDI) — preferível a forçá-lo em
    nomes que não comportam a posição.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MomentumConfig


def compute_weights(
    signal: pd.Series,
    selected: set[str],
    config: MomentumConfig,
    market_cap: pd.Series | None = None,
    adtv: pd.Series | None = None,
) -> pd.Series:
    """Pesos proporcionais ao score (truncado em zero), com teto por ativo.

    O teto é aplicado de forma iterativa: ativos que estourariam o teto são
    fixados nele, e o peso excedente é redistribuído proporcionalmente
    entre os demais, até nenhum peso violar o teto (ou não haver mais para
    quem redistribuir, caso em que o teto é relaxado igualmente entre
    todos — situação rara, só ocorre com poucos nomes selecionados).
    """
    if not selected:
        return pd.Series(dtype=float)

    scores = signal.loc[list(selected)].clip(lower=0.0)
    if scores.sum() == 0:
        scores = pd.Series(1.0, index=scores.index)

    if config.weighting_scheme == "score_sqrt_mcap":
        scores = scores * _size_tilt(scores.index, market_cap)
    elif config.weighting_scheme != "score":
        raise ValueError(f"weighting_scheme desconhecido: {config.weighting_scheme!r}")

    weights = scores / scores.sum()

    if config.target_aum_brl and adtv is not None:
        caps = liquidity_caps(weights.index, adtv, config)
        weights = _apply_cap_per_name(weights, caps)
    else:
        weights = _apply_cap(weights, config.weight_cap)
    return weights.sort_values(ascending=False)


def liquidity_caps(
    tickers: pd.Index, adtv: pd.Series, config: MomentumConfig
) -> pd.Series:
    """Teto de peso por papel: min(weight_cap, peso montável no AUM-alvo).

    peso_max = ADTV x participação_máxima x dias_para_montar / AUM_alvo
    (mesma fórmula de src/capacity.py, resolvida para o peso).
    """
    buildable = (
        adtv.reindex(tickers).fillna(0.0)
        * config.max_adtv_participation
        * config.days_to_build_position
        / config.target_aum_brl
    )
    return buildable.clip(upper=config.weight_cap)


def _size_tilt(tickers: pd.Index, market_cap: pd.Series | None) -> pd.Series:
    """sqrt(market cap); papel sem market cap recebe a mediana dos demais."""
    if market_cap is None:
        return pd.Series(1.0, index=tickers)
    mcap = market_cap.reindex(tickers)
    if mcap.notna().sum() == 0:
        return pd.Series(1.0, index=tickers)
    return np.sqrt(mcap.fillna(mcap.median()))


def _apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    weights = weights.copy()
    for _ in range(len(weights)):
        over_cap = weights > cap
        if not over_cap.any():
            break
        excess = (weights[over_cap] - cap).sum()
        weights[over_cap] = cap

        under_cap = ~over_cap
        if not under_cap.any() or weights[under_cap].sum() == 0:
            n = len(weights)
            weights[:] = 1.0 / n
            break
        weights[under_cap] += excess * (weights[under_cap] / weights[under_cap].sum())
    return weights / weights.sum()


def _apply_cap_per_name(weights: pd.Series, caps: pd.Series) -> pd.Series:
    """Como `_apply_cap`, mas com teto por papel e sem forçar soma 1.

    Se a soma dos tetos for menor que 1, a carteira não comporta 100%
    investido no AUM-alvo: cada papel fica no seu teto e o restante vira
    caixa (o backtest remunera `1 - soma dos pesos` a CDI).
    """
    weights = weights.copy()
    caps = caps.reindex(weights.index).fillna(0.0)
    if caps.sum() <= 1.0:
        return caps[caps > 0]

    for _ in range(len(weights)):
        over_cap = weights > caps + 1e-12
        if not over_cap.any():
            break
        excess = (weights[over_cap] - caps[over_cap]).sum()
        weights[over_cap] = caps[over_cap]
        room = (caps - weights).clip(lower=0.0)
        receivers = ~over_cap & (room > 0)
        if not receivers.any() or weights[receivers].sum() == 0:
            break
        weights[receivers] += excess * (weights[receivers] / weights[receivers].sum())
    return weights[weights > 0]
