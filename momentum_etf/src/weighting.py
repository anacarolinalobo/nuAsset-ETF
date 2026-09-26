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
"""

from __future__ import annotations

import pandas as pd

from src.config import MomentumConfig


def compute_weights(
    signal: pd.Series,
    selected: set[str],
    config: MomentumConfig,
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

    weights = scores / scores.sum()
    weights = _apply_cap(weights, config.weight_cap)
    return weights.sort_values(ascending=False)


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
