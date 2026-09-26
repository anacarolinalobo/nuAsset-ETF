"""Estimativa de capacidade do produto (AUM implementável na B3).

Os 3 CSVs do case trazem apenas RETORNOS, sem volume financeiro
negociado — capacidade de um índice de momentum depende diretamente de
ADTV (average daily traded value) por papel, que não vem ali. Isso foi
resolvido via `src/cotahist.py` (parser dos arquivos de pregão da B3,
usado também no filtro de liquidez em `universe.py`): quando
`scripts/build_market_data.py` já rodou, `data/derived/adtv.csv` traz
ADTV real por ticker/data, e é isso que `app.py` passa para
`estimate_capacity` — não é mais uma estimativa assumida, é ADTV
observado. Sem esse arquivo (COTAHIST não fornecido), a função ainda
aceita uma série de ADTV assumida manualmente por faixa de liquidez,
deixado explícito na UI que é uma estimativa de ordem de grandeza nesse
caso, não um número validado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CapacityResult:
    capacity_per_name: pd.Series   # ticker -> capacidade (R$), só nomes com ADTV utilizável
    product_capacity: float        # mínimo de capacity_per_name (gargalo); NaN se nenhum nome utilizável
    uncovered_tickers: pd.Series   # ticker -> peso, para nomes sem ADTV utilizável (ver docstring)
    uncovered_weight: float        # soma dos pesos sem ADTV utilizável


def estimate_capacity(
    weights: pd.Series,
    adtv_by_ticker: pd.Series,
    max_participation_rate: float = 0.10,
    days_to_build_position: int = 5,
    min_usable_adtv: float = 1.0,
) -> CapacityResult:
    """AUM máximo implementável, dado ADTV e taxa de participação por papel.

    capacidade_papel = (ADTV * participação_máxima * dias) / peso_no_índice

    A capacidade do PRODUTO é o MÍNIMO entre os papéis — o nome mais
    ilíquido da carteira é o fator limitante (gargalo clássico de
    capacidade em estratégias small/mid cap). É justamente por isso que
    ADTV ausente ou zerada precisa de tratamento cuidadoso: um único
    papel com ADTV=0 zeraria a capacidade do PRODUTO INTEIRO, mesmo que
    os demais 30-60 nomes tenham liquidez ótima.

    ADTV <= `min_usable_adtv` é tratado como "sem cobertura confiável de
    dado", não como "zero liquidez real" — na prática, isso costuma
    significar que o COTAHIST fornecido não cobre aquele ticker ou não
    chega até a data de referência, não que o papel parou de negociar de
    verdade (um papel realmente ilíquido a esse ponto muito provavelmente
    já teria sido excluído pelo filtro de liquidez em `universe.py` antes
    de entrar na carteira). Esses nomes são EXCLUÍDOS do cálculo do
    gargalo e devolvidos à parte (`uncovered_tickers`, `uncovered_weight`)
    — o chamador decide como comunicar isso, mas nunca fica escondido
    atrás de um "capacidade = R$ 0" que mistura os dois problemas
    (iliquidez real vs. buraco de cobertura de dado) num único número.
    """
    aligned = pd.concat([weights, adtv_by_ticker], axis=1, keys=["peso", "adtv"])
    aligned = aligned[aligned["peso"] > 0]

    is_usable = aligned["adtv"].fillna(0.0) > min_usable_adtv
    usable = aligned[is_usable]
    uncovered = aligned[~is_usable]

    capacity_per_name = (
        (usable["adtv"] * max_participation_rate * days_to_build_position) / usable["peso"]
    ).sort_values()

    return CapacityResult(
        capacity_per_name=capacity_per_name,
        product_capacity=capacity_per_name.min() if not capacity_per_name.empty else np.nan,
        uncovered_tickers=uncovered["peso"].sort_values(ascending=False),
        uncovered_weight=float(uncovered["peso"].sum()) if not uncovered.empty else 0.0,
    )
