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

import numpy as np
import pandas as pd


def estimate_capacity(
    weights: pd.Series,
    adtv_by_ticker: pd.Series,
    max_participation_rate: float = 0.10,
    days_to_build_position: int = 5,
) -> pd.Series:
    """AUM máximo implementável por papel, dado ADTV e taxa de participação.

    capacidade_papel = (ADTV * participação_máxima * dias) / peso_no_índice

    A capacidade do produto como um todo é o MÍNIMO entre os papéis — o
    nome mais ilíquido da carteira é o fator limitante (gargalo clássico
    de capacidade em estratégias small/mid cap).
    """
    aligned = pd.concat([weights, adtv_by_ticker], axis=1, keys=["peso", "adtv"]).dropna()
    aligned = aligned[aligned["peso"] > 0]

    capacity_per_name = (
        aligned["adtv"] * max_participation_rate * days_to_build_position
    ) / aligned["peso"]
    return capacity_per_name.sort_values()


def product_capacity(capacity_per_name: pd.Series) -> float:
    if capacity_per_name.empty:
        return np.nan
    return capacity_per_name.min()
