"""Estimativa de capacidade do produto (AUM implementável na B3).

Limitação relevante: a base fornecida traz apenas RETORNOS, sem volume
financeiro negociado. Capacidade de um índice de momentum depende
diretamente de ADTV (average daily traded value) por papel, que não temos.

Duas opções foram consideradas:
  1. Buscar volume histórico via fonte externa (ex.: dados de pregão B3 /
     provedores de mercado) e casar por ticker/data.
  2. Parametrizar um ADTV assumido por faixa de liquidez (conservador) e
     expor a fórmula de forma transparente, deixando explícito que é uma
     estimativa de ordem de grandeza, não um número validado.

Optamos pela opção 2 para este entregável — a integração de volume real é
listada no README como próximo passo natural, não coberta por restrição de
tempo/dado, e a fórmula abaixo foi escrita para aceitar um DataFrame de
ADTV real assim que disponível, sem mudar a interface.
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
