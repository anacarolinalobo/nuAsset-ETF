"""Carga e limpeza dos três CSVs fornecidos no case.

Formatos esperados (conforme o briefing):
    acoes_retornos.csv     : date, ticker, retorno   (long)
    ibov_composicao.csv    : date, ticker, peso       (long)
    benchmarks_diarios.csv : date, cdi, ima_s, idka_pre_3a, ima_b, ihfa,
                              ifix, ibovespa, sp500_brl, bitcoin_brl (wide)

Nenhuma função aqui decide metodologia de índice — só carrega, valida e
padroniza o formato dos dados brutos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_returns(path: str | Path) -> pd.DataFrame:
    """Lê acoes_retornos.csv e devolve retornos em formato wide (date x ticker)."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "retorno"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"acoes_retornos.csv sem colunas obrigatórias: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    wide = df.pivot(index="date", columns="ticker", values="retorno").sort_index()
    wide.index.name = "date"
    wide.columns.name = "ticker"
    return wide


def load_ibov_composition(path: str | Path) -> pd.DataFrame:
    """Lê ibov_composicao.csv e devolve pesos em formato wide (date x ticker)."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ibov_composicao.csv sem colunas obrigatórias: {missing}")

    weight_col = "peso" if "peso" in df.columns else "weight"
    if weight_col not in df.columns:
        raise ValueError("ibov_composicao.csv precisa de coluna 'peso' ou 'weight'")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    wide = df.pivot(index="date", columns="ticker", values=weight_col).sort_index()
    wide.index.name = "date"
    wide.columns.name = "ticker"
    return wide.fillna(0.0)


def load_benchmarks(path: str | Path) -> pd.DataFrame:
    """Lê benchmarks_diarios.csv (já wide), padroniza nomes e devolve retornos diários."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df.columns = [_normalize_colname(c) for c in df.columns]
    # O CSV real traz NÍVEIS (CDI acumulado, Ibovespa em pontos), não
    # retornos. Coluna com mediana |x| > 1 não pode ser retorno diário:
    # converte para variação diária.
    for col in df.columns:
        values = df[col].dropna()
        if not values.empty and values.abs().median() > 1:
            df[col] = df[col].pct_change(fill_method=None)
    return df


def _normalize_colname(name: str) -> str:
    name = name.strip().lower()
    replacements = {
        "ibovespa": "ibovespa",
        "ibov": "ibovespa",
        "cdi": "cdi",
        "ima-s": "ima_s",
        "imas": "ima_s",
        "idka pré 3a": "idka_pre_3a",
        "idka pre 3a": "idka_pre_3a",
        "idka_pre_3a": "idka_pre_3a",
        "ima-b": "ima_b",
        "imab": "ima_b",
        "ihfa": "ihfa",
        "ifix": "ifix",
        "s&p 500": "sp500_brl",
        "sp500": "sp500_brl",
        "s&p500": "sp500_brl",
        "bitcoin": "bitcoin_brl",
        "btc": "bitcoin_brl",
    }
    key = name.replace("_", " ").strip()
    return replacements.get(key, replacements.get(name, name.replace(" ", "_")))


def flag_suspicious_returns(
    returns_wide: pd.DataFrame,
    max_abs_daily_return: float = 1.00,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identifica possíveis erros de cotação sem alterar o dado silenciosamente.

    Retorna (df_flags, resumo). `df_flags` tem uma linha por observação
    suspeita, com o tipo de suspeita. O chamador decide o que fazer — a
    metodologia por padrão apenas EXCLUI o dia do universo elegível daquele
    papel via winsorização em `signal.py`, nunca "conserta" o número.

    Duas heurísticas simples, documentadas como limitação (não substituem
    checagem manual contra fonte primária):
      1. |retorno diário| acima de `max_abs_daily_return` (ex.: >100% em um
         dia) — quase sempre erro de cotação ou evento não capturado no
         ajuste, já que a base afirma vir "ajustada por proventos,
         desdobramentos e grupamentos".
      2. Reversão artificial: um retorno extremo seguido, no pregão
         seguinte, por um retorno próximo de -r/(1+r) — a assinatura
         clássica de um split/agrupamento não ajustado corretamente.
    """
    flags = []

    extreme = returns_wide.abs() > max_abs_daily_return
    if extreme.to_numpy().any():
        stacked = returns_wide.where(extreme).stack()
        for (date, ticker), value in stacked.items():
            flags.append(
                {"date": date, "ticker": ticker, "retorno": value, "tipo": "retorno_extremo"}
            )

    shifted = returns_wide.shift(-1)
    implied_reversal = -returns_wide / (1 + returns_wide)
    reversal_mask = (
        (returns_wide.abs() > 0.30)
        & ((shifted - implied_reversal).abs() < 0.02)
    )
    if reversal_mask.to_numpy().any():
        stacked = returns_wide.where(reversal_mask).stack()
        for (date, ticker), value in stacked.items():
            flags.append(
                {"date": date, "ticker": ticker, "retorno": value, "tipo": "possivel_split_nao_ajustado"}
            )

    df_flags = pd.DataFrame(flags)
    if df_flags.empty:
        resumo = pd.DataFrame(columns=["tipo", "n_ocorrencias"])
    else:
        resumo = (
            df_flags.groupby("tipo").size().rename("n_ocorrencias").reset_index()
        )
    return df_flags, resumo


def align_calendars(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Reindexa uma lista de DataFrames para a união de seus índices de data."""
    common_index = frames[0].index
    for f in frames[1:]:
        common_index = common_index.union(f.index)
    common_index = common_index.sort_values()
    return [f.reindex(common_index) for f in frames]
