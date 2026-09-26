"""Carrega o histórico de ações em circulação a partir do Formulário de
Referência (FRE) da CVM — dados públicos em dados.cvm.gov.br.

Espera a estrutura de pastas que o usuário já tem localmente:

    fre_cia_aberta_2010/fre_cia_aberta_capital_social_2010.csv
    fre_cia_aberta_2011/fre_cia_aberta_capital_social_2011.csv
    ...
    fre_cia_aberta_2026/fre_cia_aberta_capital_social_2026.csv

Cada arquivo é uma exportação bruta da CVM (';'-separado, latin1), com
uma linha por companhia por Data_Referencia por Tipo_Capital ("Capital
Emitido", "Capital Subscrito", etc.). Ficamos com uma linha por
(CNPJ, Data_Referencia): a mais recente publicada, priorizando o tipo de
capital mais próximo do "efetivamente em circulação" quando várias linhas
disputam a mesma data (ver `_pick_capital_type`).

Por que isso e não `acoes_retornos.csv`: a base de retornos não traz
nenhuma informação de tamanho da empresa. Sem `Quantidade_Total_Acoes`
(FRE) combinada com preço real (COTAHIST), não há como calcular valor de
mercado — só teríamos o retorno acumulado, que mede desempenho, não porte.
"""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Ordem de preferência quando mais de um "Tipo_Capital" aparece na mesma
# data de referência para a mesma empresa — capital integralizado é o que
# melhor aproxima ações efetivamente emitidas e em posse dos acionistas.
_TIPO_CAPITAL_PRIORIDADE = ["Capital Integralizado", "Capital Subscrito", "Capital Emitido"]

_ENCODINGS = ["latin1", "cp1252", "utf-8"]
_SEPARATORS = [";", ","]


def _read_csv_robusto(path: str | Path) -> pd.DataFrame:
    last_err = None
    for encoding in _ENCODINGS:
        for sep in _SEPARATORS:
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False, on_bad_lines="skip")
                if df.shape[1] > 1:
                    return df
            except Exception as exc:  # noqa: BLE001 - tentativa best-effort de leitura
                last_err = exc
    raise ValueError(f"Não foi possível ler {path}: {last_err}")


def load_capital_social(data_dir: str | Path, start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Concatena `fre_cia_aberta_capital_social_{ano}.csv` de `start_year` a `end_year`.

    Retorna um painel long: CNPJ_Companhia, Nome_Companhia, Data_Referencia,
    Quantidade_Total_Acoes (numérica, já com fallback via
    ordinárias+preferenciais quando a coluna total vier ausente/zerada).
    """
    frames = []
    for year in range(start_year, end_year + 1):
        pattern = str(Path(data_dir) / f"fre_cia_aberta_{year}" / f"fre_cia_aberta_capital_social_{year}.csv")
        matches = glob.glob(pattern)
        if not matches:
            logger.info("Nenhum arquivo de capital social para %d (%s)", year, pattern)
            continue
        try:
            df = _read_csv_robusto(matches[0])
            df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)
            df["ano_arquivo"] = year
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao ler capital social %d: %s", year, exc)

    if not frames:
        raise FileNotFoundError(
            f"Nenhum arquivo fre_cia_aberta_capital_social_*.csv encontrado em {data_dir} "
            f"para o intervalo {start_year}-{end_year}"
        )

    raw = pd.concat(frames, ignore_index=True, sort=False)
    return _clean_capital_social(raw)


def _clean_capital_social(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"CNPJ_Companhia", "Nome_Companhia", "Data_Referencia", "Quantidade_Total_Acoes"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Colunas esperadas ausentes em fre_cia_aberta_capital_social_*.csv: {missing}. "
            "Confira se o layout do arquivo real bate com o assumido aqui (ver docstring do módulo)."
        )

    df = raw.copy()
    df["CNPJ_Companhia"] = df["CNPJ_Companhia"].apply(_normalize_cnpj)
    df["Data_Referencia"] = pd.to_datetime(df["Data_Referencia"], errors="coerce")

    for col in ["Quantidade_Total_Acoes", "Quantidade_Acoes_Ordinarias", "Quantidade_Acoes_Preferenciais"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    if "Quantidade_Acoes_Ordinarias" in df.columns and "Quantidade_Acoes_Preferenciais" in df.columns:
        fallback = df["Quantidade_Acoes_Ordinarias"].fillna(0) + df["Quantidade_Acoes_Preferenciais"].fillna(0)
        df["Quantidade_Total_Acoes"] = df["Quantidade_Total_Acoes"].fillna(fallback)

    df = df.dropna(subset=["CNPJ_Companhia", "Data_Referencia", "Quantidade_Total_Acoes"])
    df = df[df["Quantidade_Total_Acoes"] > 0]

    if "Tipo_Capital" in df.columns:
        df["_prioridade"] = df["Tipo_Capital"].apply(
            lambda t: _TIPO_CAPITAL_PRIORIDADE.index(t) if t in _TIPO_CAPITAL_PRIORIDADE else len(_TIPO_CAPITAL_PRIORIDADE)
        )
    else:
        df["_prioridade"] = 0

    df = df.sort_values(["CNPJ_Companhia", "Data_Referencia", "_prioridade"])
    df = df.groupby(["CNPJ_Companhia", "Data_Referencia"], as_index=False).first()

    cols = ["CNPJ_Companhia", "Nome_Companhia", "Data_Referencia", "Quantidade_Total_Acoes"]
    if "Codigo_CVM" in df.columns:
        cols.append("Codigo_CVM")
    return df[cols].sort_values(["CNPJ_Companhia", "Data_Referencia"]).reset_index(drop=True)


def shares_outstanding_by_cnpj(capital_social_long: pd.DataFrame) -> dict[str, pd.Series]:
    """Série (Data_Referencia -> Quantidade_Total_Acoes) por CNPJ, para forward-fill diário."""
    result = {}
    for cnpj, group in capital_social_long.groupby("CNPJ_Companhia"):
        result[cnpj] = group.set_index("Data_Referencia")["Quantidade_Total_Acoes"].sort_index()
    return result


def _normalize_cnpj(value) -> str | None:
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(14) if digits else None
