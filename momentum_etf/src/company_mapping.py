"""Mapeamento ticker (COTAHIST) <-> CNPJ (CVM/FRE), via nome da empresa.

Este é o elo mais frágil da cadeia de dados externos, e é tratado como tal:
- COTAHIST não traz CNPJ, só um `nome_empresa` truncado em 12 caracteres.
- FRE/CVM não traz ticker, só `CNPJ_Companhia` e `Nome_Companhia` (nome
  oficial, não abreviado).
- Não há chave exata em comum entre as duas fontes — ISIN foi cogitado
  (ver `caseNuAsset.ipynb`) mas o FRE não carrega ISIN, então também não
  serve de ponte.

A solução aqui é fuzzy matching de nome normalizado (maiúsculas, sem
acento, sem pontuação, sem sufixos societários como "S.A."/"S/A"/"LTDA"),
com pontuação de similaridade e um limiar mínimo de confiança
(`min_score`). Usa `rapidfuzz` se disponível (melhor qualidade, usado no
notebook exploratório original) e cai para `difflib` da stdlib caso
contrário — sempre funcional, mesmo sem a dependência extra instalada.

Limitação central, documentada e não escondida: nome truncado em 12
caracteres colide com frequência entre empresas do mesmo grupo econômico
(ex.: diferentes classes de ação da mesma controladora, ou uma holding e
sua subsidiária com nome parecido). Por isso o output inclui a pontuação
de similaridade e uma flag `matched_high_confidence` — qualquer decisão
que dependa desse mapeamento em produção precisaria de uma revisão manual
da lista de matches abaixo do limiar (exportada para QA em
`scripts/build_market_data.py`), não deste matching automático sozinho.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

try:
    from rapidfuzz import fuzz, process as rf_process

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercitado só sem a dependência opcional
    import difflib

    _HAS_RAPIDFUZZ = False

_LEGAL_SUFFIXES = [
    r"\bS A\b", r"\bSA\b", r"\bLTDA\b", r"\bME\b", r"\bEPP\b",
    r"\bHOLDING[S]?\b", r"\bPARTICIPACOES\b", r"\bPART\b", r"\bCIA\b",
    r"\bCOMPANHIA\b", r"\bEM RECUPERACAO JUDICIAL\b", r"\bEM LIQUIDACAO\b",
]


def normalize_company_name(name: str | float) -> str:
    if pd.isna(name):
        return ""
    text = str(name).upper()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in _LEGAL_SUFFIXES:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _best_match(query: str, choices: list[str], min_score: float) -> tuple[str | None, float]:
    if not query or not choices:
        return None, 0.0
    if _HAS_RAPIDFUZZ:
        result = rf_process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
        if result is None:
            return None, 0.0
        match, score, _ = result
        return (match, score) if score >= min_score else (None, score)
    else:  # difflib fallback: escala 0-100 como rapidfuzz para manter a mesma interface
        best_match, best_score = None, 0.0
        for choice in choices:
            score = difflib.SequenceMatcher(None, query, choice).ratio() * 100
            if score > best_score:
                best_match, best_score = choice, score
        return (best_match, best_score) if best_score >= min_score else (None, best_score)


def build_ticker_cnpj_mapping(
    cotahist_instruments: pd.DataFrame,
    cvm_companies: pd.DataFrame,
    min_score: float = 80.0,
) -> pd.DataFrame:
    """Casa cada ticker do COTAHIST com o CNPJ mais provável da base CVM.

    `cotahist_instruments`: colunas [ticker, nome_empresa] (ver
    `src/cotahist.py::instrument_reference`).
    `cvm_companies`: colunas [CNPJ_Companhia, Nome_Companhia] únicas por
    CNPJ (ex.: `capital_social_long[["CNPJ_Companhia","Nome_Companhia"]].drop_duplicates()`).

    Retorna: ticker, nome_b3, cnpj, nome_cvm, score, matched_high_confidence.
    Tickers sem nenhum candidato acima de `min_score` vêm com `cnpj=None`
    e devem ser tratados como "sem dado de tamanho/liquidez externo" pelo
    resto do pipeline (fallback documentado em `src/universe.py`).
    """
    cvm_unique = cvm_companies.drop_duplicates(subset=["CNPJ_Companhia"]).copy()
    cvm_unique["nome_norm"] = cvm_unique["Nome_Companhia"].apply(normalize_company_name)
    cvm_unique = cvm_unique[cvm_unique["nome_norm"] != ""]

    choices = cvm_unique["nome_norm"].tolist()
    cnpj_by_norm = dict(zip(cvm_unique["nome_norm"], cvm_unique["CNPJ_Companhia"]))
    nome_cvm_by_norm = dict(zip(cvm_unique["nome_norm"], cvm_unique["Nome_Companhia"]))

    rows = []
    for _, row in cotahist_instruments.iterrows():
        query = normalize_company_name(row["nome_empresa"])
        match, score = _best_match(query, choices, min_score)
        rows.append(
            {
                "ticker": row["ticker"],
                "nome_b3": row["nome_empresa"],
                "cnpj": cnpj_by_norm.get(match),
                "nome_cvm": nome_cvm_by_norm.get(match),
                "score": round(score, 1),
                "matched_high_confidence": match is not None,
            }
        )

    return pd.DataFrame(rows)
