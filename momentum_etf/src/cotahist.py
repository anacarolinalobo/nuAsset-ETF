"""Parser dos arquivos de pregão da B3 (layout COTAHIST, registro tipo 01).

Fonte de PREÇO DE FECHAMENTO real (`preco_ultimo`) e de VOLUME FINANCEIRO
NEGOCIADO (`volume_financeiro`) — nenhum dos dois está nos 3 CSVs do case
(que só trazem retorno). É a partir daqui que os módulos `market_cap.py`
(preço × ações em circulação) e a ADTV do filtro de liquidez em
`universe.py` são construídos.

Layout fixo (colunas em posição de caractere), conforme a especificação
pública da B3 e conforme já validado manualmente em `caseNuAsset.ipynb`:
tipo_registro(2) data(8) cod_bdi(2) ticker(12) tipo_mercado(3)
nome_empresa(12) especificacao(10) prazo_termo(3) moeda(4)
preco_abertura(13) preco_maximo(13) preco_minimo(13) preco_medio(13)
preco_ultimo(13) preco_oferta_compra(13) preco_oferta_venda(13)
numero_negocios(5) quantidade_negociada(18) volume_financeiro(18)
preco_exercicio(13) indicador_correcao(1) data_vencimento(8)
fator_cotacao(7) pontos_exercicio(13) isin(12) numero_distribuicao(3).

Campos monetários e de volume vêm sem separador decimal (2 casas
implícitas) — divididos por 100 aqui. Só o registro tipo "01"
(negociação a vista/padrão) é mantido; tipo "99" (trailer) e outros
tipos de mercado (opções, termo) são descartados por padrão.

Aviso: layouts de anos muito antigos ou arquivos já em formato novo
("COTAHIST" pós-2023 tem uma variante de campo largo) podem divergir
ligeiramente destas posições — se a leitura de um ano específico vier com
muitas linhas nulas/zeradas, confira esse arquivo contra o layout oficial
antes de confiar no resultado (checagem sugerida em
`scripts/build_market_data.py`).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COLSPECS = [
    (0, 2), (2, 10), (10, 12), (12, 24), (24, 27), (27, 39), (39, 49),
    (49, 52), (52, 56), (56, 69), (69, 82), (82, 95), (95, 108), (108, 121),
    (121, 134), (134, 147), (147, 152), (152, 170), (170, 188), (188, 201),
    (201, 202), (202, 210), (210, 217), (217, 230), (230, 242), (242, 245),
]

_NAMES = [
    "tipo_registro", "data", "cod_bdi", "ticker", "tipo_mercado",
    "nome_empresa", "especificacao", "prazo_termo", "moeda",
    "preco_abertura", "preco_maximo", "preco_minimo", "preco_medio",
    "preco_ultimo", "preco_oferta_compra", "preco_oferta_venda",
    "numero_negocios", "quantidade_negociada", "volume_financeiro",
    "preco_exercicio", "indicador_correcao", "data_vencimento",
    "fator_cotacao", "pontos_exercicio", "isin", "numero_distribuicao",
]

_PRICE_COLUMNS = [
    "preco_abertura", "preco_maximo", "preco_minimo", "preco_medio",
    "preco_ultimo", "preco_oferta_compra", "preco_oferta_venda",
    "preco_exercicio",
]

# tipo_mercado == "010" é o mercado à vista (mesmo escopo de
# acoes_retornos.csv); opções e termo (outros códigos) são descartados.
MERCADO_A_VISTA = "010"


def parse_cotahist_file(path: str | Path, apenas_mercado_a_vista: bool = True) -> pd.DataFrame:
    """Lê um único arquivo COTAHIST_A{ano}.TXT e devolve um DataFrame long."""
    df = pd.read_fwf(path, colspecs=_COLSPECS, names=_NAMES, encoding="latin1", dtype=str)
    df = df[df["tipo_registro"] == "01"].copy()

    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d", errors="coerce")
    df["ticker"] = df["ticker"].str.strip()
    df["nome_empresa"] = df["nome_empresa"].str.strip()
    df["especificacao"] = df["especificacao"].str.strip()
    df["isin"] = df["isin"].str.strip()
    df["tipo_mercado"] = df["tipo_mercado"].str.strip()

    numeric_cols = _PRICE_COLUMNS + ["numero_negocios", "quantidade_negociada", "volume_financeiro"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in _PRICE_COLUMNS:
        df[col] = df[col] / 100.0
    df["volume_financeiro"] = df["volume_financeiro"] / 100.0

    if apenas_mercado_a_vista:
        df = df[df["tipo_mercado"] == MERCADO_A_VISTA]

    # Ações ON/PN líquidas (evita units, BDRs, fracionário no ticker-base
    # que já viria misturado se não filtrado; fracionário tem sufixo "F"
    # anexado ao ticker em alguns layouts — mantemos só terminação 3-8).
    df = df[df["ticker"].str.fullmatch(r"[A-Z]{4}\d{1,2}", na=False)]

    keep = [
        "data", "ticker", "nome_empresa", "especificacao", "isin",
        "preco_ultimo", "quantidade_negociada", "volume_financeiro",
        "numero_negocios",
    ]
    return df[keep].rename(columns={"data": "date"}).reset_index(drop=True)


def parse_cotahist_directory(directory: str | Path, pattern: str = "COTAHIST_A*.TXT") -> pd.DataFrame:
    """Lê e concatena todos os arquivos COTAHIST de um diretório."""
    files = sorted(glob.glob(str(Path(directory) / pattern)))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo '{pattern}' encontrado em {directory}")

    frames = []
    for f in files:
        try:
            frames.append(parse_cotahist_file(f))
            logger.info("Lido %s: %d registros", f, len(frames[-1]))
        except Exception as exc:
            logger.warning("Falha ao ler %s: %s", f, exc)

    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])


def instrument_reference(cotahist_long: pd.DataFrame) -> pd.DataFrame:
    """Um registro por ticker (nome/ISIN mais recentes) — insumo do matching CNPJ."""
    return (
        cotahist_long.sort_values("date")
        .groupby("ticker")
        .last()[["nome_empresa", "especificacao", "isin"]]
        .reset_index()
    )


def close_price_wide(cotahist_long: pd.DataFrame) -> pd.DataFrame:
    """Preço de fechamento em formato wide (date x ticker)."""
    return cotahist_long.pivot(index="date", columns="ticker", values="preco_ultimo").sort_index()


def traded_value_wide(cotahist_long: pd.DataFrame) -> pd.DataFrame:
    """Volume financeiro negociado em formato wide (date x ticker)."""
    return cotahist_long.pivot(index="date", columns="ticker", values="volume_financeiro").sort_index()


def compute_adtv(traded_value: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """ADTV (average daily traded value) móvel de `window_days` pregões.

    Dias sem negociação (NaN) contam como zero de volume — não são
    excluídos da média —, para que um papel que passou a negociar pouco
    veja a ADTV cair de verdade, em vez de a média ser calculada só sobre
    os dias em que houve pregão.
    """
    return traded_value.fillna(0.0).rolling(window_days, min_periods=1).mean()
