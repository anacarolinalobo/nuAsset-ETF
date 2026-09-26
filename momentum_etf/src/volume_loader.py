"""Volume financeiro negociado por ação, a partir dos arquivos COTAHIST da B3.

`acoes_retornos.csv` (o CSV do case) não traz volume — só retorno — então o
proxy de liquidez em `universe.py` usa presença de retorno diário como
substituto (ver docstring de `eligible_universe`). Este módulo lê o volume
financeiro NEGOCIADO de verdade (campo `VOLTOT` do layout histórico de
cotações da B3, "COTAHIST") para permitir um filtro de liquidez por ADTV
(average daily traded value) quando os arquivos brutos da B3 estiverem
disponíveis. Extraído e limpo da exploração feita em `caseNuAsset.ipynb`.

Os arquivos COTAHIST_A<ano>.TXT não fazem parte deste repositório (são
arquivos de texto de largura fixa, dezenas de MB por ano, publicados pela
própria B3) — quem for usar este módulo precisa baixá-los e apontar
`load_cotahist` para a pasta onde estiverem.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

# Layout oficial do arquivo histórico de cotações da B3 (registro tipo "01").
_COLSPECS = [
    (0, 2),  # tipo_registro
    (2, 10),  # data
    (10, 12),  # cod_bdi
    (12, 24),  # ticker
    (24, 27),  # tipo_mercado
    (27, 39),  # nome_empresa
    (39, 49),  # especificacao
    (49, 52),  # prazo_termo
    (52, 56),  # moeda
    (56, 69),  # preco_abertura
    (69, 82),  # preco_maximo
    (82, 95),  # preco_minimo
    (95, 108),  # preco_medio
    (108, 121),  # preco_ultimo
    (121, 134),  # preco_oferta_compra
    (134, 147),  # preco_oferta_venda
    (147, 152),  # numero_negocios
    (152, 170),  # quantidade_negociada
    (170, 188),  # volume_financeiro
    (188, 201),  # preco_exercicio
    (201, 202),  # indicador_correcao
    (202, 210),  # data_vencimento
    (210, 217),  # fator_cotacao
    (217, 230),  # pontos_exercicio
    (230, 242),  # isin
    (242, 245),  # numero_distribuicao
]

_COLUMNS = [
    "tipo_registro",
    "data",
    "cod_bdi",
    "ticker",
    "tipo_mercado",
    "nome_empresa",
    "especificacao",
    "prazo_termo",
    "moeda",
    "preco_abertura",
    "preco_maximo",
    "preco_minimo",
    "preco_medio",
    "preco_ultimo",
    "preco_oferta_compra",
    "preco_oferta_venda",
    "numero_negocios",
    "quantidade_negociada",
    "volume_financeiro",
    "preco_exercicio",
    "indicador_correcao",
    "data_vencimento",
    "fator_cotacao",
    "pontos_exercicio",
    "isin",
    "numero_distribuicao",
]

_PRICE_COLUMNS = [
    "preco_abertura",
    "preco_maximo",
    "preco_minimo",
    "preco_medio",
    "preco_ultimo",
    "preco_oferta_compra",
    "preco_oferta_venda",
    "preco_exercicio",
]

# Ações ON/PN/PNA/PNB/... negociadas no padrão XXXX3..XXXX8 (exclui units
# "11", BDRs "34"/"35" e outras classes fora do escopo de ações do índice).
_EQUITY_TICKER = re.compile(r"^[A-Z]{4}[3-8]$")
_MERCADO_A_VISTA = "010"
_TIPO_REGISTRO_COTACAO = "01"


def load_cotahist_file(path: str | Path) -> pd.DataFrame:
    """Lê um arquivo COTAHIST anual e devolve negociações à vista de ações.

    Filtra para o que interessa a um filtro de liquidez de ações do índice:
      - registros de cotação (exclui cabeçalho/rodapé do arquivo);
      - mercado à vista (exclui opções, termo, futuros);
      - tickers de ação ON/PN no padrão `XXXX3`..`XXXX8`.
    """
    df = pd.read_fwf(
        path,
        colspecs=_COLSPECS,
        names=_COLUMNS,
        encoding="latin1",
        dtype=str,
    )

    df = df[df["tipo_registro"] == _TIPO_REGISTRO_COTACAO].copy()
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["tipo_mercado"] = df["tipo_mercado"].str.strip()
    df = df[df["tipo_mercado"] == _MERCADO_A_VISTA]
    df = df[df["ticker"].str.match(_EQUITY_TICKER, na=False)]

    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d", errors="coerce")

    numeric_columns = _PRICE_COLUMNS + [
        "numero_negocios",
        "quantidade_negociada",
        "volume_financeiro",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Campos monetários vêm em centavos (2 casas decimais implícitas).
    for col in _PRICE_COLUMNS + ["volume_financeiro"]:
        df[col] = df[col] / 100

    df = df[df["data"].notna() & df["volume_financeiro"].notna()]

    return df[
        ["data", "ticker", "volume_financeiro", "quantidade_negociada", "numero_negocios"]
    ].reset_index(drop=True)


def load_cotahist(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Lê um ou mais arquivos COTAHIST e concatena em um único DataFrame.

    `paths` pode ser: um diretório (lê todo `COTAHIST_A*.TXT` dentro dele),
    um único arquivo, ou uma lista explícita de arquivos (ex.: um por ano).
    """
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        files = sorted(path.glob("COTAHIST_A*.TXT")) if path.is_dir() else [path]
    else:
        files = [Path(p) for p in paths]

    if not files:
        raise ValueError(f"Nenhum arquivo COTAHIST encontrado em: {paths!r}")

    frames = [load_cotahist_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["data", "ticker"]).reset_index(drop=True)


def build_volume_wide(cotahist: pd.DataFrame) -> pd.DataFrame:
    """Pivota negociações (long) para volume financeiro diário (date x ticker)."""
    wide = cotahist.pivot_table(
        index="data", columns="ticker", values="volume_financeiro", aggfunc="sum"
    ).sort_index()
    wide.index.name = "date"
    wide.columns.name = "ticker"
    return wide


def compute_adtv(volume_wide: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """ADTV (average daily traded value) móvel por ticker, série completa."""
    return volume_wide.rolling(window_days, min_periods=1).mean()


def adtv_on_date(
    volume_wide: pd.DataFrame,
    date: pd.Timestamp,
    window_days: int,
) -> pd.Series:
    """ADTV por ticker na `date`, usando só volume até essa data (sem look-ahead)."""
    window = volume_wide.loc[:date].tail(window_days)
    return window.mean(axis=0)
