"""Dados de liquidez, volume e market cap por ticker (B3 COTAHIST + CVM).

Complementa `acoes_retornos.csv` com as três variáveis que a base do case
não traz. O arquivo é gerado pela última célula de `caseNuAsset.ipynb` a
partir do COTAHIST da B3 (volume financeiro diário) e do capital social
informado à CVM (quantidade de ações x preço de fechamento = market cap).

Formato esperado (long):
    liquidez_mercado.csv : date, ticker, volume_financeiro, market_cap

    - volume_financeiro: R$ negociados no dia (0 ou ausente = não negociou).
    - market_cap: R$ (quantidade de ações da classe x preço de fechamento).
      Pode vir diário ou só em algumas datas (ex.: fim de mês); é
      propagado para frente até a próxima observação.

Toda consulta é feita "até a data" — nenhum valor posterior à data de
cálculo do rebalance entra na decisão, a mesma disciplina de
`src/universe.py` contra look-ahead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketData:
    """Painéis wide (date x ticker) de volume financeiro e market cap."""

    volume: pd.DataFrame
    market_cap: pd.DataFrame

    def adtv(self, date: pd.Timestamp, window_days: int) -> pd.Series:
        """Mediana do volume financeiro diário (R$) nos últimos `window_days` pregões.

        Dias sem negociação contam como zero — um papel que negocia R$ 50 mi
        em um dia e nada nos outros 62 não é líquido. A mediana (e não a
        média) evita que um único bloco de negociação infle a liquidez.
        """
        window = self.volume.loc[:date].tail(window_days)
        if window.empty:
            return pd.Series(dtype=float)
        return window.fillna(0.0).median(axis=0)

    def traded_ratio(self, date: pd.Timestamp, window_days: int) -> pd.Series:
        """Fração de pregões com volume > 0 nos últimos `window_days` pregões."""
        window = self.volume.loc[:date].tail(window_days)
        if window.empty:
            return pd.Series(dtype=float)
        return (window.fillna(0.0) > 0).mean(axis=0)

    def market_cap_on(self, date: pd.Timestamp) -> pd.Series:
        """Último market cap conhecido até `date` (NaN se nunca observado)."""
        history = self.market_cap.loc[:date]
        if history.empty:
            return pd.Series(dtype=float)
        return history.ffill().iloc[-1]


def load_market_data(
    path: str | Path,
    trading_days: pd.DatetimeIndex | None = None,
) -> MarketData:
    """Lê liquidez_mercado.csv (ou .parquet) e devolve um `MarketData`.

    Se `trading_days` for passado (normalmente o índice de
    `acoes_retornos.csv`), o volume é reindexado para esse calendário
    (pregão sem registro = volume zero) e o market cap é alinhado a ele por
    forward-fill — assim um market cap mensal vira uma série diária
    consistente com os retornos.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["date"])

    required = {"date", "ticker", "volume_financeiro"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} sem colunas obrigatórias: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    volume = _to_wide(df, "volume_financeiro")
    if "market_cap" in df.columns:
        market_cap = _to_wide(df.dropna(subset=["market_cap"]), "market_cap")
    else:
        logger.warning("%s sem coluna market_cap — filtro de tamanho desligado", path.name)
        market_cap = pd.DataFrame(index=volume.index, dtype=float)

    market_cap = market_cap.where(market_cap > 0)

    if trading_days is not None:
        volume = volume.reindex(trading_days).fillna(0.0)
        market_cap = (
            market_cap.reindex(market_cap.index.union(trading_days))
            .sort_index()
            .ffill()
            .reindex(trading_days)
        )

    return MarketData(volume=volume, market_cap=market_cap)


def coverage_report(market_data: MarketData, returns_wide: pd.DataFrame) -> pd.Series:
    """Quantos tickers de `acoes_retornos.csv` têm volume e market cap."""
    tickers = pd.Index(returns_wide.columns)
    has_volume = tickers.isin(market_data.volume.columns)
    has_mcap = tickers.isin(market_data.market_cap.dropna(axis=1, how="all").columns)
    return pd.Series(
        {
            "tickers_retorno": len(tickers),
            "com_volume": int(has_volume.sum()),
            "com_market_cap": int(has_mcap.sum()),
            "cobertura_volume": float(has_volume.mean()) if len(tickers) else np.nan,
            "cobertura_market_cap": float(has_mcap.mean()) if len(tickers) else np.nan,
        }
    )


def _to_wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    wide = df.pivot(index="date", columns="ticker", values=column).sort_index()
    wide.index.name = "date"
    wide.columns.name = "ticker"
    return wide.astype(float)
