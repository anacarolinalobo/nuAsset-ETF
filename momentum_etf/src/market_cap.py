"""Combina preço (COTAHIST) + ações em circulação (FRE) + mapeamento
ticker<->CNPJ (company_mapping) em uma série de valor de mercado diário
por ticker.

    market_cap(t, ticker) = preco_ultimo(t, ticker) * shares_outstanding(t, cnpj)

`shares_outstanding` é publicado em datas esparsas (a cada nova versão do
FRE, tipicamente trimestral/quando há alteração de capital) — é
propagado (forward-fill) até a próxima publicação, já que a contagem de
ações não muda diariamente fora de eventos corporativos (que por sua vez
já afetam o preço/retorno ajustado, não a contagem nominal de ações desse
jeito simplificado).
"""

from __future__ import annotations

import pandas as pd

from src.fre_capital_social import shares_outstanding_by_cnpj


def build_market_cap(
    close_price_wide: pd.DataFrame,
    capital_social_long: pd.DataFrame,
    ticker_cnpj_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Valor de mercado diário (date x ticker), só para tickers com CNPJ casado."""
    shares_by_cnpj = shares_outstanding_by_cnpj(capital_social_long)
    matched = ticker_cnpj_mapping[ticker_cnpj_mapping["matched_high_confidence"]]

    market_cap = pd.DataFrame(index=close_price_wide.index)
    for _, row in matched.iterrows():
        ticker, cnpj = row["ticker"], row["cnpj"]
        if ticker not in close_price_wide.columns or cnpj not in shares_by_cnpj:
            continue

        shares_series = shares_by_cnpj[cnpj].reindex(close_price_wide.index, method="ffill")
        market_cap[ticker] = close_price_wide[ticker] * shares_series

    return market_cap.sort_index()
