"""Atribuição de performance do índice contra o Ibovespa.

Usa uma decomposição de contribuição simples (não um Brinson setorial
completo, já que a base não traz classificação setorial): para cada
rebalanceamento, decompõe o excesso de retorno acumulado até o próximo
rebalanceamento em

    excesso = efeito_alocação + efeito_seleção + interação

onde, tomando o Ibovespa como benchmark de pesos w_b e retornos r_b, e o
índice como w_p e r_p, por ativo i:

    alocação_i = (w_p_i - w_b_i) * r_b_i
    seleção_i  = w_b_i * (r_p_i - r_b_i)
    interação_i = (w_p_i - w_b_i) * (r_p_i - r_b_i)

Interpretação: alocação mede o ganho/perda por pesar diferente do
Ibovespa em nomes que o Ibovespa já carrega; seleção mede o ganho/perda
por deter nomes fora do Ibovespa ou por captar retorno diferente do
mesmo nome (aqui, como ambos usam o mesmo retorno de mercado por ticker,
seleção_i colapsa ao efeito de deter ativos fora do índice de referência).
"""

from __future__ import annotations

import pandas as pd


def period_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    period_returns: pd.Series,
) -> pd.Series:
    """Atribuição de contribuição para um único período entre rebalances.

    `period_returns` deve conter o retorno acumulado de cada ticker no
    período, já alinhado ao universo unido de portfolio e benchmark.
    """
    all_tickers = portfolio_weights.index.union(benchmark_weights.index).union(
        period_returns.index
    )
    w_p = portfolio_weights.reindex(all_tickers, fill_value=0.0)
    w_b = benchmark_weights.reindex(all_tickers, fill_value=0.0)
    r = period_returns.reindex(all_tickers, fill_value=0.0)

    r_b_avg = (w_b * r).sum() / w_b.sum() if w_b.sum() > 0 else 0.0
    r_p_avg = (w_p * r).sum() / w_p.sum() if w_p.sum() > 0 else 0.0

    allocation = ((w_p - w_b) * (r - r_b_avg)).sum()
    selection = (w_b * 0).sum()  # mesmo retorno de mercado por ticker nos dois lados
    total_excess = (w_p * r).sum() - (w_b * r).sum()
    interaction = total_excess - allocation - selection

    return pd.Series(
        {
            "excesso_total": total_excess,
            "efeito_alocacao": allocation,
            "efeito_selecao": selection,
            "efeito_interacao": interaction,
        }
    )


def attribution_history(
    weights_history: dict[pd.Timestamp, pd.Series],
    ibov_weights_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Roda `period_attribution` em cada janela entre rebalanceamentos."""
    dates = sorted(weights_history.keys())
    rows = []
    for i, date in enumerate(dates):
        period_end = dates[i + 1] if i + 1 < len(dates) else returns_wide.index[-1]
        period_mask = (returns_wide.index > date) & (returns_wide.index <= period_end)
        period_returns = (1 + returns_wide.loc[period_mask]).prod() - 1

        bench_weights = (
            ibov_weights_wide.loc[date] if date in ibov_weights_wide.index else ibov_weights_wide.reindex([date], method="ffill").iloc[0]
        )

        row = period_attribution(weights_history[date], bench_weights, period_returns)
        row.name = date
        rows.append(row)

    return pd.DataFrame(rows)
