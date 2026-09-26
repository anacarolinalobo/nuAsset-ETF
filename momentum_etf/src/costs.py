"""Custos de transação estimados a partir do turnover.

Um único parâmetro em bps por perna (`transaction_cost_bps`) representando
a soma de corretagem + emolumentos B3 + spread bid-ask médio + impacto de
mercado esperado para o book de nomes elegíveis (tipicamente mid/small
caps mais ilíquidas que os componentes do Ibovespa). É uma simplificação
deliberada — custo de execução real varia por nome e por tamanho de
ordem — documentada no README como limitação, com a análise de
sensibilidade (src/sensitivity.py) testando o range 10-50 bps para checar
se as conclusões dependem desse número específico.
"""

from __future__ import annotations

import pandas as pd


def turnover_cost(turnover: float, transaction_cost_bps: float) -> float:
    """Custo em fração de retorno para um turnover (soma de |Δpeso|/2)."""
    return turnover * (transaction_cost_bps / 10_000.0)
