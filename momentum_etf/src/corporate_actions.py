"""Tratamento de eventos corporativos dentro do backtest diário.

Proventos, desdobramentos e grupamentos já vêm incorporados na série de
retorno total ajustado (conforme o briefing) — nenhum ajuste adicional de
preço é necessário para esses eventos.

O evento que o backtest precisa tratar explicitamente é a SAÍDA de um
papel da base (delisting, incorporação, ou troca de código sem
mapeamento disponível): quando um ticker que está na carteira deixa de
ter retorno observável de um dia para o outro.

Escolha: a posição é liquidada ao último retorno disponível (sem penalidade
adicional — não presumimos fraude ou perda total) e o caixa resultante fica
alocado ao CDI (proxy de caixa remunerado) até o próximo rebalanceamento,
quando é redistribuído entre os nomes selecionados. Alternativa descartada:
redistribuir o caixa pro-rata entre as posições remanescentes no mesmo dia
— rejeitada por assumir implicitamente uma capacidade de execução imediata
que não existe operacionalmente (a mesa não sabe do delisting no fechamento
do mesmo pregão em que ele ocorre).

Limitação documentada: quando o motivo da saída é troca de código por
incorporação (não falência/OPA de fechamento), o ideal seria mapear o
código antigo para o novo e migrar a posição sem liquidar. A base fornecida
não inclui esse mapeamento, então tratamos toda saída de forma idêntica —
uma simplificação que pode subestimar levemente o retorno em casos de
incorporação vantajosa (troca de ações) e é sinalizada no relatório de
turnover como "saídas por delisting".
"""

from __future__ import annotations

import pandas as pd


def detect_delistings(
    returns_wide: pd.DataFrame,
    holdings: set[str],
    date: pd.Timestamp,
) -> set[str]:
    """Tickers em `holdings` que não têm retorno observável em `date`."""
    if date not in returns_wide.index:
        return set()
    day = returns_wide.loc[date]
    return {t for t in holdings if t not in day.index or pd.isna(day[t])}
