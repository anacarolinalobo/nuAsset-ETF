"""Critério de seleção: reduz o universo elegível à carteira efetiva do índice.

Usa uma regra de banda ("buffer rule"), padrão em índices de momentum
(ex.: metodologia MSCI Momentum) para controlar turnover:
  - Um papel FORA da carteira só ENTRA se seu score estiver no top
    `1 - entry_percentile` do universo elegível (corte mais estreito).
  - Um papel JÁ NA carteira só SAI se seu score cair abaixo do corte mais
    largo `1 - hold_percentile` (hold_percentile < entry_percentile).

Sem essa banda, papéis que oscilam em torno de um único corte entram e
saem a cada rebalanceamento só por ruído do sinal, inflando turnover sem
ganho de retorno esperado — esse é o trade-off que a banda resolve.
"""

from __future__ import annotations

import pandas as pd

from src.config import MomentumConfig


def select_portfolio(
    signal: pd.Series,
    current_holdings: set[str],
    config: MomentumConfig,
) -> set[str]:
    """Aplica a banda de turnover e os limites min/max de nomes.

    `signal` já deve conter apenas tickers elegíveis na data de rebalance.
    """
    if signal.empty:
        return set()

    entry_cutoff = signal.quantile(config.entry_percentile)
    hold_cutoff = signal.quantile(config.hold_percentile)

    eligible_tickers = set(signal.index)
    still_eligible_holdings = current_holdings & eligible_tickers

    keep = {t for t in still_eligible_holdings if signal[t] >= hold_cutoff}
    new_entrants = {
        t for t in eligible_tickers - current_holdings if signal[t] >= entry_cutoff
    }

    selected = keep | new_entrants

    selected = _enforce_size_limits(selected, signal, config)
    return selected


def _enforce_size_limits(
    selected: set[str], signal: pd.Series, config: MomentumConfig
) -> set[str]:
    ranked = signal.loc[list(selected)].sort_values(ascending=False)

    if len(ranked) > config.max_names:
        ranked = ranked.iloc[: config.max_names]
        selected = set(ranked.index)

    if len(selected) < config.min_names:
        remaining = signal.drop(index=list(selected)).sort_values(ascending=False)
        n_needed = config.min_names - len(selected)
        selected = selected | set(remaining.iloc[:n_needed].index)

    return selected
