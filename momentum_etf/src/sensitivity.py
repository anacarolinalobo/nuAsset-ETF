"""Análise de sensibilidade a parâmetros e robustez fora da amostra.

Duas checagens complementares:

1. Grade de parâmetros (`parameter_grid_sensitivity`): reroda o backtest
   variando um parâmetro por vez (lookback, skip, corte de seleção, custo)
   e reporta Sharpe/retorno para cada valor. Se o resultado for muito
   sensível a escolhas "de calibração fina" (ex.: 252 vs 260 dias de
   lookback), é sinal de overfitting ao histórico — o time pediu
   explicitamente essa checagem ("Como você se protegeu de escolher os
   parâmetros que ficaram bonitos apenas no histórico?").

2. Janela expansível / walk-forward (`expanding_window_stability`):
   compara o Sharpe do índice calculado em sub-períodos sucessivos, para
   ver se o desempenho é estável ao longo do tempo ou concentrado em uma
   janela específica (ex.: só 2016-2018).
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pandas as pd

from src.backtest import run_backtest
from src.config import MomentumConfig
from src.metrics import annualized_return, sharpe_ratio, annualized_vol, max_drawdown


def parameter_grid_sensitivity(
    returns_wide: pd.DataFrame,
    cdi_daily: pd.Series,
    base_config: MomentumConfig,
    param_name: str,
    values: list,
) -> pd.DataFrame:
    rows = []
    for value in values:
        cfg = replace(base_config, **{param_name: value})
        result = run_backtest(returns_wide, cdi_daily, cfg)
        rows.append(
            {
                param_name: value,
                "retorno_anualizado": annualized_return(result.returns_net),
                "vol_anualizada": annualized_vol(result.returns_net),
                "sharpe": sharpe_ratio(result.returns_net, cdi_daily),
                "max_drawdown": max_drawdown(result.returns_net),
                "turnover_medio": result.turnover_history.mean(),
            }
        )
    return pd.DataFrame(rows)


def expanding_window_stability(
    returns_wide: pd.DataFrame,
    cdi_daily: pd.Series,
    config: MomentumConfig,
    n_splits: int = 4,
) -> pd.DataFrame:
    result = run_backtest(returns_wide, cdi_daily, config)
    returns = result.returns_net
    chunk_size = len(returns) // n_splits

    rows = []
    for i in range(n_splits):
        start = i * chunk_size
        end = len(returns) if i == n_splits - 1 else (i + 1) * chunk_size
        chunk = returns.iloc[start:end]
        if chunk.empty:
            continue
        rows.append(
            {
                "periodo": f"{chunk.index[0].date()} a {chunk.index[-1].date()}",
                "retorno_anualizado": annualized_return(chunk),
                "vol_anualizada": annualized_vol(chunk),
                "sharpe": sharpe_ratio(chunk, cdi_daily),
                "max_drawdown": max_drawdown(chunk),
            }
        )
    return pd.DataFrame(rows)
