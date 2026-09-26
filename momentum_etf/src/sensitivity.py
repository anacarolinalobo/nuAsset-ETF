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

3. Busca em grade com validação treino/teste (`train_test_grid_search`):
   otimiza múltiplos parâmetros simultaneamente na primeira metade da
   amostra (treino) e mede, SEM reotimizar, o Sharpe da combinação vencedora
   na segunda metade (teste) — comparando contra o Sharpe do modelo DEFAULT
   (nunca ajustado) no mesmo período de teste. Responde à pergunta que a
   checagem (1) não responde sozinha: mesmo que nenhum parâmetro isolado
   pareça "frágil", a combinação escolhida pela grade pode ainda assim estar
   ajustada a ruído específico do treino — só um teste cego fora da amostra
   revela isso.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import replace
from typing import Any

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


def train_test_grid_search(
    returns_wide: pd.DataFrame,
    cdi_daily: pd.Series,
    base_config: MomentumConfig,
    grid: dict[str, list],
) -> dict[str, Any]:
    """Otimiza `grid` no treino (1ª metade da amostra) e valida no teste (2ª).

    Roda UM backtest por combinação da grade (cobrindo o período inteiro de
    `base_config.backtest_start` a `backtest_end`) e calcula o Sharpe de cada
    combinação separadamente nas duas metades da série resultante — não é
    necessário rodar o backtest duas vezes por combinação porque o motor já
    respeita a disciplina de não-look-ahead (o sinal em cada rebalance usa só
    dado até a própria data de cálculo), então o trecho de teste de uma
    combinação treinada "vendo" só o desempenho do treino é uma validação
    cega legítima.

    A combinação vencedora (maior Sharpe de TREINO) é comparada, no teste,
    contra `base_config` sem nenhum ajuste — se o modelo "otimizado" não bate
    o default fora da amostra, é evidência de que a grade capturou ruído do
    treino, não um padrão robusto (overfitting de hiperparâmetros).
    """
    trading_days = returns_wide.index[
        (returns_wide.index >= pd.Timestamp(base_config.backtest_start))
        & (returns_wide.index <= pd.Timestamp(base_config.backtest_end))
    ]
    split_idx = len(trading_days) // 2
    train_start, train_end = trading_days[0], trading_days[split_idx - 1]
    test_start, test_end = trading_days[split_idx], trading_days[-1]

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))

    rows = []
    results_by_combo: dict[tuple, "pd.Series"] = {}
    for combo in combos:
        overrides = dict(zip(keys, combo))
        cfg = replace(base_config, **overrides)
        result = run_backtest(returns_wide, cdi_daily, cfg)
        results_by_combo[combo] = result.returns_net
        rows.append(
            {
                **overrides,
                "sharpe_treino": sharpe_ratio(
                    result.returns_net.loc[train_start:train_end], cdi_daily
                ),
                "sharpe_teste": sharpe_ratio(
                    result.returns_net.loc[test_start:test_end], cdi_daily
                ),
            }
        )
    grid_results = pd.DataFrame(rows)

    best_idx = grid_results["sharpe_treino"].idxmax()
    best_row = grid_results.loc[best_idx]
    best_params = {k: best_row[k] for k in keys}

    default_returns = run_backtest(returns_wide, cdi_daily, base_config).returns_net
    sharpe_teste_default = sharpe_ratio(default_returns.loc[test_start:test_end], cdi_daily)

    sharpe_treino_otimizado = float(best_row["sharpe_treino"])
    sharpe_teste_otimizado = float(best_row["sharpe_teste"])

    return {
        "grid_results": grid_results,
        "train_period": (train_start, train_end),
        "test_period": (test_start, test_end),
        "best_params": best_params,
        "sharpe_treino_otimizado": sharpe_treino_otimizado,
        "sharpe_teste_otimizado": sharpe_teste_otimizado,
        "sharpe_teste_default": float(sharpe_teste_default),
        "gap_treino_teste": sharpe_treino_otimizado - sharpe_teste_otimizado,
        "otimizado_bate_default_no_teste": bool(sharpe_teste_otimizado > sharpe_teste_default),
    }
