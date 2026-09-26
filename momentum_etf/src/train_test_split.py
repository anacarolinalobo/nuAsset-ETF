"""Split formal treino/validação, com otimização cega de hiperparâmetros.

Fecha a limitação declarada no README ("teste out-of-sample formal com
otimização cega" ficou de fora da primeira entrega). Os parâmetros
default do índice foram fixados por julgamento e literatura ANTES de
qualquer contato com dado real nesta sessão — o que já evita overfitting
*ativo* (nunca escolhi parâmetro olhando o Sharpe que ele produzia), mas
não é o mesmo que uma validação formal fora da amostra. Este módulo
implementa essa validação.

Protocolo:
  1. Divide o período de backtest (`config.backtest_start` a
     `config.backtest_end`) ao meio por NÚMERO DE PREGÕES (não por
     calendário, para as duas metades terem tamanho de amostra
     comparável) — metade "treino", metade "teste".
  2. Varre uma grade de hiperparâmetros SÓ NO TREINO, escolhendo a
     combinação de maior Sharpe (ou Calmar, configurável via
     `objective`) ali. "Cega" porque nenhuma informação do período de
     teste entra nessa escolha — a grade só varia parâmetros que são
     escolhas de METODOLOGIA (janela do sinal, corte de seleção,
     frequência de rebalance), nunca a premissa de custo de transação
     (não é uma variável de decisão, é uma suposição sobre fricção de
     mercado — "otimizar" o custo para zero só para melhorar o número
     seria trapacear, não validar).
  3. Roda a MESMA combinação vencedora, sem reotimizar nada, só no
     período de teste — esse número, não o de treino, é o que deveria
     entrar numa apresentação como evidência de robustez.
  4. Roda também a configuração DEFAULT (a mesma usada em `app.py` e
     `scripts/run_backtest.py`, nunca ajustada a dado nenhum) no MESMO
     período de teste, como baseline. Se o "otimizado" não bate o default
     fora da amostra, é evidência de que a otimização capturou ruído do
     treino, não sinal robusto — e a recomendação correta seria usar o
     default, não o "melhor" da grade.

O que NÃO muda: a causalidade do backtest em si (cada rebalanceamento
continua usando só dado disponível até a própria data — ver
`signal.py`/`universe.py`, já testado contra look-ahead). O que muda
aqui é só QUAL PERÍODO decide os hiperparâmetros vs. QUAL PERÍODO mede o
resultado reportado como validação.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import pandas as pd

from src import metrics as metrics_mod
from src.backtest import BacktestResult, run_backtest
from src.config import MomentumConfig
from src.rebalance import generate_rebalance_dates

_OBJECTIVES = {
    "sharpe": lambda returns, cdi: metrics_mod.sharpe_ratio(returns, cdi),
    "calmar": lambda returns, _cdi: metrics_mod.calmar_ratio(returns),
}

# Grade padrão: só parâmetros que são escolhas de metodologia, nunca a
# premissa de custo de transação (ver docstring do módulo).
DEFAULT_PARAM_GRID: dict[str, list] = {
    "lookback_days": [189, 252, 315],
    "entry_percentile": [0.60, 0.70, 0.80],
    "rebalance_freq": ["Q", "S"],
}


@dataclass
class TrainTestResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    objective: str
    grid_results: pd.DataFrame       # uma linha por combinação testada, ordenada pelo objetivo no treino
    best_config: MomentumConfig
    best_train_score: float
    optimized_test_score: float
    optimized_test_metrics: pd.Series
    baseline_test_score: float
    baseline_test_metrics: pd.Series
    overfit_gap: float                # score treino - score teste, do modelo otimizado


def split_train_test(
    returns_wide: pd.DataFrame, config: MomentumConfig
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Divide o período de backtest ao meio por número de pregões."""
    start = pd.Timestamp(config.backtest_start)
    end = min(pd.Timestamp(config.backtest_end), returns_wide.index.max())
    trading_days = returns_wide.index[(returns_wide.index >= start) & (returns_wide.index <= end)]
    if len(trading_days) < 4:
        raise ValueError("Período de backtest curto demais para dividir em treino/teste.")

    mid = len(trading_days) // 2
    return trading_days[0], trading_days[mid - 1], trading_days[mid], trading_days[-1]


def _with_consistent_buffer(base_config: MomentumConfig, overrides: dict) -> dict:
    """Se `entry_percentile` está na grade e `hold_percentile` não, deriva
    o corte de manutenção mantendo o mesmo espaçamento (0.10) do default —
    evita combinações inválidas onde manter ficaria mais exigente que entrar."""
    if "entry_percentile" in overrides and "hold_percentile" not in overrides:
        gap = base_config.entry_percentile - base_config.hold_percentile
        overrides = {**overrides, "hold_percentile": max(overrides["entry_percentile"] - gap, 0.0)}
    return overrides


def _rebalances_per_year(returns_wide: pd.DataFrame, config: MomentumConfig) -> float:
    dates = generate_rebalance_dates(returns_wide.index, config)
    return 252 / (len(returns_wide) / max(len(dates), 1))


def _summary_for_config(
    returns_wide: pd.DataFrame,
    cdi_daily: pd.Series,
    bench_returns: pd.Series,
    config: MomentumConfig,
    adtv_wide: pd.DataFrame | None,
    market_cap_wide: pd.DataFrame | None,
) -> tuple[pd.Series, BacktestResult]:
    result = run_backtest(returns_wide, cdi_daily, config, adtv_wide=adtv_wide, market_cap_wide=market_cap_wide)
    if result.returns_net.empty or not result.weights_history:
        return pd.Series(dtype=float), result

    rpy = _rebalances_per_year(returns_wide, config)
    summary = metrics_mod.summary_table(
        result.returns_net, bench_returns, cdi_daily, result.turnover_history, rpy
    )
    return summary, result


def grid_search(
    returns_wide: pd.DataFrame,
    cdi_daily: pd.Series,
    base_config: MomentumConfig,
    param_grid: dict[str, list],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    objective: str = "sharpe",
    adtv_wide: pd.DataFrame | None = None,
    market_cap_wide: pd.DataFrame | None = None,
) -> tuple[MomentumConfig, float, pd.DataFrame]:
    """Varre o produto cartesiano de `param_grid` SÓ no período de treino."""
    if objective not in _OBJECTIVES:
        raise ValueError(f"Objetivo desconhecido: {objective!r}. Use um de {list(_OBJECTIVES)}.")
    score_fn = _OBJECTIVES[objective]

    keys = list(param_grid.keys())
    rows = []
    best_config: MomentumConfig | None = None
    best_score = float("-inf")

    for combo in product(*param_grid.values()):
        overrides = _with_consistent_buffer(base_config, dict(zip(keys, combo)))
        cfg = replace(
            base_config,
            backtest_start=str(train_start.date()),
            backtest_end=str(train_end.date()),
            **overrides,
        )
        result = run_backtest(returns_wide, cdi_daily, cfg, adtv_wide=adtv_wide, market_cap_wide=market_cap_wide)

        if result.returns_net.empty:
            score = float("-inf")
        else:
            raw_score = score_fn(result.returns_net, cdi_daily)
            score = raw_score if pd.notna(raw_score) else float("-inf")

        rows.append({**dict(zip(keys, combo)), "objetivo": score})
        if score > best_score:
            best_score, best_config = score, cfg

    grid_results = pd.DataFrame(rows).sort_values("objetivo", ascending=False).reset_index(drop=True)
    if best_config is None:
        raise ValueError("Nenhuma combinação da grade produziu um backtest válido no período de treino.")
    return best_config, best_score, grid_results


def run_train_test_split(
    returns_wide: pd.DataFrame,
    cdi_daily: pd.Series,
    bench_returns: pd.Series,
    base_config: MomentumConfig,
    param_grid: dict[str, list],
    objective: str = "sharpe",
    adtv_wide: pd.DataFrame | None = None,
    market_cap_wide: pd.DataFrame | None = None,
) -> TrainTestResult:
    train_start, train_end, test_start, test_end = split_train_test(returns_wide, base_config)

    best_config, best_train_score, grid_results = grid_search(
        returns_wide, cdi_daily, base_config, param_grid, train_start, train_end,
        objective=objective, adtv_wide=adtv_wide, market_cap_wide=market_cap_wide,
    )
    score_fn = _OBJECTIVES[objective]

    optimized_test_config = replace(
        best_config, backtest_start=str(test_start.date()), backtest_end=str(test_end.date())
    )
    optimized_test_metrics, optimized_test_result = _summary_for_config(
        returns_wide, cdi_daily, bench_returns, optimized_test_config, adtv_wide, market_cap_wide
    )
    optimized_test_score = (
        score_fn(optimized_test_result.returns_net, cdi_daily)
        if not optimized_test_result.returns_net.empty else float("nan")
    )

    baseline_test_config = replace(
        base_config, backtest_start=str(test_start.date()), backtest_end=str(test_end.date())
    )
    baseline_test_metrics, baseline_test_result = _summary_for_config(
        returns_wide, cdi_daily, bench_returns, baseline_test_config, adtv_wide, market_cap_wide
    )
    baseline_test_score = (
        score_fn(baseline_test_result.returns_net, cdi_daily)
        if not baseline_test_result.returns_net.empty else float("nan")
    )

    overfit_gap = (
        best_train_score - optimized_test_score if pd.notna(optimized_test_score) else float("nan")
    )

    return TrainTestResult(
        train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
        objective=objective, grid_results=grid_results, best_config=best_config,
        best_train_score=best_train_score,
        optimized_test_score=optimized_test_score, optimized_test_metrics=optimized_test_metrics,
        baseline_test_score=baseline_test_score, baseline_test_metrics=baseline_test_metrics,
        overfit_gap=overfit_gap,
    )
