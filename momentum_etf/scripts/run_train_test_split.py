"""Split formal treino/validação com otimização cega de hiperparâmetros.

Uso:
    python scripts/run_train_test_split.py

Protocolo completo e a justificativa de cada passo em
`src/train_test_split.py`. Roda contra os mesmos dados de `data/` que
`scripts/run_backtest.py` (com a mesma limpeza de retornos suspeitos).

Como cada combinação da grade roda um backtest completo no período de
treino, e a grade default tem 18 combinações (3 janelas × 3 cortes de
seleção × 2 frequências), isto leva bem mais tempo que um único
`run_backtest.py` — rode com paciência.

Saída em `output/train_test_split/`:
  grid_results.csv            toda a grade testada no treino, ordenada
  optimized_vs_baseline.csv   comparação lado a lado no período de TESTE
                               (modelo otimizado vs. config default)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import data_loader
from src.config import DEFAULT_CONFIG
from src.train_test_split import DEFAULT_PARAM_GRID, run_train_test_split

DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUT_DIR = ROOT / "output" / "train_test_split"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    returns_wide = data_loader.load_returns(DATA_DIR / "acoes_retornos.csv")
    benchmarks_levels = data_loader.load_benchmarks(DATA_DIR / "benchmarks_diarios.csv")
    benchmarks = data_loader.benchmarks_to_returns(benchmarks_levels)

    flags, _ = data_loader.flag_suspicious_returns(returns_wide, DEFAULT_CONFIG.max_abs_daily_return)
    returns_wide = data_loader.clean_returns(returns_wide, flags)

    bench_flags, _ = data_loader.flag_suspicious_returns(benchmarks, DEFAULT_CONFIG.max_abs_daily_return)
    benchmarks = data_loader.clean_returns(benchmarks, bench_flags)

    cdi_daily = benchmarks[DEFAULT_CONFIG.risk_free_column]
    bench_returns = benchmarks[DEFAULT_CONFIG.benchmark_column]

    adtv_wide = None
    market_cap_wide = None
    if (DERIVED_DIR / "adtv.csv").exists():
        adtv_wide = pd.read_csv(DERIVED_DIR / "adtv.csv", index_col=0, parse_dates=True)
    if (DERIVED_DIR / "market_cap.csv").exists():
        market_cap_wide = pd.read_csv(DERIVED_DIR / "market_cap.csv", index_col=0, parse_dates=True)

    n_combos = 1
    for values in DEFAULT_PARAM_GRID.values():
        n_combos *= len(values)
    print(f"Grade: {DEFAULT_PARAM_GRID} ({n_combos} combinações)")
    print("Rodando busca em grade no período de treino (roda 1 backtest por combinação -- pode levar vários minutos)...")

    result = run_train_test_split(
        returns_wide, cdi_daily, bench_returns, DEFAULT_CONFIG, DEFAULT_PARAM_GRID,
        objective="sharpe", adtv_wide=adtv_wide, market_cap_wide=market_cap_wide,
    )

    print(f"\nTreino: {result.train_start.date()} a {result.train_end.date()}")
    print(f"Teste:  {result.test_start.date()} a {result.test_end.date()}")

    print(f"\nMelhor combinação no treino (Sharpe treino = {result.best_train_score:.2f}):")
    for key in DEFAULT_PARAM_GRID:
        print(f"  {key} = {getattr(result.best_config, key)}")

    print(f"\nSharpe do MESMO modelo, sem reotimizar, no teste: {result.optimized_test_score:.2f}")
    print(f"Sharpe do modelo DEFAULT (nunca ajustado), no mesmo teste: {result.baseline_test_score:.2f}")
    print(f"Gap treino -> teste do modelo otimizado: {result.overfit_gap:.2f}")

    if pd.notna(result.optimized_test_score) and pd.notna(result.baseline_test_score):
        if result.optimized_test_score > result.baseline_test_score:
            print(
                "\n=> O modelo otimizado bateu o default fora da amostra -- indício de sinal "
                "real na escolha de parâmetros, não só ruído do período de treino."
            )
        else:
            print(
                "\n=> O modelo otimizado NÃO bateu o default fora da amostra -- evidência de "
                "que a grade capturou ruído do treino, não um padrão robusto. Recomendação: "
                "usar os parâmetros default (fixados por julgamento/literatura), não os "
                "\"otimizados\" pela grade."
            )

    result.grid_results.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)

    comparison = pd.concat(
        [
            result.optimized_test_metrics.rename("otimizado_no_teste"),
            result.baseline_test_metrics.rename("default_no_teste"),
        ],
        axis=1,
    )
    comparison.to_csv(OUTPUT_DIR / "optimized_vs_baseline.csv")

    print(f"\nResultados salvos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
