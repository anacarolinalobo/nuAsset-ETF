"""Validação treino/teste da grade de parâmetros (checagem de overfitting).

Implementa o item declarado como cortado no README ("teste out-of-sample
formal com otimização de hiperparâmetros apenas na primeira metade da
amostra e validação cega na segunda"): otimiza uma grade de parâmetros no
treino, mede o Sharpe da combinação vencedora no teste sem reotimizar, e
compara contra o Sharpe do modelo DEFAULT (nunca ajustado) no mesmo teste.

Uso:
    python scripts/run_train_test_split.py

Espera encontrar em `data/` os mesmos 3 CSVs de `scripts/run_backtest.py`.

Exporta para `output/train_test_split/`:
    grid_treino_teste.csv   — Sharpe de treino e teste de cada combinação da grade
    resumo.csv              — resumo da comparação otimizado vs. default
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import data_loader, market_data, sensitivity
from src.config import DEFAULT_CONFIG

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "train_test_split"

# Grade: varia sinal (lookback), seleção (corte de entrada) e calendário
# (frequência) ao mesmo tempo -- diferente de parameter_grid_sensitivity, que
# varia um parâmetro por vez, aqui o objetivo é simular a tentação real de
# "otimizar tudo junto" e checar se o resultado sobrevive fora da amostra.
GRID = {
    "lookback_days": [189, 252, 315],
    "entry_percentile": [0.6, 0.7, 0.8],
    "rebalance_freq": ["Q", "S"],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_combos = 1
    for values in GRID.values():
        n_combos *= len(values)

    print("Carregando dados...")
    returns_wide = data_loader.load_returns(DATA_DIR / "acoes_retornos.csv")
    benchmarks = data_loader.load_benchmarks(DATA_DIR / "benchmarks_diarios.csv")
    cdi_daily = benchmarks[DEFAULT_CONFIG.risk_free_column]
    market_path = next(
        (p for p in (DATA_DIR / "liquidez_mercado.parquet", DATA_DIR / "liquidez_mercado.csv") if p.exists()),
        None,
    )
    mkt = market_data.load_market_data(market_path, returns_wide.index) if market_path else None

    print(f"Grade: {GRID} ({n_combos} combinações)")
    print(
        "Rodando busca em grade no período de treino "
        "(roda 1 backtest por combinação -- pode levar vários minutos)...\n"
    )

    result = sensitivity.train_test_grid_search(returns_wide, cdi_daily, DEFAULT_CONFIG, GRID, mkt)

    train_start, train_end = result["train_period"]
    test_start, test_end = result["test_period"]
    print(f"Treino: {train_start.date()} a {train_end.date()}")
    print(f"Teste:  {test_start.date()} a {test_end.date()}\n")

    print(f"Melhor combinação no treino (Sharpe treino = {result['sharpe_treino_otimizado']:.2f}):")
    for k, v in result["best_params"].items():
        print(f"  {k} = {v}")

    print(f"\nSharpe do MESMO modelo, sem reotimizar, no teste: {result['sharpe_teste_otimizado']:.2f}")
    print(f"Sharpe do modelo DEFAULT (nunca ajustado), no mesmo teste: {result['sharpe_teste_default']:.2f}")
    print(f"Gap treino -> teste do modelo otimizado: {result['gap_treino_teste']:.2f}")

    print()
    if result["otimizado_bate_default_no_teste"]:
        print(
            "=> O modelo otimizado bateu o default fora da amostra -- alguma evidência "
            "de padrão robusto, mas a diferença deve ser vista com cautela dado o "
            "número de combinações testadas."
        )
    else:
        print(
            "=> O modelo otimizado NÃO bateu o default fora da amostra -- evidência de "
            "que a grade capturou ruído do treino, não um padrão robusto. Recomendação: "
            "usar os parâmetros default (fixados por julgamento/literatura), não os "
            '"otimizados" pela grade.'
        )

    result["grid_results"].to_csv(OUTPUT_DIR / "grid_treino_teste.csv", index=False)
    pd.Series(
        {
            "treino_inicio": train_start.date(),
            "treino_fim": train_end.date(),
            "teste_inicio": test_start.date(),
            "teste_fim": test_end.date(),
            **{f"melhor_{k}": v for k, v in result["best_params"].items()},
            "sharpe_treino_otimizado": result["sharpe_treino_otimizado"],
            "sharpe_teste_otimizado": result["sharpe_teste_otimizado"],
            "sharpe_teste_default": result["sharpe_teste_default"],
            "gap_treino_teste": result["gap_treino_teste"],
            "otimizado_bate_default_no_teste": result["otimizado_bate_default_no_teste"],
        }
    ).to_csv(OUTPUT_DIR / "resumo.csv", header=["valor"])

    print(f"\nResultados salvos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
