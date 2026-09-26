"""Script principal: roda o backtest fim-a-fim e exporta resultados.

Uso:
    python scripts/run_backtest.py

Espera encontrar em `data/`:
    acoes_retornos.csv, ibov_composicao.csv, benchmarks_diarios.csv

Exporta para `output/`:
    dashboard.html          — relatório visual completo
    summary_metrics.csv     — tabela de métricas
    daily_returns.csv       — série diária (bruta e líquida) do índice
    turnover_history.csv
    attribution.csv
    sensitivity_lookback.csv, sensitivity_cost.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import attribution, capacity, data_loader, metrics, report, sensitivity, volume_loader
from src.backtest import run_backtest
from src.config import DEFAULT_CONFIG
from src.rebalance import generate_rebalance_dates

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    config = DEFAULT_CONFIG

    print("Carregando dados...")
    returns_wide = data_loader.load_returns(DATA_DIR / "acoes_retornos.csv")
    ibov_weights = data_loader.load_ibov_composition(DATA_DIR / "ibov_composicao.csv")
    benchmarks = data_loader.load_benchmarks(DATA_DIR / "benchmarks_diarios.csv")

    flags, flags_summary = data_loader.flag_suspicious_returns(
        returns_wide, config.max_abs_daily_return
    )
    if not flags_summary.empty:
        print("Alertas de qualidade de dado:")
        print(flags_summary.to_string(index=False))
        flags.to_csv(OUTPUT_DIR / "data_quality_flags.csv", index=False)

    cdi_daily = benchmarks[config.risk_free_column]
    bench_returns = benchmarks[config.benchmark_column]

    cotahist_files = sorted(DATA_DIR.glob("COTAHIST_A*.TXT"))
    volume_wide = None
    if cotahist_files:
        print(f"Carregando volume negociado (COTAHIST): {len(cotahist_files)} arquivo(s)...")
        cotahist = volume_loader.load_cotahist(cotahist_files)
        volume_wide = volume_loader.build_volume_wide(cotahist)

    print("Rodando backtest...")
    result = run_backtest(returns_wide, cdi_daily, config, volume_wide=volume_wide)

    rebalance_dates = generate_rebalance_dates(returns_wide.index, config)
    rebalances_per_year = 252 / (len(returns_wide) / max(len(rebalance_dates), 1))

    latest_date = max(result.weights_history.keys())
    latest_weights = result.weights_history[latest_date]
    latest_bench_weights = (
        ibov_weights.reindex([latest_date], method="ffill").iloc[0]
        if not ibov_weights.empty
        else pd.Series(dtype=float)
    )

    summary = metrics.summary_table(
        result.returns_net,
        bench_returns,
        cdi_daily,
        result.turnover_history,
        rebalances_per_year,
        latest_weights,
        latest_bench_weights,
    )
    print(summary)
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", header=["valor"])

    pd.DataFrame(
        {"retorno_bruto": result.returns_gross, "retorno_liquido": result.returns_net}
    ).to_csv(OUTPUT_DIR / "daily_returns.csv")

    result.turnover_history.to_csv(OUTPUT_DIR / "turnover_history.csv", header=["turnover"])

    if volume_wide is not None:
        adtv_latest = volume_loader.adtv_on_date(volume_wide, latest_date, config.volume_window_days)
        capacity_per_name = capacity.estimate_capacity(latest_weights, adtv_latest)
        capacity_per_name.to_csv(OUTPUT_DIR / "capacity_by_name.csv", header=["capacidade_brl"])
        print(f"Capacidade do produto (gargalo): R$ {capacity.product_capacity(capacity_per_name):,.0f}")

    print("Calculando atribuição de performance...")
    attr = attribution.attribution_history(result.weights_history, ibov_weights, returns_wide)
    attr.to_csv(OUTPUT_DIR / "attribution.csv")

    print("Rodando análise de sensibilidade (pode levar alguns minutos)...")
    sens_lookback = sensitivity.parameter_grid_sensitivity(
        returns_wide, cdi_daily, config, "lookback_days", [126, 189, 252, 315]
    )
    sens_lookback.to_csv(OUTPUT_DIR / "sensitivity_lookback.csv", index=False)

    sens_cost = sensitivity.parameter_grid_sensitivity(
        returns_wide, cdi_daily, config, "transaction_cost_bps", [10, 25, 40, 60]
    )
    sens_cost.to_csv(OUTPUT_DIR / "sensitivity_cost.csv", index=False)

    print("Gerando dashboard...")
    notes = [
        "Backtest sem look-ahead: sinal e universo em cada rebalance usam só dados até a própria data de rebalance.",
        "Survivorship bias mitigado: ações deslistadas permanecem no universo até o último dia negociado.",
        f"Custo de transação assumido: {config.transaction_cost_bps:.0f} bps por lado sobre o turnover.",
        (
            "Capacidade do produto estimada com ADTV real (COTAHIST B3) — ver capacity_by_name.csv."
            if volume_wide is not None
            else "Capacidade do produto não estimada aqui por falta de dado de volume na base fornecida (ver src/capacity.py)."
        ),
        "Ver README.md para a descrição completa da metodologia e as limitações declaradas.",
    ]
    report.build_dashboard(
        OUTPUT_DIR / "dashboard.html",
        result.returns_gross,
        result.returns_net,
        bench_returns,
        result.turnover_history,
        result.n_holdings_history,
        cdi_daily,
        summary,
        notes=notes,
    )

    print(f"\nConcluído. Resultados em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
