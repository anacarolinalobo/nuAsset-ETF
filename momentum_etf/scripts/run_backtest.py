"""Script principal: roda o backtest fim-a-fim e exporta resultados.

Uso:
    python scripts/run_backtest.py

Espera encontrar em `data/`:
    acoes_retornos.csv, ibov_composicao.csv, benchmarks_diarios.csv
e, opcionalmente, liquidez_mercado.csv (volume B3 + market cap CVM, gerado
pela última célula de caseNuAsset.ipynb). Sem ele, o universo usa o proxy
de liquidez por presença de retorno e a capacidade não é estimada.

Exporta para `output/`:
    dashboard.html          — relatório visual completo
    summary_metrics.csv     — tabela de métricas
    daily_returns.csv       — série diária (bruta e líquida) do índice
    turnover_history.csv
    attribution.csv
    sensitivity_lookback.csv, sensitivity_cost.csv
    liquidez_carteira.csv   — ADTV, market cap e capacidade por rebalance
    capacidade_ultima_carteira.csv — capacidade por papel na carteira atual
    cobertura_liquidez.csv  — tickers com volume / market cap
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import attribution, capacity, data_loader, market_data, metrics, report, sensitivity
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

    mkt = None
    market_path = next(
        (p for p in (DATA_DIR / "liquidez_mercado.parquet", DATA_DIR / "liquidez_mercado.csv") if p.exists()),
        None,
    )
    if market_path is not None:
        print(f"Carregando liquidez, volume e market cap de {market_path.name}...")
        mkt = market_data.load_market_data(market_path, returns_wide.index)
        coverage = market_data.coverage_report(mkt, returns_wide)
        print(coverage.to_string())
        coverage.to_csv(OUTPUT_DIR / "cobertura_liquidez.csv", header=["valor"])
    else:
        print("liquidez_mercado.csv não encontrado: usando proxy de liquidez por presença de retorno.")

    cdi_daily = benchmarks[config.risk_free_column]
    bench_returns = benchmarks[config.benchmark_column]

    print("Rodando backtest...")
    result = run_backtest(returns_wide, cdi_daily, config, mkt)

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

    if mkt is not None:
        result.liquidity_history.to_csv(OUTPUT_DIR / "liquidez_carteira.csv")
        latest_adtv = mkt.adtv(latest_date, config.adtv_window_days)
        cap_per_name = capacity.estimate_capacity(
            latest_weights,
            latest_adtv,
            config.max_adtv_participation,
            config.days_to_build_position,
        )
        cap_per_name.rename("capacidade_brl").to_csv(OUTPUT_DIR / "capacidade_ultima_carteira.csv")
        print(
            f"Capacidade estimada da carteira atual: "
            f"R$ {capacity.product_capacity(cap_per_name) / 1e6:,.0f} mi "
            f"(gargalo: {cap_per_name.index[0] if not cap_per_name.empty else '-'})"
        )

    print("Calculando atribuição de performance...")
    attr = attribution.attribution_history(result.weights_history, ibov_weights, returns_wide)
    attr.to_csv(OUTPUT_DIR / "attribution.csv")

    print("Rodando análise de sensibilidade (pode levar alguns minutos)...")
    sens_lookback = sensitivity.parameter_grid_sensitivity(
        returns_wide, cdi_daily, config, "lookback_days", [126, 189, 252, 315], mkt
    )
    sens_lookback.to_csv(OUTPUT_DIR / "sensitivity_lookback.csv", index=False)

    sens_cost = sensitivity.parameter_grid_sensitivity(
        returns_wide, cdi_daily, config, "transaction_cost_bps", [10, 25, 40, 60], mkt
    )
    sens_cost.to_csv(OUTPUT_DIR / "sensitivity_cost.csv", index=False)

    print("Gerando dashboard...")
    notes = [
        "Backtest sem look-ahead: sinal e universo em cada rebalance usam só dados até a própria data de rebalance.",
        "Survivorship bias mitigado: ações deslistadas permanecem no universo até o último dia negociado.",
        f"Custo de transação assumido: {config.transaction_cost_bps:.0f} bps por lado sobre o turnover.",
        (
            f"Universo filtrado por ADTV >= R$ {config.min_adtv_brl / 1e6:,.0f} mi e market cap >= "
            f"R$ {config.min_market_cap_brl / 1e6:,.0f} mi (volume B3 COTAHIST, capital social CVM)."
            if mkt is not None
            else "Sem liquidez_mercado.csv: universo usa proxy de liquidez e capacidade não é estimada (ver src/capacity.py)."
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
