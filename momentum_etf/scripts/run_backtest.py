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

from src import attribution, data_loader, metrics, report, sensitivity
from src.backtest import run_backtest
from src.config import DEFAULT_CONFIG
from src.rebalance import generate_rebalance_dates

DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUT_DIR = ROOT / "output"


def _load_derived_liquidity_data():
    """Carrega ADTV e valor de mercado gerados por `build_market_data.py`.

    Retorna (None, None) se ainda não foram gerados — nesse caso o
    backtest cai automaticamente no proxy de liquidez baseado em presença
    de retorno (ver `src/universe.py`), sem quebrar.
    """
    adtv_path = DERIVED_DIR / "adtv.csv"
    mcap_path = DERIVED_DIR / "market_cap.csv"

    adtv_wide = pd.read_csv(adtv_path, index_col=0, parse_dates=True) if adtv_path.exists() else None
    market_cap_wide = pd.read_csv(mcap_path, index_col=0, parse_dates=True) if mcap_path.exists() else None

    if adtv_wide is None and market_cap_wide is None:
        print("Sem dados derivados de liquidez/market cap (rode scripts/build_market_data.py "
              "para usá-los). Caindo no proxy de liquidez baseado em presença de retorno.")
    else:
        print(f"Usando dados externos de liquidez/tamanho: "
              f"ADTV={'sim' if adtv_wide is not None else 'não'}, "
              f"market cap={'sim' if market_cap_wide is not None else 'não'}")

    return adtv_wide, market_cap_wide


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    config = DEFAULT_CONFIG

    print("Carregando dados...")
    returns_wide = data_loader.load_returns(DATA_DIR / "acoes_retornos.csv")
    ibov_weights = data_loader.load_ibov_composition(DATA_DIR / "ibov_composicao.csv")
    # benchmarks_diarios.csv vem em NÍVEL (pontos de índice, fator
    # acumulado, preço) -- confirmado no dado real (Ibovespa em dezenas de
    # milhares de pontos, CDI como fator acumulado). Converte para retorno
    # diário antes de qualquer outro cálculo.
    benchmarks_levels = data_loader.load_benchmarks(DATA_DIR / "benchmarks_diarios.csv")
    benchmarks = data_loader.benchmarks_to_returns(benchmarks_levels)
    adtv_wide, market_cap_wide = _load_derived_liquidity_data()

    flags, flags_summary = data_loader.flag_suspicious_returns(
        returns_wide, config.max_abs_daily_return
    )
    if not flags_summary.empty:
        print("Alertas de qualidade de dado em acoes_retornos.csv (removidos do backtest):")
        print(flags_summary.to_string(index=False))
        flags.to_csv(OUTPUT_DIR / "data_quality_flags.csv", index=False)
        returns_wide = data_loader.clean_returns(returns_wide, flags)

    # Mesma checagem sobre os benchmarks (CDI, Ibovespa, etc.) — um dado
    # ruim ali não é pego pela limpeza acima (que só olha as ações) e ainda
    # assim contamina beta, tracking error e o gráfico de retorno
    # acumulado via cumprod, que é justamente o que travou aqui antes.
    bench_flags, bench_flags_summary = data_loader.flag_suspicious_returns(
        benchmarks, config.max_abs_daily_return
    )
    if not bench_flags_summary.empty:
        print("Alertas de qualidade de dado em benchmarks_diarios.csv (removidos):")
        print(bench_flags_summary.to_string(index=False))
        bench_flags.to_csv(OUTPUT_DIR / "data_quality_flags_benchmarks.csv", index=False)
        benchmarks = data_loader.clean_returns(benchmarks, bench_flags)

    cdi_daily = benchmarks[config.risk_free_column]
    bench_returns = benchmarks[config.benchmark_column]

    print("Rodando backtest...")
    result = run_backtest(returns_wide, cdi_daily, config, adtv_wide=adtv_wide, market_cap_wide=market_cap_wide)

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

    print("Calculando atribuição de performance...")
    attr = attribution.attribution_history(result.weights_history, ibov_weights, returns_wide)
    attr.to_csv(OUTPUT_DIR / "attribution.csv")

    print("Rodando análise de sensibilidade (pode levar alguns minutos)...")
    sens_lookback = sensitivity.parameter_grid_sensitivity(
        returns_wide, cdi_daily, config, "lookback_days", [126, 189, 252, 315],
        adtv_wide=adtv_wide, market_cap_wide=market_cap_wide,
    )
    sens_lookback.to_csv(OUTPUT_DIR / "sensitivity_lookback.csv", index=False)

    sens_cost = sensitivity.parameter_grid_sensitivity(
        returns_wide, cdi_daily, config, "transaction_cost_bps", [10, 25, 40, 60],
        adtv_wide=adtv_wide, market_cap_wide=market_cap_wide,
    )
    sens_cost.to_csv(OUTPUT_DIR / "sensitivity_cost.csv", index=False)

    print("Gerando dashboard...")
    notes = [
        "Backtest sem look-ahead: sinal e universo em cada rebalance usam só dados até a própria data de rebalance.",
        "Survivorship bias mitigado: ações deslistadas permanecem no universo até o último dia negociado.",
        f"Custo de transação assumido: {config.transaction_cost_bps:.0f} bps por lado sobre o turnover.",
        "Capacidade do produto não estimada aqui por falta de dado de volume na base fornecida (ver src/capacity.py).",
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
