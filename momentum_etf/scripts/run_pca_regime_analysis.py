"""Fatores de mercado via PCA e desempenho do momentum condicional a eles.

Constrói fatores estatísticos de mercado (PCA sobre os retornos das ações,
padronizados) e mede o Sharpe/retorno do índice de momentum em cada regime
(tercil) de cada fator, além da exposição (beta/R²) do índice a esses
fatores — ver `src/market_factors.py` para a metodologia e as limitações
declaradas (é uma análise diagnóstica/post-hoc, não um sinal de regime ao
vivo).

Uso:
    python scripts/run_pca_regime_analysis.py

Espera encontrar em `data/` os mesmos 3 CSVs de `scripts/run_backtest.py`.

Exporta para `output/pca_regimes/`:
    variancia_explicada.csv
    desempenho_por_regime.csv
    exposicao_fatores.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_loader, market_factors
from src.backtest import run_backtest
from src.config import DEFAULT_CONFIG

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "pca_regimes"
N_COMPONENTS = 5


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG

    print("Carregando dados...")
    returns_wide = data_loader.load_returns(DATA_DIR / "acoes_retornos.csv")
    benchmarks = data_loader.load_benchmarks(DATA_DIR / "benchmarks_diarios.csv")
    cdi_daily = benchmarks[config.risk_free_column]

    print("Rodando backtest do índice de momentum (parâmetros default)...")
    result = run_backtest(returns_wide, cdi_daily, config)
    returns_net = result.returns_net

    print(f"Ajustando PCA sobre os retornos padronizados ({N_COMPONENTS} componentes)...")
    factors = market_factors.fit_market_factors(returns_wide, n_components=N_COMPONENTS)

    print("\nVariância explicada por componente:")
    var_pct = (factors.explained_variance_ratio * 100).round(1)
    print(var_pct.to_string())
    print(
        f"\nPC1: {factors.pct_tickers_positive_pc1:.0%} das ações têm carga positiva "
        "-- quanto mais perto de 100%, mais o PC1 se comporta como um fator de "
        "mercado amplo (todas as ações se movendo mais ou menos juntas)."
    )

    print("\nClassificando dias em tercis (baixo/médio/alto) de cada fator...")
    regimes = market_factors.classify_factor_regimes(factors.factor_scores, n_buckets=3)

    perf_by_regime = market_factors.momentum_performance_by_regime(
        returns_net, regimes, cdi_daily
    )
    print("\nDesempenho do momentum por regime de cada fator:")
    print(perf_by_regime.to_string(index=False))

    exposure = market_factors.factor_exposure(returns_net, factors.factor_scores)
    print("\nExposição do índice de momentum aos fatores (regressão OLS diária):")
    print(exposure.to_string())
    print(
        f"\nR² = {exposure['r_quadrado']:.2f}: {exposure['r_quadrado']:.0%} da variância "
        "diária do índice é explicada pelos fatores estatísticos de mercado; o "
        "restante é idiossincrático à estratégia (seleção/timing de momentum)."
    )

    var_pct.to_csv(OUTPUT_DIR / "variancia_explicada.csv", header=["pct_variancia_explicada"])
    perf_by_regime.to_csv(OUTPUT_DIR / "desempenho_por_regime.csv", index=False)
    exposure.to_csv(OUTPUT_DIR / "exposicao_fatores.csv", header=["valor"])

    print(f"\nResultados salvos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
