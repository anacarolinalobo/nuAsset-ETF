"""Gera dados SINTÉTICOS no formato dos 3 CSVs do case, só para smoke-test.

IMPORTANTE: isto NÃO é dado real de mercado. Serve apenas para exercitar o
pipeline (data_loader -> universe -> signal -> selection -> weighting ->
backtest -> metrics -> report) de ponta a ponta neste ambiente, já que os
CSVs reais do case (acoes_retornos.csv, ibov_composicao.csv,
benchmarks_diarios.csv) não foram anexados a esta sessão.

Ao rodar o case de verdade, coloque os 3 arquivos reais em `data/` e pule
este script — `scripts/run_backtest.py` os lê diretamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_returns(n_tickers: int = 120, start: str = "2008-01-01", end: str = "2026-06-30") -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    tickers = [f"SINT{i:03d}3" for i in range(n_tickers)]

    # Cada ticker tem uma data de entrada e (às vezes) uma data de saída,
    # para exercitar o tratamento de survivorship / delisting.
    records = []
    for i, ticker in enumerate(tickers):
        entry_idx = RNG.integers(0, max(1, len(dates) // 3))
        lifespan = RNG.integers(int(len(dates) * 0.4), len(dates) - entry_idx)
        exit_idx = min(entry_idx + lifespan, len(dates))

        ticker_dates = dates[entry_idx:exit_idx]
        drift = RNG.normal(0.0003, 0.0003)
        vol = RNG.uniform(0.015, 0.045)

        # injeta autocorrelação positiva de curto/médio prazo (proxy de
        # momentum) via um AR(1) fraco sobre o drift latente
        latent = np.zeros(len(ticker_dates))
        shocks = RNG.normal(0, 1, len(ticker_dates))
        for t in range(1, len(latent)):
            latent[t] = 0.985 * latent[t - 1] + shocks[t]
        rets = drift + vol * (0.3 * latent / (latent.std() + 1e-9) + 0.7 * shocks)

        df_t = pd.DataFrame({"date": ticker_dates, "ticker": ticker, "retorno": rets})
        records.append(df_t)

    long_df = pd.concat(records, ignore_index=True)
    return long_df


def make_ibov_composition(returns_long: pd.DataFrame, n_components: int = 40) -> pd.DataFrame:
    tickers = sorted(returns_long["ticker"].unique())[:n_components]
    dates = sorted(returns_long["date"].unique())
    quarter_ends = pd.DatetimeIndex(dates).to_period("Q").unique()

    rows = []
    for q in quarter_ends:
        q_end_dates = [d for d in dates if pd.Timestamp(d).to_period("Q") == q]
        if not q_end_dates:
            continue
        date = max(q_end_dates)
        weights = RNG.dirichlet(np.ones(len(tickers)))
        for t, w in zip(tickers, weights):
            rows.append({"date": date, "ticker": t, "peso": w})
    return pd.DataFrame(rows)


def make_benchmarks(returns_long: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(returns_long["date"].unique()))
    n = len(dates)

    cdi = np.full(n, 0.0004)
    ibov = RNG.normal(0.0003, 0.014, n)
    ima_s = cdi * 1.02
    idka = RNG.normal(0.0003, 0.006, n)
    ima_b = RNG.normal(0.0004, 0.007, n)
    ihfa = RNG.normal(0.0003, 0.006, n)
    ifix = RNG.normal(0.0003, 0.009, n)
    sp500 = RNG.normal(0.0004, 0.011, n)
    btc = RNG.normal(0.0006, 0.035, n)

    return pd.DataFrame(
        {
            "date": dates,
            "cdi": cdi,
            "ima_s": ima_s,
            "idka_pre_3a": idka,
            "ima_b": ima_b,
            "ihfa": ihfa,
            "ifix": ifix,
            "ibovespa": ibov,
            "sp500_brl": sp500,
            "bitcoin_brl": btc,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    returns_long = make_returns()
    returns_long.to_csv(DATA_DIR / "acoes_retornos.csv", index=False)

    ibov = make_ibov_composition(returns_long)
    ibov.to_csv(DATA_DIR / "ibov_composicao.csv", index=False)

    benchmarks = make_benchmarks(returns_long)
    benchmarks.to_csv(DATA_DIR / "benchmarks_diarios.csv", index=False)

    print(f"Dados sintéticos gravados em {DATA_DIR}")


if __name__ == "__main__":
    main()
