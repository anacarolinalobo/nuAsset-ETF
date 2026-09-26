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

import string
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _make_ticker(i: int) -> str:
    """Ticker sintético no formato real de ações B3 (4 letras + 1 dígito),
    para passar no filtro de regex de `src/cotahist.py::parse_cotahist_file`."""
    letters = string.ascii_uppercase
    a = letters[(i // (26 * 26)) % 26]
    b = letters[(i // 26) % 26]
    c = letters[i % 26]
    digit = 3 if i % 2 == 0 else 4
    return f"S{a}{b}{c}{digit}"


def make_returns(n_tickers: int = 120, start: str = "2008-01-01", end: str = "2026-06-30") -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    tickers = [_make_ticker(i) for i in range(n_tickers)]

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


# ============================================================
# COTAHIST sintético (preço + volume financeiro negociado)
# ============================================================

# Mesmo layout fixo de src/cotahist.py — repetido aqui (não importado)
# para manter este gerador desacoplado do pacote principal.
_COTAHIST_COLSPECS = [
    (0, 2), (2, 10), (10, 12), (12, 24), (24, 27), (27, 39), (39, 49),
    (49, 52), (52, 56), (56, 69), (69, 82), (82, 95), (95, 108), (108, 121),
    (121, 134), (134, 147), (147, 152), (152, 170), (170, 188), (188, 201),
    (201, 202), (202, 210), (210, 217), (217, 230), (230, 242), (242, 245),
]


def _fmt_field(value: str, width: int, numeric: bool) -> str:
    text = str(value)[:width]
    return text.rjust(width, "0") if numeric else text.ljust(width)


def make_cotahist_files(
    returns_long: pd.DataFrame,
    out_dir: Path,
    company_names: dict[str, str],
    n_tickers_subset: int = 30,
):
    """Gera COTAHIST_A{ano}.TXT sintéticos para um subconjunto de tickers.

    Preço reconstruído por cumprod(1+retorno) a partir de um preço-base
    aleatório por ticker (para o valor de mercado sintético ter dispersão
    de nível de preço entre papéis, como no mercado real). Volume
    financeiro simulado como preço x quantidade aleatória, com alguns
    tickers deliberadamente ilíquidos para exercitar o filtro.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = sorted(returns_long["ticker"].unique())[:n_tickers_subset]
    wide = returns_long[returns_long["ticker"].isin(tickers)].pivot(
        index="date", columns="ticker", values="retorno"
    ).sort_index()

    base_prices = {t: RNG.uniform(3, 120) for t in tickers}
    liquidity_tier = {t: RNG.choice(["alta", "baixa"], p=[0.7, 0.3]) for t in tickers}

    prices = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for t in tickers:
        prices[t] = base_prices[t] * (1 + wide[t].fillna(0)).cumprod()

    years = sorted(wide.index.year.unique())
    for year in years:
        year_dates = wide.index[wide.index.year == year]
        lines = []
        for date in year_dates:
            for t in tickers:
                price = prices.loc[date, t]
                if pd.isna(price):
                    continue
                qty_low, qty_high = (1_000, 50_000) if liquidity_tier[t] == "alta" else (50, 500)
                qty = int(RNG.integers(qty_low, qty_high))
                volume = price * qty
                nome = company_names.get(t, t)[:12]

                fields = [
                    "01", date.strftime("%Y%m%d"), "02", t, "010", nome,
                    "ON      NM", "0", "R$  ",
                    f"{price*100:.0f}", f"{price*100:.0f}", f"{price*100:.0f}", f"{price*100:.0f}",
                    f"{price*100:.0f}", f"{price*100:.0f}", f"{price*100:.0f}",
                    f"{RNG.integers(1, 500)}", f"{qty}", f"{volume*100:.0f}",
                    "0", "0", "99991231", "0000001", "0", f"BR{t}000{RNG.integers(0,9)}", "126",
                ]
                numeric_flags = [
                    False, False, False, False, False, False, False, False, False,
                    True, True, True, True, True, True, True, True, True, True,
                    True, False, False, True, True, False, False,
                ]
                line = "".join(
                    _fmt_field(val, spec[1] - spec[0], is_num)
                    for val, spec, is_num in zip(fields, _COTAHIST_COLSPECS, numeric_flags)
                )
                lines.append(line)

        (out_dir / f"COTAHIST_A{year}.TXT").write_text("\n".join(lines), encoding="latin1")

    return tickers, liquidity_tier


# ============================================================
# FRE / capital social sintético
# ============================================================

def make_fre_capital_social(out_dir: Path, tickers: list[str], company_names: dict[str, str], years: range):
    """Gera fre_cia_aberta_{ano}/fre_cia_aberta_capital_social_{ano}.csv sintéticos."""
    cnpjs = {t: f"{RNG.integers(10**13, 10**14 - 1)}" for t in tickers}
    base_shares = {t: RNG.integers(50_000_000, 2_000_000_000) for t in tickers}

    for year in years:
        rows = []
        for t in tickers:
            shares = int(base_shares[t] * (1 + RNG.normal(0, 0.02)))
            rows.append(
                {
                    "CNPJ_Companhia": cnpjs[t],
                    "Nome_Companhia": company_names[t],
                    "Codigo_CVM": str(abs(hash(t)) % 100000),
                    "Data_Referencia": f"{year}-04-30",
                    "Tipo_Capital": "Capital Integralizado",
                    "Quantidade_Acoes_Ordinarias": shares,
                    "Quantidade_Acoes_Preferenciais": 0,
                    "Quantidade_Total_Acoes": shares,
                }
            )
        year_dir = out_dir / f"fre_cia_aberta_{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            year_dir / f"fre_cia_aberta_capital_social_{year}.csv", sep=";", encoding="latin1", index=False
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

    print("Gerando COTAHIST e FRE sintéticos (liquidez/market cap)...")
    all_tickers = sorted(returns_long["ticker"].unique())
    # Nome curto (cabe nos 12 caracteres truncados do COTAHIST e é idêntico
    # no FRE) — só para o matching fuzzy ter uma âncora clara neste
    # smoke-test; dados reais têm nomes divergentes de verdade entre as
    # duas fontes, daí o matching aproximado documentado em company_mapping.py.
    company_names = {t: f"CIA {t}" for t in all_tickers}

    cotahist_dir = DATA_DIR / "cotahist"
    subset_tickers, _ = make_cotahist_files(returns_long, cotahist_dir, company_names, n_tickers_subset=30)

    fre_years = range(2010, 2027)
    make_fre_capital_social(DATA_DIR, subset_tickers, company_names, fre_years)

    print(f"COTAHIST sintético em {cotahist_dir}, FRE sintético em {DATA_DIR}/fre_cia_aberta_*")
    print("Rode 'python scripts/build_market_data.py' em seguida para gerar ADTV/market cap.")


if __name__ == "__main__":
    main()
