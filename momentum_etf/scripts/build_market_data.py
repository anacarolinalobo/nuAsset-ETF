"""Constrói os dados derivados de liquidez e tamanho a partir de fontes externas.

Roda ANTES de `run_backtest.py`. Lê:
  - `data/cotahist/COTAHIST_A*.TXT`      (arquivos de pregão da B3)
  - `data/fre_cia_aberta_{ano}/...csv`   (Formulário de Referência da CVM,
                                           2010-2026, um diretório por ano)

E escreve em `data/derived/`:
  - `ticker_cnpj_mapping.csv`  — para QA manual do matching (ver
    src/company_mapping.py); revise as linhas com `matched_high_confidence`
    falso ou `score` baixo antes de confiar cegamente no resultado.
  - `adtv.csv`                 — ADTV móvel (date x ticker)
  - `market_cap.csv`           — valor de mercado diário (date x ticker)

Reexecutar este script é caro (parsing de COTAHIST é pesado) — os arquivos
derivados ficam em CSV para serem lidos direto por `run_backtest.py` sem
reprocessar tudo a cada backtest.

Se `data/cotahist/` ou as pastas `fre_cia_aberta_*` não existirem, o
script avisa e simplesmente não gera os arquivos — `run_backtest.py`
detecta a ausência e cai automaticamente no proxy de liquidez antigo
(ver `src/universe.py`), então o pipeline principal não quebra.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import company_mapping, cotahist, fre_capital_social, market_cap
from src.config import DEFAULT_CONFIG

DATA_DIR = ROOT / "data"
COTAHIST_DIR = DATA_DIR / "cotahist"
DERIVED_DIR = DATA_DIR / "derived"


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    if not COTAHIST_DIR.exists() or not list(COTAHIST_DIR.glob("COTAHIST_A*.TXT")):
        print(f"Nenhum arquivo COTAHIST encontrado em {COTAHIST_DIR} — abortando.")
        print("Coloque os arquivos COTAHIST_A{ano}.TXT nessa pasta e rode de novo.")
        return

    fre_years_present = sorted(DATA_DIR.glob("fre_cia_aberta_*"))
    if not fre_years_present:
        print(f"Nenhuma pasta fre_cia_aberta_{{ano}} encontrada em {DATA_DIR} — abortando.")
        return

    print("Lendo COTAHIST...")
    cotahist_long = cotahist.parse_cotahist_directory(COTAHIST_DIR)
    print(f"  {len(cotahist_long):,} registros, {cotahist_long['ticker'].nunique()} tickers")

    print("Lendo capital social (FRE/CVM)...")
    capital_social = fre_capital_social.load_capital_social(DATA_DIR, start_year=2010, end_year=2026)
    print(f"  {len(capital_social):,} registros, {capital_social['CNPJ_Companhia'].nunique()} CNPJs")

    print("Construindo mapeamento ticker <-> CNPJ (fuzzy match de nome)...")
    instruments = cotahist.instrument_reference(cotahist_long)
    companies = capital_social[["CNPJ_Companhia", "Nome_Companhia"]].drop_duplicates()
    mapping = company_mapping.build_ticker_cnpj_mapping(instruments, companies)
    mapping.to_csv(DERIVED_DIR / "ticker_cnpj_mapping.csv", index=False)

    n_matched = mapping["matched_high_confidence"].sum()
    print(f"  {n_matched}/{len(mapping)} tickers casados com confiança "
          f"(revise {DERIVED_DIR / 'ticker_cnpj_mapping.csv'} antes de confiar no resultado)")

    print("Calculando ADTV...")
    traded_value = cotahist.traded_value_wide(cotahist_long)
    adtv = cotahist.compute_adtv(traded_value, DEFAULT_CONFIG.liquidity_lookback_days)
    adtv.to_csv(DERIVED_DIR / "adtv.csv")

    print("Calculando valor de mercado...")
    close_price = cotahist.close_price_wide(cotahist_long)
    mcap = market_cap.build_market_cap(close_price, capital_social, mapping)
    mcap.to_csv(DERIVED_DIR / "market_cap.csv")

    print(f"\nConcluído. Arquivos derivados em {DERIVED_DIR}")


if __name__ == "__main__":
    main()
