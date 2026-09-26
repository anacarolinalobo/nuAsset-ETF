"""Parâmetros da metodologia do índice de momentum.

Centraliza todas as escolhas metodológicas em um único lugar, para que
mudar um parâmetro não exija tocar em nenhum outro módulo e para que a
análise de sensibilidade (src/sensitivity.py) possa varrer esses valores
de forma sistemática.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MomentumConfig:
    # --- Sinal de momentum ---
    # Momentum "12-1": retorno acumulado dos últimos `lookback_days` pregões,
    # pulando os `skip_days` mais recentes (evita reversão de curto prazo /
    # microestrutura documentada em Jegadeesh & Titman 1993).
    lookback_days: int = 252
    skip_days: int = 21
    # Ajusta o sinal pela volatilidade realizada na mesma janela (momentum
    # "risk-adjusted" / information-ratio-like). Reduz o viés do sinal cru
    # para papéis de alta vol que sobem/caem por ruído, não por tendência.
    risk_adjust: bool = True
    vol_window_days: int = 252
    # Piso de volatilidade diária para evitar explosão numérica ao dividir
    # por desvios-padrão muito pequenos (papéis quase sem negociação).
    min_daily_vol: float = 1e-4

    # --- Universo elegível ---
    # Histórico mínimo de retornos antes da data de rebalanceamento, para
    # que o sinal de 12 meses já esteja plenamente formado.
    min_history_days: int = 252 + 21
    # Proxy de liquidez de FALLBACK: fração mínima de pregões com retorno
    # não-nulo na janela de observação. Usado quando ADTV real (COTAHIST)
    # não está disponível para o ticker/data — ver `use_external_liquidity`
    # abaixo e o módulo `src/universe.py`.
    liquidity_window_days: int = 126
    min_active_ratio: float = 0.90
    # Descarta retornos diários absurdos antes de qualquer cálculo (ver
    # src/data_loader.py::flag_suspicious_returns). Um |retorno| acima disso
    # é tratado como possível erro de cotação, não como sinal real.
    max_abs_daily_return: float = 1.00

    # --- Filtro de liquidez e tamanho (aplicado ANTES do ranking de momentum) ---
    # Quando ADTV (COTAHIST) e/ou valor de mercado (FRE) estão disponíveis
    # (ver src/cotahist.py, src/market_cap.py), o universo elegível passa
    # primeiro por este filtro de liquidez/tamanho, e só o que sobra entra
    # no cálculo e ranking do sinal de momentum — dois estágios distintos,
    # não um único corte combinado, para que a decisão de "é negociável"
    # não dependa de ter tido bom ou mau momentum recente.
    #
    # Cortes por PERCENTIL da seção transversal do dia (e não valor
    # absoluto em R$) por padrão: um piso nominal fixo perderia sentido ao
    # longo de 2010-2026 por causa de inflação/crescimento do mercado; um
    # corte relativo se mantém comparável ano a ano. Um piso absoluto
    # opcional pode ser somado por cima (None = desativado).
    liquidity_lookback_days: int = 63  # ~3 meses de pregão, para a ADTV móvel
    adtv_min_percentile: float = 0.30      # exclui o terço menos líquido
    market_cap_min_percentile: float = 0.20  # exclui o quinto menor em valor de mercado
    min_adtv_reais: float | None = None
    min_market_cap_reais: float | None = None

    # --- Seleção (com banda de turnover / buffer rule) ---
    # Percentil de corte para ENTRAR no índice (papel novo): top 30% do
    # universo elegível por score de momentum.
    entry_percentile: float = 0.70
    # Percentil de corte para PERMANECER no índice (papel já presente):
    # banda mais larga, top 40%. Só sai quem cair abaixo disso.
    hold_percentile: float = 0.60
    min_names: int = 30
    max_names: int = 60

    # --- Ponderação ---
    # Peso proporcional ao score de momentum (score-weighted), truncado em
    # zero, com teto por ativo para limitar concentração em nomes de sinal
    # muito extremo (geralmente os mais ilíquidos).
    weight_cap: float = 0.08

    # --- Rebalanceamento ---
    rebalance_freq: str = "Q"  # fim de trimestre
    # Defasagem entre o cálculo (fechamento da data de rebalance) e a data
    # em que os novos pesos passam a valer — simula o tempo operacional de
    # implementação da carteira.
    execution_lag_days: int = 2
    # Tolerância mínima de rebalanceamento: ignora ajustes de peso menores
    # que isso para reduzir giro por ruído numérico.
    min_weight_change: float = 0.0020

    # --- Custos de transação ---
    transaction_cost_bps: float = 25.0

    # --- Backtest ---
    backtest_start: str = "2010-01-01"
    backtest_end: str = "2026-12-31"

    # --- Benchmark ---
    benchmark_column: str = "ibovespa"
    risk_free_column: str = "cdi"


DEFAULT_CONFIG = MomentumConfig()
