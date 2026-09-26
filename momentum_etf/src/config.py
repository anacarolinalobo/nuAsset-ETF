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
    # Proxy de liquidez: fração mínima de pregões com retorno não-nulo e
    # não-ausente na janela de observação. Como a base não traz volume,
    # usamos a presença de negociação (retorno registrado) como proxy.
    liquidity_window_days: int = 126
    min_active_ratio: float = 0.90
    # Filtro de liquidez por volume real (opcional): exige ADTV (average
    # daily traded value) mínimo em R$ na janela de `volume_window_days`
    # pregões. Só é aplicado quando `eligible_universe` recebe um
    # `volume_wide` (ver src/volume_loader.py, que lê o COTAHIST da B3).
    # `None` desliga o filtro — é o padrão porque a base do case
    # (`acoes_retornos.csv`) não traz volume, só o proxy por retorno acima.
    min_adtv_brl: float | None = None
    volume_window_days: int = 21
    # Descarta retornos diários absurdos antes de qualquer cálculo (ver
    # src/data_loader.py::flag_suspicious_returns). Um |retorno| acima disso
    # é tratado como possível erro de cotação, não como sinal real.
    max_abs_daily_return: float = 1.00

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
