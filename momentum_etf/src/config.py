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
    # Descarta retornos diários absurdos antes de qualquer cálculo (ver
    # src/data_loader.py::flag_suspicious_returns). Um |retorno| acima disso
    # é tratado como possível erro de cotação, não como sinal real.
    max_abs_daily_return: float = 1.00

    # --- Liquidez, volume e market cap (B3 COTAHIST + CVM) ---
    # Só usados quando `liquidez_mercado.csv` é fornecido (ver
    # src/market_data.py); sem esse arquivo o universo cai de volta no
    # proxy de presença de retorno acima.
    # Janela do ADTV (mediana do volume financeiro diário): ~3 meses, o
    # padrão de índices como MSCI/S&P para medir liquidez corrente.
    adtv_window_days: int = 63
    # Liquidez mínima para entrar no universo: R$ 5 mi/dia de mediana.
    # Abaixo disso, montar/desmontar uma posição de um produto de porte
    # institucional leva dias demais e o custo de impacto domina o prêmio.
    min_adtv_brl: float = 5_000_000.0
    # Tamanho mínimo: R$ 500 mi de market cap. Corta micro caps, onde o
    # sinal de momentum é mais ruidoso e o preço mais manipulável.
    min_market_cap_brl: float = 500_000_000.0
    # Papel sem market cap na base (cobertura CVM x B3 incompleta) passa
    # pelo filtro de tamanho se False; é excluído se True.
    require_market_cap: bool = False

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
    # Esquema de ponderação:
    #   "score"           : peso ∝ score (padrão original)
    #   "score_sqrt_mcap" : peso ∝ score x sqrt(market cap) — inclina para
    #                       nomes maiores sem virar cap-weight
    weighting_scheme: str = "score"
    # Teto de peso por liquidez: se `target_aum_brl` for definido, nenhum
    # papel pode ter peso maior que o que se monta em
    # `days_to_build_position` pregões negociando no máximo
    # `max_adtv_participation` do ADTV. O que não couber fica em caixa (CDI).
    target_aum_brl: float | None = None
    max_adtv_participation: float = 0.10
    days_to_build_position: int = 5

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
