"""Motor de backtest: simula a carteira dia a dia entre rebalanceamentos.

Fluxo por pregão:
  1. Aplica o retorno do dia aos pesos vigentes (drift) — entre
     rebalanceamentos a carteira é buy-and-hold, os pesos variam com o
     preço relativo dos componentes.
  2. Detecta delistings: papel sem retorno observável hoje tem seu peso
     movido para caixa (proxy CDI) a partir de hoje (ver
     src/corporate_actions.py).
  3. Se hoje é uma data efetiva de rebalanceamento (calculada com a
     defasagem de execução), recalcula universo elegível, sinal, seleção e
     pesos usando dados até a DATA DE CÁLCULO do rebalance (não até hoje),
     aplica o turnover resultante e desconta o custo de transação do
     retorno do dia.

Nenhum dado posterior à data de cálculo de cada rebalance é usado para
decidir a composição daquele rebalance — é essa disciplina, linha a linha,
que elimina look-ahead bias no motor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.capacity import estimate_capacity, product_capacity
from src.config import MomentumConfig
from src.corporate_actions import detect_delistings
from src.costs import turnover_cost
from src.market_data import MarketData
from src.rebalance import effective_date, generate_rebalance_dates
from src.selection import select_portfolio
from src.signal import compute_signal_on_date
from src.universe import eligible_universe


@dataclass
class BacktestResult:
    returns_gross: pd.Series
    returns_net: pd.Series
    weights_history: dict[pd.Timestamp, pd.Series]
    turnover_history: pd.Series
    n_holdings_history: pd.Series
    cash_weight: pd.Series
    delistings_log: list[dict] = field(default_factory=list)
    config: MomentumConfig | None = None
    # Por rebalance (só quando há `market_data`): ADTV e market cap da
    # carteira e a capacidade estimada do produto naquela data.
    liquidity_history: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_backtest(
    returns_wide: pd.DataFrame,
    cdi_daily: pd.Series,
    config: MomentumConfig,
    market_data: MarketData | None = None,
) -> BacktestResult:
    trading_days = returns_wide.index[
        (returns_wide.index >= pd.Timestamp(config.backtest_start))
        & (returns_wide.index <= pd.Timestamp(config.backtest_end))
    ]
    calc_dates = generate_rebalance_dates(returns_wide.index, config)
    calc_dates = calc_dates[calc_dates < trading_days[-1]]
    effective_dates = {
        effective_date(d, returns_wide.index, config): d for d in calc_dates
    }

    first_effective = min(effective_dates) if effective_dates else trading_days[0]
    sim_days = trading_days[trading_days >= first_effective]

    cdi_daily = cdi_daily.reindex(returns_wide.index).fillna(0.0)

    weights = pd.Series(dtype=float)
    cash_weight = 1.0
    current_holdings: set[str] = set()

    gross_returns = pd.Series(index=sim_days, dtype=float)
    net_returns = pd.Series(index=sim_days, dtype=float)
    cash_weight_series = pd.Series(index=sim_days, dtype=float)
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_history: dict[pd.Timestamp, float] = {}
    n_holdings_history: dict[pd.Timestamp, int] = {}
    delistings_log: list[dict] = []
    liquidity_rows: list[dict] = []

    for date in sim_days:
        cost_today = 0.0

        delisted = detect_delistings(returns_wide, set(weights.index), date)
        if delisted:
            for t in delisted:
                cash_weight += weights.get(t, 0.0)
                weights = weights.drop(index=t, errors="ignore")
                delistings_log.append({"date": date, "ticker": t})
            current_holdings -= delisted

        day_returns = returns_wide.loc[date, weights.index].fillna(0.0) if not weights.empty else pd.Series(dtype=float)
        cdi_today = cdi_daily.loc[date]

        port_return_pre_cost = (
            (weights * day_returns).sum() + cash_weight * cdi_today
        )

        if date in effective_dates:
            calc_date = effective_dates[date]
            eligible = eligible_universe(returns_wide, calc_date, config, market_data)
            signal = compute_signal_on_date(returns_wide, calc_date, eligible, config)

            from src.weighting import compute_weights  # local import avoids cycle

            selected = select_portfolio(signal, current_holdings, config)
            adtv = mcap = None
            if market_data is not None:
                adtv = market_data.adtv(calc_date, config.adtv_window_days)
                mcap = market_data.market_cap_on(calc_date)
            target_weights = compute_weights(signal, selected, config, mcap, adtv)
            if market_data is not None:
                liquidity_rows.append(
                    _liquidity_snapshot(calc_date, target_weights, adtv, mcap, len(eligible), config)
                )

            drifted_weights = (weights * (1 + day_returns)) / (1 + port_return_pre_cost)
            drifted_weights = drifted_weights.reindex(target_weights.index.union(drifted_weights.index), fill_value=0.0)
            target_aligned = target_weights.reindex(drifted_weights.index, fill_value=0.0)

            turnover = (target_aligned - drifted_weights).abs().sum() / 2.0
            cost_today = turnover_cost(turnover, config.transaction_cost_bps)

            weights = target_weights
            cash_weight = 1.0 - weights.sum()
            current_holdings = selected

            weights_history[calc_date] = weights.copy()
            turnover_history[calc_date] = turnover
            n_holdings_history[calc_date] = len(selected)
        else:
            new_weights = (weights * (1 + day_returns)) / (1 + port_return_pre_cost)
            new_cash = cash_weight * (1 + cdi_today) / (1 + port_return_pre_cost)
            weights = new_weights
            cash_weight = new_cash

        gross_returns.loc[date] = port_return_pre_cost
        net_returns.loc[date] = port_return_pre_cost - cost_today
        cash_weight_series.loc[date] = cash_weight

    return BacktestResult(
        returns_gross=gross_returns,
        returns_net=net_returns,
        weights_history=weights_history,
        turnover_history=pd.Series(turnover_history),
        n_holdings_history=pd.Series(n_holdings_history),
        cash_weight=cash_weight_series,
        delistings_log=delistings_log,
        config=config,
        liquidity_history=pd.DataFrame(liquidity_rows).set_index("date") if liquidity_rows else pd.DataFrame(),
    )


def _liquidity_snapshot(
    calc_date: pd.Timestamp,
    weights: pd.Series,
    adtv: pd.Series,
    mcap: pd.Series,
    n_eligible: int,
    config: MomentumConfig,
) -> dict:
    held_adtv = adtv.reindex(weights.index)
    held_mcap = mcap.reindex(weights.index)
    capacity = estimate_capacity(
        weights,
        held_adtv,
        config.max_adtv_participation,
        config.days_to_build_position,
    )
    known_mcap = held_mcap.notna()
    return {
        "date": calc_date,
        "n_elegiveis": n_eligible,
        "n_carteira": len(weights),
        "peso_investido": float(weights.sum()),
        "adtv_mediano_carteira": float(held_adtv.median()) if len(held_adtv) else np.nan,
        "adtv_minimo_carteira": float(held_adtv.min()) if len(held_adtv) else np.nan,
        "adtv_ponderado_carteira": float((weights * held_adtv.fillna(0.0)).sum()),
        "market_cap_ponderado_carteira": (
            float((weights[known_mcap] * held_mcap[known_mcap]).sum() / weights[known_mcap].sum())
            if known_mcap.any()
            else np.nan
        ),
        "cobertura_market_cap_carteira": float(weights[known_mcap].sum() / weights.sum()) if weights.sum() else np.nan,
        "capacidade_produto_brl": product_capacity(capacity),
        "papel_gargalo": capacity.index[0] if not capacity.empty else None,
    }
