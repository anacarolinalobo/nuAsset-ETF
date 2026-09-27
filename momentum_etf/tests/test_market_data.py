import numpy as np
import pandas as pd
import pytest

from dataclasses import replace

from src.backtest import run_backtest
from src.market_data import MarketData, load_market_data
from src.universe import eligible_universe
from src.weighting import compute_weights, liquidity_caps


@pytest.fixture
def market(synthetic_returns) -> MarketData:
    idx = synthetic_returns.index
    tickers = synthetic_returns.columns
    # T0-T7: R$ 20 mi/dia; T8: R$ 1 mi/dia (ilíquido)
    volume = pd.DataFrame(20e6, index=idx, columns=tickers)
    volume["T8"] = 1e6
    volume = volume.where(synthetic_returns.notna(), 0.0)
    # market cap mensal; T1 é micro cap, T2 não tem market cap
    month_ends = idx[idx.to_period("M") != idx.shift(-1, freq="B").to_period("M")]
    mcap = pd.DataFrame(5e9, index=month_ends, columns=tickers)
    mcap["T1"] = 1e8
    mcap = mcap.drop(columns=["T2"])
    mcap = mcap.reindex(idx).ffill()
    return MarketData(volume=volume, market_cap=mcap)


def test_filters_illiquid_and_small_caps(synthetic_returns, small_config, market):
    date = synthetic_returns.index[synthetic_returns.index >= "2020-06-01"][0]
    universe = eligible_universe(synthetic_returns, date, small_config, market)
    assert "T8" not in universe  # ADTV abaixo do mínimo
    assert "T1" not in universe  # market cap abaixo do mínimo
    assert "T2" in universe  # sem market cap: passa por padrão
    assert "T0" in universe


def test_require_market_cap_excludes_missing(synthetic_returns, small_config, market):
    cfg = replace(small_config, require_market_cap=True)
    date = synthetic_returns.index[synthetic_returns.index >= "2020-06-01"][0]
    universe = eligible_universe(synthetic_returns, date, cfg, market)
    assert "T2" not in universe


def test_market_cap_uses_only_past_values(market):
    date = market.market_cap.index[40]
    mcap = market.market_cap_on(date)
    assert mcap["T0"] == 5e9


def test_sqrt_mcap_tilts_toward_larger_names(small_config):
    cfg = replace(small_config, weighting_scheme="score_sqrt_mcap", weight_cap=1.0)
    signal = pd.Series({"A": 1.0, "B": 1.0})
    mcap = pd.Series({"A": 4e9, "B": 1e9})
    w = compute_weights(signal, {"A", "B"}, cfg, market_cap=mcap)
    assert w["A"] == pytest.approx(2 / 3)


def test_liquidity_cap_leaves_residual_in_cash(small_config):
    cfg = replace(small_config, target_aum_brl=100e6, weight_cap=1.0)
    signal = pd.Series({"A": 1.0, "B": 1.0})
    adtv = pd.Series({"A": 20e6, "B": 20e6})  # 20mi*10%*5/100mi = 10% cada
    caps = liquidity_caps(signal.index, adtv, cfg)
    assert caps["A"] == pytest.approx(0.10)
    w = compute_weights(signal, {"A", "B"}, cfg, adtv=adtv)
    assert w.sum() == pytest.approx(0.20)


def test_liquidity_cap_redistributes_when_room(small_config):
    cfg = replace(small_config, target_aum_brl=10e6, weight_cap=0.6)
    signal = pd.Series({"A": 3.0, "B": 1.0})
    adtv = pd.Series({"A": 20e6, "B": 20e6})  # tetos de liquidez folgados -> weight_cap manda
    w = compute_weights(signal, {"A", "B"}, cfg, adtv=adtv)
    assert w["A"] == pytest.approx(0.6)
    assert w["B"] == pytest.approx(0.4)


def test_backtest_records_liquidity_history(synthetic_returns, synthetic_cdi, small_config, market):
    result = run_backtest(synthetic_returns, synthetic_cdi, small_config, market)
    hist = result.liquidity_history
    assert not hist.empty
    assert {"capacidade_produto_brl", "adtv_mediano_carteira"} <= set(hist.columns)
    for w in result.weights_history.values():
        assert "T8" not in w.index


def test_load_market_data_aligns_calendar(tmp_path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-31"],
            "ticker": ["abcd3", "ABCD3", "ABCD3"],
            "volume_financeiro": [1e6, 2e6, 3e6],
            "market_cap": [1e9, np.nan, 2e9],
        }
    )
    path = tmp_path / "liquidez_mercado.csv"
    df.to_csv(path, index=False)
    days = pd.bdate_range("2020-01-02", "2020-02-05")
    md = load_market_data(path, days)
    assert md.volume.loc["2020-01-06", "ABCD3"] == 0.0
    assert md.market_cap.loc["2020-01-15", "ABCD3"] == 1e9
    assert md.market_cap.loc["2020-02-05", "ABCD3"] == 2e9
