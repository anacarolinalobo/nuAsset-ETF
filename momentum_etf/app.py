"""Dashboard interativo do índice de momentum (Streamlit).

Uso:
    streamlit run app.py

Lê os mesmos dados de `data/` que `scripts/run_backtest.py` (e os
derivados de `data/derived/`, se `scripts/build_market_data.py` já rodou).
A frequência de rebalanceamento é o controle principal pedido — mensal,
trimestral ou semestral — mas outros parâmetros centrais da metodologia
também ficam expostos, porque "quão sensível é o resultado à
parametrização" é uma das perguntas que o case pede para responder, e a
forma mais direta de responder isso é deixar o próprio leitor mexer nos
parâmetros e ver o que muda.

O backtest é cacheado por combinação de parâmetros (`st.cache_data`) — a
primeira rodada de cada configuração leva alguns segundos; repetir uma
combinação já vista é instantâneo.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import attribution, data_loader, metrics
from src.backtest import run_backtest
from src.config import DEFAULT_CONFIG, MomentumConfig
from src.rebalance import generate_rebalance_dates

DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"

REBALANCE_LABELS = {"Mensal": "M", "Trimestral": "Q", "Semestral": "S"}

COLOR_INDEX = "#2a78d6"
COLOR_BENCHMARK = "#898781"
COLOR_DRAWDOWN = "#e34948"
COLOR_TURNOVER = "#eb6834"


st.set_page_config(page_title="Índice de Momentum — Ações Brasileiras", layout="wide")


# ============================================================
# Carga de dados (cacheada — roda uma vez por sessão)
# ============================================================

@st.cache_data(show_spinner="Carregando e limpando dados...")
def load_all_data():
    returns_path = DATA_DIR / "acoes_retornos.csv"
    ibov_path = DATA_DIR / "ibov_composicao.csv"
    bench_path = DATA_DIR / "benchmarks_diarios.csv"

    missing = [p.name for p in [returns_path, ibov_path, bench_path] if not p.exists()]
    if missing:
        return None, missing

    returns_wide = data_loader.load_returns(returns_path)
    ibov_weights = data_loader.load_ibov_composition(ibov_path)

    benchmarks_levels = data_loader.load_benchmarks(bench_path)
    benchmarks = data_loader.benchmarks_to_returns(benchmarks_levels)

    flags, flags_summary = data_loader.flag_suspicious_returns(
        returns_wide, DEFAULT_CONFIG.max_abs_daily_return
    )
    returns_wide = data_loader.clean_returns(returns_wide, flags)

    bench_flags, bench_flags_summary = data_loader.flag_suspicious_returns(
        benchmarks, DEFAULT_CONFIG.max_abs_daily_return
    )
    benchmarks = data_loader.clean_returns(benchmarks, bench_flags)

    adtv_wide = None
    market_cap_wide = None
    adtv_path = DERIVED_DIR / "adtv.csv"
    mcap_path = DERIVED_DIR / "market_cap.csv"
    if adtv_path.exists():
        adtv_wide = pd.read_csv(adtv_path, index_col=0, parse_dates=True)
    if mcap_path.exists():
        market_cap_wide = pd.read_csv(mcap_path, index_col=0, parse_dates=True)

    data = {
        "returns_wide": returns_wide,
        "ibov_weights": ibov_weights,
        "benchmarks": benchmarks,
        "adtv_wide": adtv_wide,
        "market_cap_wide": market_cap_wide,
        "flags_summary": flags_summary,
        "bench_flags_summary": bench_flags_summary,
        "n_flags": len(flags),
        "n_bench_flags": len(bench_flags),
    }
    return data, []


# ============================================================
# Backtest (cacheado por configuração — dados grandes ficam fora
# do hash via prefixo "_", só o `config` decide o cache)
# ============================================================

@st.cache_data(show_spinner="Rodando backtest...")
def run_cached_backtest(config: MomentumConfig, _returns_wide, _cdi_daily, _adtv_wide, _market_cap_wide):
    return run_backtest(_returns_wide, _cdi_daily, config, adtv_wide=_adtv_wide, market_cap_wide=_market_cap_wide)


def build_config(rebalance_freq: str, lookback_days: int, skip_days: int, entry_percentile: float,
                  weight_cap: float, transaction_cost_bps: float) -> MomentumConfig:
    hold_percentile = max(entry_percentile - 0.10, 0.0)
    return replace(
        DEFAULT_CONFIG,
        rebalance_freq=rebalance_freq,
        lookback_days=lookback_days,
        skip_days=skip_days,
        entry_percentile=entry_percentile,
        hold_percentile=hold_percentile,
        weight_cap=weight_cap,
        transaction_cost_bps=transaction_cost_bps,
    )


# ============================================================
# Gráficos (Plotly — interativos: hover, zoom, pan)
# ============================================================

def cumulative_returns_figure(port_returns: pd.Series, bench_returns: pd.Series) -> go.Figure:
    port_curve = metrics.cumulative_curve(port_returns)
    bench_curve = metrics.cumulative_curve(bench_returns.reindex(port_returns.index).fillna(0.0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_curve.index, y=port_curve.values, name="Índice Momentum",
                              line=dict(color=COLOR_INDEX, width=2)))
    fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve.values, name="Ibovespa",
                              line=dict(color=COLOR_BENCHMARK, width=2)))
    fig.update_layout(title="Retorno acumulado", template="plotly_white", height=420,
                       hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig


def drawdown_figure(port_returns: pd.Series) -> go.Figure:
    dd = metrics.drawdown_series(port_returns) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown",
                              line=dict(color=COLOR_DRAWDOWN, width=1.2)))
    fig.update_layout(title="Drawdown (%)", template="plotly_white", height=300, hovermode="x unified")
    return fig


def turnover_figure(turnover_history: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=turnover_history.index, y=turnover_history.values * 100,
                          name="Turnover", marker_color=COLOR_TURNOVER))
    fig.update_layout(title="Turnover por rebalanceamento (%)", template="plotly_white", height=300)
    return fig


def n_holdings_figure(n_holdings: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_holdings.index, y=n_holdings.values, mode="lines+markers",
                              line=dict(color=COLOR_INDEX, width=2)))
    fig.update_layout(title="Número de ativos na carteira", template="plotly_white", height=300)
    return fig


def rolling_sharpe_figure(port_returns: pd.Series, risk_free: pd.Series, window: int = 252) -> go.Figure:
    excess = port_returns - risk_free.reindex(port_returns.index).fillna(0.0)
    rolling_sharpe = (excess.rolling(window).mean() / excess.rolling(window).std(ddof=1)) * (252 ** 0.5)
    rolling_sharpe = rolling_sharpe.replace([float("inf"), float("-inf")], pd.NA)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, line=dict(color=COLOR_INDEX, width=1.6)))
    fig.add_hline(y=0, line_color="#c3c2b7")
    fig.update_layout(title=f"Sharpe móvel ({window} pregões)", template="plotly_white", height=300)
    return fig


# ============================================================
# App
# ============================================================

def main() -> None:
    st.title("Índice de Momentum — Ações Brasileiras")
    st.caption("Backtest long-only interativo. Ajuste os parâmetros na barra lateral.")

    data, missing = load_all_data()
    if data is None:
        st.error(
            "Faltam arquivos em `data/`: " + ", ".join(missing) +
            ". Rode `python scripts/make_sample_data.py` para gerar dados sintéticos de teste, "
            "ou coloque os CSVs reais do case em `data/`."
        )
        st.stop()

    with st.sidebar:
        st.header("Parâmetros")

        rebalance_label = st.selectbox(
            "Frequência de rebalanceamento",
            list(REBALANCE_LABELS.keys()),
            index=1,
            help="Mensal reage mais rápido ao sinal mas gira mais (mais custo); "
                 "semestral gira menos mas fica mais tempo desatualizado.",
        )
        rebalance_freq = REBALANCE_LABELS[rebalance_label]

        st.subheader("Sinal de momentum")
        lookback_days = st.slider("Janela do sinal (pregões)", 126, 378, DEFAULT_CONFIG.lookback_days, step=21)
        skip_days = st.slider("Skip — últimos N dias ignorados", 0, 42, DEFAULT_CONFIG.skip_days, step=21)

        st.subheader("Seleção e ponderação")
        entry_percentile = st.slider("Percentil de corte (entrada)", 0.50, 0.95, DEFAULT_CONFIG.entry_percentile, step=0.05)
        weight_cap = st.slider("Teto de peso por ativo (%)", 3, 20, int(DEFAULT_CONFIG.weight_cap * 100), step=1) / 100

        st.subheader("Custos")
        transaction_cost_bps = st.slider("Custo de transação (bps por lado)", 0, 100, int(DEFAULT_CONFIG.transaction_cost_bps), step=5)

        compare_frequencies = st.checkbox("Comparar as 3 frequências de rebalanceamento", value=False)

    config = build_config(rebalance_freq, lookback_days, skip_days, entry_percentile, weight_cap, transaction_cost_bps)

    cdi_daily = data["benchmarks"][config.risk_free_column]
    bench_returns = data["benchmarks"][config.benchmark_column]

    result = run_cached_backtest(config, data["returns_wide"], cdi_daily, data["adtv_wide"], data["market_cap_wide"])

    if not result.weights_history:
        st.warning("Nenhum rebalanceamento ocorreu no período configurado — ajuste os parâmetros ou o intervalo de datas.")
        st.stop()

    rebalance_dates = generate_rebalance_dates(data["returns_wide"].index, config)
    rebalances_per_year = 252 / (len(data["returns_wide"]) / max(len(rebalance_dates), 1))

    latest_date = max(result.weights_history.keys())
    latest_weights = result.weights_history[latest_date]
    latest_bench_weights = (
        data["ibov_weights"].reindex([latest_date], method="ffill").iloc[0]
        if not data["ibov_weights"].empty else pd.Series(dtype=float)
    )

    summary = metrics.summary_table(
        result.returns_net, bench_returns, cdi_daily, result.turnover_history,
        rebalances_per_year, latest_weights, latest_bench_weights,
    )

    # --- KPIs no topo ---
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Retorno anualizado", f"{summary['Retorno anualizado']:.2%}")
    kpi_cols[1].metric("Sharpe", f"{summary['Sharpe']:.2f}")
    kpi_cols[2].metric("Máximo drawdown", f"{summary['Máximo drawdown']:.2%}")
    kpi_cols[3].metric("Turnover anualizado", f"{summary['Turnover anualizado']:.2%}")
    kpi_cols[4].metric("Nº de ativos (atual)", f"{len(latest_weights)}")

    tab_desempenho, tab_carteira, tab_atribuicao, tab_comparar, tab_dados = st.tabs(
        ["Desempenho", "Carteira", "Atribuição", "Comparar frequências", "Qualidade de dado"]
    )

    with tab_desempenho:
        st.plotly_chart(cumulative_returns_figure(result.returns_net, bench_returns), use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(drawdown_figure(result.returns_net), use_container_width=True)
        col_b.plotly_chart(rolling_sharpe_figure(result.returns_net, cdi_daily), use_container_width=True)
        st.subheader("Todas as métricas")
        st.dataframe(summary.to_frame("valor"), use_container_width=True)

    with tab_carteira:
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(turnover_figure(result.turnover_history), use_container_width=True)
        col_b.plotly_chart(n_holdings_figure(result.n_holdings_history), use_container_width=True)
        st.subheader(f"Composição na última data de rebalance ({latest_date.date()})")
        st.dataframe(
            latest_weights.rename("peso").to_frame().sort_values("peso", ascending=False).style.format({"peso": "{:.2%}"}),
            use_container_width=True,
        )
        if result.delistings_log:
            st.subheader(f"Delistings detectados ({len(result.delistings_log)})")
            st.dataframe(pd.DataFrame(result.delistings_log), use_container_width=True)

    with tab_atribuicao:
        st.caption("Atribuição de contribuição vs. Ibovespa por janela entre rebalanceamentos (ver src/attribution.py).")
        attr = attribution.attribution_history(result.weights_history, data["ibov_weights"], data["returns_wide"])
        st.dataframe(attr.style.format("{:.2%}"), use_container_width=True)

    with tab_comparar:
        if compare_frequencies:
            rows = []
            for label, freq in REBALANCE_LABELS.items():
                cfg_i = replace(config, rebalance_freq=freq)
                res_i = run_cached_backtest(cfg_i, data["returns_wide"], cdi_daily, data["adtv_wide"], data["market_cap_wide"])
                dates_i = generate_rebalance_dates(data["returns_wide"].index, cfg_i)
                rpy_i = 252 / (len(data["returns_wide"]) / max(len(dates_i), 1))
                rows.append({
                    "Frequência": label,
                    "Retorno anualizado": metrics.annualized_return(res_i.returns_net),
                    "Volatilidade anualizada": metrics.annualized_vol(res_i.returns_net),
                    "Sharpe": metrics.sharpe_ratio(res_i.returns_net, cdi_daily),
                    "Máximo drawdown": metrics.max_drawdown(res_i.returns_net),
                    "Turnover anualizado": metrics.turnover_annualized(res_i.turnover_history, rpy_i),
                })
            comp = pd.DataFrame(rows).set_index("Frequência")
            st.dataframe(
                comp.style.format({
                    "Retorno anualizado": "{:.2%}", "Volatilidade anualizada": "{:.2%}",
                    "Sharpe": "{:.2f}", "Máximo drawdown": "{:.2%}", "Turnover anualizado": "{:.2%}",
                }),
                use_container_width=True,
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp.index, y=comp["Sharpe"], marker_color=COLOR_INDEX))
            fig.update_layout(title="Sharpe por frequência de rebalanceamento", template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Os demais parâmetros (janela do sinal, corte de seleção, custo) ficam fixos nos "
                "valores da barra lateral — essa comparação isola só o efeito da frequência."
            )
        else:
            st.info("Marque \"Comparar as 3 frequências\" na barra lateral para rodar mensal, trimestral e "
                    "semestral lado a lado (roda o backtest 3x, por isso fica atrás de um checkbox).")

    with tab_dados:
        st.subheader("Ações (acoes_retornos.csv)")
        if data["flags_summary"].empty:
            st.success("Nenhum retorno suspeito detectado.")
        else:
            st.warning(f"{data['n_flags']} observações sinalizadas e removidas do backtest (ver README, seção de limitações).")
            st.dataframe(data["flags_summary"], use_container_width=True)

        st.subheader("Benchmarks (benchmarks_diarios.csv)")
        if data["bench_flags_summary"].empty:
            st.success("Nenhum retorno suspeito detectado.")
        else:
            st.warning(f"{data['n_bench_flags']} observações sinalizadas e removidas.")
            st.dataframe(data["bench_flags_summary"], use_container_width=True)

        st.subheader("Cobertura de dados externos de liquidez/tamanho")
        st.write(f"ADTV (COTAHIST): {'disponível' if data['adtv_wide'] is not None else 'não encontrado — usando proxy de histórico'}")
        st.write(f"Valor de mercado (FRE/CVM): {'disponível' if data['market_cap_wide'] is not None else 'não encontrado — usando proxy de histórico'}")

        mapping_path = DERIVED_DIR / "ticker_cnpj_mapping.csv"
        if mapping_path.exists():
            mapping = pd.read_csv(mapping_path)
            n_matched = int(mapping["matched_high_confidence"].sum())
            st.write(f"Mapeamento ticker↔CNPJ: {n_matched}/{len(mapping)} tickers casados com confiança.")
            with st.expander("Ver mapeamento completo (revisar matches de baixa confiança)"):
                st.dataframe(mapping.sort_values("score"), use_container_width=True)


if __name__ == "__main__":
    main()
