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

from src import attribution, data_loader, metrics, pca_analysis
from src.backtest import run_backtest
from src.config import DEFAULT_CONFIG, MomentumConfig
from src.rebalance import generate_rebalance_dates
from src.signal import compute_signal_on_date
from src.universe import eligible_universe

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


def pca_scree_figure(explained_variance_ratio: pd.Series) -> go.Figure:
    cumulative = explained_variance_ratio.cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=explained_variance_ratio.index, y=explained_variance_ratio.values * 100,
                          name="Variância explicada", marker_color=COLOR_INDEX))
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values * 100, name="Acumulada",
                              line=dict(color=COLOR_TURNOVER, width=2)))
    fig.update_layout(title="Variância explicada por componente", template="plotly_white", height=350,
                       yaxis_title="%", legend=dict(orientation="h", y=1.15))
    return fig


def pca_scatter_figure(merged: pd.DataFrame, corr: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["carga_pc1"], y=merged["score_momentum"], mode="markers", text=merged.index,
        hovertemplate="%{text}<br>carga PC1: %{x:.2f}<br>score momentum: %{y:.2f}<extra></extra>",
        marker=dict(color=COLOR_INDEX, size=7, opacity=0.7),
    ))
    fig.update_layout(
        title=f"Carga no PC1 (proxy de beta de mercado) vs. score de momentum — correlação {corr:.2f}",
        template="plotly_white", height=420,
        xaxis_title="Carga no PC1", yaxis_title="Score de momentum (z-score)",
    )
    return fig


def pca_loadings_heatmap_figure(loadings: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=loadings.values, x=list(loadings.columns), y=list(loadings.index),
        colorscale="RdBu", zmid=0, colorbar=dict(title="carga"),
    ))
    fig.update_layout(title="Cargas por componente", template="plotly_white",
                       height=max(320, 26 * len(loadings)))
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

    tab_desempenho, tab_carteira, tab_atribuicao, tab_pca, tab_comparar, tab_dados = st.tabs(
        ["Desempenho", "Carteira", "Atribuição", "PCA", "Comparar frequências", "Qualidade de dado"]
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

    with tab_pca:
        st.caption(
            "PCA (componentes principais) sobre retornos padronizados — decompõe o risco em fatores "
            "ortogonais. Metodologia completa e limitações em src/pca_analysis.py."
        )
        pca_universe_tab, pca_asset_tab = st.tabs(["Risco da carteira", "Classes de ativos"])

        full_eligible = eligible_universe(
            data["returns_wide"], latest_date, config,
            adtv_wide=data["adtv_wide"], market_cap_wide=data["market_cap_wide"],
        )
        momentum_score_full = compute_signal_on_date(data["returns_wide"], latest_date, full_eligible, config)

        with pca_universe_tab:
            st.markdown("**Quanto do risco da carteira é sistemático (poucos fatores) vs. diversificado?** "
                        "E o sinal de momentum é distinto de só \"comprar ações de maior beta\"?")
            universe_choice = st.radio(
                "Conjunto de ações", ["Carteira atual", "Universo elegível (mesma data)"], horizontal=True,
            )
            pca_window = st.slider("Janela (pregões)", 126, 504, 252, step=21, key="pca_window_universe")

            tickers_for_pca = (
                list(latest_weights.index) if universe_choice == "Carteira atual" else list(full_eligible)
            )

            try:
                pca_result = pca_analysis.compute_universe_pca(
                    data["returns_wide"], tickers_for_pca, latest_date, window_days=pca_window, n_components=10,
                )
            except ValueError as exc:
                st.warning(str(exc))
            else:
                n90 = pca_result.n_components_for_variance(0.90)
                kpi_a, kpi_b, kpi_c = st.columns(3)
                kpi_a.metric("PC1 explica", f"{pca_result.explained_variance_ratio.iloc[0]:.1%}")
                kpi_b.metric("Componentes p/ 90% da variância", f"{n90} de {pca_result.n_assets}")
                kpi_c.metric("Ativos usados no PCA", f"{pca_result.n_assets} de {len(tickers_for_pca)}")

                st.plotly_chart(pca_scree_figure(pca_result.explained_variance_ratio), use_container_width=True)

                merged = pca_analysis.pc1_loading_vs_momentum_score(pca_result.loadings, momentum_score_full)
                if len(merged) >= 3:
                    corr = merged["carga_pc1"].corr(merged["score_momentum"])
                    st.plotly_chart(pca_scatter_figure(merged, corr), use_container_width=True)
                    if corr > 0.5:
                        st.info(f"Correlação alta ({corr:.2f}) entre carga no PC1 e score de momentum — "
                                "indício de que parte do sinal aqui é redundante com simplesmente "
                                "comprar ações de maior beta de mercado, não um fator distinto.")
                    else:
                        st.success(f"Correlação baixa ({corr:.2f}) entre carga no PC1 e score de momentum — "
                                   "o sinal parece captar algo além de só \"beta alto\".")

                if pca_result.dropped_assets:
                    with st.expander(f"{len(pca_result.dropped_assets)} ativo(s) descartado(s) por histórico incompleto na janela"):
                        st.write(pca_result.dropped_assets)

        with pca_asset_tab:
            st.markdown("**Com quais classes de ativos o índice compete (mesmo fator de risco) e quais "
                        "complementa (fatores praticamente ortogonais)?**")
            pca_window_asset = st.slider("Janela (pregões)", 126, 756, 504, step=21, key="pca_window_asset")

            try:
                asset_pca = pca_analysis.compute_asset_class_pca(
                    data["benchmarks"], result.returns_net, latest_date,
                    window_days=pca_window_asset, n_components=10,
                )
            except ValueError as exc:
                st.warning(str(exc))
            else:
                st.plotly_chart(pca_scree_figure(asset_pca.explained_variance_ratio), use_container_width=True)
                st.plotly_chart(pca_loadings_heatmap_figure(asset_pca.loadings), use_container_width=True)

                dominant_pc, ranked = pca_analysis.dominant_component_for_asset(asset_pca.loadings, "Índice Momentum")
                momentum_loading = ranked["Índice Momentum"]
                competitors = ranked.drop("Índice Momentum")
                close = competitors[(competitors - momentum_loading).abs() < 0.25].index.tolist()
                far = competitors[competitors.abs() < 0.15].index.tolist()

                st.markdown(f"O Índice Momentum carrega mais forte em **{dominant_pc}** (carga {momentum_loading:.2f}).")
                if close:
                    st.markdown(f"**Compete por risco com** (carga parecida no mesmo componente): {', '.join(close)}")
                if far:
                    st.markdown(f"**Complementa** (carga baixa nesse componente, fator quase ortogonal): {', '.join(far)}")
                if not close and not far:
                    st.markdown("Nenhum outro ativo com carga claramente próxima ou claramente ortogonal nesse componente.")

                if asset_pca.dropped_assets:
                    st.caption(f"Descartado por histórico incompleto na janela: {', '.join(asset_pca.dropped_assets)}")

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
