"""Gera um dashboard HTML autocontido com os resultados do backtest.

Usa matplotlib (sem dependência de servidor) e embute as figuras como PNG
base64 dentro de um único arquivo HTML — abre em qualquer navegador sem
precisar rodar nada.

Paleta segue o padrão validado de acessibilidade (contraste e
distinguibilidade para daltonismo) do design system de referência: azul
para a série principal (índice), cinza-texto para o benchmark, com
vermelho reservado só para drawdown.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import metrics

COLOR_INDEX = "#2a78d6"      # slot categórico 1 (azul)
COLOR_BENCHMARK = "#898781"  # tinta neutra (muted)
COLOR_DRAWDOWN = "#e34948"   # slot categórico 8 (vermelho)
COLOR_TURNOVER = "#eb6834"   # slot categórico 2 (laranja)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "grid.color": GRID,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sanitize_for_plot(series: pd.Series, label: str) -> pd.Series:
    """Vira NaN qualquer +-inf antes de plotar.

    Um `inf` isolado (ex.: divisão por desvio-padrão zero numa janela
    móvel, ou um dado de entrada que escapou da limpeza em
    `data_loader.clean_returns`) faz o eixo do matplotlib tentar calcular
    `log10(inf)` e quebrar a geração do dashboard inteiro — sintoma visto
    em produção antes desta blindagem. Isso é uma rede de segurança, não
    substitui investigar a causa: se `n` for maior que zero, o aviso
    abaixo aparece no console para o problema ser rastreado na fonte.
    """
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan)
    n_removed = int(np.isinf(series.astype(float)).sum())
    if n_removed:
        print(f"  aviso: {n_removed} valor(es) não-finito(s) removido(s) do gráfico '{label}' "
              f"— investigue a fonte de dado correspondente.")
    return clean


def _cumulative_returns_chart(port_returns: pd.Series, bench_returns: pd.Series) -> str:
    port_curve = _sanitize_for_plot(metrics.cumulative_curve(port_returns), "retorno acumulado (índice)")
    bench_curve = _sanitize_for_plot(
        metrics.cumulative_curve(bench_returns.reindex(port_returns.index).fillna(0.0)),
        "retorno acumulado (Ibovespa)",
    )

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(port_curve.index, port_curve.values, color=COLOR_INDEX, linewidth=2, label="Índice Momentum")
    ax.plot(bench_curve.index, bench_curve.values, color=COLOR_BENCHMARK, linewidth=2, label="Ibovespa")
    ax.set_title("Retorno acumulado", fontsize=13, loc="left", color=TEXT_PRIMARY)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def _drawdown_chart(port_returns: pd.Series) -> str:
    dd = _sanitize_for_plot(metrics.drawdown_series(port_returns), "drawdown")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(dd.index, dd.values * 100, 0, color=COLOR_DRAWDOWN, alpha=0.35)
    ax.plot(dd.index, dd.values * 100, color=COLOR_DRAWDOWN, linewidth=1.2)
    ax.set_title("Drawdown (%)", fontsize=13, loc="left", color=TEXT_PRIMARY)
    return _fig_to_base64(fig)


def _turnover_chart(turnover_history: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(turnover_history.index, turnover_history.values * 100, color=COLOR_TURNOVER, width=40)
    ax.set_title("Turnover por rebalanceamento (%)", fontsize=13, loc="left", color=TEXT_PRIMARY)
    return _fig_to_base64(fig)


def _n_holdings_chart(n_holdings: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(n_holdings.index, n_holdings.values, color=COLOR_INDEX, linewidth=2, marker="o", markersize=4)
    ax.set_title("Número de ativos na carteira", fontsize=13, loc="left", color=TEXT_PRIMARY)
    return _fig_to_base64(fig)


def _rolling_sharpe_chart(port_returns: pd.Series, risk_free: pd.Series, window: int = 252) -> str:
    excess = port_returns - risk_free.reindex(port_returns.index).fillna(0.0)
    rolling_sharpe = (
        excess.rolling(window).mean() / excess.rolling(window).std(ddof=1)
    ) * (252 ** 0.5)
    rolling_sharpe = _sanitize_for_plot(rolling_sharpe, "Sharpe móvel")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLOR_INDEX, linewidth=1.6)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_title(f"Sharpe móvel ({window} pregões)", fontsize=13, loc="left", color=TEXT_PRIMARY)
    return _fig_to_base64(fig)


def _summary_table_html(summary: pd.Series) -> str:
    pct_rows = {
        "Retorno acumulado", "Retorno anualizado", "Volatilidade anualizada",
        "Máximo drawdown", "Turnover anualizado", "Tracking error",
        "Hit rate mensal vs Ibovespa", "Up-capture", "Down-capture",
        "Active share (última carteira)",
    }
    lines = []
    for label, value in summary.items():
        if pd.isna(value):
            formatted = "n/d"
        elif label in pct_rows:
            formatted = f"{value:.2%}"
        else:
            formatted = f"{value:.2f}"
        lines.append(f"<tr><td>{label}</td><td>{formatted}</td></tr>")
    return "\n".join(lines)


def build_dashboard(
    output_path: str | Path,
    port_returns_gross: pd.Series,
    port_returns_net: pd.Series,
    bench_returns: pd.Series,
    turnover_history: pd.Series,
    n_holdings_history: pd.Series,
    risk_free: pd.Series,
    summary: pd.Series,
    notes: list[str] | None = None,
) -> None:
    charts = {
        "cumulative": _cumulative_returns_chart(port_returns_net, bench_returns),
        "drawdown": _drawdown_chart(port_returns_net),
        "turnover": _turnover_chart(turnover_history),
        "n_holdings": _n_holdings_chart(n_holdings_history),
        "rolling_sharpe": _rolling_sharpe_chart(port_returns_net, risk_free),
    }
    table_html = _summary_table_html(summary)
    notes_html = "".join(f"<li>{n}</li>" for n in (notes or []))

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Índice de Momentum — Dashboard</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          background: #f9f9f7; color: {TEXT_PRIMARY}; margin: 0; padding: 32px; }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: {TEXT_SECONDARY}; margin-top: 0; margin-bottom: 28px; }}
  .card {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 10px;
           padding: 16px 20px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid {GRID}; }}
  td:first-child {{ color: {TEXT_SECONDARY}; }}
  td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
  img {{ width: 100%; height: auto; display: block; }}
  ul {{ font-size: 13px; color: {TEXT_SECONDARY}; }}
</style>
</head>
<body>
<div class="container">
  <h1>Índice de Momentum — Ações Brasileiras</h1>
  <p class="subtitle">Backtest long-only, rebalanceamento trimestral, líquido de custos de transação</p>

  <div class="card">
    <h2>Métricas de desempenho</h2>
    <table>{table_html}</table>
  </div>

  <div class="card"><img src="data:image/png;base64,{charts['cumulative']}"></div>
  <div class="card"><img src="data:image/png;base64,{charts['drawdown']}"></div>
  <div class="card"><img src="data:image/png;base64,{charts['rolling_sharpe']}"></div>
  <div class="card"><img src="data:image/png;base64,{charts['turnover']}"></div>
  <div class="card"><img src="data:image/png;base64,{charts['n_holdings']}"></div>

  <div class="card">
    <h2>Notas e limitações</h2>
    <ul>{notes_html}</ul>
  </div>
</div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
