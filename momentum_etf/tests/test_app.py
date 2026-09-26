"""Testes de fumaça do dashboard Streamlit via streamlit.testing.v1.AppTest.

Rodam contra os dados sintéticos em `data/` (gerados por
`scripts/make_sample_data.py` + `scripts/build_market_data.py`) — pulam
com uma mensagem clara se esses arquivos ainda não foram gerados, em vez
de falhar com um erro de arquivo ausente difícil de interpretar.
"""

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"

DATA_READY = (
    (ROOT / "data" / "acoes_retornos.csv").exists()
    and (ROOT / "data" / "ibov_composicao.csv").exists()
    and (ROOT / "data" / "benchmarks_diarios.csv").exists()
)

pytestmark = pytest.mark.skipif(
    not DATA_READY,
    reason="Rode scripts/make_sample_data.py antes (dados sintéticos ausentes em data/)",
)


def _run_app() -> "AppTest":
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    return at


def test_app_loads_without_exception():
    at = _run_app()
    assert not at.exception


def test_app_shows_kpi_metrics():
    at = _run_app()
    assert len(at.metric) >= 5
    labels = {m.label for m in at.metric}
    assert "Sharpe" in labels
    assert "Máximo drawdown" in labels


def test_app_has_rebalance_frequency_selector_with_three_options():
    at = _run_app()
    selectboxes = at.selectbox
    assert len(selectboxes) >= 1
    freq_box = selectboxes[0]
    assert set(freq_box.options) == {"Mensal", "Trimestral", "Semestral"}


def test_changing_rebalance_frequency_reruns_without_exception():
    at = _run_app()
    at.selectbox[0].set_value("Mensal").run()
    assert not at.exception

    at.selectbox[0].set_value("Semestral").run()
    assert not at.exception


def test_pca_tab_renders_with_universe_choice_radio():
    at = _run_app()
    radios = [r for r in at.radio if "Conjunto de ações" in r.label]
    assert radios
    assert set(radios[0].options) == {"Carteira atual", "Universo elegível (mesma data)"}


def test_switching_pca_universe_choice_reruns_without_exception():
    at = _run_app()
    radios = [r for r in at.radio if "Conjunto de ações" in r.label]
    radios[0].set_value("Universo elegível (mesma data)").run()
    assert not at.exception


def test_compare_frequencies_checkbox_renders_table():
    at = _run_app()
    checkboxes = [c for c in at.checkbox if "Comparar" in c.label]
    assert checkboxes
    checkboxes[0].set_value(True).run(timeout=180)
    assert not at.exception
    assert len(at.dataframe) >= 1
