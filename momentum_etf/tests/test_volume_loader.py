import pandas as pd

from src.volume_loader import (
    adtv_on_date,
    build_volume_wide,
    compute_adtv,
    load_cotahist,
    load_cotahist_file,
)

_WIDTHS = [2, 8, 2, 12, 3, 12, 10, 3, 4, 13, 13, 13, 13, 13, 13, 13, 5, 18, 18, 13, 1, 8, 7, 13, 12, 3]


def _field(value: str, width: int, numeric: bool) -> str:
    value = str(value)
    return value.rjust(width, "0") if numeric else value.ljust(width)[:width]


def _cotahist_row(
    tipo_registro="01",
    data="20200102",
    ticker="PETR4",
    tipo_mercado="010",
    volume_financeiro="150000",  # em centavos -> R$ 1.500,00
    quantidade_negociada="1000",
    numero_negocios="10",
) -> str:
    values_numeric = [
        (tipo_registro, False),
        (data, True),
        ("99", True),  # cod_bdi
        (ticker, False),  # ticker
        (tipo_mercado, True),
        ("EMPRESA X", False),
        ("ON", False),  # especificacao
        ("", False),  # prazo_termo
        ("R$", False),  # moeda
        ("1000", True),  # preco_abertura
        ("1000", True),  # preco_maximo
        ("1000", True),  # preco_minimo
        ("1000", True),  # preco_medio
        ("1000", True),  # preco_ultimo
        ("1000", True),  # preco_oferta_compra
        ("1000", True),  # preco_oferta_venda
        (numero_negocios, True),
        (quantidade_negociada, True),
        (volume_financeiro, True),
        ("0", True),  # preco_exercicio
        ("", False),  # indicador_correcao
        ("99991231", True),  # data_vencimento
        ("1", True),  # fator_cotacao
        ("0", True),  # pontos_exercicio
        ("BRPETRACNPR6", False),  # isin
        ("126", True),  # numero_distribuicao
    ]
    assert len(values_numeric) == len(_WIDTHS)
    return "".join(_field(v, w, is_num) for (v, is_num), w in zip(values_numeric, _WIDTHS))


def test_load_cotahist_file_parses_volume(tmp_path):
    path = tmp_path / "COTAHIST_A2020.TXT"
    lines = [
        _cotahist_row(ticker="PETR4", data="20200102", volume_financeiro="150000"),
        _cotahist_row(ticker="PETR4", data="20200103", volume_financeiro="200000"),
        # deve ser excluído: mercado a termo, não a vista
        _cotahist_row(ticker="PETR4", data="20200103", tipo_mercado="020"),
        # deve ser excluído: não é ticker de ação (unit)
        _cotahist_row(ticker="XPTO11", data="20200103"),
        # deve ser excluído: registro de cabeçalho/rodapé
        _cotahist_row(tipo_registro="99", data="20200103"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="latin1")

    df = load_cotahist_file(path)

    assert set(df["ticker"]) == {"PETR4"}
    assert len(df) == 2
    first = df.sort_values("data").iloc[0]
    assert first["volume_financeiro"] == 1500.0
    assert first["quantidade_negociada"] == 1000
    assert first["numero_negocios"] == 10


def test_load_cotahist_concatenates_multiple_files(tmp_path):
    (tmp_path / "COTAHIST_A2020.TXT").write_text(
        _cotahist_row(ticker="VALE3", data="20200102") + "\n", encoding="latin1"
    )
    (tmp_path / "COTAHIST_A2021.TXT").write_text(
        _cotahist_row(ticker="VALE3", data="20210104") + "\n", encoding="latin1"
    )

    df = load_cotahist(tmp_path)

    assert len(df) == 2
    assert list(df["data"].dt.year) == [2020, 2021]


def test_build_volume_wide_and_adtv(tmp_path):
    lines = [
        _cotahist_row(ticker="PETR4", data="20200102", volume_financeiro="100000"),
        _cotahist_row(ticker="PETR4", data="20200103", volume_financeiro="200000"),
        _cotahist_row(ticker="VALE3", data="20200102", volume_financeiro="500000"),
    ]
    path = tmp_path / "COTAHIST_A2020.TXT"
    path.write_text("\n".join(lines) + "\n", encoding="latin1")

    volume_wide = build_volume_wide(load_cotahist_file(path))

    assert volume_wide.loc[pd.Timestamp("2020-01-02"), "PETR4"] == 1000.0
    assert volume_wide.loc[pd.Timestamp("2020-01-03"), "PETR4"] == 2000.0

    adtv = adtv_on_date(volume_wide, pd.Timestamp("2020-01-03"), window_days=5)
    assert adtv["PETR4"] == 1500.0

    adtv_series = compute_adtv(volume_wide, window_days=5)
    assert adtv_series.loc[pd.Timestamp("2020-01-03"), "PETR4"] == 1500.0
