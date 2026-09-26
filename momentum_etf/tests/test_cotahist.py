from pathlib import Path

import pandas as pd
import pytest

from src.cotahist import (
    _COLSPECS,
    close_price_wide,
    compute_adtv,
    instrument_reference,
    parse_cotahist_file,
    traded_value_wide,
)


def _fmt(value: str, width: int, numeric: bool) -> str:
    text = str(value)[:width]
    return text.rjust(width, "0") if numeric else text.ljust(width)


def _write_cotahist_line(date: str, ticker: str, preco: float, qty: int, volume: float) -> str:
    fields = [
        "01", date, "02", ticker, "010", f"CIA {ticker}", "ON      NM", "0", "R$  ",
        f"{preco*100:.0f}", f"{preco*100:.0f}", f"{preco*100:.0f}", f"{preco*100:.0f}",
        f"{preco*100:.0f}", f"{preco*100:.0f}", f"{preco*100:.0f}",
        "10", f"{qty}", f"{volume*100:.0f}",
        "0", "0", "99991231", "0000001", "0", f"BR{ticker}00", "126",
    ]
    numeric_flags = [
        False, False, False, False, False, False, False, False, False,
        True, True, True, True, True, True, True, True, True, True,
        True, False, False, True, True, False, False,
    ]
    return "".join(
        _fmt(val, spec[1] - spec[0], is_num) for val, spec, is_num in zip(fields, _COLSPECS, numeric_flags)
    )


@pytest.fixture
def cotahist_file(tmp_path) -> Path:
    lines = [
        _write_cotahist_line("20200102", "AAAA3", 10.0, 1000, 10_000.0),
        _write_cotahist_line("20200102", "BBBB4", 20.0, 500, 10_000.0),
        _write_cotahist_line("20200103", "AAAA3", 10.5, 1000, 10_500.0),
    ]
    path = tmp_path / "COTAHIST_A2020.TXT"
    path.write_text("\n".join(lines), encoding="latin1")
    return path


def test_parse_cotahist_file_extracts_expected_fields(cotahist_file):
    df = parse_cotahist_file(cotahist_file)
    assert set(df["ticker"]) == {"AAAA3", "BBBB4"}
    assert len(df) == 3

    row = df[(df["ticker"] == "AAAA3") & (df["date"] == pd.Timestamp("2020-01-02"))].iloc[0]
    assert abs(row["preco_ultimo"] - 10.0) < 1e-6
    assert abs(row["volume_financeiro"] - 10_000.0) < 1e-6


def test_close_price_and_traded_value_wide(cotahist_file):
    df = parse_cotahist_file(cotahist_file)
    prices = close_price_wide(df)
    volumes = traded_value_wide(df)

    assert prices.loc[pd.Timestamp("2020-01-03"), "AAAA3"] == 10.5
    assert volumes.loc[pd.Timestamp("2020-01-02"), "BBBB4"] == 10_000.0


def test_compute_adtv_rolling_mean(cotahist_file):
    df = parse_cotahist_file(cotahist_file)
    volumes = traded_value_wide(df)
    adtv = compute_adtv(volumes, window_days=2)

    expected = (10_000.0 + 10_500.0) / 2
    assert abs(adtv.loc[pd.Timestamp("2020-01-03"), "AAAA3"] - expected) < 1e-6


def test_instrument_reference_keeps_latest_name(cotahist_file):
    df = parse_cotahist_file(cotahist_file)
    ref = instrument_reference(df)
    assert set(ref["ticker"]) == {"AAAA3", "BBBB4"}
    assert (ref["nome_empresa"].str.startswith("CIA")).all()
