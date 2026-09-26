import pandas as pd

from src.market_cap import build_market_cap


def test_market_cap_combines_price_and_shares():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    close_price = pd.DataFrame({"AAAA3": [10.0, 11.0, 12.0]}, index=dates)

    capital_social = pd.DataFrame(
        {
            "CNPJ_Companhia": ["111"],
            "Nome_Companhia": ["CIA ALPHA"],
            "Data_Referencia": [pd.Timestamp("2019-12-01")],
            "Quantidade_Total_Acoes": [1_000_000],
        }
    )

    mapping = pd.DataFrame(
        {
            "ticker": ["AAAA3"],
            "cnpj": ["111"],
            "matched_high_confidence": [True],
        }
    )

    mcap = build_market_cap(close_price, capital_social, mapping)

    assert list(mcap["AAAA3"]) == [10_000_000.0, 11_000_000.0, 12_000_000.0]


def test_market_cap_skips_unmatched_tickers():
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    close_price = pd.DataFrame({"AAAA3": [10.0, 11.0]}, index=dates)

    capital_social = pd.DataFrame(
        {
            "CNPJ_Companhia": ["111"],
            "Nome_Companhia": ["CIA ALPHA"],
            "Data_Referencia": [pd.Timestamp("2019-12-01")],
            "Quantidade_Total_Acoes": [1_000_000],
        }
    )
    mapping = pd.DataFrame({"ticker": ["AAAA3"], "cnpj": [None], "matched_high_confidence": [False]})

    mcap = build_market_cap(close_price, capital_social, mapping)
    assert "AAAA3" not in mcap.columns


def test_market_cap_forward_fills_share_count_between_filings():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    close_price = pd.DataFrame({"AAAA3": [10.0] * 5}, index=dates)

    capital_social = pd.DataFrame(
        {
            "CNPJ_Companhia": ["111", "111"],
            "Nome_Companhia": ["CIA ALPHA", "CIA ALPHA"],
            "Data_Referencia": [pd.Timestamp("2019-12-01"), dates[2]],
            "Quantidade_Total_Acoes": [1_000_000, 2_000_000],
        }
    )
    mapping = pd.DataFrame({"ticker": ["AAAA3"], "cnpj": ["111"], "matched_high_confidence": [True]})

    mcap = build_market_cap(close_price, capital_social, mapping)

    assert mcap["AAAA3"].iloc[1] == 10_000_000.0
    assert mcap["AAAA3"].iloc[3] == 20_000_000.0
