import pandas as pd

from src.company_mapping import build_ticker_cnpj_mapping, normalize_company_name


def test_normalize_strips_legal_suffix_and_accents():
    assert normalize_company_name("Petróleo Brasileiro S.A.") == "PETROLEO BRASILEIRO"
    assert normalize_company_name("Vale S/A") == "VALE"


def test_normalize_handles_nan():
    assert normalize_company_name(float("nan")) == ""


def test_build_mapping_matches_identical_names():
    instruments = pd.DataFrame({"ticker": ["AAAA3", "BBBB4"], "nome_empresa": ["CIA ALPHA", "CIA BETA"]})
    companies = pd.DataFrame(
        {"CNPJ_Companhia": ["111", "222"], "Nome_Companhia": ["CIA ALPHA", "CIA BETA"]}
    )

    mapping = build_ticker_cnpj_mapping(instruments, companies, min_score=80.0)

    assert mapping.set_index("ticker").loc["AAAA3", "cnpj"] == "111"
    assert mapping.set_index("ticker").loc["BBBB4", "cnpj"] == "222"
    assert mapping["matched_high_confidence"].all()


def test_build_mapping_leaves_unmatched_below_threshold():
    instruments = pd.DataFrame({"ticker": ["ZZZZ9"], "nome_empresa": ["COMPLETAMENTE DIFERENTE"]})
    companies = pd.DataFrame({"CNPJ_Companhia": ["999"], "Nome_Companhia": ["OUTRA EMPRESA XPTO"]})

    mapping = build_ticker_cnpj_mapping(instruments, companies, min_score=95.0)

    row = mapping.iloc[0]
    assert not row["matched_high_confidence"]
    assert row["cnpj"] is None
