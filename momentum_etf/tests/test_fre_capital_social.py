import pandas as pd
import pytest

from src.fre_capital_social import load_capital_social, shares_outstanding_by_cnpj


@pytest.fixture
def fre_dir(tmp_path):
    year_dir = tmp_path / "fre_cia_aberta_2020"
    year_dir.mkdir()
    df = pd.DataFrame(
        {
            "CNPJ_Companhia": ["11.111.111/0001-11", "11.111.111/0001-11", "22.222.222/0001-22"],
            "Nome_Companhia": ["EMPRESA A", "EMPRESA A", "EMPRESA B"],
            "Data_Referencia": ["2020-03-31", "2020-03-31", "2020-06-30"],
            "Tipo_Capital": ["Capital Subscrito", "Capital Integralizado", "Capital Integralizado"],
            "Quantidade_Acoes_Ordinarias": [1_000_000, 1_000_000, 500_000],
            "Quantidade_Acoes_Preferenciais": [0, 0, 0],
            "Quantidade_Total_Acoes": [1_000_000, 1_000_000, 500_000],
        }
    )
    df.to_csv(year_dir / "fre_cia_aberta_capital_social_2020.csv", sep=";", encoding="latin1", index=False)
    return tmp_path


def test_load_capital_social_dedups_by_priority(fre_dir):
    result = load_capital_social(fre_dir, start_year=2020, end_year=2020)
    # duas linhas para a mesma empresa/data -> deve colapsar em uma
    empresa_a = result[result["Nome_Companhia"] == "EMPRESA A"]
    assert len(empresa_a) == 1


def test_cnpj_normalized_to_14_digits(fre_dir):
    result = load_capital_social(fre_dir, start_year=2020, end_year=2020)
    assert (result["CNPJ_Companhia"].str.len() == 14).all()
    assert (result["CNPJ_Companhia"].str.isdigit()).all()


def test_missing_folder_for_year_is_skipped(fre_dir):
    # 2021 não existe, não deve quebrar - só não traz dado daquele ano
    result = load_capital_social(fre_dir, start_year=2020, end_year=2021)
    assert result["Data_Referencia"].dt.year.max() == 2020


def test_shares_outstanding_by_cnpj_indexed_by_date(fre_dir):
    result = load_capital_social(fre_dir, start_year=2020, end_year=2020)
    by_cnpj = shares_outstanding_by_cnpj(result)
    assert len(by_cnpj) == 2
    for series in by_cnpj.values():
        assert isinstance(series.index, pd.DatetimeIndex)


def test_raises_when_no_files_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_capital_social(tmp_path, start_year=2099, end_year=2099)
