"""Carga e limpeza dos três CSVs fornecidos no case.

Formatos esperados (conforme o briefing):
    acoes_retornos.csv     : date, ticker, retorno   (long)
    ibov_composicao.csv    : date, ticker, peso       (long)
    benchmarks_diarios.csv : date, cdi, ima_s, idka_pre_3a, ima_b, ihfa,
                              ifix, ibovespa, sp500_brl, bitcoin_brl (wide)

Nenhuma função aqui decide metodologia de índice — só carrega, valida e
padroniza o formato dos dados brutos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_returns(path: str | Path) -> pd.DataFrame:
    """Lê acoes_retornos.csv e devolve retornos em formato wide (date x ticker)."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "retorno"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"acoes_retornos.csv sem colunas obrigatórias: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    wide = df.pivot(index="date", columns="ticker", values="retorno").sort_index()
    wide.index.name = "date"
    wide.columns.name = "ticker"
    return wide


def load_ibov_composition(path: str | Path) -> pd.DataFrame:
    """Lê ibov_composicao.csv e devolve pesos em formato wide (date x ticker)."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ibov_composicao.csv sem colunas obrigatórias: {missing}")

    weight_col = "peso" if "peso" in df.columns else "weight"
    if weight_col not in df.columns:
        raise ValueError("ibov_composicao.csv precisa de coluna 'peso' ou 'weight'")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    wide = df.pivot(index="date", columns="ticker", values=weight_col).sort_index()
    wide.index.name = "date"
    wide.columns.name = "ticker"
    return wide.fillna(0.0)


def load_benchmarks(path: str | Path) -> pd.DataFrame:
    """Lê benchmarks_diarios.csv (já wide) e padroniza nomes de coluna.

    Devolve os valores EXATAMENTE como estão no arquivo — em NÍVEL (pontos
    de índice, fator acumulado, preço), não em retorno percentual.
    Confirmado inspecionando dado real: Ibovespa na casa de dezenas/
    centenas de milhares de pontos, CDI como fator acumulado crescendo de
    ~3.5 para ~21 ao longo de 2008-2026, bitcoin em preço R$. Use
    `benchmarks_to_returns()` para obter a série que o resto do código
    (Sharpe, beta, tracking error, curva acumulada) espera.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df.columns = [_normalize_colname(c) for c in df.columns]
    return df


def benchmarks_to_returns(benchmarks_levels: pd.DataFrame) -> pd.DataFrame:
    """Converte as séries de nível de `load_benchmarks` em retorno diário.

    `.pct_change()` é a transformação correta para qualquer um dos 3 tipos
    de série presentes (pontos de índice, fator acumulado, preço) — o
    retorno diário é a variação percentual dia a dia em qualquer um dos
    três casos, independente da escala/base arbitrária do nível.

    Precisa rodar logo após `load_benchmarks`, ANTES de qualquer outro
    cálculo (inclusive antes de `flag_suspicious_returns`/`clean_returns`
    — aplicar a checagem de retorno suspeito sobre o NÍVEL, não sobre o
    retorno derivado, sinalizava a série inteira como "extrema" e zerava
    CDI/Ibovespa por completo, silenciosamente).
    """
    return benchmarks_levels.pct_change()


def _normalize_colname(name: str) -> str:
    name = name.strip().lower()
    replacements = {
        "ibovespa": "ibovespa",
        "ibov": "ibovespa",
        "cdi": "cdi",
        "ima-s": "ima_s",
        "imas": "ima_s",
        "idka pré 3a": "idka_pre_3a",
        "idka pre 3a": "idka_pre_3a",
        "idka_pre_3a": "idka_pre_3a",
        "ima-b": "ima_b",
        "imab": "ima_b",
        "ihfa": "ihfa",
        "ifix": "ifix",
        "s&p 500": "sp500_brl",
        "sp500": "sp500_brl",
        "s&p500": "sp500_brl",
        "bitcoin": "bitcoin_brl",
        "btc": "bitcoin_brl",
        "btc_brl": "bitcoin_brl",
        "bitcoin_brl": "bitcoin_brl",
    }
    key = name.replace("_", " ").strip()
    return replacements.get(key, replacements.get(name, name.replace(" ", "_")))


def _mask_to_flag_rows(returns_wide: pd.DataFrame, mask: pd.DataFrame, tipo: str) -> list[dict]:
    """Extrai (date, ticker, retorno) onde `mask` é True, sem depender de
    `DataFrame.stack()` — o tratamento de NaN em `.where(...).stack()`
    mudou de comportamento padrão entre versões do pandas (dropava NaN
    antes, passou a manter em versões mais novas), o que fazia esta função
    devolver linhas fantasma com `retorno=NaN` dependendo do ambiente.
    `np.where` sobre o array booleano é estável em qualquer versão.
    """
    rows_idx, cols_idx = np.where(mask.to_numpy())
    return [
        {
            "date": returns_wide.index[i],
            "ticker": returns_wide.columns[j],
            "retorno": returns_wide.iat[i, j],
            "tipo": tipo,
        }
        for i, j in zip(rows_idx, cols_idx)
    ]


def flag_suspicious_returns(
    returns_wide: pd.DataFrame,
    max_abs_daily_return: float = 1.00,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identifica possíveis erros de cotação sem alterar o dado silenciosamente.

    Retorna (df_flags, resumo). `df_flags` tem uma linha por observação
    suspeita, com o tipo de suspeita. O chamador decide o que fazer — a
    metodologia por padrão apenas EXCLUI o dia do universo elegível daquele
    papel via winsorização em `signal.py`, nunca "conserta" o número.

    Duas heurísticas simples, documentadas como limitação (não substituem
    checagem manual contra fonte primária):
      1. |retorno diário| acima de `max_abs_daily_return` (ex.: >100% em um
         dia) — quase sempre erro de cotação ou evento não capturado no
         ajuste, já que a base afirma vir "ajustada por proventos,
         desdobramentos e grupamentos".
      2. Reversão artificial: um retorno extremo seguido, no pregão
         seguinte, por um retorno próximo de -r/(1+r) — a assinatura
         clássica de um split/agrupamento não ajustado corretamente.
    """
    flags = []

    extreme = returns_wide.abs() > max_abs_daily_return
    flags.extend(_mask_to_flag_rows(returns_wide, extreme, "retorno_extremo"))

    shifted = returns_wide.shift(-1)
    implied_reversal = -returns_wide / (1 + returns_wide)
    spike_mask = (
        (returns_wide.abs() > 0.30)
        & ((shifted - implied_reversal).abs() < 0.02)
    )
    # O dia SEGUINTE ao pico (a "reversão" em si) também não é um retorno
    # real — é o desfazer aritmético do erro do dia anterior — então marca
    # os dois dias do par, não só o pico.
    reversal_day_mask = spike_mask.shift(1, fill_value=False)
    flags.extend(
        _mask_to_flag_rows(returns_wide, spike_mask | reversal_day_mask, "possivel_split_nao_ajustado")
    )

    df_flags = pd.DataFrame(flags)
    if df_flags.empty:
        resumo = pd.DataFrame(columns=["tipo", "n_ocorrencias"])
    else:
        resumo = (
            df_flags.groupby("tipo").size().rename("n_ocorrencias").reset_index()
        )
    return df_flags, resumo


def clean_returns(returns_wide: pd.DataFrame, df_flags: pd.DataFrame) -> pd.DataFrame:
    """Remove (vira NaN) as observações sinalizadas por `flag_suspicious_returns`.

    Sem este passo, os pontos suspeitos continuavam entrando sem filtro no
    cálculo do sinal e no backtest — exatamente o risco que o case avisa
    ("uma cotação errada entra no retorno e ainda pode fazer a ação ser
    selecionada"): um retorno de centenas de % em um dia domina o momentum
    acumulado se a janela de sinal terminar logo depois do evento, mesmo
    que ele "se cancele" matematicamente no dia seguinte (caso de split não
    ajustado) — a assinatura de reversão só aparece OLHANDO OS DOIS DIAS
    JUNTOS, mas o sinal em uma data intermediária já teria sido contaminado.
    Também explica overflow numérico em `cumprod`: um retorno de milhares
    de % composto ao longo de milhares de pregões estoura float64.

    Por isso os DOIS dias de um par de reversão (`possivel_split_nao_ajustado`)
    viram NaN, não só o primeiro — o segundo dia também não é um retorno
    real, é o "desfazer" aritmético de um erro no primeiro.

    Tratamos a ausência (NaN) como "sem informação nesse dia para esse
    papel", não como "retorno zero" — quem decide o que fazer com isso é
    cada consumidor: `_cumulative_return` (signal.py) trata NaN como 0% via
    `fillna(0)`, o que é razoável para um ponto isolado removido no meio de
    uma série; `universe.py` conta como um dia a menos de atividade, o que
    é o comportamento correto (não sabemos o retorno real daquele dia).

    Limitação declarada: por não termos como confirmar o valor correto
    contra a fonte primária (B3/CVM), a escolha conservadora é excluir a
    observação em vez de tentar "adivinhar" o valor certo. Isso pode
    ocasionalmente descartar um movimento genuíno muito grande (ex.: IPO
    de primeiro dia, penny stock com notícia real) — o corte de
    `max_abs_daily_return` (100%/dia) foi calibrado para minimizar esse
    risco, mas não elimina.
    """
    if df_flags.empty:
        return returns_wide

    cleaned = returns_wide.copy()
    for _, row in df_flags.iterrows():
        cleaned.loc[row["date"], row["ticker"]] = np.nan
    return cleaned


def align_calendars(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Reindexa uma lista de DataFrames para a união de seus índices de data."""
    common_index = frames[0].index
    for f in frames[1:]:
        common_index = common_index.union(f.index)
    common_index = common_index.sort_values()
    return [f.reindex(common_index) for f in frames]
