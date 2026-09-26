"""Fatores de mercado via PCA e desempenho do momentum condicional a eles.

Constrói fatores de mercado ESTATÍSTICOS (não fundamentais/setoriais, já que
a base não traz classificação setorial) a partir dos retornos padronizados
do universo de ações: PCA sobre o painel padronizado extrai as direções de
variação comum entre os papéis (tipicamente o 1º componente se aproxima de
um "fator de mercado" amplo, já que a maioria das ações brasileiras se move
junto no mesmo sentido na maior parte do tempo).

Uso pretendido: análise DIAGNÓSTICA / post-hoc — responder "em que ambiente
de mercado (medido pelos componentes) o momentum funcionou ou falhou no
passado?". Por isso o ajuste do PCA usa a amostra inteira (não é refeito de
forma expansível/rolling): como o resultado não realimenta a construção da
carteira (não é um sinal ao vivo), não há look-ahead bias relevante para a
integridade do backtest em si — mas o mesmo não vale se algum dia isto for
usado para DECIDIR posições em tempo real; nesse caso seria necessário
reajustar o PCA de forma expansível/rolling, usando só dado até cada data,
como já é feito em `signal.py`/`universe.py`.

Limitações declaradas:
  - Painel desbalanceado (nem toda ação tem retorno em toda data, por
    entrada/delisting): dias sem retorno observado são tratados como 0 após
    a padronização (equivalente a "sem informação nova" naquele dia para
    aquele papel) — um `SimpleImputer` implícito, viés conhecido e aceito
    para esta análise exploratória.
  - Os componentes são estatísticos, sem rótulo econômico automático; o
    módulo reporta a fração de ações com carga positiva no 1º componente
    como uma checagem de sanidade de que ele de fato se comporta como um
    fator de mercado amplo, mas não atribui nome a componentes de ordem
    mais alta (isso exigiria julgamento humano/dados setoriais).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import annualized_return, annualized_vol, sharpe_ratio


def standardize_returns(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Padroniza (z-score) cada ação pela própria média/desvio-padrão.

    Usa só os dias em que a ação tem retorno observado (histórico próprio),
    para não distorcer o z-score de papéis que entraram/saíram no meio da
    amostra. Remove a diferença de nível de volatilidade entre papéis antes
    do PCA — sem isso, os componentes seriam dominados pelas ações mais
    voláteis, não pelas mais CO-movidas.
    """
    mean = returns_wide.mean(axis=0)
    std = returns_wide.std(axis=0, ddof=1).replace(0.0, np.nan)
    return (returns_wide - mean) / std


@dataclass
class MarketFactors:
    loadings: pd.DataFrame  # (componente x ticker)
    factor_scores: pd.DataFrame  # (data x componente)
    explained_variance_ratio: pd.Series  # indexado por componente
    pct_tickers_positive_pc1: float  # checagem de sanidade do PC1 como "fator de mercado"


def fit_market_factors(returns_wide: pd.DataFrame, n_components: int = 5) -> MarketFactors:
    """Ajusta PCA (via SVD) sobre o painel de retornos padronizados.

    Dias/papéis sem retorno observado viram 0 pós-padronização (ver aviso do
    módulo). `n_components` é truncado ao rank disponível (min(T, N)).
    """
    standardized = standardize_returns(returns_wide).fillna(0.0)
    n_components = min(n_components, standardized.shape[0], standardized.shape[1])

    # SVD em vez de decompor a matriz de covariância explicitamente: mesmo
    # resultado, mais estável numericamente e não precisa materializar uma
    # matriz N x N (N = nº de ações, pode ser grande).
    x = standardized.to_numpy()
    u, s, vt = np.linalg.svd(x, full_matrices=False)

    components = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(
        vt[:n_components], index=components, columns=standardized.columns
    )
    factor_scores = pd.DataFrame(
        x @ vt[:n_components].T, index=standardized.index, columns=components
    )

    total_var = (s**2).sum()
    explained_variance_ratio = pd.Series(
        (s[:n_components] ** 2) / total_var, index=components
    )

    pc1_loadings = loadings.loc["PC1"]
    pct_positive = float((pc1_loadings > 0).mean())
    # PCA não fixa o sinal do componente (v e -v são igualmente válidos) —
    # convenciona-se que a maioria das cargas do PC1 seja positiva, para que
    # "PC1 alto" leia-se como "mercado em alta ampla", não o oposto.
    if pct_positive < 0.5:
        loadings.loc["PC1"] *= -1
        factor_scores["PC1"] *= -1
        pct_positive = 1.0 - pct_positive

    return MarketFactors(
        loadings=loadings,
        factor_scores=factor_scores,
        explained_variance_ratio=explained_variance_ratio,
        pct_tickers_positive_pc1=pct_positive,
    )


def _bucket_names(n_buckets: int) -> list[str]:
    named = {2: ["baixo", "alto"], 3: ["baixo", "médio", "alto"]}
    return named.get(n_buckets, [f"q{i + 1}" for i in range(n_buckets)])


def classify_factor_regimes(
    factor_scores: pd.DataFrame, n_buckets: int = 3
) -> pd.DataFrame:
    """Classifica cada data em tercis (ou n_buckets) por fator, na própria escala do fator.

    Usa os cortes da amostra INTEIRA (ver aviso do módulo: análise
    diagnóstica, não sinal ao vivo). Rótulos ordinais "baixo" < "médio" <
    "alto" (ou intermediários, se `n_buckets` > 3), independente do nome do
    fator. Se a série do fator tiver poucos valores distintos (amostra
    pequena), `duplicates="drop"` pode gerar menos grupos que `n_buckets` —
    os rótulos se ajustam automaticamente à contagem real de grupos.
    """
    regimes = {}
    for col in factor_scores.columns:
        bin_idx = pd.qcut(factor_scores[col], q=n_buckets, labels=False, duplicates="drop")
        names = _bucket_names(int(bin_idx.max()) + 1)
        mapped = bin_idx.map(dict(enumerate(names)))
        regimes[col] = pd.Categorical(mapped, categories=names, ordered=True)
    return pd.DataFrame(regimes, index=factor_scores.index)


def momentum_performance_by_regime(
    returns_net: pd.Series,
    regime_labels: pd.DataFrame,
    cdi_daily: pd.Series,
) -> pd.DataFrame:
    """Sharpe/retorno/hit-rate do índice de momentum em cada regime de cada fator.

    Uma linha por combinação (fator, regime), alinhando os retornos do
    índice às datas em que aquele regime esteve vigente.
    """
    aligned_regimes = regime_labels.reindex(returns_net.index)
    rows = []
    for factor in regime_labels.columns:
        for regime in regime_labels[factor].cat.categories:
            mask = aligned_regimes[factor] == regime
            chunk = returns_net[mask]
            if chunk.empty:
                continue
            cdi_chunk = cdi_daily.reindex(chunk.index).fillna(0.0)
            rows.append(
                {
                    "fator": factor,
                    "regime": regime,
                    "n_dias": len(chunk),
                    "retorno_anualizado": annualized_return(chunk),
                    "vol_anualizada": annualized_vol(chunk),
                    "sharpe": sharpe_ratio(chunk, cdi_chunk),
                    "hit_rate_diario": float((chunk > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def factor_exposure(returns_net: pd.Series, factor_scores: pd.DataFrame) -> pd.Series:
    """Regressão OLS do retorno diário do índice nos fatores (beta + R²).

    Quantifica quanto do retorno do momentum é explicado por movimentos
    sistemáticos comuns às ações (os fatores) vs. idiossincrático à
    estratégia (o que a regressão não explica, resíduo) — um R² baixo
    sugere que o timing do momentum (entrar/sair de nomes) importa mais que
    exposição passiva aos fatores de mercado; um R² alto sugere que grande
    parte do resultado do índice é só "beta" a esses fatores.
    """
    aligned = pd.concat([returns_net, factor_scores], axis=1).dropna()
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1:].to_numpy()
    x_with_intercept = np.column_stack([np.ones(len(x)), x])

    coefs, residuals_ss, _, _ = np.linalg.lstsq(x_with_intercept, y, rcond=None)
    y_pred = x_with_intercept @ coefs
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result = {"intercepto_diario": coefs[0]}
    for factor, beta in zip(factor_scores.columns, coefs[1:]):
        result[f"beta_{factor}"] = beta
    result["r_quadrado"] = r_squared
    return pd.Series(result)
