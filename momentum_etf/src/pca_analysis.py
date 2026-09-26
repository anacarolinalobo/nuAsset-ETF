"""Análise de componentes principais (PCA) sobre retornos.

Duas aplicações distintas, respondendo perguntas diferentes do case:

1. `compute_universe_pca`: PCA sobre o retorno diário de um subconjunto de
   ações (a carteira do índice, ou o universo elegível inteiro) numa
   janela fixa. Responde "quão diversificado é o risco de verdade, além
   do número de nomes na carteira?" — se poucos componentes já explicam a
   maior parte da variância, a carteira tem menos graus de liberdade de
   risco do que o número de posições sugere (tipicamente um "fator
   mercado" domina). Também cruza a carga de cada ação no primeiro
   componente (proxy de beta de mercado) com o score de momentum, para
   checar se o fator momentum é distinto de simplesmente comprar ações de
   maior beta — pergunta central para argumentar que o produto é mesmo um
   fator novo, não uma releitura de risco de mercado com um nome
   diferente.

2. `compute_asset_class_pca`: PCA sobre os 9 benchmarks do case + o
   retorno do próprio índice de momentum. Responde diretamente "que papel
   a posição cumpre num portfólio diversificado? Com quais classes de
   ativos ela compete, e quais ela complementa?" — ativos que carregam
   forte no MESMO componente do índice de momentum competem por risco
   (mover-se em conjunto); ativos em componentes praticamente ortogonais
   complementam (adicionam diversificação de verdade).

Ambas usam PCA padronizada (equivalente a decompor a matriz de
CORRELAÇÃO, não a covariância bruta) porque as séries envolvidas têm
escalas de volatilidade muito diferentes entre si (CDI quase não varia
dia a dia, bitcoin varia muito) — sem padronizar, o primeiro componente
seria dominado trivialmente pelo ativo mais volátil, não pelo padrão de
comovimento que a análise quer capturar.

Abordagem "complete case": qualquer ativo com retorno ausente em algum
dia da janela escolhida é descartado da matriz usada no PCA — imputar
retorno ausente introduziria um viés que esta análise não se propõe a
corrigir. `PCAResult.dropped_assets` registra o que foi descartado, para
transparência.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    explained_variance_ratio: pd.Series   # índice: PC1..PCn
    loadings: pd.DataFrame                # índice: ativo, colunas: PC1..PCn
    factor_returns: pd.DataFrame          # índice: data, colunas: PC1..PCn
    n_observations: int
    n_assets: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    dropped_assets: list[str]

    def n_components_for_variance(self, target: float = 0.90) -> int:
        cumulative = self.explained_variance_ratio.cumsum()
        above = cumulative[cumulative >= target]
        return int(above.index.get_loc(above.index[0]) + 1) if len(above) else len(cumulative)


def _fit_pca(returns: pd.DataFrame, n_components: int) -> PCAResult:
    """PCA padronizada via SVD sobre `returns` (linhas=data, colunas=ativo)."""
    clean = returns.dropna(axis=1, how="any").dropna(axis=0, how="any")
    dropped = [c for c in returns.columns if c not in clean.columns]

    std = clean.std(ddof=1).replace(0, np.nan)
    standardized = ((clean - clean.mean()) / std).dropna(axis=1, how="any")
    dropped += [c for c in clean.columns if c not in standardized.columns]

    n_components = min(n_components, standardized.shape[0] - 1, standardized.shape[1])
    if n_components < 1 or standardized.empty:
        raise ValueError(
            "Dados insuficientes para PCA nesta janela — poucos ativos ou "
            "observações com histórico completo. Tente uma janela menor ou "
            "um conjunto de ativos com mais histórico em comum."
        )

    X = standardized.to_numpy()
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    explained_variance = (S ** 2) / (X.shape[0] - 1)
    explained_variance_ratio_full = explained_variance / explained_variance.sum()

    pc_labels = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(Vt[:n_components].T, index=standardized.columns, columns=pc_labels)
    factor_returns = pd.DataFrame(
        X @ Vt[:n_components].T, index=standardized.index, columns=pc_labels
    )

    # Sinal de cada componente é arbitrário na SVD — fixa a convenção para
    # que a carga média seja positiva, coerente com a leitura de "PC1 =
    # fator de mercado" (todo mundo se move junto na mesma direção).
    for col in pc_labels:
        if loadings[col].mean() < 0:
            loadings[col] = -loadings[col]
            factor_returns[col] = -factor_returns[col]

    explained_variance_ratio = pd.Series(explained_variance_ratio_full[:n_components], index=pc_labels)

    return PCAResult(
        explained_variance_ratio=explained_variance_ratio,
        loadings=loadings,
        factor_returns=factor_returns,
        n_observations=standardized.shape[0],
        n_assets=standardized.shape[1],
        window_start=standardized.index.min(),
        window_end=standardized.index.max(),
        dropped_assets=sorted(set(dropped)),
    )


def compute_universe_pca(
    returns_wide: pd.DataFrame,
    tickers: list[str] | pd.Index,
    end_date: pd.Timestamp,
    window_days: int = 252,
    n_components: int = 10,
) -> PCAResult:
    window = returns_wide.loc[:end_date, list(tickers)].tail(window_days)
    return _fit_pca(window, n_components)


def compute_asset_class_pca(
    benchmarks_returns: pd.DataFrame,
    index_returns: pd.Series,
    end_date: pd.Timestamp,
    window_days: int = 504,
    n_components: int = 10,
    index_label: str = "Índice Momentum",
) -> PCAResult:
    combined = benchmarks_returns.copy()
    combined[index_label] = index_returns.reindex(combined.index)
    window = combined.loc[:end_date].tail(window_days)
    return _fit_pca(window, n_components)


def pc1_loading_vs_momentum_score(loadings: pd.DataFrame, momentum_score: pd.Series) -> pd.DataFrame:
    """Carga no PC1 (proxy de beta de mercado) x score de momentum, por ativo.

    Usado para checar se o fator momentum é redundante com "comprar ações
    de maior beta" (alta correlação positiva entre as duas colunas) ou se
    é um fator de fato distinto (correlação baixa).
    """
    aligned = pd.concat(
        [loadings["PC1"], momentum_score], axis=1, keys=["carga_pc1", "score_momentum"]
    ).dropna()
    return aligned


def dominant_component_for_asset(loadings: pd.DataFrame, asset: str) -> tuple[str, pd.Series]:
    """Componente onde `asset` tem a maior carga em módulo, e as cargas de
    todos os ativos nesse mesmo componente — para identificar quem "compete"
    (mesmo sinal, carga alta) e quem "complementa" (carga baixa/oposta)."""
    row = loadings.loc[asset]
    dominant_pc = row.abs().idxmax()
    return dominant_pc, loadings[dominant_pc].sort_values(ascending=False)
