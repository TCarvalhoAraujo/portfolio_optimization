"""
covariance/factor_model.py
==========================
Estimação de covariância via modelo de fatores estatísticos (PCA).

Conceito:
    Em vez de estimar N*(N+1)/2 parâmetros da covariância diretamente,
    um modelo de fatores decompõe os retornos em dois componentes:

        r_i = α_i + β_i1·F1 + β_i2·F2 + ... + β_iK·FK + ε_i

    Onde:
        F_k   = fator comum k (ex: mercado, valor, momentum)
        β_ik  = sensibilidade do ativo i ao fator k (factor loading)
        ε_i   = retorno idiossincrático (específico do ativo, não correlacionado)

    A covariância resultante é:
        Σ = B · Σ_F · B^T + D

    Onde:
        B    = matriz de loadings (N x K)
        Σ_F  = covariância dos fatores (K x K) — muito menor que N x N
        D    = matriz diagonal de variâncias idiossincráticas

    Com K=5 fatores e N=50 ativos, você estima apenas:
        - 50*5 = 250 loadings
        - 5*6/2 = 15 parâmetros de Σ_F
        - 50 variâncias idiossincráticas
    Total: ~315 vs 1275 da covariância amostral completa.

Abordagem aqui: PCA estatístico
    Usamos PCA para extrair os fatores diretamente dos retornos,
    sem precisar de dados fundamentalistas externos (Fama-French).
    Os fatores do PCA são combinações lineares dos retornos que
    capturam a maior parte da variância total.

Limitação:
    Fatores do PCA não têm interpretação econômica direta.
    Para fatores interpretáveis, use Fama-French (requer download
    separado dos dados do Kenneth French Data Library).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Tuple


def pca_factor_covariance(
    returns: pd.DataFrame,
    n_factors: int = 5,
    annualize: bool = True,
    trading_days: int = 252,
) -> Tuple[pd.DataFrame, dict]:
    """
    Estima a matriz de covariância usando modelo de fatores via PCA.

    Algoritmo:
        1. Aplica PCA sobre os retornos → extrai K fatores
        2. Calcula loadings B (regressão de cada ativo sobre os fatores)
        3. Calcula variância idiossincrática D (resíduo não explicado)
        4. Reconstrói: Σ = B·Σ_F·B^T + D

    Parâmetros
    ----------
    returns : pd.DataFrame
        Retornos diários. Shape (T, N).
    n_factors : int
        Número de fatores. Regra prática: começa com 5, aumenta até
        capturar ~80% da variância total explicada.
    annualize : bool
        Se True, multiplica por trading_days.
    trading_days : int
        Dias úteis por ano.

    Retorna
    -------
    cov : pd.DataFrame
        Matriz de covariância N x N estimada pelo modelo de fatores.
    diagnostics : dict
        Informações sobre o modelo:
            - explained_variance_ratio: variância explicada por fator
            - cumulative_variance: variância acumulada
            - n_factors_used: K efetivo
            - loadings: DataFrame (N x K)
            - idiosyncratic_vols: volatilidades idiossincráticas
    """
    X = returns.values  # (T, N)
    T, N = X.shape

    # PCA — sklearn centraliza automaticamente
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(X)       # (T, K) — scores dos fatores
    loadings = pca.components_.T         # (N, K) — loadings de cada ativo

    # Variância explicada
    evr = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(evr)

    # Covariância dos fatores: Σ_F = (1/T) · F^T · F
    # Os fatores do PCA são ortogonais → Σ_F é diagonal
    factor_cov = np.cov(factors.T, ddof=1)  # (K, K)

    # Componente sistemática: B · Σ_F · B^T
    systematic_cov = loadings @ factor_cov @ loadings.T  # (N, N)

    # Resíduos idiossincráticos: ε = r - F·B^T
    residuals = X - factors @ loadings.T   # (T, N)

    # Variância idiossincrática: diagonal de cov(ε)
    idio_var = np.var(residuals, axis=0, ddof=1)  # (N,)
    D = np.diag(idio_var)  # (N, N) diagonal

    # Covariância total do modelo de fatores
    cov_values = systematic_cov + D  # (N, N)

    if annualize:
        cov_values = cov_values * trading_days

    cov = pd.DataFrame(cov_values, index=returns.columns, columns=returns.columns)

    # Diagnósticos
    loadings_df = pd.DataFrame(
        loadings,
        index=returns.columns,
        columns=[f"Factor_{k+1}" for k in range(n_factors)],
    )

    idio_vols = pd.Series(
        np.sqrt(idio_var * (trading_days if annualize else 1)),
        index=returns.columns,
        name="idiosyncratic_vol",
    )

    diagnostics = {
        "explained_variance_ratio": pd.Series(
            evr, index=[f"Factor_{k+1}" for k in range(n_factors)]
        ),
        "cumulative_variance": cumulative_var[-1],
        "n_factors_used": n_factors,
        "loadings": loadings_df,
        "idiosyncratic_vols": idio_vols,
    }

    return cov, diagnostics


def select_n_factors(
    returns: pd.DataFrame,
    min_variance_explained: float = 0.80,
    max_factors: int = 20,
) -> int:
    """
    Escolhe automaticamente o número de fatores para explicar
    um percentual mínimo da variância total.

    Parâmetros
    ----------
    returns : pd.DataFrame
    min_variance_explained : float
        Percentual mínimo de variância a ser explicado. Default: 80%.
    max_factors : int
        Limite máximo de fatores a considerar.

    Retorna
    -------
    int
        Número mínimo de fatores para atingir min_variance_explained.

    Exemplo
    -------
    >>> k = select_n_factors(returns, min_variance_explained=0.80)
    >>> print(f"Use {k} fatores para capturar 80% da variância")
    """
    n_max = min(max_factors, returns.shape[1], returns.shape[0] - 1)
    pca = PCA(n_components=n_max)
    pca.fit(returns.values)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_factors = int(np.searchsorted(cumvar, min_variance_explained) + 1)

    return min(n_factors, n_max)


def factor_model_summary(diagnostics: dict) -> pd.DataFrame:
    """
    Formata os diagnósticos do modelo de fatores em tabela legível.

    Parâmetros
    ----------
    diagnostics : dict
        Retornado por pca_factor_covariance().

    Retorna
    -------
    pd.DataFrame
        Tabela com variância explicada por fator.
    """
    evr = diagnostics["explained_variance_ratio"]
    cumvar = np.cumsum(evr.values)

    summary = pd.DataFrame({
        "Explained Variance": evr.values,
        "Cumulative Variance": cumvar,
    }, index=evr.index)

    summary["Explained Variance"] = summary["Explained Variance"].map("{:.2%}".format)
    summary["Cumulative Variance"] = summary["Cumulative Variance"].map("{:.2%}".format)

    return summary
