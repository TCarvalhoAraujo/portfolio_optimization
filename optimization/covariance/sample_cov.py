"""
covariance/sample_cov.py
========================
Covariância amostral e utilitários básicos de matriz.

Conceito:
    A covariância amostral é o estimador mais direto: usa os dados
    históricos sem nenhum ajuste. É o ponto de partida, mas sofre
    de dois problemas sérios quando N (ativos) é grande em relação
    a T (observações):

    1. Ruído de estimação: com N=50 e T=252 (1 ano), você estima
       50*51/2 = 1275 parâmetros com apenas 252 observações.
       O resultado é uma matriz barulhenta.

    2. Mal condicionamento: a matriz pode ter autovalores próximos
       de zero, tornando-a quase singular. Inverter uma matriz
       quase singular amplifica erros — problema sério para
       Markowitz e Black-Litterman que precisam de Σ⁻¹.

    Use sempre como baseline. Para otimização real, prefira
    Ledoit-Wolf (ledoit_wolf.py) ou modelo de fatores (factor_model.py).
"""

import numpy as np
import pandas as pd
from typing import Tuple


def sample_covariance(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Calcula a matriz de covariância amostral dos retornos.

    Usa ddof=1 (correção de Bessel) — divide por T-1 em vez de T,
    produzindo um estimador não-viesado da covariância populacional.

    Parâmetros
    ----------
    returns : pd.DataFrame
        Retornos diários. Shape: (T, N). Sem NaN.
    annualize : bool
        Se True, multiplica por trading_days para anualizar.
        Covariância anualizada é mais intuitiva para relatórios.
    trading_days : int
        Número de dias úteis por ano. Default: 252.

    Retorna
    -------
    pd.DataFrame
        Matriz de covariância (N x N), indexada pelos tickers.

    Exemplo
    -------
    >>> cov = sample_covariance(returns, annualize=True)
    >>> cov.shape  # (50, 50) para 50 ativos
    >>> np.sqrt(cov.loc["AAPL", "AAPL"])  # volatilidade anualizada da AAPL
    """
    cov = returns.cov()  # pandas usa ddof=1 por padrão

    if annualize:
        cov = cov * trading_days

    return cov


def sample_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Matriz de correlação amostral.

    Correlação é covariância normalizada:
        ρ_ij = σ_ij / (σ_i * σ_j)

    Valores entre -1 e 1. Diagonal sempre = 1.
    Não precisa ser anualizada (é adimensional).

    Parâmetros
    ----------
    returns : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Matriz de correlação (N x N).
    """
    return returns.corr()


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Converte matriz de covariância para correlação.

    Útil quando você tem uma covariância ajustada (ex: Ledoit-Wolf)
    e quer visualizar as correlações implícitas.

    Parâmetros
    ----------
    cov : pd.DataFrame
        Matriz de covariância (N x N).

    Retorna
    -------
    pd.DataFrame
        Matriz de correlação equivalente.
    """
    std = np.sqrt(np.diag(cov.values))
    # Divide cada elemento cov[i,j] por std[i] * std[j]
    outer_std = np.outer(std, std)
    corr_values = cov.values / outer_std
    # Garante diagonal exatamente 1.0 (evita erros numéricos)
    np.fill_diagonal(corr_values, 1.0)
    return pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)


def corr_to_cov(corr: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    """
    Reconstrói covariância a partir de correlação + volatilidades.

    Útil em Black-Litterman e modelos de fatores onde você manipula
    correlação e volatilidade separadamente.

    σ_ij = ρ_ij * σ_i * σ_j

    Parâmetros
    ----------
    corr : pd.DataFrame
        Matriz de correlação (N x N).
    vols : pd.Series
        Volatilidades anualizadas por ativo. Index = tickers.

    Retorna
    -------
    pd.DataFrame
        Matriz de covariância reconstruída.
    """
    v = vols.values
    outer_v = np.outer(v, v)
    cov_values = corr.values * outer_v
    return pd.DataFrame(cov_values, index=corr.index, columns=corr.columns)


def is_positive_definite(matrix: np.ndarray) -> bool:
    """
    Verifica se uma matriz é positiva definida.

    Uma matriz de covariância válida deve ser positiva definida:
    todos os autovalores > 0. Se não for, a otimização pode falhar
    ou produzir resultados sem sentido.

    Parâmetros
    ----------
    matrix : np.ndarray

    Retorna
    -------
    bool
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def condition_number(matrix: np.ndarray) -> float:
    """
    Número de condicionamento da matriz.

    Razão entre o maior e o menor autovalor.
    - Próximo de 1: matriz bem condicionada, inversão estável.
    - > 1000: matriz mal condicionada, inversão amplifica erros.
    - > 1e6: numericamente quase singular.

    Use para decidir se precisa de shrinkage.

    Parâmetros
    ----------
    matrix : np.ndarray

    Retorna
    -------
    float
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    # eigvalsh retorna valores em ordem crescente
    min_eig = eigenvalues[0]
    max_eig = eigenvalues[-1]

    if min_eig <= 0:
        return np.inf  # não positiva definida

    return max_eig / min_eig


def matrix_summary(cov: pd.DataFrame) -> dict:
    """
    Diagnóstico rápido da matriz de covariância.

    Retorna métricas para decidir se a matriz é confiável
    para uso em otimização.

    Parâmetros
    ----------
    cov : pd.DataFrame

    Retorna
    -------
    dict com:
        - is_positive_definite: bool
        - condition_number: float
        - min_eigenvalue: float
        - max_eigenvalue: float
        - mean_correlation: float  (excluindo diagonal)
        - vols_annualized: pd.Series  (volatilidades implícitas)
    """
    mat = cov.values
    eigenvalues = np.linalg.eigvalsh(mat)

    corr = cov_to_corr(cov).values
    # Correlação média excluindo diagonal
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_corr = corr[mask].mean()

    vols = pd.Series(
        np.sqrt(np.diag(mat)),
        index=cov.index,
        name="vol_annualized",
    )

    return {
        "is_positive_definite": is_positive_definite(mat),
        "condition_number": condition_number(mat),
        "min_eigenvalue": eigenvalues[0],
        "max_eigenvalue": eigenvalues[-1],
        "mean_correlation": mean_corr,
        "vols_annualized": vols,
    }
