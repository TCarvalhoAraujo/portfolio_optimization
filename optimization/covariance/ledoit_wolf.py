"""
covariance/ledoit_wolf.py
=========================
Estimação de covariância com shrinkage de Ledoit-Wolf.

Conceito:
    O estimador de Ledoit-Wolf resolve o problema da covariância amostral
    ruidosa combinando dois estimadores:

        Σ̂ = (1 - α) · Σ_amostral + α · T

    Onde:
        Σ_amostral  = covariância histórica (alta variância, baixo viés)
        T           = alvo estruturado (baixa variância, alto viés)
        α           = coeficiente de shrinkage ∈ [0, 1] (estimado analiticamente)

    O alvo T mais comum é a matriz de identidade escalada:
        T = μ̄ · I
    onde μ̄ é a média dos autovalores de Σ_amostral.

    O α ótimo é calculado analiticamente para minimizar o erro quadrático
    médio esperado entre Σ̂ e a verdadeira covariância Σ — sem precisar
    de dados extras ou cross-validation.

    Intuição: "encolhe" a covariância amostral em direção a uma estrutura
    mais simples e estável. Ativos muito correlacionados ficam um pouco
    menos correlacionados; volatilidades extremas ficam um pouco mais
    próximas da média.

Por que usar antes de qualquer otimização com 20+ ativos:
    - Reduz o número de condicionamento da matriz
    - Produz portfólios mais estáveis out-of-sample
    - Custo computacional negligível
    - Disponível diretamente no scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS
from typing import Literal

from .sample_cov import matrix_summary


def ledoit_wolf_cov(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
    assume_centered: bool = False,
) -> pd.DataFrame:
    """
    Estima matriz de covariância com shrinkage de Ledoit-Wolf.

    Usa a implementação analítica do scikit-learn (Ledoit & Wolf, 2004),
    que calcula o coeficiente de shrinkage ótimo sem cross-validation.

    Parâmetros
    ----------
    returns : pd.DataFrame
        Retornos diários. Shape (T, N). Sem NaN.
    annualize : bool
        Se True, multiplica por trading_days após o shrinkage.
        O shrinkage é aplicado sobre retornos diários (escala natural),
        depois anualizamos — essa é a ordem correta.
    trading_days : int
        Dias úteis por ano. Default: 252.
    assume_centered : bool
        Se True, não subtrai a média antes de calcular. Use False
        (default) para séries de retornos que não foram centradas.

    Retorna
    -------
    pd.DataFrame
        Matriz de covariância shrinkada (N x N).

    Exemplo
    -------
    >>> cov_lw = ledoit_wolf_cov(returns)
    >>> summary = matrix_summary(cov_lw)
    >>> print(summary["condition_number"])  # deve ser muito menor que amostral
    """
    lw = LedoitWolf(assume_centered=assume_centered)
    lw.fit(returns.values)

    cov_values = lw.covariance_

    if annualize:
        cov_values = cov_values * trading_days

    return pd.DataFrame(cov_values, index=returns.columns, columns=returns.columns)


def ledoit_wolf_shrinkage_coefficient(returns: pd.DataFrame) -> float:
    """
    Retorna apenas o coeficiente de shrinkage α estimado.

    Útil para diagnóstico: α próximo de 1 significa que a covariância
    amostral é muito ruidosa e o modelo está confiando quase só no alvo.
    α próximo de 0 significa que a amostral já é confiável.

    Parâmetros
    ----------
    returns : pd.DataFrame

    Retorna
    -------
    float
        Coeficiente de shrinkage ∈ [0, 1].
    """
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.shrinkage_


def oas_cov(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Oracle Approximating Shrinkage (OAS) — alternativa ao Ledoit-Wolf.

    OAS é uma variante que, em alguns cenários com N grande e T pequeno,
    tem erro quadrático médio ligeiramente menor que Ledoit-Wolf.
    Ambos são boas escolhas; LW é mais comum na prática.

    Parâmetros
    ----------
    returns : pd.DataFrame
    annualize : bool
    trading_days : int

    Retorna
    -------
    pd.DataFrame
        Matriz de covariância com shrinkage OAS.
    """
    oas = OAS()
    oas.fit(returns.values)

    cov_values = oas.covariance_

    if annualize:
        cov_values = cov_values * trading_days

    return pd.DataFrame(cov_values, index=returns.columns, columns=returns.columns)


def compare_estimators(
    returns: pd.DataFrame,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compara métricas das covariâncias amostral, LW e OAS.

    Útil para decidir qual estimador usar e para diagnóstico
    da qualidade dos dados.

    Parâmetros
    ----------
    returns : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Tabela comparativa com condition number, PD, correlação média
        e coeficiente de shrinkage.
    """
    from .sample_cov import sample_covariance

    cov_sample = sample_covariance(returns, annualize=True, trading_days=trading_days)
    cov_lw = ledoit_wolf_cov(returns, annualize=True, trading_days=trading_days)
    cov_oas = oas_cov(returns, annualize=True, trading_days=trading_days)

    alpha_lw = ledoit_wolf_shrinkage_coefficient(returns)

    rows = []
    for name, cov in [("Sample", cov_sample), ("Ledoit-Wolf", cov_lw), ("OAS", cov_oas)]:
        s = matrix_summary(cov)
        rows.append({
            "Estimator": name,
            "Positive Definite": s["is_positive_definite"],
            "Condition Number": round(s["condition_number"], 1),
            "Min Eigenvalue": round(s["min_eigenvalue"], 6),
            "Mean Correlation": round(s["mean_correlation"], 4),
            "Shrinkage α": round(alpha_lw, 4) if name == "Ledoit-Wolf" else "—",
        })

    return pd.DataFrame(rows).set_index("Estimator")
