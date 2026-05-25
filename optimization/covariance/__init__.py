"""
covariance/
===========
Estimadores de matriz de covariância para otimização de portfólio.

Hierarquia de qualidade (do mais simples ao mais robusto):
    1. sample_cov      — baseline, ruidosa com N grande
    2. ledoit_wolf_cov — shrinkage analítico, recomendado para 20-100 ativos
    3. oas_cov         — variante do shrinkage, similar ao LW
    4. pca_factor_cov  — modelo de fatores, recomendado para 50+ ativos

Uso típico:
    from optimization.covariance import ledoit_wolf_cov
    cov = ledoit_wolf_cov(returns)
"""

from .sample_cov import (
    sample_covariance,
    sample_correlation,
    cov_to_corr,
    corr_to_cov,
    is_positive_definite,
    condition_number,
    matrix_summary,
)

from .ledoit_wolf import (
    ledoit_wolf_cov,
    ledoit_wolf_shrinkage_coefficient,
    oas_cov,
    compare_estimators,
)

from .factor_model import (
    pca_factor_covariance,
    select_n_factors,
    factor_model_summary,
)

__all__ = [
    "sample_covariance",
    "sample_correlation",
    "cov_to_corr",
    "corr_to_cov",
    "is_positive_definite",
    "condition_number",
    "matrix_summary",
    "ledoit_wolf_cov",
    "ledoit_wolf_shrinkage_coefficient",
    "oas_cov",
    "compare_estimators",
    "pca_factor_covariance",
    "select_n_factors",
    "factor_model_summary",
]
