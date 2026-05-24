"""
optimization/
=============
Módulo de otimização de portfólio.

Hierarquia de complexidade:
    1. MinVarianceOptimizer   — apenas Σ, sem μ, mais estável
    2. RiskParityOptimizer    — apenas Σ, contribuição igual de risco
    3. HRPOptimizer           — clusterização hierárquica, sem inversão de Σ
    4. MarkowitzOptimizer     — μ + Σ, máximo Sharpe
    5. RobustOptimizer        — μ + Σ + incerteza, mais conservador
    6. BlackLittermanOptimizer — prior de mercado + views do investidor

Uso típico:
    from optimization import MarkowitzOptimizer, HRPOptimizer
    from optimization import long_only_box, View

    constraints = long_only_box(w_max=0.10)

    opt = MarkowitzOptimizer()
    weights = opt.optimize(returns, constraints)
    metrics = opt.portfolio_metrics(weights, returns)

    hrp = HRPOptimizer(linkage_method="ward")
    weights_hrp = hrp.optimize(returns)

Comparando múltiplas estratégias:
    optimizers = {
        "Markowitz": MarkowitzOptimizer(),
        "MinVar": MinVarianceOptimizer(),
        "RiskParity": RiskParityOptimizer(),
        "HRP": HRPOptimizer(),
    }
    results = {
        name: opt.fit_and_metrics(returns, constraints)
        for name, opt in optimizers.items()
    }
"""

from .base_optimizer import BaseOptimizer
from .markowitz import MarkowitzOptimizer
from .min_variance import MinVarianceOptimizer
from .risk_parity import RiskParityOptimizer
from .hrp import HRPOptimizer
from .black_litterman import BlackLittermanOptimizer, View
from .robust_optimizer import RobustOptimizer

from .views import (
    momentum_views,
    sector_momentum_views,
    build_views,
    views_summary,
)

from .constraints import (
    PortfolioConstraints,
    build_scipy_constraints,
    build_scipy_bounds,
    long_only_box,
    long_only_sector,
    with_turnover,
)

from .covariance import (
    sample_covariance,
    ledoit_wolf_cov,
    oas_cov,
    pca_factor_covariance,
    compare_estimators,
)

__all__ = [
    # Otimizadores
    "BaseOptimizer",
    "MarkowitzOptimizer",
    "MinVarianceOptimizer",
    "RiskParityOptimizer",
    "HRPOptimizer",
    "BlackLittermanOptimizer",
    "RobustOptimizer",
    # Views (Black-Litterman)
    "View",
    "momentum_views",
    "sector_momentum_views",
    "build_views",
    "views_summary",
    # Constraints
    "PortfolioConstraints",
    "build_scipy_constraints",
    "build_scipy_bounds",
    "long_only_box",
    "long_only_sector",
    "with_turnover",
    # Covariância
    "sample_covariance",
    "ledoit_wolf_cov",
    "oas_cov",
    "pca_factor_covariance",
    "compare_estimators",
]