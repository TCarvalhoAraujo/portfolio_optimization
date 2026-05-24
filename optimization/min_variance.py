"""
min_variance.py
===============
Carteira de Mínima Variância Global (GMV — Global Minimum Variance).

Conceito:
    Resolve apenas:
        min  w^T Σ w
        s.t. Σw = 1, restrições

    Ignora completamente as estimativas de retorno esperado.

Por que isso é uma vantagem:
    Estimativas de retorno esperado são notoriamente imprecisas.
    A covariância, embora também ruidosa, é estimada com menos erro
    relativo. A GMV aposta apenas na estrutura de dependência entre
    os ativos, sem se expor ao erro de previsão de μ.

    Empiricamente, a GMV muitas vezes supera carteiras de Máximo Sharpe
    out-of-sample precisamente porque evita esse erro de estimação.

Quando usar:
    - Alta incerteza sobre retornos futuros
    - Regimes de mercado voláteis
    - Como baseline de risco mínimo em comparações
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

from .base_optimizer import BaseOptimizer
from .constraints import PortfolioConstraints, long_only_box, build_scipy_bounds, build_scipy_constraints
from .covariance import ledoit_wolf_cov


class MinVarianceOptimizer(BaseOptimizer):
    """
    Carteira de Mínima Variância Global.

    Não usa retornos esperados — apenas a matriz de covariância.

    Parâmetros
    ----------
    cov_estimator : callable
        Estimador de covariância. Default: ledoit_wolf_cov.
    """

    def __init__(self, cov_estimator=None):
        super().__init__(cov_estimator)

    @property
    def name(self) -> str:
        return "Min Variance (GMV)"

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Encontra os pesos que minimizam a variância da carteira.

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None

        Retorna
        -------
        pd.Series com pesos de mínima variância.
        """
        self._validate_returns(returns)
        constraints = constraints or self._default_constraints(returns.shape[1])
        constraints.validate(returns.shape[1])

        tickers = list(returns.columns)
        n = len(tickers)
        cov = self.cov_estimator(returns).values

        def portfolio_variance(w):
            return w @ cov @ w

        def variance_gradient(w):
            """Gradiente analítico: ∂(w^TΣw)/∂w = 2Σw"""
            return 2 * cov @ w

        bounds = build_scipy_bounds(constraints, n)
        cons = build_scipy_constraints(constraints, tickers)

        best_result = None
        best_var = np.inf

        starts = [np.ones(n) / n] + [
            np.random.dirichlet(np.ones(n)) for _ in range(4)
        ]

        for w0 in starts:
            result = minimize(
                portfolio_variance,
                w0,
                jac=variance_gradient,   # gradiente analítico — mais rápido e estável
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-14, "maxiter": 1000},
            )

            if result.success and result.fun < best_var:
                best_var = result.fun
                best_result = result.x

        if best_result is None:
            raise RuntimeError("Otimização GMV não convergiu.")

        # Limpeza numérica
        best_result = np.where(np.abs(best_result) < 1e-6, 0.0, best_result)
        best_result = best_result / best_result.sum()

        return pd.Series(best_result, index=tickers, name=self.name)

    def analytical_solution(self, returns: pd.DataFrame) -> pd.Series:
        """
        Solução analítica da GMV sem constraints (exceto soma = 1).

        Quando não há restrições de box ou setor, a GMV tem solução
        fechada via multiplicadores de Lagrange:

            w* = Σ⁻¹ · 1 / (1^T · Σ⁻¹ · 1)

        Onde 1 é o vetor de uns.

        Parâmetros
        ----------
        returns : pd.DataFrame

        Retorna
        -------
        pd.Series
            Pode conter pesos negativos (short positions).

        Nota:
            Use apenas para fins educativos ou quando short selling
            é permitido. Para long-only, use optimize().
        """
        tickers = list(returns.columns)
        cov = self.cov_estimator(returns).values
        n = len(tickers)

        ones = np.ones(n)
        cov_inv = np.linalg.inv(cov)

        w = cov_inv @ ones
        w = w / (ones @ w)  # normaliza para soma = 1

        return pd.Series(w, index=tickers, name=f"{self.name} (analytical)")
