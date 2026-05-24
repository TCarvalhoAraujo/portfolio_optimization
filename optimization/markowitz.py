"""
markowitz.py
============
Otimização de Markowitz: Máximo Sharpe e Fronteira Eficiente.

Conceito:
    Resolve dois problemas clássicos de otimização quadrática:

    1. Máximo Sharpe (tangency portfolio):
       max  (μ_p - rf) / σ_p
       s.t. Σw = 1, restrições

    2. Fronteira Eficiente:
       Para cada alvo de retorno μ_target:
       min  w^T Σ w
       s.t. w^T μ = μ_target, Σw = 1, restrições

    A fronteira eficiente é o conjunto de portfólios que maximizam
    retorno para cada nível de risco. A carteira de máximo Sharpe
    é o ponto onde a linha do ativo livre de risco tangencia a fronteira.

Implementação:
    Usa scipy.optimize.minimize com método SLSQP (Sequential Least
    Squares Programming) — adequado para problemas quadráticos com
    restrições de igualdade e desigualdade.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Optional, Tuple

from .base_optimizer import BaseOptimizer
from .constraints import PortfolioConstraints, long_only_box, build_scipy_bounds, build_scipy_constraints
from .covariance import ledoit_wolf_cov


class MarkowitzOptimizer(BaseOptimizer):
    """
    Otimizador de Markowitz: Máximo Sharpe e Mínima Variância.

    Parâmetros
    ----------
    cov_estimator : callable
        Estimador de covariância. Default: ledoit_wolf_cov.
    rf : float
        Taxa livre de risco anualizada. Default: 0.05 (5%).
    """

    def __init__(self, cov_estimator=None, rf: float = 0.05):
        super().__init__(cov_estimator)
        self.rf = rf

    @property
    def name(self) -> str:
        return "Markowitz (Max Sharpe)"

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Maximiza o Índice de Sharpe da carteira.

        Internamente, minimiza o negativo do Sharpe:
            min  -(μ_p - rf) / σ_p

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None

        Retorna
        -------
        pd.Series com pesos que maximizam o Sharpe.
        """
        self._validate_returns(returns)
        constraints = constraints or self._default_constraints(returns.shape[1])
        constraints.validate(returns.shape[1])

        tickers = list(returns.columns)
        n = len(tickers)

        mu = returns.mean().values * 252          # retornos anualizados
        cov = self.cov_estimator(returns).values  # covariância anualizada

        def neg_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(port_return - self.rf) / port_vol

        result = self._run_optimization(
            objective=neg_sharpe,
            n=n,
            tickers=tickers,
            constraints=constraints,
        )

        return pd.Series(result, index=tickers, name=self.name)

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Traça a fronteira eficiente variando o retorno alvo.

        Para cada ponto, resolve:
            min  w^T Σ w
            s.t. w^T μ = μ_target, restrições

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None
        n_points : int
            Número de pontos na fronteira. Default: 50.

        Retorna
        -------
        pd.DataFrame com colunas:
            - return: retorno esperado anualizado
            - volatility: volatilidade anualizada
            - sharpe: Índice de Sharpe
            - weights_*: peso de cada ativo (uma coluna por ativo)
        """
        self._validate_returns(returns)
        constraints = constraints or self._default_constraints(returns.shape[1])
        tickers = list(returns.columns)
        n = len(tickers)

        mu = returns.mean().values * 252
        cov = self.cov_estimator(returns).values

        # Range de retornos: do mínimo ao máximo possível
        mu_min = mu.min()
        mu_max = mu.max()
        targets = np.linspace(mu_min, mu_max, n_points)

        records = []

        for mu_target in targets:
            def portfolio_variance(w):
                return w @ cov @ w

            # Adiciona constraint de retorno alvo
            extra_constraint = {
                "type": "eq",
                "fun": lambda w, t=mu_target: w @ mu - t,
            }

            base_cons = build_scipy_constraints(constraints, tickers)
            all_cons = base_cons + [extra_constraint]
            bounds = build_scipy_bounds(constraints, n)

            w0 = np.ones(n) / n
            result = minimize(
                portfolio_variance,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=all_cons,
                options={"ftol": 1e-12, "maxiter": 1000},
            )

            if result.success:
                w = result.x
                port_vol = np.sqrt(w @ cov @ w)
                port_ret = w @ mu
                sharpe = (port_ret - self.rf) / port_vol if port_vol > 0 else 0

                record = {
                    "return": port_ret,
                    "volatility": port_vol,
                    "sharpe": sharpe,
                }
                for i, t in enumerate(tickers):
                    record[f"w_{t}"] = w[i]

                records.append(record)

        return pd.DataFrame(records)

    def _run_optimization(
        self,
        objective,
        n: int,
        tickers: List[str],
        constraints: PortfolioConstraints,
    ) -> np.ndarray:
        """
        Executa scipy.optimize.minimize com reinicializações.

        Roda múltiplos starts aleatórios para evitar mínimos locais.
        Retorna os melhores pesos encontrados.
        """
        bounds = build_scipy_bounds(constraints, n)
        cons = build_scipy_constraints(constraints, tickers)

        best_result = None
        best_value = np.inf

        # Múltiplos starts: equal weight + 4 aleatórios
        starts = [np.ones(n) / n] + [
            np.random.dirichlet(np.ones(n)) for _ in range(4)
        ]

        for w0 in starts:
            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_result = result.x

        if best_result is None:
            raise RuntimeError(
                "Otimização não convergiu. Verifique as constraints e os dados."
            )

        # Limpeza numérica: zera pesos muito pequenos
        best_result = np.where(np.abs(best_result) < 1e-6, 0.0, best_result)
        # Renormaliza para garantir soma = 1
        best_result = best_result / best_result.sum()

        return best_result
