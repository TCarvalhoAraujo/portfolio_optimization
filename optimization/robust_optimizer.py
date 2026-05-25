"""
robust_optimizer.py
===================
Otimização Robusta — maximiza retorno no pior cenário possível.

Conceito:
    O Markowitz clássico otimiza para um único cenário (μ estimado).
    A otimização robusta reconhece que μ é incerto e otimiza para
    o pior caso dentro de um conjunto de incerteza U:

        max_w  min_{μ ∈ U}  (w^T μ - λ · w^T Σ w)

    O conjunto U é uma elipsoide ao redor da estimativa μ̂:
        U = {μ : (μ - μ̂)^T Θ⁻¹ (μ - μ̂) ≤ κ²}

    Onde:
        μ̂ = retorno esperado estimado (ex: média histórica)
        Θ = matriz de incerteza (ex: τ·Σ/T ou variância amostral de μ̂)
        κ = "tamanho" do conjunto de incerteza (parâmetro de robustez)

    O problema se reduz a:
        max_w  w^T μ̂ - κ · √(w^T Θ w) - λ · w^T Σ w

    O termo κ · √(w^T Θ w) é o "penalty de robustez": penaliza
    portfólios que dependem de estimativas incertas de μ.

Intuição:
    κ = 0  → Markowitz clássico (nenhuma robustez)
    κ = ∞  → Mínima variância (ignora μ completamente)
    κ ∈ (0, ∞) → trade-off controlável entre Sharpe esperado e robustez

Quando usar:
    - Muita incerteza sobre retornos futuros
    - Poucos dados históricos confiáveis
    - Períodos de grande instabilidade macroeconômica
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

from .base_optimizer import BaseOptimizer
from .constraints import PortfolioConstraints, long_only_box, build_scipy_bounds, build_scipy_constraints
from .covariance import ledoit_wolf_cov


class RobustOptimizer(BaseOptimizer):
    """
    Otimizador Robusto — maximiza retorno no pior cenário.

    Parâmetros
    ----------
    cov_estimator : callable
    rf : float
        Taxa livre de risco. Default: 0.05.
    kappa : float
        Parâmetro de robustez. Controla o tamanho do conjunto de incerteza.
        0.0 = Markowitz clássico.
        1.0 = robustez moderada (recomendado como ponto de partida).
        3.0 = robustez alta.
        Default: 1.0.
    risk_aversion : float
        Coeficiente de aversão ao risco λ na função objetivo.
        Balanceia retorno esperado vs. variância.
        Default: 1.0.
    """

    def __init__(
        self,
        cov_estimator=None,
        rf: float = 0.05,
        kappa: float = 1.0,
        risk_aversion: float = 1.0,
    ):
        super().__init__(cov_estimator)
        self.rf = rf
        self.kappa = kappa
        self.risk_aversion = risk_aversion

    @property
    def name(self) -> str:
        return f"Robust (κ={self.kappa})"

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Maximiza o retorno robusto ajustado ao risco.

        Objetivo:
            max  w^T μ̂ - κ · √(w^T Θ w) - λ · w^T Σ w

        Onde Θ = Σ / T (incerteza de estimação da média).

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None

        Retorna
        -------
        pd.Series com pesos robustos.
        """
        self._validate_returns(returns)
        constraints = constraints or long_only_box(w_max=0.15)
        constraints.validate(returns.shape[1])

        tickers = list(returns.columns)
        n = len(tickers)
        T = returns.shape[0]

        mu = returns.mean().values * 252
        cov = self.cov_estimator(returns).values

        # Matriz de incerteza Θ: variância amostral da estimativa de μ
        # Θ ≈ Σ / T (erro padrão da média amostral)
        Theta = cov / T

        def neg_robust_objective(w):
            """
            Negativo do objetivo robusto (minimizar = maximizar).

            Termos:
                w^T μ̂         : retorno esperado (positivo → queremos max)
                κ·√(w^T Θ w)  : penalty de incerteza (negativo)
                λ·w^T Σ w     : penalty de variância (negativo)
            """
            expected_ret = w @ mu
            uncertainty_penalty = self.kappa * np.sqrt(w @ Theta @ w + 1e-12)
            variance_penalty = self.risk_aversion * (w @ cov @ w)
            return -(expected_ret - uncertainty_penalty - variance_penalty)

        bounds = build_scipy_bounds(constraints, n)
        cons = build_scipy_constraints(constraints, tickers)

        best_result = None
        best_value = np.inf

        starts = [np.ones(n) / n] + [
            np.random.dirichlet(np.ones(n)) for _ in range(4)
        ]

        for w0 in starts:
            result = minimize(
                neg_robust_objective,
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
            raise RuntimeError("Otimização robusta não convergiu.")

        best_result = np.where(np.abs(best_result) < 1e-6, 0.0, best_result)
        best_result = best_result / best_result.sum()

        return pd.Series(best_result, index=tickers, name=self.name)

    def kappa_sensitivity(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        kappas: Optional[list] = None,
        rf: float = 0.05,
    ) -> pd.DataFrame:
        """
        Analisa como os pesos e métricas mudam com κ.

        Útil para escolher o nível de robustez adequado ao seu
        grau de confiança nas estimativas de retorno.

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None
        kappas : list[float] ou None
            Valores de κ a testar. Default: [0, 0.5, 1, 2, 3, 5].
        rf : float

        Retorna
        -------
        pd.DataFrame com métricas por κ.
        """
        kappas = kappas or [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        records = []

        original_kappa = self.kappa

        for k in kappas:
            self.kappa = k
            weights = self.optimize(returns, constraints)
            metrics = self.portfolio_metrics(weights, returns, rf=rf)
            metrics["kappa"] = k
            metrics["n_assets"] = int((weights > 0.001).sum())
            records.append(metrics)

        self.kappa = original_kappa  # restaura κ original

        df = pd.DataFrame(records).set_index("kappa")
        return df
