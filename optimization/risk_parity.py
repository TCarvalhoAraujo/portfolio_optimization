"""
risk_parity.py
==============
Carteira de Risk Parity (Equal Risk Contribution — ERC).

Conceito:
    Em vez de equalizar capital, equaliza a contribuição de risco de
    cada ativo para o risco total da carteira.

    Definições:
        σ_p = √(w^T Σ w)           volatilidade da carteira
        MRC_i = (Σw)_i / σ_p       contribuição marginal de risco do ativo i
        TRC_i = w_i · MRC_i        contribuição total de risco do ativo i

    Identidade importante: Σ TRC_i = σ_p (o risco total se decompõe)

    Objetivo do Risk Parity:
        TRC_i = σ_p / N   para todo i
        (cada ativo contribui igualmente para o risco total)

    Problema de otimização equivalente:
        min  Σ_i Σ_j (TRC_i - TRC_j)²
        ou equivalentemente:
        min  Σ_i [w_i · (Σw)_i - σ_p²/N]²

Intuição:
    Uma carteira 60/40 (ações/bonds) parece diversificada em capital,
    mas ações têm volatilidade 3-4x maior que bonds. O resultado é que
    ações contribuem com ~90% do risco. Risk Parity corrige isso
    alocando mais capital para ativos menos voláteis.

Vantagem sobre Markowitz:
    Não depende de estimativas de retorno esperado (μ) — apenas de Σ.
    Mais estável e mais diversificado em termos de risco real.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

from .base_optimizer import BaseOptimizer
from .constraints import PortfolioConstraints, long_only_box, build_scipy_bounds, build_scipy_constraints
from .covariance import ledoit_wolf_cov


def _risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Calcula a contribuição de risco de cada ativo.

    TRC_i = w_i * (Σw)_i / σ_p

    Retorna array de TRC normalizado (soma = 1 = σ_p/σ_p).
    """
    port_var = w @ cov @ w
    port_vol = np.sqrt(port_var)
    marginal = cov @ w          # (Σw) — gradiente da volatilidade
    trc = w * marginal / port_vol
    return trc


def _erc_objective(w: np.ndarray, cov: np.ndarray) -> float:
    """
    Objetivo do ERC: minimiza a dispersão das contribuições de risco.

    Minimiza: Σ_i Σ_j (TRC_i - TRC_j)²

    Quando todas as TRC são iguais, o objetivo = 0.
    """
    trc = _risk_contributions(w, cov)
    n = len(w)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += (trc[i] - trc[j]) ** 2
    return total


def _erc_objective_fast(w: np.ndarray, cov: np.ndarray) -> float:
    """
    Versão vetorizada do objetivo ERC (mais rápida para N grande).

    Equivalente a _erc_objective mas sem loops Python explícitos.
    """
    trc = _risk_contributions(w, cov)
    # Variância das contribuições de risco = medida de desigualdade
    return float(np.sum((trc - trc.mean()) ** 2))


class RiskParityOptimizer(BaseOptimizer):
    """
    Otimizador de Risk Parity (Equal Risk Contribution).

    Não usa retornos esperados. Apenas a matriz de covariância.

    Parâmetros
    ----------
    cov_estimator : callable
        Estimador de covariância. Default: ledoit_wolf_cov.
    """

    def __init__(self, cov_estimator=None):
        super().__init__(cov_estimator)

    @property
    def name(self) -> str:
        return "Risk Parity (ERC)"

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Encontra pesos com contribuição de risco igual entre ativos.

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None

        Retorna
        -------
        pd.Series com pesos ERC.
        """
        self._validate_returns(returns)
        constraints = constraints or long_only_box(w_max=1.0)
        constraints.validate(returns.shape[1])

        tickers = list(returns.columns)
        n = len(tickers)
        cov = self.cov_estimator(returns).values

        bounds = build_scipy_bounds(constraints, n)
        cons = build_scipy_constraints(constraints, tickers)

        best_result = None
        best_value = np.inf

        # Múltiplos starts — ERC tem tendência a mínimos locais
        starts = [np.ones(n) / n] + [
            np.random.dirichlet(np.ones(n)) for _ in range(9)
        ]

        for w0 in starts:
            result = minimize(
                _erc_objective_fast,
                w0,
                args=(cov,),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-14, "maxiter": 2000},
            )

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_result = result.x

        if best_result is None:
            raise RuntimeError("Risk Parity não convergiu.")

        # Limpeza numérica
        best_result = np.where(best_result < 1e-6, 0.0, best_result)
        best_result = best_result / best_result.sum()

        return pd.Series(best_result, index=tickers, name=self.name)

    def risk_contribution_report(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Relatório de contribuição de risco por ativo.

        Parâmetros
        ----------
        weights : pd.Series
            Pesos do portfólio.
        returns : pd.DataFrame

        Retorna
        -------
        pd.DataFrame com colunas:
            - weight: peso do ativo
            - risk_contribution: TRC_i
            - risk_pct: TRC_i / Σ TRC_i (percentual do risco total)
            - marginal_risk: MRC_i
        """
        common = weights.index.intersection(returns.columns)
        w = weights[common].values
        cov = self.cov_estimator(returns[common]).values

        port_vol = np.sqrt(w @ cov @ w)
        marginal = cov @ w / port_vol
        trc = w * marginal

        df = pd.DataFrame({
            "weight": weights[common].values,
            "risk_contribution": trc,
            "risk_pct": trc / trc.sum(),
            "marginal_risk": marginal,
        }, index=common)

        return df.sort_values("risk_contribution", ascending=False)
