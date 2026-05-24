"""
black_litterman.py
==================
Modelo de Black-Litterman com views configuráveis.

Conceito:
    Black-Litterman (1990) resolve o problema mais sério do Markowitz:
    a hipersensibilidade às estimativas de retorno esperado.

    A ideia Bayesiana:
        Prior:    retornos de equilíbrio implícitos no mercado
        Likelihood: views do investidor (opiniões quantitativas)
        Posterior: média ponderada pela confiança em cada fonte

    Retornos de Equilíbrio (Prior):
        O mercado, em equilíbrio, precifica ativos de forma que a
        carteira de mercado (ponderada por market cap) seja ótima.
        Isso implica retornos de equilíbrio:

            Π = λ · Σ · w_mkt

        Onde:
            λ     = coeficiente de aversão ao risco do mercado
            Σ     = matriz de covariância
            w_mkt = pesos de mercado (market cap)

    Views do Investidor:
        Expressas como: P · μ = Q + ε,  ε ~ N(0, Ω)
        Onde:
            P = matriz de seleção (quais ativos a view afeta)
            Q = retorno esperado pela view
            Ω = incerteza da view (diagonal → views independentes)

    Posterior (Black-Litterman):
        μ_BL = [(τΣ)⁻¹ + P^T Ω⁻¹ P]⁻¹ · [(τΣ)⁻¹ Π + P^T Ω⁻¹ Q]

        Onde τ é um escalar pequeno (≈ 1/T) que escala a incerteza
        sobre o prior.

    Os retornos μ_BL são então usados no otimizador de Markowitz.

Tipos de Views suportados:
    1. View absoluta: "AAPL vai retornar 15% ao ano"
    2. View relativa: "AAPL vai superar MSFT em 5% ao ano"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base_optimizer import BaseOptimizer
from .constraints import PortfolioConstraints, long_only_box, build_scipy_bounds, build_scipy_constraints
from .covariance import ledoit_wolf_cov


# ---------------------------------------------------------------------------
# Estrutura de View
# ---------------------------------------------------------------------------

@dataclass
class View:
    """
    Uma view do investidor sobre retornos esperados.

    Parâmetros
    ----------
    assets : list[str]
        Ativos envolvidos na view.
    weights : list[float]
        Pesos de cada ativo na view.
        Para view absoluta: [1.0] para o único ativo.
        Para view relativa (A supera B): [1.0, -1.0].
    expected_return : float
        Retorno esperado anualizado pela view.
        Ex: 0.15 = 15% ao ano.
    confidence : float
        Confiança na view. ∈ (0, 1].
        0.5 = moderada confiança, 0.9 = alta confiança.
        Internamente converte para a variância da view (Ω_kk).
    name : str
        Descrição legível da view (opcional).

    Exemplos
    --------
    # View absoluta: AAPL retorna 18% ao ano (alta confiança)
    View(assets=["AAPL"], weights=[1.0], expected_return=0.18, confidence=0.8)

    # View relativa: NVDA supera AMD em 10% ao ano (confiança moderada)
    View(assets=["NVDA", "AMD"], weights=[1.0, -1.0],
         expected_return=0.10, confidence=0.5)
    """
    assets: List[str]
    weights: List[float]
    expected_return: float
    confidence: float = 0.5
    name: str = ""

    def __post_init__(self):
        if len(self.assets) != len(self.weights):
            raise ValueError("assets e weights devem ter o mesmo tamanho.")
        if not (0 < self.confidence <= 1):
            raise ValueError("confidence deve ser ∈ (0, 1].")


# ---------------------------------------------------------------------------
# Otimizador Black-Litterman
# ---------------------------------------------------------------------------

class BlackLittermanOptimizer(BaseOptimizer):
    """
    Otimizador Black-Litterman com views configuráveis.

    Parâmetros
    ----------
    cov_estimator : callable
    rf : float
        Taxa livre de risco anualizada. Default: 0.05.
    tau : float
        Escalar de incerteza do prior. Convenção: 1/T onde T = nº observações.
        Valores típicos: 0.01 a 0.05. Default: 0.05.
    risk_aversion : float
        Coeficiente de aversão ao risco λ do mercado.
        Estimado como (μ_mkt - rf) / σ²_mkt. Default: 2.5 (proxy histórico).
    """

    def __init__(
        self,
        cov_estimator=None,
        rf: float = 0.05,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ):
        super().__init__(cov_estimator)
        self.rf = rf
        self.tau = tau
        self.risk_aversion = risk_aversion
        self._view_method: Optional[str] = None  # populado por optimize_with_views

    @property
    def name(self) -> str:
        if self._view_method:
            return f"BL-{self._view_method}"
        return "Black-Litterman"

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        views: Optional[List[View]] = None,
        market_weights: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Otimiza usando retornos Black-Litterman como input do Markowitz.

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None
        views : list[View] ou None
            Views do investidor. Se None, usa apenas o prior de equilíbrio
            (equivale a carteira de mercado rebalanceada).
        market_weights : pd.Series ou None
            Pesos de mercado (market cap) por ticker.
            Se None, usa equal-weight como proxy.

        Retorna
        -------
        pd.Series com pesos ótimos Black-Litterman.
        """
        self._validate_returns(returns)
        constraints = constraints or long_only_box(w_max=0.20)
        constraints.validate(returns.shape[1])
        views = views or []

        tickers = list(returns.columns)
        n = len(tickers)
        cov = self.cov_estimator(returns).values

        # Pesos de mercado (prior)
        if market_weights is not None:
            w_mkt = market_weights.reindex(tickers).fillna(0).values
            w_mkt = w_mkt / w_mkt.sum()
        else:
            w_mkt = np.ones(n) / n  # equal-weight como proxy

        # Retornos de equilíbrio: Π = λ · Σ · w_mkt
        pi = self.risk_aversion * cov @ w_mkt

        # Retornos Black-Litterman
        mu_bl = self._bl_returns(pi, cov, views, tickers)

        # Otimiza Sharpe usando μ_BL
        bounds = build_scipy_bounds(constraints, n)
        cons = build_scipy_constraints(constraints, tickers)

        def neg_sharpe(w):
            port_ret = w @ mu_bl
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(port_ret - self.rf) / port_vol

        best_result = None
        best_value = np.inf
        starts = [w_mkt] + [np.ones(n) / n] + [
            np.random.dirichlet(np.ones(n)) for _ in range(3)
        ]

        for w0 in starts:
            result = minimize(
                neg_sharpe, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_result = result.x

        if best_result is None:
            raise RuntimeError("Black-Litterman não convergiu.")

        best_result = np.where(np.abs(best_result) < 1e-6, 0.0, best_result)
        best_result = best_result / best_result.sum()

        return pd.Series(best_result, index=tickers, name=self.name)

    def optimize_with_views(
        self,
        returns: pd.DataFrame,
        method: str                        = "momentum",
        sector_map: Optional[dict]         = None,
        constraints: Optional[PortfolioConstraints] = None,
        market_weights: Optional[pd.Series] = None,
        momentum_kwargs: Optional[dict]    = None,
        sector_momentum_kwargs: Optional[dict] = None,
        verbose: bool                      = True,
    ) -> pd.Series:
        """
        Atalho: gera views automaticamente e otimiza.

        Combina build_views() + optimize() em uma única chamada.
        Ideal para uso no pipeline sem precisar instanciar views manualmente.

        Parâmetros
        ----------
        returns : pd.DataFrame
        method : str
            "momentum" | "sector_momentum" | "combined"
        sector_map : dict ou None
            {ticker: setor}. Necessário para sector_momentum e combined.
        constraints : PortfolioConstraints ou None
        market_weights : pd.Series ou None
        momentum_kwargs : dict ou None
            Passado para momentum_views(). Ex: {"n_winners": 3, "confidence": 0.4}
        sector_momentum_kwargs : dict ou None
            Passado para sector_momentum_views().
        verbose : bool

        Retorna
        -------
        pd.Series com pesos ótimos BL com views de momentum.

        Exemplo
        -------
        >>> opt = BlackLittermanOptimizer(rf=0.05)
        >>> weights = opt.optimize_with_views(
        ...     returns,
        ...     method="combined",
        ...     sector_map=sector_map,
        ...     constraints=long_only_box(w_max=0.15),
        ... )
        """
        from .views import build_views

        views = build_views(
            returns,
            method=method,
            sector_map=sector_map,
            momentum_kwargs=momentum_kwargs,
            sector_momentum_kwargs=sector_momentum_kwargs,
            verbose=verbose,
        )

        self._view_method = method

        return self.optimize(
            returns,
            constraints=constraints,
            views=views,
            market_weights=market_weights,
        )

    def equilibrium_returns(
        self,
        returns: pd.DataFrame,
        market_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Retorna os retornos de equilíbrio implícitos (prior BL).

        Útil para comparar com retornos históricos e identificar
        onde o mercado está "precificando diferente" do histórico.
        """
        tickers = list(returns.columns)
        cov = self.cov_estimator(returns).values
        n = len(tickers)

        if market_weights is not None:
            w_mkt = market_weights.reindex(tickers).fillna(0).values
            w_mkt = w_mkt / w_mkt.sum()
        else:
            w_mkt = np.ones(n) / n

        pi = self.risk_aversion * cov @ w_mkt
        return pd.Series(pi, index=tickers, name="equilibrium_returns")

    # ------------------------------------------------------------------
    # Cálculo BL interno
    # ------------------------------------------------------------------

    def _bl_returns(
        self,
        pi: np.ndarray,
        cov: np.ndarray,
        views: List[View],
        tickers: List[str],
    ) -> np.ndarray:
        """
        Calcula retornos Black-Litterman combinando prior e views.

        Fórmula:
            μ_BL = [(τΣ)⁻¹ + P^T Ω⁻¹ P]⁻¹ · [(τΣ)⁻¹ Π + P^T Ω⁻¹ Q]

        Se não há views, retorna Π (prior de equilíbrio).
        """
        if not views:
            return pi  # sem views → usa o prior diretamente

        n = len(tickers)
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        K = len(views)  # número de views

        # Monta matriz P (K x N): cada linha é uma view
        P = np.zeros((K, n))
        Q = np.zeros(K)
        omega_diag = np.zeros(K)

        for k, view in enumerate(views):
            for asset, w in zip(view.assets, view.weights):
                if asset in ticker_idx:
                    P[k, ticker_idx[asset]] = w
            Q[k] = view.expected_return

            # Ω_kk: variância da view
            # Convenção: Ω_kk = (1 - conf) / conf · P_k Σ P_k^T
            # Quanto maior a confiança, menor a variância
            view_var = P[k] @ cov @ P[k]
            omega_diag[k] = view_var * (1 - view.confidence) / view.confidence

        Omega = np.diag(omega_diag)

        # Prior precision: (τΣ)⁻¹
        tau_cov_inv = np.linalg.inv(self.tau * cov)

        # View precision: P^T Ω⁻¹ P
        omega_inv = np.diag(1.0 / omega_diag)
        view_precision = P.T @ omega_inv @ P

        # Posterior precision
        post_precision = tau_cov_inv + view_precision

        # Posterior mean
        post_mean = np.linalg.solve(
            post_precision,
            tau_cov_inv @ pi + P.T @ omega_inv @ Q
        )

        return post_mean