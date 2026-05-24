"""
base_optimizer.py
=================
Classe abstrata que define a interface comum a todos os otimizadores.

Por que uma classe base?
    Todos os otimizadores recebem os mesmos inputs (retornos históricos,
    constraints) e devem retornar os mesmos outputs (pesos, métricas).
    Definir uma interface comum permite:
    - Trocar otimizadores sem mudar o código de backtesting
    - Comparar estratégias com a mesma régua
    - Garantir que todos implementem os métodos necessários

Design:
    BaseOptimizer define o contrato (interface).
    Cada subclasse implementa optimize() com sua própria lógica.
    portfolio_metrics() é implementado aqui — igual para todos.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import pandas as pd

from .constraints import PortfolioConstraints, long_only_box
from .covariance import ledoit_wolf_cov


class BaseOptimizer(ABC):
    """
    Interface comum para todos os otimizadores de portfólio.

    Subclasses devem implementar:
        optimize() → pd.Series com pesos por ticker

    Subclasses herdam automaticamente:
        portfolio_metrics() — retorno esperado, vol, Sharpe da carteira
        __repr__()          — representação legível

    Parâmetros do construtor
    ------------------------
    cov_estimator : callable
        Função que recebe returns (DataFrame) e retorna cov (DataFrame).
        Default: ledoit_wolf_cov — recomendado para uso geral.
        Alternativas: sample_covariance, pca_factor_covariance, etc.
    """

    def __init__(self, cov_estimator=None):
        self.cov_estimator = cov_estimator or ledoit_wolf_cov
        self._weights: Optional[pd.Series] = None
        self._returns: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Interface obrigatória — cada subclasse implementa isso
    # ------------------------------------------------------------------

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Executa a otimização e retorna os pesos do portfólio.

        Parâmetros
        ----------
        returns : pd.DataFrame
            Retornos históricos. Shape (T, N). Index = datas, Columns = tickers.
            Deve estar limpo (sem NaN, sem outliers extremos).
        constraints : PortfolioConstraints ou None
            Restrições de portfólio. Se None, usa long_only_box(w_max=1.0).
        **kwargs
            Parâmetros específicos de cada otimizador.

        Retorna
        -------
        pd.Series
            Pesos otimizados. Index = tickers, valores somam 1.0.
            Ex: pd.Series({"AAPL": 0.05, "MSFT": 0.08, ...})
        """
        ...

    @property
    def name(self) -> str:
        """Nome legível do otimizador. Override nas subclasses."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Métodos compartilhados — implementados aqui, herdados por todos
    # ------------------------------------------------------------------

    def portfolio_metrics(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        rf: float = 0.05,
        trading_days: int = 252,
    ) -> Dict[str, float]:
        """
        Calcula métricas financeiras de um portfólio dado seus pesos.

        Métricas calculadas:
            - expected_return: retorno anualizado esperado (média histórica)
            - volatility: volatilidade anualizada da carteira
            - sharpe_ratio: (retorno - rf) / volatilidade
            - min_weight: menor peso não-zero
            - max_weight: maior peso
            - n_assets_active: número de ativos com peso > 0.1%

        Parâmetros
        ----------
        weights : pd.Series
            Pesos do portfólio. Index = tickers.
        returns : pd.DataFrame
            Retornos históricos.
        rf : float
            Taxa livre de risco anualizada. Default: 5% (Treasury 2024).
        trading_days : int
            Dias úteis por ano.

        Retorna
        -------
        dict com as métricas.

        Exemplo
        -------
        >>> weights = optimizer.optimize(returns)
        >>> metrics = optimizer.portfolio_metrics(weights, returns)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        """
        # Alinha tickers — garante que weights e returns têm os mesmos ativos
        common_tickers = weights.index.intersection(returns.columns)
        w = weights[common_tickers].values
        ret = returns[common_tickers]

        # Retorno esperado anualizado: média dos retornos diários * 252
        mu = ret.mean().values
        expected_return = float(w @ mu) * trading_days

        # Covariância e volatilidade da carteira
        cov = self.cov_estimator(ret, annualize=True, trading_days=trading_days)
        portfolio_var = float(w @ cov.values @ w)
        volatility = np.sqrt(portfolio_var)

        # Sharpe Ratio
        sharpe = (expected_return - rf) / volatility if volatility > 0 else 0.0

        # Métricas de concentração
        nonzero = weights[weights > 0.001]

        return {
            "expected_return": round(expected_return, 6),
            "volatility": round(volatility, 6),
            "sharpe_ratio": round(sharpe, 4),
            "min_weight": round(nonzero.min() if len(nonzero) > 0 else 0.0, 6),
            "max_weight": round(weights.max(), 6),
            "n_assets_active": int((weights > 0.001).sum()),
        }

    def fit_and_metrics(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        rf: float = 0.05,
        **kwargs,
    ) -> Dict:
        """
        Conveniência: otimiza e calcula métricas em uma chamada.

        Retorna
        -------
        dict com:
            "weights": pd.Series
            "metrics": dict de métricas
        """
        weights = self.optimize(returns, constraints, **kwargs)
        metrics = self.portfolio_metrics(weights, returns, rf=rf)
        self._weights = weights
        self._returns = returns
        return {"weights": weights, "metrics": metrics}

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def _validate_returns(self, returns: pd.DataFrame) -> None:
        """
        Verificações básicas nos dados de entrada.
        Lança erros descritivos antes de tentar otimizar.
        """
        if returns.empty:
            raise ValueError("DataFrame de retornos está vazio.")

        if returns.isnull().any().any():
            n_nan = returns.isnull().sum().sum()
            raise ValueError(
                f"Returns contém {n_nan} NaN(s). "
                "Limpe os dados antes de otimizar."
            )

        if returns.shape[0] < returns.shape[1]:
            import warnings
            warnings.warn(
                f"T={returns.shape[0]} observações < N={returns.shape[1]} ativos. "
                "A matriz de covariância amostral será singular. "
                "Use Ledoit-Wolf ou aumente o período histórico.",
                UserWarning,
            )

    def _default_constraints(self, n_assets: int) -> PortfolioConstraints:
        """Constraints padrão se nenhum for fornecido."""
        return long_only_box(w_max=1.0)

    def __repr__(self) -> str:
        return (
            f"{self.name}("
            f"cov_estimator={self.cov_estimator.__name__})"
        )
