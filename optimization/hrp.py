"""
hrp.py
======
Hierarchical Risk Parity (HRP) — Lopez de Prado, 2016.

Conceito:
    HRP resolve o problema de instabilidade do Markowitz de forma elegante:
    em vez de inverter a matriz de covariância (operação instável), usa
    apenas a estrutura de correlações para agrupar ativos e distribuir
    risco hierarquicamente.

    O algoritmo tem 3 etapas:

    1. CLUSTERIZAÇÃO (Tree Clustering):
       Converte a matriz de correlação em distâncias:
           d_ij = √(0.5 · (1 - ρ_ij))
       Aplica clusterização hierárquica aglomerativa (linkage).
       Resultado: dendrograma que organiza ativos por similaridade.

    2. QUASI-DIAGONALIZAÇÃO:
       Reordena a matriz de covariância para que ativos similares
       (próximos no dendrograma) fiquem adjacentes.
       Resultado: matriz com blocos de alta covariância na diagonal.

    3. ALOCAÇÃO RECURSIVA BISSETORA:
       Divide a lista reordenada em dois subgrupos.
       Aloca risco entre eles inversamente proporcional à variância
       de cada subgrupo (igual ao Risk Parity, mas aplicado recursivamente).
       Repete dentro de cada subgrupo até chegar nos ativos individuais.

Vantagens sobre Markowitz e Risk Parity:
    - Não inverte a matriz de covariância → mais estável
    - Respeita a estrutura de clusters → mais diversificado
    - Funciona bem out-of-sample (Lopez de Prado, 2016)
    - Sem parâmetros livres para otimizar → sem overfitting

Limitação:
    - Não maximiza Sharpe explicitamente
    - A qualidade depende do método de linkage escolhido
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from typing import List, Optional

from .base_optimizer import BaseOptimizer
from .constraints import PortfolioConstraints
from .covariance import ledoit_wolf_cov, sample_correlation


class HRPOptimizer(BaseOptimizer):
    """
    Hierarchical Risk Parity Optimizer.

    Parâmetros
    ----------
    cov_estimator : callable
        Estimador de covariância. Default: ledoit_wolf_cov.
    linkage_method : str
        Método de linkage para clusterização hierárquica.
        Opções: 'single', 'complete', 'average', 'ward'.
        'single' = Lopez de Prado original.
        'ward'   = tende a produzir clusters mais equilibrados.
        Default: 'single'.
    """

    def __init__(self, cov_estimator=None, linkage_method: str = "single"):
        super().__init__(cov_estimator)
        self.linkage_method = linkage_method

    @property
    def name(self) -> str:
        return f"HRP ({self.linkage_method} linkage)"

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Executa HRP e retorna pesos hierárquicos.

        HRP não suporta constraints arbitrárias de forma nativa.
        Se constraints for fornecido, aplica normalização pós-otimização
        para respeitar w_max (clipping + renormalização).

        Parâmetros
        ----------
        returns : pd.DataFrame
        constraints : PortfolioConstraints ou None

        Retorna
        -------
        pd.Series com pesos HRP.
        """
        self._validate_returns(returns)

        tickers = list(returns.columns)
        corr = sample_correlation(returns)
        cov = self.cov_estimator(returns)

        # Etapa 1: Clusterização
        ordered_tickers = self._cluster(corr)

        # Etapa 2 & 3: Alocação recursiva sobre a ordem do dendrograma
        weights = self._recursive_bisection(cov, ordered_tickers)

        # Aplica w_max por clipping se necessário
        if constraints is not None and constraints.w_max < 1.0:
            weights = self._apply_weight_cap(weights, constraints.w_max)

        return pd.Series(weights, name=self.name)

    # ------------------------------------------------------------------
    # Etapa 1: Clusterização hierárquica
    # ------------------------------------------------------------------

    def _cluster(self, corr: pd.DataFrame) -> List[str]:
        """
        Converte correlação em distância e aplica linkage hierárquico.

        Distância: d_ij = √(0.5 · (1 - ρ_ij))
            - ρ = 1  → d = 0  (ativos perfeitamente correlacionados)
            - ρ = 0  → d = 0.707
            - ρ = -1 → d = 1  (máxima distância)

        Retorna a lista de tickers na ordem do dendrograma.
        """
        corr_values = corr.values
        # Garante simetria perfeita e diagonal = 1 (erros numéricos)
        np.fill_diagonal(corr_values, 1.0)
        corr_values = (corr_values + corr_values.T) / 2

        distance = np.sqrt(0.5 * (1 - corr_values))

        # squareform converte matriz quadrada em vetor condensado
        # que é o formato esperado pelo linkage do scipy
        dist_condensed = squareform(distance, checks=False)

        Z = linkage(dist_condensed, method=self.linkage_method)

        # leaves_list retorna a ordem das folhas do dendrograma
        order = leaves_list(Z)
        tickers = list(corr.index)

        return [tickers[i] for i in order]

    # ------------------------------------------------------------------
    # Etapa 2 & 3: Alocação recursiva bissetora
    # ------------------------------------------------------------------

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        ordered_tickers: List[str],
    ) -> pd.Series:
        """
        Distribui pesos recursivamente dividindo a lista em dois subgrupos.

        Algoritmo:
            1. Divide a lista ordenada em duas metades
            2. Calcula a variância de cada metade (como sub-portfólio EW)
            3. Aloca risco entre as metades inversamente proporcional à variância
            4. Repete em cada metade recursivamente
            5. Caso base: lista com 1 ativo → peso = alocado

        Parâmetros
        ----------
        cov : pd.DataFrame
        ordered_tickers : list[str]
            Tickers na ordem do dendrograma.

        Retorna
        -------
        pd.Series com pesos por ticker.
        """
        weights = pd.Series(1.0, index=ordered_tickers)
        self._bisect(weights, cov, ordered_tickers)
        return weights

    def _bisect(
        self,
        weights: pd.Series,
        cov: pd.DataFrame,
        items: List[str],
    ) -> None:
        """Recursão bissetora — modifica weights in-place."""
        if len(items) <= 1:
            return

        # Divide em duas metades
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        # Variância de cada metade como sub-portfólio equal-weight
        var_left = self._cluster_variance(cov, left)
        var_right = self._cluster_variance(cov, right)

        # Alocação inversamente proporcional à variância
        # Mais variância → menos peso (risk parity entre clusters)
        total_var = var_left + var_right
        alpha = 1 - var_left / total_var   # fração para o lado esquerdo

        # Multiplica os pesos de cada metade pela fração alocada
        weights[left] *= alpha
        weights[right] *= (1 - alpha)

        # Recursão
        self._bisect(weights, cov, left)
        self._bisect(weights, cov, right)

    def _cluster_variance(self, cov: pd.DataFrame, tickers: List[str]) -> float:
        """
        Variância de um sub-portfólio equal-weight com os tickers dados.

        w_cluster = 1/N para cada ativo no cluster.
        var = w^T Σ_cluster w
        """
        sub_cov = cov.loc[tickers, tickers].values
        n = len(tickers)
        w = np.ones(n) / n
        return float(w @ sub_cov @ w)

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def _apply_weight_cap(self, weights: pd.Series, w_max: float) -> pd.Series:
        """
        Aplica limite máximo de peso por clipping iterativo.

        Algoritmo:
            1. Clipa todos os pesos acima de w_max para w_max
            2. Redistribui o excesso proporcionalmente entre os demais
            3. Repete até não haver mais violações
        """
        w = weights.copy()
        for _ in range(100):
            excess = (w - w_max).clip(lower=0)
            if excess.sum() < 1e-8:
                break
            w = w.clip(upper=w_max)
            below_cap = w < w_max
            if below_cap.sum() == 0:
                break
            # Redistribui o excesso proporcionalmente
            w[below_cap] += excess.sum() * (w[below_cap] / w[below_cap].sum())

        return w / w.sum()

    def plot_dendrogram(self, returns: pd.DataFrame):
        """
        Plota o dendrograma da clusterização hierárquica.

        Útil para visualizar quais ativos o modelo considera similares.

        Parâmetros
        ----------
        returns : pd.DataFrame
        """
        import matplotlib.pyplot as plt

        corr = sample_correlation(returns)
        corr_values = corr.values.copy()
        np.fill_diagonal(corr_values, 1.0)
        distance = np.sqrt(0.5 * (1 - corr_values))
        dist_condensed = squareform(distance, checks=False)
        Z = linkage(dist_condensed, method=self.linkage_method)

        fig, ax = plt.subplots(figsize=(14, 5))
        dendrogram(
            Z,
            labels=list(returns.columns),
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=9,
        )
        ax.set_title(f"HRP Dendrogram ({self.linkage_method} linkage)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        return fig
