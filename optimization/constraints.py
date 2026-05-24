"""
constraints.py
==============
Define restrições de portfólio reutilizáveis por todos os otimizadores.

Conceito:
    Restrições transformam o problema de otimização livre em algo realista.
    Sem restrições, o otimizador pode alocar 100% em 1 ativo, shortar
    ativos sem limite, ou ignorar setores inteiros. As restrições abaixo
    cobrem os casos mais comuns em gestão de portfólios reais.

Estrutura:
    PortfolioConstraints  — dataclass de configuração (o que o usuário define)
    build_scipy_constraints()  — converte para formato scipy.optimize
    build_scipy_bounds()       — converte limites por ativo para scipy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Dataclass de configuração
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConstraints:
    """
    Centraliza todas as restrições de portfólio em um único objeto.

    Parâmetros
    ----------
    long_only : bool
        Se True, proíbe posições vendidas (w_i >= 0). Default: True.

    w_min : float
        Peso mínimo por ativo. Só faz sentido se long_only=True.
        Ex: 0.01 força pelo menos 1% em cada ativo se incluído.
        Default: 0.0

    w_max : float
        Peso máximo por ativo. Evita concentração excessiva.
        Ex: 0.10 = nenhum ativo passa de 10% da carteira.
        Default: 1.0 (sem restrição)

    sector_constraints : dict
        Mapa de {nome_setor: (min_weight, max_weight)}.
        Ex: {"Technology": (0.05, 0.30)} limita tech entre 5% e 30%.
        Default: {} (sem restrição setorial)

    sector_map : dict
        Mapa de {ticker: nome_setor}.
        Necessário se sector_constraints for definido.
        Ex: {"AAPL": "Technology", "JPM": "Financials"}

    max_turnover : float ou None
        Turnover máximo permitido vs. portfólio atual.
        Ex: 0.20 = no máximo 20% do portfólio pode mudar em cada rebalanceamento.
        Requer w_current para ser usado.
        Default: None (sem restrição de turnover)

    w_current : np.ndarray ou None
        Pesos atuais do portfólio (para calcular turnover).
        Deve ser fornecido se max_turnover não for None.

    budget : float
        Soma dos pesos. 1.0 = fully invested. Default: 1.0.
    """

    long_only: bool = True
    w_min: float = 0.0
    w_max: float = 1.0
    sector_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sector_map: Dict[str, str] = field(default_factory=dict)
    max_turnover: Optional[float] = None
    w_current: Optional[np.ndarray] = None
    budget: float = 1.0

    def validate(self, n_assets: int) -> None:
        """
        Verifica consistência das restrições antes de otimizar.
        Lança ValueError com mensagem clara se algo estiver errado.
        """
        if self.w_min < 0 and self.long_only:
            raise ValueError("w_min não pode ser negativo com long_only=True.")

        if self.w_min > self.w_max:
            raise ValueError(f"w_min ({self.w_min}) > w_max ({self.w_max}).")

        if self.w_min * n_assets > self.budget:
            raise ValueError(
                f"w_min={self.w_min} * {n_assets} ativos = {self.w_min * n_assets:.2f} "
                f"> budget={self.budget}. Impossível satisfazer."
            )

        if self.sector_constraints and not self.sector_map:
            raise ValueError(
                "sector_constraints definido mas sector_map está vazio. "
                "Forneça o mapa {ticker: setor}."
            )

        if self.max_turnover is not None and self.w_current is None:
            raise ValueError(
                "max_turnover definido mas w_current não foi fornecido."
            )

        if self.w_current is not None and len(self.w_current) != n_assets:
            raise ValueError(
                f"w_current tem {len(self.w_current)} elementos, "
                f"esperado {n_assets}."
            )


# ---------------------------------------------------------------------------
# Builders para scipy.optimize
# ---------------------------------------------------------------------------

def build_scipy_bounds(
    constraints: PortfolioConstraints,
    n_assets: int,
) -> List[Tuple[float, float]]:
    """
    Constrói lista de bounds por ativo no formato scipy.optimize.

    Retorna
    -------
    list of (min, max) para cada ativo.
    Ex: [(0.0, 0.10), (0.0, 0.10), ...] para long-only com w_max=10%.
    """
    lower = 0.0 if constraints.long_only else -1.0
    lower = max(lower, constraints.w_min)
    upper = constraints.w_max

    return [(lower, upper)] * n_assets


def build_scipy_constraints(
    constraints: PortfolioConstraints,
    tickers: List[str],
) -> List[Dict]:
    """
    Constrói lista de constraints no formato scipy.optimize.minimize.

    Cada constraint é um dict com:
        'type': 'eq' (igualdade) ou 'ineq' (>=0)
        'fun':  função que retorna 0 quando satisfeita (eq)
                ou valor >= 0 quando satisfeita (ineq)

    Parâmetros
    ----------
    constraints : PortfolioConstraints
    tickers : list[str]
        Lista de tickers na mesma ordem dos pesos w.

    Retorna
    -------
    list de dicts compatíveis com scipy.optimize.minimize(constraints=...)
    """
    n = len(tickers)
    cons = []

    # 1. Budget constraint: soma dos pesos = budget (geralmente 1.0)
    cons.append({
        "type": "eq",
        "fun": lambda w: np.sum(w) - constraints.budget,
    })

    # 2. Restrições setoriais
    for sector, (s_min, s_max) in constraints.sector_constraints.items():
        # Identifica índices dos ativos pertencentes ao setor
        sector_idx = [
            i for i, t in enumerate(tickers)
            if constraints.sector_map.get(t) == sector
        ]

        if not sector_idx:
            continue  # setor definido mas sem ativos na lista — ignora

        # Peso total do setor >= s_min
        cons.append({
            "type": "ineq",
            "fun": lambda w, idx=sector_idx, mn=s_min: np.sum(w[idx]) - mn,
        })

        # Peso total do setor <= s_max
        cons.append({
            "type": "ineq",
            "fun": lambda w, idx=sector_idx, mx=s_max: mx - np.sum(w[idx]),
        })

    # 3. Restrição de turnover
    if constraints.max_turnover is not None and constraints.w_current is not None:
        w0 = constraints.w_current
        max_to = constraints.max_turnover

        # turnover = sum(|w_new - w_old|) / 2 <= max_turnover
        # Equivalente (ineq >= 0): max_turnover - sum(|w - w0|)/2 >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda w, w0=w0, mt=max_to: mt - np.sum(np.abs(w - w0)) / 2,
        })

    return cons


# ---------------------------------------------------------------------------
# Preset de configurações comuns
# ---------------------------------------------------------------------------

def long_only_box(w_max: float = 0.10, w_min: float = 0.0) -> PortfolioConstraints:
    """
    Preset mais comum: long-only com limite máximo por ativo.

    Parâmetros
    ----------
    w_max : float
        Peso máximo por ativo. Default: 10%.
    w_min : float
        Peso mínimo por ativo. Default: 0% (ativo pode ser excluído).
    """
    return PortfolioConstraints(long_only=True, w_min=w_min, w_max=w_max)


def long_only_sector(
    sector_map: Dict[str, str],
    sector_limits: Dict[str, Tuple[float, float]],
    w_max: float = 0.10,
) -> PortfolioConstraints:
    """
    Preset com restrições setoriais.

    Parâmetros
    ----------
    sector_map : dict
        {ticker: setor}
    sector_limits : dict
        {setor: (min_weight, max_weight)}
    w_max : float
        Peso máximo por ativo individual.
    """
    return PortfolioConstraints(
        long_only=True,
        w_max=w_max,
        sector_map=sector_map,
        sector_constraints=sector_limits,
    )


def with_turnover(
    base: PortfolioConstraints,
    w_current: np.ndarray,
    max_turnover: float = 0.20,
) -> PortfolioConstraints:
    """
    Adiciona restrição de turnover a um conjunto de constraints existente.

    Útil para rebalanceamento periódico onde você quer limitar custos
    de transação implicitamente pelo turnover máximo permitido.

    Parâmetros
    ----------
    base : PortfolioConstraints
        Constraints base (ex: long_only_box()).
    w_current : np.ndarray
        Pesos atuais antes do rebalanceamento.
    max_turnover : float
        Turnover máximo permitido. Default: 20%.
    """
    from dataclasses import replace
    return replace(base, w_current=w_current, max_turnover=max_turnover)
