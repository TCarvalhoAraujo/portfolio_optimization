"""
metrics.py
==========
Métricas financeiras calculadas sobre os resultados da simulação Monte Carlo.

Todas as funções operam sobre arrays NumPy e são agnósticas de backend
(funcionam igual com resultados de CPU ou GPU).

Métricas implementadas
----------------------
- Retorno esperado (média e mediana)
- Volatilidade realizada
- Value at Risk (VaR) histórico — percentil dos retornos
- Conditional VaR / Expected Shortfall (CVaR/ES)
- Índice de Sharpe (anualizado)
- Índice de Sortino (penaliza só downside)
- Maximum Drawdown (sobre trajetórias, quando disponível)
- Probabilidade de perda
- Distribuição de retornos finais
"""

import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# ESTRUTURA DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioMetrics:
    """
    Conjunto completo de métricas de risco/retorno de um portfólio simulado.

    Todos os valores de retorno são frações (ex: 0.15 = 15%).
    """
    # Retorno
    expected_return   : float   # média dos retornos finais
    median_return     : float   # mediana (mais robusta a outliers)
    std_return        : float   # desvio padrão dos retornos

    # Risco
    var_95            : float   # VaR 95%: pior retorno em 95% dos cenários
    var_99            : float   # VaR 99%
    cvar_95           : float   # CVaR 95% (perda esperada além do VaR)
    cvar_99           : float   # CVaR 99%
    prob_loss         : float   # probabilidade de retorno negativo

    # Índices ajustados ao risco
    sharpe            : float   # (E[R] - rf) / std   (anualizado)
    sortino           : float   # (E[R] - rf) / downside_std

    # Percentis de distribuição
    p5                : float   # percentil 5%
    p25               : float   # percentil 25%
    p75               : float   # percentil 75%
    p95               : float   # percentil 95%

    # Contexto
    n_sims            : int
    n_steps           : int
    rf_annual         : float   # taxa livre de risco usada

    def print(self) -> None:
        """Exibe as métricas formatadas."""
        print("\n" + "="*50)
        print("  MÉTRICAS DO PORTFÓLIO")
        print("="*50)
        print(f"  Simulações        : {self.n_sims:,}")
        print(f"  Horizonte (dias)  : {self.n_steps}")
        print(f"  Taxa livre risco  : {self.rf_annual:.2%} a.a.")
        print()
        print(f"  Retorno esperado  : {self.expected_return:>+10.2%}")
        print(f"  Retorno mediano   : {self.median_return:>+10.2%}")
        print(f"  Volatilidade      : {self.std_return:>10.2%}")
        print()
        print(f"  VaR  95%          : {self.var_95:>+10.2%}")
        print(f"  VaR  99%          : {self.var_99:>+10.2%}")
        print(f"  CVaR 95%          : {self.cvar_95:>+10.2%}")
        print(f"  CVaR 99%          : {self.cvar_99:>+10.2%}")
        print(f"  P(perda)          : {self.prob_loss:>10.2%}")
        print()
        print(f"  Sharpe            : {self.sharpe:>10.3f}")
        print(f"  Sortino           : {self.sortino:>10.3f}")
        print()
        print(f"  Percentis: p5={self.p5:+.2%}  p25={self.p25:+.2%}"
              f"  p75={self.p75:+.2%}  p95={self.p95:+.2%}")
        print("="*50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS INDIVIDUAIS
# ─────────────────────────────────────────────────────────────────────────────

def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Value at Risk histórico (método percentil).

    Interpreta: com probabilidade `confidence`, a perda não excederá |VaR|.
    VaR é retornado como número negativo (representa perda).

    Ex: var_historical(r, 0.95) = -0.12  →  perda máxima de 12% em 95% dos cenários
    """
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) — média das perdas além do VaR.

    Mais informativo que o VaR porque captura a severidade das caudas.
    Também retornado como número negativo.
    """
    threshold = var_historical(returns, confidence)
    tail      = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else threshold


def sharpe_ratio(
    returns: np.ndarray,
    n_steps: int,
    trading_days: int = 252,
    rf_annual: float  = 0.05,
) -> float:
    """
    Índice de Sharpe anualizado.

        Sharpe = (E[R_anual] - rf) / sigma_anual

    Converte retornos do horizonte simulado para base anual antes de calcular.
    """
    # Anualiza: R_anual ≈ R_horizonte * (trading_days / n_steps)
    scale       = trading_days / n_steps
    mu_annual   = returns.mean() * scale
    std_annual  = returns.std()  * np.sqrt(scale)

    if std_annual < 1e-8:
        return 0.0
    return float((mu_annual - rf_annual) / std_annual)


def sortino_ratio(
    returns: np.ndarray,
    n_steps: int,
    trading_days: int = 252,
    rf_annual: float  = 0.05,
) -> float:
    """
    Índice de Sortino — como o Sharpe, mas penaliza apenas o downside.

        Sortino = (E[R_anual] - rf) / downside_std_anual

    downside_std usa apenas os retornos abaixo do retorno alvo (aqui: rf).
    """
    scale      = trading_days / n_steps
    mu_annual  = returns.mean() * scale
    rf_horizon = rf_annual / scale

    downside   = returns[returns < rf_horizon] - rf_horizon
    if len(downside) == 0:
        return float("inf")

    downside_std_annual = np.sqrt((downside ** 2).mean()) * np.sqrt(scale)
    if downside_std_annual < 1e-8:
        return float("inf")

    return float((mu_annual - rf_annual) / downside_std_annual)


def max_drawdown(price_paths: np.ndarray) -> np.ndarray:
    """
    Maximum Drawdown por trajetória.

    Parâmetros
    ----------
    price_paths : shape (n_steps, n_sims, n_assets) ou (n_steps, n_sims)
                  Preços normalizados (S_t / S_0)

    Retorna
    -------
    Array shape (n_sims,) com o MDD de cada simulação (número negativo)
    """
    if price_paths.ndim == 3:
        # Agrega ativos usando pesos iguais (simplificação)
        paths = price_paths.mean(axis=-1)   # [n_steps, n_sims]
    else:
        paths = price_paths                  # [n_steps, n_sims]

    # Para cada simulação: rolling max e drawdown a partir dele
    running_max = np.maximum.accumulate(paths, axis=0)  # [n_steps, n_sims]
    drawdowns   = (paths - running_max) / running_max   # [n_steps, n_sims]
    return drawdowns.min(axis=0)                         # [n_sims]


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    result,                         # SimulationResult (importado de monte_carlo_cpu)
    rf_annual: float  = 0.05,
    trading_days: int = 252,
) -> PortfolioMetrics:
    """
    Calcula o conjunto completo de métricas a partir de um SimulationResult.

    Parâmetros
    ----------
    result       : SimulationResult de simulate_vectorized / simulate_batched
    rf_annual    : taxa livre de risco anual (padrão: 5%)
    trading_days : dias úteis por ano
    """
    r        = result.portfolio_returns.astype(np.float64)
    n_steps  = result.n_steps

    return PortfolioMetrics(
        # Retorno
        expected_return = float(r.mean()),
        median_return   = float(np.median(r)),
        std_return      = float(r.std()),

        # Risco
        var_95   = var_historical(r, 0.95),
        var_99   = var_historical(r, 0.99),
        cvar_95  = cvar_historical(r, 0.95),
        cvar_99  = cvar_historical(r, 0.99),
        prob_loss = float((r < 0).mean()),

        # Índices
        sharpe  = sharpe_ratio(r,  n_steps, trading_days, rf_annual),
        sortino = sortino_ratio(r, n_steps, trading_days, rf_annual),

        # Percentis
        p5  = float(np.percentile(r, 5)),
        p25 = float(np.percentile(r, 25)),
        p75 = float(np.percentile(r, 75)),
        p95 = float(np.percentile(r, 95)),

        # Contexto
        n_sims    = result.n_sims,
        n_steps   = n_steps,
        rf_annual = rf_annual,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPARAÇÃO DE PORTFÓLIOS
# ─────────────────────────────────────────────────────────────────────────────

def compare_portfolios(
    metrics_list: list[PortfolioMetrics],
    labels: list[str],
) -> None:
    """
    Imprime tabela comparativa de múltiplos portfólios lado a lado.
    Útil para avaliar diferentes alocações de pesos.
    """
    col_w = 14
    header = f"  {'Métrica':<22}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    print("\n" + "="*(22 + col_w * len(labels) + 2))
    print("  COMPARAÇÃO DE PORTFÓLIOS")
    print("="*(22 + col_w * len(labels) + 2))
    print(header)
    print("  " + "-"*(20 + col_w * len(labels)))

    def row(name, getter, fmt="{:>+.2%}"):
        vals = "".join(fmt.format(getter(m)) if fmt else f"{getter(m):>{col_w}}" for m in metrics_list)
        print(f"  {name:<22}{vals}")

    row("Retorno esperado",  lambda m: m.expected_return)
    row("Volatilidade",      lambda m: m.std_return,      fmt="{:>+.2%}")
    row("VaR 95%",           lambda m: m.var_95)
    row("CVaR 95%",          lambda m: m.cvar_95)
    row("P(perda)",          lambda m: m.prob_loss)
    row("Sharpe",            lambda m: m.sharpe,          fmt="{:>+.3f}")
    row("Sortino",           lambda m: m.sortino,         fmt="{:>+.3f}")
    print("="*(22 + col_w * len(labels) + 2) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.synthetic import generate_synthetic_data
    from simulation.monte_carlo_cpu import (
        simulate_vectorized, make_equal_weights, make_random_weights
    )

    data    = generate_synthetic_data(n_assets=20, n_days=1008, seed=0)
    mu      = data["mu"]
    sigma   = data["sigma"]
    chol    = data["chol_lower"]

    # Portfólio 1: 1/N
    w1  = make_equal_weights(20)
    r1  = simulate_vectorized(mu, sigma, chol, w1, n_sims=5_000, seed=1)
    m1  = compute_metrics(r1)
    m1.print()

    # Portfólio 2: aleatório
    w2  = make_random_weights(20, rng=np.random.default_rng(7))
    r2  = simulate_vectorized(mu, sigma, chol, w2, n_sims=5_000, seed=1)
    m2  = compute_metrics(r2)

    compare_portfolios([m1, m2], ["Igualitário (1/N)", "Aleatório"])