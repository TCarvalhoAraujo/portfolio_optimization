"""
dataset.py
==========
Geração do dataset de treinamento para o modelo de ML.

IDEIA CENTRAL
=============
Simulamos N portfólios aleatórios (cada um com pesos w diferentes)
usando Monte Carlo. Para cada portfólio extraímos features baseadas
nos pesos e nas propriedades dos ativos, e calculamos métricas
financeiras como label (Sharpe, VaR, retorno esperado).

O modelo de ML aprende a mapear:

    features(w, ativos) → qualidade do portfólio (Sharpe ratio)

Uma vez treinado, podemos gerar milhares de portfólios candidatos,
predizer seu Sharpe em microssegundos (sem simular), e selecionar
o melhor — em vez de otimizar por gradiente descente ou enumeração exaustiva.

FEATURES POR PORTFÓLIO
=======================
Baseadas nos pesos e nas propriedades dos ativos:

    Pesos:
        w_max, w_min, w_std          — concentração
        herfindahl                   — índice de diversificação (sum w²)
        n_effective                  — nº efetivo de ativos (1/herfindahl)
        entropy                      — -sum(w * log(w))

    Propriedades ponderadas dos ativos:
        mu_w       = w · mu          — retorno médio ponderado
        sigma_w    = w · sigma       — volatilidade média ponderada
        sharpe_w   = mu_w / sigma_w  — Sharpe ingênuo (sem correlação)

    Correlação:
        corr_mean_w                  — correlação média ponderada entre pares
        port_var_approx              — variância aproximada do portfólio: w.T @ Σ @ w

LABELS
======
    sharpe_sim   — Sharpe ratio calculado sobre retornos simulados  (target principal)
    ret_mean     — retorno esperado
    ret_std      — volatilidade
    var95        — VaR 95%
    cvar95       — CVaR 95%

O target do modelo é `sharpe_sim` — maximizá-lo é equivalente a
encontrar o portfólio na fronteira eficiente com melhor risco/retorno.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.monte_carlo_cpu import (
    simulate_batched, simulate_vectorized, SimulationResult
)
from portfolio.metrics import (
    var_historical, cvar_historical, sharpe_ratio, sortino_ratio
)


# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DE PESOS ALEATÓRIOS DIVERSIFICADOS
# ─────────────────────────────────────────────────────────────────────────────

def sample_weights(
    n_assets: int,
    n_portfolios: int,
    concentration: str = "mixed",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Gera n_portfolios vetores de pesos aleatórios normalizados.

    Parâmetros
    ----------
    concentration : estratégia de amostragem
        "uniform"       — Dirichlet(1,...,1): uniforme no simplex
        "concentrated"  — Dirichlet(0.5,...): pesos mais concentrados
        "diversified"   — Dirichlet(2,...):   pesos mais distribuídos
        "mixed"         — mistura das três (recomendado para treino)

    Retorna
    -------
    array float32 shape (n_portfolios, n_assets), cada linha soma 1
    """
    rng = rng or np.random.default_rng(42)
    n   = n_assets

    if concentration == "uniform":
        alpha = np.ones(n)
        W = rng.dirichlet(alpha, size=n_portfolios)

    elif concentration == "concentrated":
        alpha = np.full(n, 0.3)
        W = rng.dirichlet(alpha, size=n_portfolios)

    elif concentration == "diversified":
        alpha = np.full(n, 3.0)
        W = rng.dirichlet(alpha, size=n_portfolios)

    else:  # mixed — 1/3 de cada
        n1 = n_portfolios // 3
        n2 = n_portfolios // 3
        n3 = n_portfolios - n1 - n2

        W = np.vstack([
            rng.dirichlet(np.ones(n),       size=n1),
            rng.dirichlet(np.full(n, 0.3),  size=n2),
            rng.dirichlet(np.full(n, 3.0),  size=n3),
        ])
        rng.shuffle(W)

    return W.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRAÇÃO DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    weights_matrix: np.ndarray,    # (n_portfolios, n_assets)
    mu: np.ndarray,                # (n_assets,)
    sigma: np.ndarray,             # (n_assets,)
    cov_matrix: np.ndarray,        # (n_assets, n_assets)
    rf_annual: float = 0.05,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Extrai features analíticas para cada portfólio — sem simulação.

    Rápido o suficiente para rodar em 100k portfólios em segundos.
    Estas features são a entrada do modelo de ML.

    Retorna DataFrame com shape (n_portfolios, n_features).
    """
    W  = weights_matrix   # (P, N)
    P, N = W.shape

    features = {}

    # ── Features de concentração dos pesos ───────────────────────────────
    features["w_max"]        = W.max(axis=1)
    features["w_min"]        = W.min(axis=1)
    features["w_std"]        = W.std(axis=1)
    features["herfindahl"]   = (W ** 2).sum(axis=1)           # sum(w²)
    features["n_effective"]  = 1.0 / features["herfindahl"]   # 1/HHI
    # Entropia de Shannon: -sum(w * log(w))
    W_safe = np.clip(W, 1e-10, 1.0)
    features["entropy"]      = -(W_safe * np.log(W_safe)).sum(axis=1)

    # ── Retorno e risco ponderados ────────────────────────────────────────
    features["mu_w"]         = W @ mu                          # retorno médio ponderado
    features["sigma_w"]      = W @ sigma                       # vol média ponderada
    features["sharpe_naive"] = (features["mu_w"] - rf_annual) / (features["sigma_w"] + 1e-8)

    # ── Variância real do portfólio: w.T @ Σ @ w ─────────────────────────
    # Para P portfólios de uma vez: diag(W @ Σ @ W.T)
    port_var = np.einsum("pi,ij,pj->p", W, cov_matrix, W)
    features["port_var"]     = port_var
    features["port_std"]     = np.sqrt(np.clip(port_var, 0, None))
    features["sharpe_cov"]   = (features["mu_w"] - rf_annual) / (features["port_std"] + 1e-8)

    # ── Correlação média ponderada ────────────────────────────────────────
    # corr = cov / (sigma_i * sigma_j)
    sigma_outer = np.outer(sigma, sigma)
    corr_matrix = cov_matrix / (sigma_outer + 1e-10)
    np.fill_diagonal(corr_matrix, 0.0)   # ignora diagonal

    # Para cada portfólio: sum_{i≠j} w_i * w_j * corr_ij
    W_corr_W = np.einsum("pi,ij,pj->p", W, corr_matrix, W)
    features["corr_mean_w"]  = W_corr_W / (features["herfindahl"] + 1e-8)

    # ── Pesos por quartil de Sharpe individual ────────────────────────────
    # Quanto do portfólio está em ativos "bons" (alto Sharpe individual)?
    sharpe_ind = (mu - rf_annual) / (sigma + 1e-8)     # Sharpe por ativo
    q75 = np.percentile(sharpe_ind, 75)
    q25 = np.percentile(sharpe_ind, 25)
    features["w_in_top_quartile"]    = W[:, sharpe_ind >= q75].sum(axis=1)
    features["w_in_bottom_quartile"] = W[:, sharpe_ind <= q25].sum(axis=1)

    return pd.DataFrame(features)


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE LABELS (via simulação)
# ─────────────────────────────────────────────────────────────────────────────

def compute_labels(
    weights_matrix: np.ndarray,    # (n_portfolios, n_assets)
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    n_sims_per_portfolio: int = 2_000,
    n_steps: int = 252,
    trading_days: int = 252,
    rf_annual: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calcula métricas financeiras simuladas para cada portfólio.

    Para n_portfolios portfólios, executa n_sims_per_portfolio simulações
    Monte Carlo cada — e extrai Sharpe, VaR, retorno, etc.

    Custo computacional: n_portfolios × n_sims_per_portfolio simulações.
    Para 5.000 portfólios × 2.000 sims = 10M simulações.
    → Use a GPU para este passo quando possível!

    Retorna DataFrame com shape (n_portfolios, n_labels).
    """
    n_portfolios = len(weights_matrix)
    rng          = np.random.default_rng(seed)

    rows = []
    for i, w in enumerate(weights_matrix):
        if verbose and (i % max(1, n_portfolios // 10) == 0):
            print(f"  [labels] Portfólio {i+1:>5}/{n_portfolios} "
                  f"({100*i/n_portfolios:.0f}%)...")

        result = simulate_vectorized(
            mu, sigma, chol_lower, w,
            n_sims=n_sims_per_portfolio,
            n_steps=n_steps,
            trading_days=trading_days,
            seed=int(rng.integers(0, 2**31)),
        )
        r = result.portfolio_returns

        rows.append({
            "sharpe_sim"  : sharpe_ratio(r,  n_steps, trading_days, rf_annual),
            "sortino_sim" : sortino_ratio(r, n_steps, trading_days, rf_annual),
            "ret_mean"    : float(r.mean()),
            "ret_std"     : float(r.std()),
            "var95"       : var_historical(r, 0.95),
            "cvar95"      : cvar_historical(r, 0.95),
            "prob_loss"   : float((r < 0).mean()),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO DE GERAÇÃO DE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    data: dict,
    n_portfolios: int          = 3_000,
    n_sims_per_portfolio: int  = 2_000,
    n_steps: int               = 252,
    rf_annual: float           = 0.05,
    concentration: str         = "mixed",
    seed: int                  = 42,
    save_path: Optional[Path]  = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Pipeline completo: gera pesos → features → labels → dataset.

    Parâmetros
    ----------
    data         : dicionário de fetcher.prepare_data() ou synthetic
    n_portfolios : número de portfólios a amostrar
    n_sims_per_portfolio : Monte Carlo por portfólio (precisão dos labels)
    n_steps      : horizonte temporal (dias)
    rf_annual    : taxa livre de risco anual
    concentration: estratégia de amostragem de pesos
    seed         : semente global
    save_path    : se fornecido, salva o dataset em parquet

    Retorna
    -------
    X            : DataFrame de features  (n_portfolios, n_features)
    y            : DataFrame de labels    (n_portfolios, n_labels)
    weights_mat  : array de pesos         (n_portfolios, n_assets)
    """
    mu         = data["mu"]
    sigma      = data["sigma"]
    chol_lower = data["chol_lower"]
    cov_matrix = data["cov_matrix"]
    n_assets   = data["n_assets"]

    rng = np.random.default_rng(seed)

    print(f"\n[dataset] Gerando {n_portfolios:,} portfólios "
          f"({n_assets} ativos, {n_sims_per_portfolio:,} sims cada)...")

    # 1. Amostra pesos aleatórios
    print(f"[dataset] Amostrando pesos ({concentration})...")
    W = sample_weights(n_assets, n_portfolios, concentration, rng)

    # 2. Extrai features analíticas (sem simulação — rápido)
    print(f"[dataset] Extraindo features...")
    X = extract_features(W, mu, sigma, cov_matrix, rf_annual)

    # 3. Calcula labels via simulação (custoso — use GPU!)
    print(f"[dataset] Calculando labels via Monte Carlo...")
    y = compute_labels(
        W, mu, sigma, chol_lower,
        n_sims_per_portfolio=n_sims_per_portfolio,
        n_steps=n_steps,
        rf_annual=rf_annual,
        seed=int(rng.integers(0, 2**31)),
    )

    print(f"\n[dataset] Dataset gerado:")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Sharpe  : min={y['sharpe_sim'].min():.3f}  "
          f"mean={y['sharpe_sim'].mean():.3f}  "
          f"max={y['sharpe_sim'].max():.3f}")

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        X.to_parquet(save_path / "features.parquet")
        y.to_parquet(save_path / "labels.parquet")
        np.save(save_path / "weights.npy", W)
        print(f"[dataset] Salvo em {save_path}")

    return X, y, W


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.synthetic import generate_synthetic_data

    data = generate_synthetic_data(n_assets=20, n_days=1260, seed=0)

    X, y, W = build_dataset(
        data,
        n_portfolios=500,
        n_sims_per_portfolio=1_000,
        n_steps=252,
        seed=42,
    )

    print("\nAmostra de features:")
    print(X.describe().T[["mean", "std", "min", "max"]].round(4))

    print("\nAmostra de labels:")
    print(y.describe().T[["mean", "std", "min", "max"]].round(4))