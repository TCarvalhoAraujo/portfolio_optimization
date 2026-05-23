"""
monte_carlo_cpu.py
==================
Simulação de Monte Carlo em CPU usando NumPy vetorizado.

Este módulo é o BASELINE do projeto: implementa o mesmo algoritmo
que será paralelizado no Módulo 3 (CUDA), permitindo comparação
direta de tempo de execução e correção dos resultados.

Modelo: Movimento Browniano Geométrico (GBM) multivariado
----------------------------------------------------------
Para cada simulação s e passo de tempo t:

    r_t = (mu - 0.5 * sigma²) * dt  +  sigma * sqrt(dt) * (L @ Z_t)

    S_t = S_{t-1} * exp(r_t)

onde:
    mu, sigma  — parâmetros anualizados por ativo  [n_assets]
    dt         — 1 / trading_days  (fração de ano por dia)
    L          — fator de Cholesky de Σ  [n_assets × n_assets]
    Z_t        — vetor iid N(0,1)         [n_assets]

Estratégia de vetorização
-------------------------
Em vez de simular um portfólio de cada vez (loop sobre n_sims),
geramos TODOS os choques aleatórios de uma só vez com shape
(n_steps, n_sims, n_assets) e aplicamos as operações em batch.

Isso explora ao máximo o BLAS/LAPACK por trás do NumPy e serve
como referência justa para o speedup da GPU.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ESTRUTURA DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Contém todas as saídas de uma rodada de Monte Carlo.

    Atributos
    ---------
    price_paths   : trajetórias de preços normalizadas (S_t / S_0)
                    shape (n_steps, n_sims, n_assets)  — float32
    final_prices  : preços no último passo (S_T / S_0)
                    shape (n_sims, n_assets)             — float32
    portfolio_returns : retorno total de cada simulação (pesos fixos)
                    shape (n_sims,)                      — float32
    weights       : pesos usados na simulação
                    shape (n_assets,)                    — float32
    elapsed_sec   : tempo de execução em segundos
    n_sims        : número de simulações
    n_steps       : número de passos de tempo
    n_assets      : número de ativos
    backend       : "cpu" ou "gpu"
    """
    price_paths       : np.ndarray
    final_prices      : np.ndarray
    portfolio_returns : np.ndarray
    weights           : np.ndarray
    elapsed_sec       : float
    n_sims            : int
    n_steps           : int
    n_assets          : int
    backend           : str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Checa dimensões e valores antes de simular."""
    n = len(mu)
    assert len(sigma)        == n,      f"sigma deve ter {n} elementos"
    assert chol_lower.shape  == (n, n), f"chol_lower deve ser ({n},{n})"
    assert len(weights)      == n,      f"weights deve ter {n} elementos"
    assert np.all(sigma > 0),           "sigma deve ser positivo"
    assert abs(weights.sum() - 1.0) < 1e-4, "pesos devem somar 1"


def make_equal_weights(n_assets: int) -> np.ndarray:
    """Portfólio igualmente ponderado (1/N)."""
    return np.ones(n_assets, dtype=np.float32) / n_assets


def make_random_weights(n_assets: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Pesos aleatórios normalizados — útil para varrer a fronteira eficiente."""
    rng = rng or np.random.default_rng()
    w = rng.exponential(1.0, size=n_assets).astype(np.float32)
    return w / w.sum()


# ─────────────────────────────────────────────────────────────────────────────
# SIMULAÇÃO PRINCIPAL — MODO VETORIZADO (recomendado)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_vectorized(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    weights: np.ndarray,
    n_sims: int       = 10_000,
    n_steps: int      = 252,
    trading_days: int = 252,
    seed: Optional[int] = None,
    store_paths: bool = False,
) -> SimulationResult:
    """
    Monte Carlo vetorizado — gera TODAS as trajetórias de uma vez.

    Complexidade de memória: O(n_steps × n_sims × n_assets) float32
        Exemplo: 252 × 10.000 × 30 = ~287 MB  (gerenciável)
        Para n_sims > 100.000, prefira simulate_batched().

    Parâmetros
    ----------
    mu, sigma    : arrays float32 [n_assets] — anualizados
    chol_lower   : array  float32 [n_assets, n_assets]
    weights      : array  float32 [n_assets] — pesos do portfólio (soma=1)
    n_sims       : número de trajetórias a simular
    n_steps      : número de dias simulados (horizonte temporal)
    trading_days : dias úteis por ano (para calcular dt)
    seed         : semente para reproducibilidade
    store_paths  : se False, descarta trajetórias intermediárias (economiza RAM)

    Retorna
    -------
    SimulationResult com price_paths, final_prices, portfolio_returns e métricas
    """
    mu         = mu.astype(np.float32)
    sigma      = sigma.astype(np.float32)
    chol_lower = chol_lower.astype(np.float32)
    weights    = weights.astype(np.float32)

    _validate_inputs(mu, sigma, chol_lower, weights)

    n_assets = len(mu)
    dt       = np.float32(1.0 / trading_days)
    rng      = np.random.default_rng(seed)

    t0 = time.perf_counter()

    # ── Parâmetros do GBM (drift e difusão) ──────────────────────────────────
    # drift[i]  = (mu[i] - 0.5 * sigma[i]²) * dt       shape: (n_assets,)
    # diffusion[i] = sigma[i] * sqrt(dt)                shape: (n_assets,)
    drift     = (mu - 0.5 * sigma ** 2) * dt            # [n_assets]
    diffusion = sigma * np.sqrt(dt)                      # [n_assets]

    # ── Geração dos choques correlacionados ──────────────────────────────────
    # Z shape: (n_steps, n_sims, n_assets) — iid N(0,1)
    Z = rng.standard_normal((n_steps, n_sims, n_assets)).astype(np.float32)

    # eps = Z @ L.T  →  correlaciona os ativos via Cholesky
    # Usamos einsum para aplicar L em cada (passo, sim) com um único op:
    #   eps[t, s, :] = chol_lower @ Z[t, s, :]
    # equivale a: eps = (Z.reshape(-1, n_assets) @ chol_lower.T).reshape(n_steps, n_sims, n_assets)
    eps = Z @ chol_lower.T                               # [n_steps, n_sims, n_assets]

    # ── Retornos log por passo ────────────────────────────────────────────────
    # log_ret[t, s, i] = drift[i] + diffusion[i] * eps[t, s, i]
    log_ret = drift + diffusion * eps                    # broadcasting: [n_steps, n_sims, n_assets]

    # ── Integração: preços acumulados (S_t / S_0) ────────────────────────────
    # cumsum ao longo do eixo de tempo → exp → normalizado em S_0=1
    cum_log = np.cumsum(log_ret, axis=0)                 # [n_steps, n_sims, n_assets]
    price_paths_full = np.exp(cum_log, dtype=np.float32) # [n_steps, n_sims, n_assets]

    # ── Resultado final ───────────────────────────────────────────────────────
    final_prices = price_paths_full[-1]                  # [n_sims, n_assets]  (S_T / S_0)

    # Retorno do portfólio: soma ponderada dos retornos finais de cada ativo
    # portfolio_return[s] = sum_i(w[i] * (S_T[s,i] / S_0[i] - 1))
    portfolio_returns = (final_prices - 1.0) @ weights   # [n_sims]

    elapsed = time.perf_counter() - t0

    return SimulationResult(
        price_paths       = price_paths_full if store_paths else price_paths_full[[-1]],
        final_prices      = final_prices,
        portfolio_returns = portfolio_returns,
        weights           = weights,
        elapsed_sec       = elapsed,
        n_sims            = n_sims,
        n_steps           = n_steps,
        n_assets          = n_assets,
        backend           = "cpu",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIMULAÇÃO EM LOTES — para n_sims muito grande (RAM limitada)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_batched(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    weights: np.ndarray,
    n_sims: int       = 100_000,
    n_steps: int      = 252,
    trading_days: int = 252,
    batch_size: int   = 10_000,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Monte Carlo em lotes — divide n_sims em batches para controlar uso de RAM.

    Útil quando n_sims > 50.000 e a memória disponível é limitada.
    Descarta trajetórias intermediárias (armazena apenas final_prices).

    Parâmetros adicionais
    ---------------------
    batch_size : número de simulações por lote (padrão: 10.000)
    """
    mu         = mu.astype(np.float32)
    sigma      = sigma.astype(np.float32)
    chol_lower = chol_lower.astype(np.float32)
    weights    = weights.astype(np.float32)

    _validate_inputs(mu, sigma, chol_lower, weights)

    n_assets  = len(mu)
    dt        = np.float32(1.0 / trading_days)
    drift     = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    rng = np.random.default_rng(seed)

    all_final   = np.empty((n_sims, n_assets), dtype=np.float32)
    all_returns = np.empty(n_sims,             dtype=np.float32)

    t0 = time.perf_counter()

    processed = 0
    while processed < n_sims:
        bs = min(batch_size, n_sims - processed)

        Z   = rng.standard_normal((n_steps, bs, n_assets)).astype(np.float32)
        eps = Z @ chol_lower.T
        log_ret     = drift + diffusion * eps
        cum_log     = np.cumsum(log_ret, axis=0)
        final_batch = np.exp(cum_log[-1], dtype=np.float32)   # [bs, n_assets]

        all_final[processed : processed + bs]   = final_batch
        all_returns[processed : processed + bs] = (final_batch - 1.0) @ weights

        processed += bs

    elapsed = time.perf_counter() - t0

    return SimulationResult(
        price_paths       = all_final[np.newaxis],   # [1, n_sims, n_assets]
        final_prices      = all_final,
        portfolio_returns = all_returns,
        weights           = weights,
        elapsed_sec       = elapsed,
        n_sims            = n_sims,
        n_steps           = n_steps,
        n_assets          = n_assets,
        backend           = "cpu",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIMULAÇÃO SEQUENCIAL (LOOP) — referência didática, NÃO usar em produção
# ─────────────────────────────────────────────────────────────────────────────

def simulate_sequential(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    weights: np.ndarray,
    n_sims: int       = 1_000,
    n_steps: int      = 252,
    trading_days: int = 252,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Monte Carlo com loop explícito sobre simulações — versão didática.

    Propósito: mostrar a versão "ingênua" do algoritmo, que é exatamente
    o que uma thread CUDA executa. O speedup do vetorizado vs este loop
    é análogo (conceitualmente) ao speedup GPU vs CPU.

    NÃO usar para n_sims > 5.000 (muito lento).
    """
    mu         = mu.astype(np.float32)
    sigma      = sigma.astype(np.float32)
    chol_lower = chol_lower.astype(np.float32)
    weights    = weights.astype(np.float32)

    n_assets  = len(mu)
    dt        = np.float32(1.0 / trading_days)
    drift     = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    rng = np.random.default_rng(seed)

    final_prices      = np.empty((n_sims, n_assets), dtype=np.float32)
    portfolio_returns = np.empty(n_sims,             dtype=np.float32)

    t0 = time.perf_counter()

    for s in range(n_sims):                              # ← loop que a GPU elimina
        log_price = np.zeros(n_assets, dtype=np.float32)

        for t in range(n_steps):                         # ← dependência temporal
            Z   = rng.standard_normal(n_assets).astype(np.float32)
            eps = chol_lower @ Z
            log_price += drift + diffusion * eps

        S_T = np.exp(log_price)
        final_prices[s]      = S_T
        portfolio_returns[s] = ((S_T - 1.0) * weights).sum()

    elapsed = time.perf_counter() - t0

    return SimulationResult(
        price_paths       = final_prices[np.newaxis],
        final_prices      = final_prices,
        portfolio_returns = portfolio_returns,
        weights           = weights,
        elapsed_sec       = elapsed,
        n_sims            = n_sims,
        n_steps           = n_steps,
        n_assets          = n_assets,
        backend           = "cpu-sequential",
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA — teste e mini-benchmark
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.synthetic import generate_synthetic_data

    print("=" * 60)
    print("  MÓDULO 2 — Monte Carlo CPU  |  Teste e Mini-Benchmark")
    print("=" * 60)

    # Dados sintéticos: 20 ativos, 4 anos de histórico
    data = generate_synthetic_data(n_assets=20, n_days=1008, seed=0)

    mu         = data["mu"]
    sigma      = data["sigma"]
    chol_lower = data["chol_lower"]
    n_assets   = data["n_assets"]
    weights    = make_equal_weights(n_assets)

    configs = [
        ("Sequential (loop)",  "seq",  1_000,  252),
        ("Vetorizado   1k",    "vec",  1_000,  252),
        ("Vetorizado  10k",    "vec",  10_000, 252),
        ("Batched     10k",    "bat",  10_000, 252),
    ]

    print(f"\n  Ativos: {n_assets}  |  Pesos: 1/N\n")
    print(f"  {'Modo':<24} {'n_sims':>8}  {'Tempo':>8}  {'Ret. médio':>12}  {'Ret. p5':>10}  {'Ret. p95':>10}")
    print(f"  {'-'*75}")

    results = {}
    for label, mode, n_sims, n_steps in configs:
        if mode == "seq":
            res = simulate_sequential(mu, sigma, chol_lower, weights,
                                      n_sims=n_sims, n_steps=n_steps, seed=42)
        elif mode == "vec":
            res = simulate_vectorized(mu, sigma, chol_lower, weights,
                                      n_sims=n_sims, n_steps=n_steps, seed=42)
        else:
            res = simulate_batched(mu, sigma, chol_lower, weights,
                                   n_sims=n_sims, n_steps=n_steps, seed=42)

        r = res.portfolio_returns
        print(f"  {label:<24} {n_sims:>8,}  {res.elapsed_sec:>7.3f}s  "
              f"{r.mean():>11.2%}  {np.percentile(r,5):>9.2%}  {np.percentile(r,95):>9.2%}")
        results[label] = res

    # Speedup vetorizado vs sequencial
    t_seq = results["Sequential (loop)"].elapsed_sec
    t_vec = results["Vetorizado  10k"].elapsed_sec / 10   # normaliza por n_sims
    t_seq_unit = t_seq / 1_000
    speedup = t_seq_unit / t_vec if t_vec > 0 else float("inf")
    print(f"\n  Speedup vetorizado vs loop (por simulação): ~{speedup:.1f}x")
    print("\n  [ok] Módulo 2 validado.\n")