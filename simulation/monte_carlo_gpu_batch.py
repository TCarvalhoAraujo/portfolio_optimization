"""
monte_carlo_gpu_batch.py
========================
Simulação de Monte Carlo em lote para N portfólios em um único kernel launch.

Diferença de simulate_gpu (monte_carlo_gpu.py)
----------------------------------------------
    simulate_gpu       : 1 portfólio × n_sims   →  1 kernel launch
    simulate_gpu_batch : N portfólios × n_sims   →  1 kernel launch

O loop Python que chamava simulate_gpu N vezes tinha dois custos:
    1. Overhead de launch de kernel por portfólio (~microsegundos cada)
    2. Sincronização CPU/GPU a cada chamada

Com o kernel batch, todos os N × n_sims trabalhos rodam em paralelo.
O ganho real é proporcional a N — com N=3000, o speedup é grande.

Caso de uso
-----------
    compute_labels() em ml/dataset.py precisa avaliar N portfólios
    aleatórios para gerar o dataset de treinamento do modelo ML.
    Esta função é o backend GPU para essa operação.
"""

import math
import time
import numpy as np
from pathlib import Path
from typing import Optional

from simulation.monte_carlo_gpu import (
    HAS_CUDA,
    init_cuda,
    _prepare_gbm_params,
)

if HAS_CUDA:
    import pycuda.driver   as cuda
    import pycuda.compiler as compiler
    import pycuda.gpuarray as gpuarray


# Cache por n_assets para evitar recompilação desnecessária.
# Chave: n_assets — pois MAX_ASSETS é definido em compile-time.
_batch_kernel_cache: dict = {}


def _compile_batch_kernel(n_assets: int):
    """
    Compila mc_multi_portfolio_kernel.cu e retorna o handle da função.

    O resultado é cacheado por n_assets: recompila apenas quando o número
    de ativos muda entre chamadas (raro em produção).
    """
    if n_assets in _batch_kernel_cache:
        return _batch_kernel_cache[n_assets].get_function("mc_gbm_batch_kernel")

    kernel_path = Path(__file__).parent / "kernels" / "mc_multi_portfolio_kernel.cu"
    kernel_code = kernel_path.read_text()

    options = [
        f"-DMAX_ASSETS={n_assets}",
        "-O3",
        "-use_fast_math",
        "--ptxas-options=-v",
    ]

    print(f"[cuda-batch] Compilando kernel (MAX_ASSETS={n_assets})...")
    t0 = time.perf_counter()

    mod = compiler.SourceModule(
        kernel_code,
        options=options,
        include_dirs=[str(kernel_path.parent)],
        no_extern_c=True,
    )

    _batch_kernel_cache[n_assets] = mod
    print(f"[cuda-batch] Kernel compilado em {time.perf_counter() - t0:.2f}s")
    return mod.get_function("mc_gbm_batch_kernel")


def simulate_gpu_batch(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    weights_matrix: np.ndarray,    # (n_portfolios, n_assets)
    n_sims: int            = 2_000,
    n_steps: int           = 252,
    trading_days: int      = 252,
    threads_per_block: int = 128,
    seed: Optional[int]    = None,
    device_id: int         = 0,
) -> np.ndarray:
    """
    Simula n_portfolios × n_sims trajetórias GBM em um único kernel launch.

    Equivalente a chamar simulate_vectorized() para cada linha de
    weights_matrix em loop, mas sem o overhead de N launches separados.

    Parâmetros
    ----------
    mu, sigma, chol_lower  : parâmetros do modelo (mesmos do CPU)
    weights_matrix         : (n_portfolios, n_assets) — linha i = pesos do portfólio i
    n_sims                 : simulações por portfólio
    n_steps                : horizonte temporal em dias
    trading_days           : dias úteis por ano
    threads_per_block      : threads CUDA por block — 128 por padrão (menor que
                             simulate_gpu pois cada thread usa mais registros)
    seed                   : semente base para cuRAND
    device_id              : índice da GPU

    Retorna
    -------
    np.ndarray shape (n_portfolios, n_sims) — retornos de portfólio.
    Linha i corresponde à linha i de weights_matrix.

    Limites práticos
    ----------------
    n_assets ≤ ~109 para caber em 48 KB de shared memory por block.
    Para portfólios típicos (10–60 ativos) há folga ampla.
    """
    if not HAS_CUDA:
        raise RuntimeError("PyCUDA não disponível.")

    mu             = mu.astype(np.float32)
    sigma          = sigma.astype(np.float32)
    chol_lower     = chol_lower.astype(np.float32)
    weights_matrix = np.ascontiguousarray(weights_matrix, dtype=np.float32)

    n_portfolios, n_assets = weights_matrix.shape
    seed_base = np.uint64(seed if seed is not None else int(time.time()) % (2**32))

    # Verifica shared memory antes de compilar
    shmem_bytes = (n_assets * n_assets + n_assets) * 4
    if shmem_bytes > 48 * 1024:
        raise ValueError(
            f"n_assets={n_assets} requer {shmem_bytes/1024:.1f} KB de shared memory "
            f"(limite: 48 KB). Use simulate_gpu em loop ou reduza o número de ativos."
        )

    init_cuda(device_id)
    batch_fn = _compile_batch_kernel(n_assets)

    drift, diffusion = _prepare_gbm_params(mu, sigma, n_steps, trading_days)

    # Upload CPU → GPU
    d_drift      = gpuarray.to_gpu(drift)
    d_diffusion  = gpuarray.to_gpu(diffusion)
    d_chol       = gpuarray.to_gpu(chol_lower.flatten())
    d_weights    = gpuarray.to_gpu(weights_matrix)       # (n_portfolios, n_assets)
    d_returns    = gpuarray.empty(n_portfolios * n_sims, dtype=np.float32)

    # Grid 2D: eixo x = simulações, eixo y = portfólios
    blocks_x = math.ceil(n_sims / threads_per_block)
    block    = (threads_per_block, 1, 1)
    grid     = (blocks_x, n_portfolios, 1)

    total_threads = blocks_x * threads_per_block * n_portfolios
    print(
        f"[cuda-batch] Grid ({blocks_x}, {n_portfolios}) × {threads_per_block} threads"
        f" = {total_threads:,} threads  |  {n_portfolios:,} portfólios × {n_sims:,} sims"
    )

    cuda.Context.synchronize()
    t0 = time.perf_counter()

    batch_fn(
        d_drift, d_diffusion, d_chol, d_weights,
        np.int32(n_assets),
        np.int32(n_steps),
        np.int32(n_sims),
        np.int32(n_portfolios),
        np.uint64(seed_base),
        d_returns,
        block=block,
        grid=grid,
    )

    cuda.Context.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[cuda-batch] Kernel concluído em {elapsed:.3f}s")

    return d_returns.get().reshape(n_portfolios, n_sims)


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA — teste com dados sintéticos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.synthetic import generate_synthetic_data
    from simulation.monte_carlo_cpu import simulate_vectorized
    from ml.dataset import sample_weights

    if not HAS_CUDA:
        print("[aviso] PyCUDA não instalado. Execute em máquina com GPU.")
        sys.exit(0)

    print("=" * 60)
    print("  TESTE — simulate_gpu_batch vs loop CPU")
    print("=" * 60)

    data = generate_synthetic_data(n_assets=20, n_days=1260, seed=0)
    mu, sigma, chol = data["mu"], data["sigma"], data["chol_lower"]
    n = data["n_assets"]

    N_PORT  = 200
    N_SIMS  = 1_000
    N_STEPS = 252

    rng = np.random.default_rng(42)
    W   = sample_weights(n, N_PORT, "mixed", rng)

    # CPU loop (referência)
    print(f"\n[cpu] {N_PORT} portfólios × {N_SIMS} sims...")
    t0 = time.perf_counter()
    cpu_returns = np.stack([
        simulate_vectorized(mu, sigma, chol, W[i],
                            n_sims=N_SIMS, n_steps=N_STEPS, seed=i).portfolio_returns
        for i in range(N_PORT)
    ])
    t_cpu = time.perf_counter() - t0
    print(f"[cpu] {t_cpu:.2f}s  ({int(N_PORT * N_SIMS / t_cpu):,} sims/s)")

    # GPU batch
    print(f"\n[gpu-batch] {N_PORT} portfólios × {N_SIMS} sims...")
    gpu_returns = simulate_gpu_batch(mu, sigma, chol, W,
                                     n_sims=N_SIMS, n_steps=N_STEPS, seed=0)
    t_gpu = float(gpu_returns.shape[0])  # placeholder — tempo já impresso pelo kernel

    # Verifica distribuições (CPU e GPU usam RNGs diferentes → compara estatísticas)
    print("\n  Verificação estatística (média do Sharpe por portfólio):")
    sharpe_cpu = cpu_returns.mean(axis=1) / (cpu_returns.std(axis=1) + 1e-8)
    sharpe_gpu = gpu_returns.mean(axis=1) / (gpu_returns.std(axis=1) + 1e-8)
    diff = np.abs(sharpe_cpu.mean() - sharpe_gpu.mean())
    print(f"  Sharpe médio CPU : {sharpe_cpu.mean():.4f}")
    print(f"  Sharpe médio GPU : {sharpe_gpu.mean():.4f}")
    print(f"  Diferença absoluta: {diff:.4f}  {'✓ OK' if diff < 0.05 else '✗ DIVERGÊNCIA'}")
