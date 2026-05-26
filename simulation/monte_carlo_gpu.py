"""
monte_carlo_gpu.py
==================
Simulação de Monte Carlo em GPU via PyCUDA.

Este módulo é o espelho direto de monte_carlo_cpu.py:
mesma interface, mesmo algoritmo, resultados estatisticamente
equivalentes — mas executado em paralelo na GPU.

COMO PYCUDA FUNCIONA
====================

PyCUDA permite escrever kernels CUDA em C dentro de strings Python,
compilá-los em tempo de execução e chamá-los diretamente:

    1. pycuda.compiler.SourceModule(codigo_cu)  → compila o .cu
    2. mod.get_function("nome")                  → obtém handle do kernel
    3. kernel(args, block=(...), grid=(...))      → lança na GPU

Transferência de dados CPU ↔ GPU:
    - numpy array → GPU : pycuda.driver.to_gpu(array)   ou gpuarray.to_gpu()
    - GPU → numpy array : gpu_array.get()

CONFIGURAÇÃO DE GRID E BLOCK
==============================

    threads_per_block = 256   (potência de 2, recomendado 128-512)
    blocks_per_grid   = ceil(n_sims / threads_per_block)

    Total de threads lançadas = blocks × threads ≥ n_sims
    Threads "extras" são descartadas dentro do kernel (if s >= n_sims: return)

    Regra prática para threads_per_block:
      - Múltiplo de 32 (warp size)
      - 256 é um bom padrão para kernels com uso médio de registros
      - Kernels com muitos registros por thread → use 128 ou menos
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional

# Garante que nvcc e cl.exe estejam no PATH antes de importar pycuda
def _setup_cuda_path() -> None:
    cuda_toolkit = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if cuda_toolkit.exists():
        versions = sorted(cuda_toolkit.iterdir(), reverse=True)
        for v in versions:
            bin_dir = v / "bin"
            if (bin_dir / "nvcc.exe").exists():
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                break

    msvc_root = Path("C:/Program Files (x86)/Microsoft Visual Studio")
    if msvc_root.exists():
        for vs_ver in sorted(msvc_root.iterdir(), reverse=True):
            tools = vs_ver / "BuildTools" / "VC" / "Tools" / "MSVC"
            if not tools.exists():
                continue
            for msvc_ver in sorted(tools.iterdir(), reverse=True):
                cl = msvc_ver / "bin" / "HostX64" / "x64" / "cl.exe"
                if cl.exists():
                    os.environ["PATH"] = str(cl.parent) + os.pathsep + os.environ.get("PATH", "")
                    break
            break

    winkits = Path("C:/Program Files (x86)/Windows Kits/10/bin")
    if winkits.exists():
        for kit in sorted(winkits.iterdir(), reverse=True):
            x64 = kit / "x64"
            if x64.exists():
                os.environ["PATH"] = str(x64) + os.pathsep + os.environ.get("PATH", "")
                break

_setup_cuda_path()

# Importação condicional — permite importar o módulo mesmo sem GPU instalada
try:
    import pycuda.driver   as cuda
    import pycuda.compiler as compiler
    import pycuda.gpuarray as gpuarray
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

from simulation.monte_carlo_cpu import SimulationResult


# ─────────────────────────────────────────────────────────────────────────────
# GERENCIAMENTO DO CONTEXTO CUDA
# ─────────────────────────────────────────────────────────────────────────────

_cuda_context = None   # contexto global (inicializado uma vez)


def _cleanup_cuda():
    global _cuda_context
    if _cuda_context is not None:
        try:
            _cuda_context.pop()
        except Exception:
            pass
        _cuda_context = None


if HAS_CUDA:
    import atexit
    atexit.register(_cleanup_cuda)


def init_cuda(device_id: int = 0) -> None:
    """
    Inicializa o contexto CUDA para o dispositivo especificado.

    Deve ser chamado uma vez antes de qualquer operação GPU.
    PyCUDA requer init explícito — diferente de bibliotecas como CuPy.

    Parâmetros
    ----------
    device_id : índice da GPU (0 para a primeira, 1 para a segunda, etc.)
    """
    global _cuda_context
    if _cuda_context is not None:
        return   # já inicializado

    if not HAS_CUDA:
        raise RuntimeError(
            "PyCUDA não está instalado. Execute: pip install pycuda\n"
            "Também é necessário CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
        )

    cuda.init()
    device  = cuda.Device(device_id)
    _cuda_context = device.make_context()

    props = device.get_attributes()
    name  = device.name()
    mem_total = device.total_memory() / (1024**3)

    print(f"[cuda] GPU inicializada: {name}")
    print(f"[cuda] Memória total   : {mem_total:.1f} GB")
    print(f"[cuda] Compute cap.   : {device.compute_capability()}")


def get_gpu_info() -> dict:
    """Retorna informações sobre a GPU ativa."""
    if not HAS_CUDA:
        return {"available": False}

    init_cuda()
    dev = cuda.Context.get_device()
    return {
        "available"          : True,
        "name"               : dev.name(),
        "total_memory_gb"    : dev.total_memory() / (1024**3),
        "compute_capability" : dev.compute_capability(),
        "multiprocessors"    : dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
        "max_threads_per_block": dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK),
        "warp_size"          : dev.get_attribute(cuda.device_attribute.WARP_SIZE),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPILAÇÃO DO KERNEL
# ─────────────────────────────────────────────────────────────────────────────

_compiled_module = None   # cache do módulo compilado

def _compile_kernel(n_assets: int) -> tuple:
    """
    Compila o kernel CUDA e retorna o handle da função mc_gbm_kernel.

    A compilação usa nvcc em background via PyCUDA.SourceModule.
    O resultado é cacheado para evitar recompilação desnecessária.

    Parâmetros
    ----------
    n_assets : define MAX_ASSETS em tempo de compilação (otimiza registros)
    """
    global _compiled_module

    # Lê o código .cu externo
    kernel_path = Path(__file__).parent / "kernels" / "mc_kernel.cu"
    kernel_code = kernel_path.read_text(encoding="utf-8").encode("ascii", "replace").decode("ascii")

    options = [
        f"-DMAX_ASSETS={n_assets}",   # define MAX_ASSETS para o compilador
        "-O3",                         # otimização máxima
        "-use_fast_math",              # exp/sin aproximados (mais rápido, ligeira perda de precisão)
        "--ptxas-options=-v",          # verbose: mostra uso de registros e shared memory
    ]

    print(f"[cuda] Compilando kernel (MAX_ASSETS={n_assets})...")
    t0 = time.perf_counter()

    _compiled_module = compiler.SourceModule(
        kernel_code,
        options=options,
        include_dirs=[str(kernel_path.parent)],
        no_extern_c=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"[cuda] Kernel compilado em {elapsed:.2f}s")

    mc_kernel  = _compiled_module.get_function("mc_gbm_kernel")
    wmu_kernel = _compiled_module.get_function("warmup_kernel")

    return mc_kernel, wmu_kernel


# ─────────────────────────────────────────────────────────────────────────────
# WARM-UP DA GPU
# ─────────────────────────────────────────────────────────────────────────────

def _warmup(warmup_kernel_fn) -> None:
    """
    Executa um kernel trivial para inicializar o contexto CUDA completamente.

    A primeira chamada a qualquer kernel inclui overhead de JIT, inicialização
    de driver etc. O warm-up garante que esse custo não apareça no benchmark.
    """
    dummy = gpuarray.zeros(1, dtype=np.float32)
    warmup_kernel_fn(
        dummy,
        block=(1, 1, 1),
        grid=(1, 1, 1),
    )
    cuda.Context.synchronize()
    print("[cuda] Warm-up concluído.")


# ─────────────────────────────────────────────────────────────────────────────
# PREPARAÇÃO DOS PARÂMETROS GBM
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_gbm_params(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_steps: int,
    trading_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula drift e diffusion diários a partir dos parâmetros anualizados.

        drift[i]     = (mu[i] - 0.5 * sigma[i]²) * dt
        diffusion[i] = sigma[i] * sqrt(dt)

    Retorna arrays float32 prontos para upload à GPU.
    """
    dt        = np.float32(1.0 / trading_days)
    drift     = ((mu - 0.5 * sigma**2) * dt).astype(np.float32)
    diffusion = (sigma * np.sqrt(dt)).astype(np.float32)
    return drift, diffusion


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE GRID/BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def _grid_config(n_sims: int, threads_per_block: int = 256) -> tuple[tuple, tuple]:
    """
    Calcula a configuração de grid e block para lançar n_sims threads.

    Retorna (block, grid) no formato esperado pelo PyCUDA.

    Exemplo:
        n_sims=10000, threads_per_block=256
        → blocks = ceil(10000/256) = 40
        → grid = (40, 1, 1), block = (256, 1, 1)
        → total threads = 40 × 256 = 10.240  (240 threads ociosas no último block)
    """
    import math
    blocks = math.ceil(n_sims / threads_per_block)
    block  = (threads_per_block, 1, 1)
    grid   = (blocks, 1, 1)
    return block, grid


# ─────────────────────────────────────────────────────────────────────────────
# SIMULAÇÃO PRINCIPAL — GPU
# ─────────────────────────────────────────────────────────────────────────────

def simulate_gpu(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    weights: np.ndarray,
    n_sims: int            = 100_000,
    n_steps: int           = 252,
    trading_days: int      = 252,
    threads_per_block: int = 256,
    seed: Optional[int]    = None,
    device_id: int         = 0,
    do_warmup: bool        = True,
) -> SimulationResult:
    """
    Monte Carlo paralelo na GPU via PyCUDA.

    Interface idêntica a simulate_vectorized() — substituto direto
    para comparação de desempenho.

    Parâmetros
    ----------
    mu, sigma, chol_lower, weights : mesmos de monte_carlo_cpu
    n_sims             : número de simulações (recomendado: ≥ 10.000)
    n_steps            : horizonte temporal em dias
    trading_days       : dias úteis por ano
    threads_per_block  : threads por block CUDA (padrão: 256)
    seed               : semente base para cuRAND
    device_id          : índice da GPU a usar
    do_warmup          : executa warm-up antes de medir tempo

    Retorna
    -------
    SimulationResult com backend="gpu"

    Fluxo de execução
    -----------------
    1. init_cuda()           — inicializa contexto (uma vez)
    2. _compile_kernel()     — compila .cu via nvcc (uma vez)
    3. _warmup()             — elimina overhead de inicialização
    4. to_gpu()              — transfere parâmetros CPU → GPU
    5. kernel launch         — lança n_sims threads em paralelo
    6. synchronize()         — aguarda conclusão de todas as threads
    7. .get()                — transfere resultados GPU → CPU
    """
    if not HAS_CUDA:
        raise RuntimeError(
            "PyCUDA não disponível. Verifique a instalação do CUDA Toolkit."
        )

    mu         = mu.astype(np.float32)
    sigma      = sigma.astype(np.float32)
    chol_lower = chol_lower.astype(np.float32)
    weights    = weights.astype(np.float32)
    n_assets   = len(mu)

    seed_base = np.uint64(seed if seed is not None else int(time.time()) % (2**32))

    # ── Inicialização ─────────────────────────────────────────────────────
    init_cuda(device_id)
    mc_kernel_fn, warmup_fn = _compile_kernel(n_assets)

    if do_warmup:
        _warmup(warmup_fn)

    # ── Prepara parâmetros ────────────────────────────────────────────────
    drift, diffusion = _prepare_gbm_params(mu, sigma, n_steps, trading_days)

    # ── Upload para GPU (CPU → GPU) ───────────────────────────────────────
    print(f"[cuda] Transferindo dados para GPU...")
    d_drift      = gpuarray.to_gpu(drift)
    d_diffusion  = gpuarray.to_gpu(diffusion)
    d_chol       = gpuarray.to_gpu(chol_lower.flatten())   # row-major flat
    d_weights    = gpuarray.to_gpu(weights)

    # Arrays de saída (alocados diretamente na GPU, sem dados iniciais)
    d_final_prices = gpuarray.empty((n_sims, n_assets), dtype=np.float32)
    d_returns      = gpuarray.empty(n_sims,             dtype=np.float32)

    # ── Configura grid/block ──────────────────────────────────────────────
    block, grid = _grid_config(n_sims, threads_per_block)
    print(f"[cuda] Grid: {grid[0]} blocks × {block[0]} threads = "
          f"{grid[0]*block[0]:,} threads totais para {n_sims:,} simulações")

    # ── LANÇAMENTO DO KERNEL ──────────────────────────────────────────────
    print(f"[cuda] Lançando kernel...")
    cuda.Context.synchronize()   # garante que uploads terminaram
    t0 = time.perf_counter()

    mc_kernel_fn(
        # Parâmetros do modelo
        d_drift,
        d_diffusion,
        d_chol,
        d_weights,
        # Dimensões (passados como int32 via np.int32)
        np.int32(n_assets),
        np.int32(n_steps),
        np.int32(n_sims),
        # Semente
        np.uint64(seed_base),
        # Saídas
        d_final_prices,
        d_returns,
        # Configuração
        block=block,
        grid=grid,
    )

    cuda.Context.synchronize()   # aguarda TODAS as threads terminarem
    elapsed = time.perf_counter() - t0
    print(f"[cuda] Kernel concluído em {elapsed:.4f}s")

    # ── Download resultados (GPU → CPU) ───────────────────────────────────
    final_prices      = d_final_prices.get()   # [n_sims, n_assets]
    portfolio_returns = d_returns.get()        # [n_sims]

    return SimulationResult(
        price_paths       = final_prices[np.newaxis],
        final_prices      = final_prices,
        portfolio_returns = portfolio_returns,
        weights           = weights,
        elapsed_sec       = elapsed,
        n_sims            = n_sims,
        n_steps           = n_steps,
        n_assets          = n_assets,
        backend           = "gpu",
    )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO — compara GPU vs CPU para verificar correção numérica
# ─────────────────────────────────────────────────────────────────────────────

def validate_gpu_vs_cpu(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    weights: np.ndarray,
    n_sims: int   = 10_000,
    n_steps: int  = 252,
    tol_mean: float = 0.005,   # tolerância: 0.5% no retorno médio
    tol_std:  float = 0.005,   # tolerância: 0.5% no desvio padrão
) -> bool:
    """
    Verifica se a GPU produz resultados estatisticamente equivalentes à CPU.

    Como CPU e GPU usam geradores de números aleatórios diferentes
    (NumPy vs cuRAND), não comparamos trajetória a trajetória —
    comparamos as DISTRIBUIÇÕES de retornos (média e std).

    Retorna True se a validação passou.
    """
    from simulation.monte_carlo_cpu import simulate_vectorized

    print("[validação] Rodando CPU (referência)...")
    res_cpu = simulate_vectorized(mu, sigma, chol_lower, weights,
                                  n_sims=n_sims, n_steps=n_steps, seed=42)

    print("[validação] Rodando GPU...")
    res_gpu = simulate_gpu(mu, sigma, chol_lower, weights,
                           n_sims=n_sims, n_steps=n_steps, seed=42)

    r_cpu = res_cpu.portfolio_returns
    r_gpu = res_gpu.portfolio_returns

    diff_mean = abs(r_cpu.mean() - r_gpu.mean())
    diff_std  = abs(r_cpu.std()  - r_gpu.std())

    print(f"\n  {'Métrica':<20} {'CPU':>12} {'GPU':>12} {'Diferença':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Retorno médio':<20} {r_cpu.mean():>12.4f} {r_gpu.mean():>12.4f} {diff_mean:>12.4f}")
    print(f"  {'Desvio padrão':<20} {r_cpu.std():>12.4f}  {r_gpu.std():>12.4f} {diff_std:>12.4f}")
    print(f"  {'VaR 95%':<20} {np.percentile(r_cpu,5):>12.4f} {np.percentile(r_gpu,5):>12.4f}")
    print(f"  {'P(retorno < 0)':<20} {(r_cpu<0).mean():>12.4f} {(r_gpu<0).mean():>12.4f}")

    ok = diff_mean < tol_mean and diff_std < tol_std
    status = "✓ PASSOU" if ok else "✗ FALHOU"
    print(f"\n  Validação: {status} (tol_mean={tol_mean}, tol_std={tol_std})\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.synthetic import generate_synthetic_data
    from simulation.monte_carlo_cpu import make_equal_weights

    if not HAS_CUDA:
        print("[aviso] PyCUDA não instalado — execute na sua máquina com GPU.")
        print("        Mostrando estrutura do código apenas.\n")
        sys.exit(0)

    print("=" * 60)
    print("  MÓDULO 3 — Monte Carlo GPU  |  Validação e Benchmark")
    print("=" * 60)

    data     = generate_synthetic_data(n_assets=20, n_days=1008, seed=0)
    mu       = data["mu"]
    sigma    = data["sigma"]
    chol     = data["chol_lower"]
    weights  = make_equal_weights(data["n_assets"])

    # Exibe info da GPU
    info = get_gpu_info()
    print(f"\n  GPU: {info['name']}")
    print(f"  SMs: {info['multiprocessors']} | Warp: {info['warp_size']}")
    print(f"  Compute: {info['compute_capability']}\n")

    # Validação estatística
    print("─" * 60)
    print("  VALIDAÇÃO GPU vs CPU")
    print("─" * 60)
    validate_gpu_vs_cpu(mu, sigma, chol, weights, n_sims=10_000, n_steps=252)

    # Benchmark rápido
    print("─" * 60)
    print("  BENCHMARK RÁPIDO")
    print("─" * 60)
    for n_sims in [10_000, 100_000, 1_000_000]:
        res = simulate_gpu(mu, sigma, chol, weights,
                           n_sims=n_sims, n_steps=252, seed=0, do_warmup=False)
        r = res.portfolio_returns
        print(f"  n_sims={n_sims:>9,} | tempo={res.elapsed_sec:.4f}s | "
              f"ret_médio={r.mean():+.2%} | VaR95={np.percentile(r,5):+.2%}")