"""
benchmark.py
============
Comparação de desempenho CPU vs GPU para a simulação Monte Carlo.

Métricas avaliadas
------------------
- Tempo de execução (wall-clock)
- Speedup  = t_cpu / t_gpu
- Throughput = simulações por segundo
- Escalabilidade com n_sims e n_assets
- Uso de memória GPU estimado

Gera uma tabela de resultados e, opcionalmente, salva um CSV.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_synthetic_data
from simulation.monte_carlo_cpu import (
    simulate_vectorized, simulate_batched, make_equal_weights
)

try:
    from simulation.monte_carlo_gpu import simulate_gpu, init_cuda, get_gpu_info, HAS_CUDA
except ImportError:
    HAS_CUDA = False


# ─────────────────────────────────────────────────────────────────────────────
# ESTIMATIVA DE MEMÓRIA
# ─────────────────────────────────────────────────────────────────────────────

def estimate_memory_mb(n_sims: int, n_steps: int, n_assets: int) -> dict:
    """
    Estima uso de memória para CPU (vetorizado) e GPU.

    CPU (vetorizado): armazena Z e price_paths completos
        Z           : n_steps × n_sims × n_assets × 4 bytes
        price_paths : n_steps × n_sims × n_assets × 4 bytes

    GPU: armazena apenas os resultados finais
        out_final   : n_sims × n_assets × 4 bytes
        out_returns : n_sims × 4 bytes
        + parâmetros estáticos: negligíveis
    """
    float_bytes = 4
    cpu_z      = n_steps * n_sims * n_assets * float_bytes / (1024**2)
    cpu_paths  = n_steps * n_sims * n_assets * float_bytes / (1024**2)
    cpu_total  = cpu_z + cpu_paths

    gpu_out    = (n_sims * n_assets + n_sims) * float_bytes / (1024**2)

    return {
        "cpu_mb": cpu_total,
        "gpu_mb": gpu_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DE UM CASO DE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def _run_case(
    label: str,
    fn,
    mu, sigma, chol, weights,
    n_sims: int,
    n_steps: int,
    n_runs: int = 3,
    **kwargs,
) -> dict:
    """
    Executa `fn` por `n_runs` vezes e retorna a mediana do tempo.
    Usar a mediana descarta outliers de cache frio ou interrupções do SO.
    """
    times = []
    result = None
    for _ in range(n_runs):
        result = fn(mu, sigma, chol, weights,
                    n_sims=n_sims, n_steps=n_steps, **kwargs)
        times.append(result.elapsed_sec)

    t_median = float(np.median(times))
    r = result.portfolio_returns

    return {
        "label"      : label,
        "n_sims"     : n_sims,
        "n_steps"    : n_steps,
        "n_assets"   : len(mu),
        "elapsed_s"  : round(t_median, 4),
        "throughput" : int(n_sims / t_median),
        "ret_mean"   : float(r.mean()),
        "ret_std"    : float(r.std()),
        "var95"      : float(np.percentile(r, 5)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    n_assets_list : list[int]  = [10, 20, 50],
    n_sims_list   : list[int]  = [10_000, 100_000, 500_000, 1_000_000],
    n_steps       : int        = 252,
    n_runs        : int        = 3,
    seed          : int        = 42,
    output_csv    : Optional[Path] = None,
) -> pd.DataFrame:
    """
    Executa o benchmark completo CPU vs GPU em diversas configurações.

    Para cada combinação de (n_assets, n_sims):
        - CPU vetorizado  (simulate_vectorized ou simulate_batched)
        - GPU             (simulate_gpu) — apenas se HAS_CUDA

    Parâmetros
    ----------
    n_assets_list : tamanhos de portfólio a testar
    n_sims_list   : volumes de simulação a testar
    n_steps       : horizonte temporal fixo
    n_runs        : repetições para calcular mediana de tempo
    output_csv    : se fornecido, salva resultados em CSV
    """
    rows = []

    for n_assets in n_assets_list:
        print(f"\n{'='*65}")
        print(f"  PORTFÓLIO COM {n_assets} ATIVOS")
        print(f"{'='*65}")

        data     = generate_synthetic_data(n_assets=n_assets, n_days=1260, seed=seed)
        mu       = data["mu"]
        sigma    = data["sigma"]
        chol     = data["chol_lower"]
        n_actual = data["n_assets"]   # pode ser menor que n_assets (limite de tickers sintéticos)
        weights  = make_equal_weights(n_actual)

        for n_sims in n_sims_list:
            mem = estimate_memory_mb(n_sims, n_steps, n_actual)
            print(f"\n  n_sims={n_sims:>9,} | RAM CPU estimada: {mem['cpu_mb']:.0f} MB"
                  f" | VRAM GPU estimada: {mem['gpu_mb']:.0f} MB")

            # ── CPU: usa batched para n_sims grandes (evita OOM) ──────────
            batch = 20_000 if n_sims > 50_000 else n_sims
            cpu_fn   = simulate_batched if n_sims > 50_000 else simulate_vectorized
            cpu_label = f"CPU-batched(b={batch//1000}k)" if n_sims > 50_000 else "CPU-vetorizado"

            row_cpu = _run_case(
                cpu_label, cpu_fn,
                mu, sigma, chol, weights,
                n_sims=n_sims, n_steps=n_steps, n_runs=n_runs,
                seed=seed,
                **( {"batch_size": batch} if n_sims > 50_000 else {} ),
            )
            rows.append(row_cpu)
            print(f"  {cpu_label:<28} {row_cpu['elapsed_s']:>8.4f}s  "
                  f"{row_cpu['throughput']:>12,} sims/s")

            # ── GPU ───────────────────────────────────────────────────────
            if HAS_CUDA:
                row_gpu = _run_case(
                    "GPU-CUDA", simulate_gpu,
                    mu, sigma, chol, weights,
                    n_sims=n_sims, n_steps=n_steps, n_runs=n_runs,
                    seed=seed, do_warmup=False,
                )
                rows.append(row_gpu)
                speedup = row_cpu["elapsed_s"] / row_gpu["elapsed_s"]
                print(f"  {'GPU-CUDA':<28} {row_gpu['elapsed_s']:>8.4f}s  "
                      f"{row_gpu['throughput']:>12,} sims/s  →  speedup: {speedup:.1f}x")
            else:
                # Linha placeholder para manter estrutura do DataFrame
                rows.append({
                    "label": "GPU-CUDA (indisponível)",
                    "n_sims": n_sims, "n_steps": n_steps,
                    "n_assets": n_assets,
                    "elapsed_s": None, "throughput": None,
                    "ret_mean": None, "ret_std": None, "var95": None,
                })

    df = pd.DataFrame(rows)

    # Calcula speedup onde ambos estão disponíveis
    if HAS_CUDA:
        cpu_mask = ~df["label"].str.startswith("GPU")
        gpu_mask =  df["label"].str.startswith("GPU")
        cpu_times = df[cpu_mask]["elapsed_s"].values
        gpu_times = df[gpu_mask]["elapsed_s"].values
        if len(cpu_times) == len(gpu_times):
            df.loc[gpu_mask, "speedup"] = (cpu_times / gpu_times).round(1)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n[benchmark] Resultados salvos em {output_csv}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXIBIÇÃO DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame) -> None:
    """Exibe tabela formatada de resultados do benchmark."""
    print("\n" + "="*75)
    print("  RESULTADOS DO BENCHMARK — CPU vs GPU")
    print("="*75)
    print(f"  {'Backend':<28} {'n_assets':>8} {'n_sims':>10} "
          f"{'Tempo':>9} {'Throughput':>14} {'Speedup':>8}")
    print(f"  {'-'*72}")

    for _, row in df.iterrows():
        t = f"{row['elapsed_s']:.4f}s" if row['elapsed_s'] else "N/A"
        th = f"{row['throughput']:,}" if row['throughput'] else "N/A"
        sp = f"{row.get('speedup', ''):.1f}x" if pd.notna(row.get("speedup")) else ""
        print(f"  {row['label']:<28} {row['n_assets']:>8} {row['n_sims']:>10,} "
              f"{t:>9} {th:>14} {sp:>8}")

    print("="*75)

    if HAS_CUDA and "speedup" in df.columns:
        max_speedup = df["speedup"].max()
        best = df.loc[df["speedup"].idxmax()]
        print(f"\n  Speedup máximo: {max_speedup:.1f}x"
              f" ({int(best['n_sims']):,} sims, {int(best['n_assets'])} ativos)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if HAS_CUDA:
        info = get_gpu_info()
        print(f"GPU detectada: {info['name']} ({info['total_memory_gb']:.1f} GB)\n")
    else:
        print("[aviso] GPU não disponível — benchmark rodará apenas CPU.\n")

    df = run_benchmark(
        n_assets_list = [10, 20, 50],
        n_sims_list   = [10_000, 100_000, 500_000, 1_000_000],
        n_steps       = 252,
        n_runs        = 3,
        output_csv    = Path(__file__).parent / "benchmark_results.csv",
    )

    print_summary_table(df)