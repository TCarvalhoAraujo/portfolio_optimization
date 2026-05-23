"""
main.py
=======
Ponto de entrada do projeto portfolio_cuda.

Orquestra todos os módulos em um fluxo coeso e configurável
via argumentos de linha de comando.

USO
===
    python main.py --mode full                     # pipeline completo (sintético)
    python main.py --mode full --tickers AAPL MSFT NVDA GOOGL AMZN
    python main.py --mode benchmark --n-sims 100000 500000 1000000
    python main.py --mode ml --n-portfolios 5000
    python main.py --mode simulate --n-sims 50000
    python main.py --preset diversified-30 --mode full

MODOS
=====
    full        pipeline completo: dados -> simulação -> benchmark -> ML
    data        apenas coleta e EDA
    simulate    apenas Monte Carlo CPU (e GPU se disponível)
    benchmark   benchmark CPU vs GPU em múltiplas escalas
    ml          geração de dataset + treino + seleção de portfólio
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.fetcher    import prepare_data, print_summary, export_to_csv
from data.synthetic  import generate_synthetic_data
from data.tickers    import get_diversified_portfolio, SP500_TOP20, SP500_DIVERSIFIED_30, MY_DIVERSIFIED_50
from data.eda        import full_eda_report

from simulation.monte_carlo_cpu import (
    simulate_vectorized, simulate_batched, make_equal_weights
)
from portfolio.metrics import compute_metrics

from ml.dataset            import build_dataset
from ml.portfolio_selector import (
    train, evaluate, select_best_portfolio,
    compare_with_baselines, save_model
)

try:
    from simulation.monte_carlo_gpu import simulate_gpu, get_gpu_info, HAS_CUDA
except ImportError:
    HAS_CUDA = False

from benchmark.benchmark import run_benchmark, print_summary_table


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio_cuda",
        description="Simulação e Otimização de Portfólios via Monte Carlo + CUDA",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    data_grp = p.add_argument_group("Dados")
    data_grp.add_argument("--tickers", nargs="+", default=None,
        help="Tickers Yahoo Finance (ex: AAPL MSFT NVDA). Omitir = sintético.")
    data_grp.add_argument("--preset",
        choices=["top20", "diversified-30", "my50", "synthetic"], default=None,
        help="top20 | diversified-30 | my50 | synthetic")
    data_grp.add_argument("--n-assets",  type=int, default=20,
        help="Nº de ativos sintéticos (padrão: 20)")
    data_grp.add_argument("--start",  default="2020-01-01",
        help="Data inicial YYYY-MM-DD")
    data_grp.add_argument("--end",    default=None,
        help="Data final YYYY-MM-DD (padrão: hoje)")
    data_grp.add_argument("--no-cache", action="store_true",
        help="Não usa cache local de preços")

    p.add_argument("--mode",
        choices=["full", "data", "simulate", "benchmark", "ml"],
        default="full", help="Modo de execução (padrão: full)")

    sim_grp = p.add_argument_group("Simulação")
    sim_grp.add_argument("--n-sims",  type=int, nargs="+", default=[100_000],
        help="Nº de simulações (múltiplos valores para benchmark)")
    sim_grp.add_argument("--n-steps", type=int, default=252,
        help="Horizonte temporal em dias úteis (padrão: 252)")
    sim_grp.add_argument("--rf",      type=float, default=0.05,
        help="Taxa livre de risco anual (padrão: 0.05)")
    sim_grp.add_argument("--seed",    type=int, default=42)
    sim_grp.add_argument("--gpu-device", type=int, default=0,
        help="Índice da GPU (padrão: 0)")

    ml_grp = p.add_argument_group("Machine Learning")
    ml_grp.add_argument("--n-portfolios",         type=int, default=3_000,
        help="Portfólios de treino (padrão: 3000)")
    ml_grp.add_argument("--n-sims-per-portfolio", type=int, default=2_000,
        help="Simulações MC por portfólio (padrão: 2000)")
    ml_grp.add_argument("--n-candidates",         type=int, default=50_000,
        help="Candidatos na busca final (padrão: 50000)")
    ml_grp.add_argument("--skip-ml", action="store_true",
        help="Pula ML no modo full")

    out_grp = p.add_argument_group("Saída")
    out_grp.add_argument("--output-dir", type=Path, default=Path("results"),
        help="Diretório de resultados (padrão: ./results)")
    out_grp.add_argument("--save-model", action="store_true",
        help="Salva modelo treinado em --output-dir/model/")
    out_grp.add_argument("--no-eda", action="store_true",
        help="Pula relatório EDA")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# ETAPAS
# ─────────────────────────────────────────────────────────────────────────────

def step_banner(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def step_data(args) -> dict:
    step_banner("ETAPA 1 — DADOS")

    use_synthetic = True
    tickers = None

    if args.tickers:
        tickers, use_synthetic = args.tickers, False
    elif args.preset == "top20":
        tickers, use_synthetic = SP500_TOP20, False
    elif args.preset == "diversified-30":
        tickers, use_synthetic = SP500_DIVERSIFIED_30, False
    elif args.preset == "my50":
        tickers, use_synthetic = MY_DIVERSIFIED_50, False

    if use_synthetic:
        print(f"[dados] Gerando dados sintéticos ({args.n_assets} ativos)...")
        data = generate_synthetic_data(
            n_assets=args.n_assets, n_days=1260, seed=args.seed)
    else:
        print(f"[dados] Coletando {len(tickers)} tickers via yahooquery...")
        data = prepare_data(tickers=tickers, start=args.start,
                            end=args.end, use_cache=not args.no_cache)

    print_summary(data)
    if not use_synthetic:
        export_to_csv(data, args.output_dir)
    if not args.no_eda:
        full_eda_report(data)
    return data


def step_simulate(args, data: dict) -> dict:
    step_banner("ETAPA 2 — SIMULAÇÃO MONTE CARLO")

    n_sims   = args.n_sims[0]
    n_steps  = args.n_steps
    n_assets = data["n_assets"]
    mu, sigma, chol = data["mu"], data["sigma"], data["chol_lower"]
    weights  = make_equal_weights(n_assets)
    results  = {}

    # CPU
    print(f"\n[cpu] {n_sims:,} simulações × {n_steps} dias × {n_assets} ativos...")
    fn = simulate_batched if n_sims > 50_000 else simulate_vectorized
    kw = {"batch_size": 20_000} if n_sims > 50_000 else {}
    res_cpu = fn(mu, sigma, chol, weights,
                 n_sims=n_sims, n_steps=n_steps, seed=args.seed, **kw)
    print(f"[cpu] {res_cpu.elapsed_sec:.4f}s  "
          f"({int(n_sims/res_cpu.elapsed_sec):,} sims/s)")
    compute_metrics(res_cpu, rf_annual=args.rf).print()
    results["cpu"] = res_cpu

    # GPU
    if HAS_CUDA:
        info = get_gpu_info()
        print(f"[gpu] {info['name']} detectada")
        res_gpu = simulate_gpu(mu, sigma, chol, weights,
                               n_sims=n_sims, n_steps=n_steps,
                               seed=args.seed, device_id=args.gpu_device)
        speedup = res_cpu.elapsed_sec / res_gpu.elapsed_sec
        print(f"[gpu] {res_gpu.elapsed_sec:.4f}s  "
              f"({int(n_sims/res_gpu.elapsed_sec):,} sims/s)  "
              f"speedup: {speedup:.1f}x")
        results["gpu"] = res_gpu
    else:
        print("[gpu] PyCUDA não disponível — pulando.")

    return results


def step_benchmark(args, data: dict) -> pd.DataFrame:
    step_banner("ETAPA 3 — BENCHMARK CPU vs GPU")

    n_sims_list = (args.n_sims if len(args.n_sims) > 1
                   else [10_000, 100_000, 500_000])

    df = run_benchmark(
        n_assets_list=[data["n_assets"]],
        n_sims_list=n_sims_list,
        n_steps=args.n_steps,
        n_runs=3,
        seed=args.seed,
        output_csv=args.output_dir / "benchmark_results.csv",
    )
    print_summary_table(df)
    return df


def step_ml(args, data: dict) -> dict:
    step_banner("ETAPA 4 — MACHINE LEARNING")

    X, y, W = build_dataset(
        data,
        n_portfolios=args.n_portfolios,
        n_sims_per_portfolio=args.n_sims_per_portfolio,
        n_steps=args.n_steps,
        rf_annual=args.rf,
        seed=args.seed,
        save_path=args.output_dir / "dataset",
    )

    train_result = train(X, y, target="sharpe_sim", verbose=True)
    feat_imp = evaluate(train_result)
    feat_imp.to_csv(args.output_dir / "feature_importance.csv", index=False)

    if args.save_model:
        save_model(train_result, args.output_dir / "model")

    ml_result = select_best_portfolio(
        train_result, data,
        n_candidates=args.n_candidates,
        n_sims_verify=min(10_000, args.n_sims[0]),
        n_steps=args.n_steps,
        rf_annual=args.rf,
        seed=args.seed + 1,
    )

    cmp_df = compare_with_baselines(
        ml_result, data,
        n_sims=min(10_000, args.n_sims[0]),
        n_steps=args.n_steps,
        rf_annual=args.rf,
    )
    cmp_df.to_csv(args.output_dir / "portfolio_comparison.csv")

    # Salva pesos com barra visual
    weights_df = pd.DataFrame({
        "ticker": data["tickers"],
        "weight": ml_result["weights"],
    }).sort_values("weight", ascending=False)
    weights_df.to_csv(args.output_dir / "best_portfolio_weights.csv", index=False)

    print("\n  PESOS DO PORTFÓLIO SELECIONADO:")
    print(f"  {'Ticker':<12} {'Peso':>7}")
    print(f"  {'─'*22}")
    for _, row in weights_df.iterrows():
        bar = "█" * int(row["weight"] * 40)
        print(f"  {row['ticker']:<12} {row['weight']:>6.2%}  {bar}")

    return {"train_result": train_result, "ml_result": ml_result, "comparison": cmp_df}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    print("\n" + "═"*60)
    print("  PORTFOLIO CUDA — Monte Carlo + CUDA + ML")
    print("═"*60)
    print(f"  Modo  : {args.mode}")
    print(f"  GPU   : {'disponível ✓' if HAS_CUDA else 'não disponível'}")
    print(f"  Saída : {args.output_dir.resolve()}")

    if args.mode == "data":
        step_data(args)
    elif args.mode == "simulate":
        data = step_data(args)
        step_simulate(args, data)
    elif args.mode == "benchmark":
        data = step_data(args)
        step_benchmark(args, data)
    elif args.mode == "ml":
        data = step_data(args)
        step_ml(args, data)
    elif args.mode == "full":
        data = step_data(args)
        step_simulate(args, data)
        step_benchmark(args, data)
        if not args.skip_ml:
            step_ml(args, data)

    print(f"\n{'═'*60}")
    print(f"  Concluído em {time.perf_counter() - t0:.1f}s")
    print(f"  Resultados em: {args.output_dir.resolve()}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()