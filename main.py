"""
main.py
=======
Ponto de entrada do projeto portfolio_optimization.

Orquestra todos os módulos em um fluxo coeso e configurável
via argumentos de linha de comando.

USO
===
    python main.py --mode full                              # pipeline completo
    python main.py --mode full --tickers AAPL MSFT NVDA
    python main.py --preset my50 --mode optimize            # só otimizadores
    python main.py --preset my50 --mode visualize           # otimiza + plota
    python main.py --preset my50 --mode full
    python main.py --mode benchmark --n-sims 100000 500000 1000000
    python main.py --mode ml --n-portfolios 5000
    python main.py --mode simulate --n-sims 50000
    python main.py --preset my50 --mode full --no-viz       # sem gráficos

MODOS
=====
    full        pipeline completo: dados → simulação → otimização → viz → benchmark → ML
    data        apenas coleta e EDA
    simulate    apenas Monte Carlo CPU (e GPU se disponível)
    optimize    apenas otimizadores: compara todos e gera relatório de pesos
    visualize   otimiza + gera todos os gráficos das camadas implementadas
    benchmark   benchmark CPU vs GPU em múltiplas escalas
    ml          geração de dataset + treino + seleção de portfólio + comparação

FLUXO DO MODO FULL
==================
    Dados ──► Simulação ──► Otimização ──► Visualização ──► Benchmark ──► ML
                                │                │
                          (pesos e             (G1–G4 salvos
                          sim_results)          em results/charts/)
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.fetcher  import prepare_data, print_summary, export_to_csv
from data.tickers  import get_diversified_portfolio, SP500_TOP20, SP500_DIVERSIFIED_30, MY_DIVERSIFIED_50
from data.eda      import full_eda_report

from simulation.monte_carlo_cpu import (
    simulate_vectorized, simulate_batched, make_equal_weights
)
from portfolio.metrics import compute_metrics

# Importa todos os otimizadores
from optimization import (
    MarkowitzOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    HRPOptimizer,
    BlackLittermanOptimizer,
    RobustOptimizer,
    long_only_box,
)

from ml.dataset import build_dataset
from ml.portfolio_selector import (
    train, evaluate, select_best_portfolio,
    compare_with_baselines, compare_weights, save_model,
    _returns_from_data,
)

try:
    from data.synthetic import generate_synthetic_data
    HAS_SYNTHETIC = True
except ImportError:
    HAS_SYNTHETIC = False

try:
    from simulation.monte_carlo_gpu import simulate_gpu, get_gpu_info, HAS_CUDA
except ImportError:
    HAS_CUDA = False

from benchmark.benchmark import run_benchmark, print_summary_table

# Visualização — import condicional para não quebrar se matplotlib ausente
try:
    from visualization.distribution import plot_distribution_layer
    from visualization.risk_metrics  import plot_risk_layer
    from visualization.portfolio     import plot_portfolio_layer
    from visualization.correlation   import plot_correlation_layer
    from visualization.gpu_benchmark import plot_time_speedup, plot_throughput
    from visualization.ml_results    import (
        plot_pred_vs_actual, plot_feature_importance,
        plot_baseline_comparison, plot_weight_distribution,
        plot_selected_stocks,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio_optimization",
        description="Simulação e Otimização de Portfólios via Monte Carlo + ML",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    data_grp = p.add_argument_group("Dados")
    data_grp.add_argument("--tickers", nargs="+", default=None,
        help="Tickers Yahoo Finance (ex: AAPL MSFT NVDA). Omitir = sintético.")
    data_grp.add_argument("--preset",
        choices=["top20", "diversified-30", "my50", "synthetic"], default=None,
        help="top20 | diversified-30 | my50 | synthetic")
    data_grp.add_argument("--n-assets", type=int, default=20,
        help="Nº de ativos sintéticos (padrão: 20)")
    data_grp.add_argument("--start",  default="2020-01-01",
        help="Data inicial YYYY-MM-DD")
    data_grp.add_argument("--end",    default=None,
        help="Data final YYYY-MM-DD (padrão: hoje)")
    data_grp.add_argument("--no-cache", action="store_true",
        help="Não usa cache local de preços")

    p.add_argument("--mode",
        choices=["full", "data", "simulate", "optimize", "benchmark", "ml", "visualize"],
        default="full",
        help="Modo de execução (padrão: full)")

    sim_grp = p.add_argument_group("Simulação")
    sim_grp.add_argument("--n-sims",  type=int, nargs="+", default=[100_000],
        help="Nº de simulações (múltiplos valores para benchmark)")
    sim_grp.add_argument("--n-steps", type=int, default=252,
        help="Horizonte temporal em dias úteis (padrão: 252)")
    sim_grp.add_argument("--rf",      type=float, default=0.05,
        help="Taxa livre de risco anual (padrão: 0.05)")
    sim_grp.add_argument("--seed",    type=int, default=42)

    opt_grp = p.add_argument_group("Otimização")
    opt_grp.add_argument("--w-max", type=float, default=0.15,
        help="Peso máximo por ativo nos otimizadores (padrão: 0.15 = 15%%)")
    opt_grp.add_argument("--skip-bl", action="store_true",
        help="Pula Black-Litterman (mais lento com N grande)")
    opt_grp.add_argument("--bl-views",
        choices=["none", "momentum", "sector_momentum", "combined"],
        default="combined",
        help="Views para Black-Litterman (padrão: combined)")

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
    out_grp.add_argument("--no-viz", action="store_true",
        help="Pula geração de gráficos de visualização")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def step_banner(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — DADOS
# ─────────────────────────────────────────────────────────────────────────────

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
        if not HAS_SYNTHETIC:
            raise ImportError("data.synthetic não encontrado.")
        print(f"[dados] Gerando dados sintéticos ({args.n_assets} ativos)...")
        data = generate_synthetic_data(
            n_assets=args.n_assets, n_days=1260, seed=args.seed
        )
    else:
        print(f"[dados] Coletando {len(tickers)} tickers via yahooquery...")
        data = prepare_data(
            tickers=tickers, start=args.start,
            end=args.end, use_cache=not args.no_cache,
        )

    print_summary(data)
    if not use_synthetic:
        export_to_csv(data, args.output_dir)
    if not args.no_eda:
        full_eda_report(data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — SIMULAÇÃO MONTE CARLO (portfólio 1/N)
# ─────────────────────────────────────────────────────────────────────────────

def step_simulate(args, data: dict) -> dict:
    step_banner("ETAPA 2 — SIMULAÇÃO MONTE CARLO (1/N)")

    n_sims   = args.n_sims[0]
    n_steps  = args.n_steps
    n_assets = data["n_assets"]
    mu, sigma, chol = data["mu"], data["sigma"], data["chol_lower"]
    weights  = make_equal_weights(n_assets)
    results  = {}

    print(f"\n[cpu] {n_sims:,} simulações × {n_steps} dias × {n_assets} ativos...")
    fn = simulate_batched if n_sims > 50_000 else simulate_vectorized
    kw = {"batch_size": 20_000} if n_sims > 50_000 else {}
    res_cpu = fn(mu, sigma, chol, weights,
                 n_sims=n_sims, n_steps=n_steps, seed=args.seed, **kw)
    print(f"[cpu] {res_cpu.elapsed_sec:.4f}s  "
          f"({int(n_sims/res_cpu.elapsed_sec):,} sims/s)")
    compute_metrics(res_cpu, rf_annual=args.rf).print()
    results["cpu"] = res_cpu

    if HAS_CUDA:
        info = get_gpu_info()
        print(f"[gpu] {info['name']} detectada")
        res_gpu = simulate_gpu(mu, sigma, chol, weights,
                               n_sims=n_sims, n_steps=n_steps, seed=args.seed)
        speedup = res_cpu.elapsed_sec / res_gpu.elapsed_sec
        print(f"[gpu] {res_gpu.elapsed_sec:.4f}s  "
              f"({int(n_sims/res_gpu.elapsed_sec):,} sims/s)  "
              f"speedup: {speedup:.1f}x")
        results["gpu"] = res_gpu
    else:
        print("[gpu] PyCUDA não disponível — pulando.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — OTIMIZAÇÃO DE PORTFÓLIO
# ─────────────────────────────────────────────────────────────────────────────

def step_optimize(args, data: dict) -> dict:
    """
    Roda todos os otimizadores e compara resultados via Monte Carlo.

    Cada otimizador produz um vetor de pesos via a interface comum
    .optimize(returns, constraints). Os pesos são então usados para
    simular Monte Carlo e calcular métricas reais comparáveis.

    Retorna dict com:
        "weights"     : {nome: np.ndarray} — pesos por otimizador
        "metrics"     : {nome: PortfolioMetrics}
        "weights_df"  : DataFrame de pesos por ativo
        "comparison"  : DataFrame de métricas comparativas
    """
    step_banner("ETAPA 3 — OTIMIZAÇÃO DE PORTFÓLIO")

    mu      = data["mu"]
    sigma   = data["sigma"]
    chol    = data["chol_lower"]
    n       = data["n_assets"]
    tickers = data.get("tickers", [f"A{i}" for i in range(n)])
    returns = _returns_from_data(data)

    constraints = long_only_box(w_max=args.w_max)

    # Define otimizadores (Black-Litterman pode ser pulado por CLI)
    optimizers = {
        "1/N"        : None,  # baseline manual
        "MinVariance": MinVarianceOptimizer(),
        "RiskParity" : RiskParityOptimizer(),
        "HRP"        : HRPOptimizer(linkage_method="ward"),
        "Markowitz"  : MarkowitzOptimizer(rf=args.rf),
        "Robust"     : RobustOptimizer(rf=args.rf, kappa=1.0),
    }

    if not args.skip_bl:
        optimizers["BlackLitterman"] = BlackLittermanOptimizer(rf=args.rf)

    # sector_map para BL com views setoriais
    sector_map = None
    try:
        from data.tickers import SECTORS
        sector_map = {t: s for s, ts in SECTORS.items() for t in ts}
    except ImportError:
        pass

    # Escolha do método de views para BL
    bl_method = getattr(args, "bl_views", "combined")
    use_views = bl_method != "none"

    all_weights  = {}
    all_metrics  = {}
    all_sim_res  = {}   # ← guarda SimulationResult por otimizador para visualização
    weight_cols  = {}
    comparison_rows = []

    # 1/N manual
    w_equal = np.ones(n, dtype=np.float32) / n
    all_weights["1/N"] = w_equal
    weight_cols["1/N"] = pd.Series(w_equal, index=tickers)

    print(f"\n[opt] Rodando {len(optimizers)-1} otimizadores "
          f"({n} ativos, w_max={args.w_max:.0%})...\n")

    for name, opt in optimizers.items():
        if opt is None:
            continue
        try:
            # Black-Litterman: usa optimize_with_views se views ativadas
            if isinstance(opt, BlackLittermanOptimizer) and use_views:
                w_series = opt.optimize_with_views(
                    returns, method=bl_method,
                    sector_map=sector_map,
                    constraints=constraints,
                    verbose=True,
                )
                name = opt.name   # atualiza nome para refletir o método de view
            else:
                w_series = opt.optimize(returns, constraints)
            w_arr    = w_series.reindex(tickers).fillna(0).values.astype(np.float32)
            w_arr   /= w_arr.sum()
            all_weights[name] = w_arr
            weight_cols[name] = pd.Series(w_arr, index=tickers)
            n_active = int((w_arr > 0.001).sum())
            print(f"  ✓ {name:<20} OK  ({n_active:>2} ativos ativos)")
        except Exception as e:
            print(f"  ✗ {name:<20} FALHOU: {e}")

    # Simula Monte Carlo para cada portfólio otimizado
    n_sims_opt = args.n_sims[0]
    _cpu_fn    = simulate_batched if n_sims_opt > 50_000 else simulate_vectorized
    _cpu_kw    = {"batch_size": 20_000} if n_sims_opt > 50_000 else {}
    _n_total   = n_sims_opt * len(all_weights)

    print(f"\n[opt] Avaliando {len(all_weights)} portfólios via Monte Carlo "
          f"({n_sims_opt:,} sims × {args.n_steps} dias)...\n")

    def _run_sim_pass(sim_fn, sim_kw):
        metrics, sim_res, rows = {}, {}, []
        for name, w in all_weights.items():
            try:
                res = sim_fn(mu, sigma, chol, w,
                             n_sims=n_sims_opt, n_steps=args.n_steps,
                             seed=args.seed, **sim_kw)
                m = compute_metrics(res, rf_annual=args.rf)
                metrics[name] = m
                sim_res[name]  = res
                rows.append({
                    "Portfólio"   : name,
                    "Sharpe"      : round(m.sharpe, 3),
                    "Sortino"     : round(m.sortino, 3),
                    "Retorno"     : f"{m.expected_return:.2%}",
                    "Volatilidade": f"{m.std_return:.2%}",
                    "VaR 95%"     : f"{m.var_95:.2%}",
                    "CVaR 95%"    : f"{m.cvar_95:.2%}",
                    "P(perda)"    : f"{m.prob_loss:.2%}",
                    "N ativos"    : int((w > 0.001).sum()),
                })
            except Exception as e:
                print(f"  ✗ {name}: simulação falhou — {e}")
        return metrics, sim_res, rows

    # ── CPU ──────────────────────────────────────────────────────────────────
    t0_sim = time.perf_counter()
    all_metrics, all_sim_res, comparison_rows = _run_sim_pass(_cpu_fn, _cpu_kw)
    t_cpu = time.perf_counter() - t0_sim
    print(f"[opt] CPU : {t_cpu:.2f}s  ({int(_n_total / t_cpu):,} sims/s)")

    # ── GPU (sobrescreve resultados da CPU quando disponível) ─────────────────
    t_gpu = None
    if HAS_CUDA:
        t0_sim = time.perf_counter()
        all_metrics, all_sim_res, comparison_rows = _run_sim_pass(simulate_gpu, {})
        t_gpu = time.perf_counter() - t0_sim
        print(f"[opt] GPU : {t_gpu:.2f}s  ({int(_n_total / t_gpu):,} sims/s)  "
              f"speedup: {t_cpu / t_gpu:.1f}x")

    comparison_df = pd.DataFrame(comparison_rows).set_index("Portfólio")
    comparison_df = comparison_df.sort_values("Sharpe", ascending=False)

    weights_df = pd.DataFrame(weight_cols).round(4)
    weights_df.index.name = "Ticker"

    # Exibe resultados
    print("\n" + "="*85)
    print("  COMPARAÇÃO DOS OTIMIZADORES")
    print("="*85)
    print(comparison_df.to_string())
    print("="*85)

    # Melhor portfólio
    best_name = comparison_df["Sharpe"].idxmax()
    best_sharpe = comparison_df.loc[best_name, "Sharpe"]
    print(f"\n  → Melhor: {best_name}  (Sharpe={best_sharpe:.3f})\n")

    # Salva resultados
    comparison_df.to_csv(args.output_dir / "optimizer_comparison.csv")
    weights_df.to_csv(args.output_dir / "optimizer_weights.csv")
    print(f"[opt] Resultados salvos em {args.output_dir}/")

    return {
        "weights"     : all_weights,
        "metrics"     : all_metrics,
        "sim_results" : all_sim_res,   # ← SimulationResult por otimizador
        "weights_df"  : weights_df,
        "comparison"  : comparison_df,
        "best_name"   : best_name,
        "timing"      : {
            "cpu_time": t_cpu,
            "gpu_time": t_gpu,
            "n_sims"  : n_sims_opt * len(all_weights),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 4 — BENCHMARK CPU vs GPU
# ─────────────────────────────────────────────────────────────────────────────

def step_benchmark(args, data: dict) -> pd.DataFrame:
    step_banner("ETAPA 4 — BENCHMARK CPU vs GPU")

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


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 5 — MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

def step_ml(args, data: dict) -> dict:
    step_banner("ETAPA 5 — MACHINE LEARNING")

    X, y, W, labels_timing = build_dataset(
        data,
        n_portfolios=args.n_portfolios,
        n_sims_per_portfolio=args.n_sims_per_portfolio,
        n_steps=args.n_steps,
        rf_annual=args.rf,
        seed=args.seed,
        save_path=args.output_dir / "dataset",
        w_max=args.w_max,
    )

    train_result = train(X, y, target="sharpe_sim", verbose=True)
    feat_imp = evaluate(train_result)
    feat_imp.to_csv(args.output_dir / "feature_importance.csv", index=False)

    if args.save_model:
        save_model(train_result, args.output_dir / "model")

    ml_result = select_best_portfolio(
        train_result, data,
        n_candidates=args.n_candidates,
        n_sims_verify=args.n_sims[0],
        n_steps=args.n_steps,
        rf_annual=args.rf,
        seed=args.seed + 1,
        w_max=args.w_max,
    )

    # Comparação ML vs todos os otimizadores (com views BL)
    bl_method = getattr(args, "bl_views", "combined")
    cmp_df = compare_with_baselines(
        ml_result, data,
        n_sims=args.n_sims[0],
        n_steps=args.n_steps,
        rf_annual=args.rf,
        w_max=args.w_max,
        bl_method=bl_method,
    )
    cmp_df.to_csv(args.output_dir / "portfolio_comparison.csv")

    # Salva pesos com barra visual
    weights_df = pd.DataFrame({
        "ticker": data["tickers"],
        "weight": ml_result["weights"],
    }).sort_values("weight", ascending=False)
    weights_df.to_csv(args.output_dir / "best_portfolio_weights.csv", index=False)

    print("\n  PESOS DO PORTFÓLIO ML SELECIONADO:")
    print(f"  {'Ticker':<12} {'Peso':>7}")
    print(f"  {'─'*22}")
    for _, row in weights_df.iterrows():
        bar = "█" * int(row["weight"] * 40)
        print(f"  {row['ticker']:<12} {row['weight']:>6.2%}  {bar}")

    if HAS_VIZ and not getattr(args, "no_viz", False):
        step_banner("ETAPA 5b — VISUALIZAÇÃO ML")
        try:
            plot_pred_vs_actual(train_result, args.output_dir)
            print("  ✓ G11 — Sharpe predito vs real")
        except Exception as e:
            print(f"  ✗ G11 falhou: {e}")
        try:
            plot_feature_importance(feat_imp, args.output_dir)
            print("  ✓ G12 — Importância de features")
        except Exception as e:
            print(f"  ✗ G12 falhou: {e}")
        try:
            plot_baseline_comparison(cmp_df, args.output_dir)
            print("  ✓ G13 — ML vs otimizadores (Sharpe)")
        except Exception as e:
            print(f"  ✗ G13 falhou: {e}")
        try:
            plot_weight_distribution(ml_result, data, args.output_dir)
            print("  ✓ G14 — Distribuição de pesos: ML vs 1/N")
        except Exception as e:
            print(f"  ✗ G14 falhou: {e}")
        try:
            plot_selected_stocks(ml_result, data, args.output_dir)
            print("  ✓ G15 — Ações selecionadas pelo modelo")
        except Exception as e:
            print(f"  ✗ G15 falhou: {e}")

    return {
        "train_result"  : train_result,
        "ml_result"     : ml_result,
        "comparison"    : cmp_df,
        "labels_timing" : labels_timing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 6 — VISUALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def step_visualize(args, data: dict, opt_result: dict) -> dict:
    """
    Gera todos os gráficos das 4 camadas de visualização.

    Camadas geradas
    ---------------
    Camada 1 — distribution.py
        G1: KDE sobreposto dos retornos por otimizador
        G2: Boxplot comparativo (p5/p25/p50/p75/p95)

    Camada 2 — risk_metrics.py
        G3: Scatter risco (vol) × retorno com fronteira eficiente
        G4: Barchart de Sharpe, Sortino, VaR 95% e CVaR 95%

    Camada 3 — portfolio.py
        G5: Heatmap de pesos (tickers × otimizadores)
        G6: Fan chart — cone de incerteza por otimizador ao longo do tempo

    Camada 4 — correlation.py
        G7: Heatmap de correlação (ordenação setorial + hierárquica)
        G8: Análise de drawdown — distribuição, scatter retorno×DD, probabilidades

    Todos os gráficos são salvos em output_dir/charts/ como PNG (dpi=150).
    """
    step_banner("ETAPA 6 — VISUALIZAÇÃO")

    if not HAS_VIZ:
        print("[viz] matplotlib não disponível — pulando visualização.")
        return {}

    sim_results  = opt_result.get("sim_results", {})
    weights_dict = opt_result.get("weights", {})

    if not sim_results:
        print("[viz] sim_results vazio — certifique-se que step_optimize rodou.")
        return {}

    returns_df = data.get("log_returns", None)
    tickers    = data.get("tickers", [])

    sector_map = None
    try:
        from data.tickers import SECTORS
        sector_map = {t: s for s, ts in SECTORS.items() for t in ts}
    except ImportError:
        pass

    all_figs = {}

    # ── Camada 1 ─────────────────────────────────────────────────────────────
    print("\n[viz] Camada 1 — distribuição de retornos (G1, G2)...")
    try:
        figs1 = plot_distribution_layer(
            sim_results, rf=args.rf,
            n_steps=args.n_steps,
            output_dir=args.output_dir,
        )
        all_figs.update(figs1)
        print("  ✓ G1 — KDE sobreposto")
        print("  ✓ G2 — Boxplot comparativo")
    except Exception as e:
        print(f"  ✗ Camada 1 falhou: {e}")

    # ── Camada 2 ─────────────────────────────────────────────────────────────
    print("\n[viz] Camada 2 — risco e métricas (G3, G4)...")
    try:
        figs2 = plot_risk_layer(
            sim_results,
            returns_df=returns_df,
            rf=args.rf,
            n_steps=args.n_steps,
            output_dir=args.output_dir,
        )
        all_figs.update(figs2)
        print("  ✓ G3 — Scatter risco × retorno")
        print("  ✓ G4 — Barchart de métricas")
    except Exception as e:
        print(f"  ✗ Camada 2 falhou: {e}")

    # ── Camada 3 ─────────────────────────────────────────────────────────────
    print("\n[viz] Camada 3 — composição e trajetórias (G5, G6)...")
    try:
        n_fan_paths = min(2_000, args.n_sims[0])
        figs3 = plot_portfolio_layer(
            weights_dict,
            data=data,
            tickers=tickers,
            sector_map=sector_map,
            n_paths=n_fan_paths,
            n_steps=args.n_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        all_figs.update(figs3)
        print("  ✓ G5 — Heatmap de pesos")
        print(f"  ✓ G6 — Fan chart ({n_fan_paths:,} trajetórias × {args.n_steps} dias)")
    except Exception as e:
        print(f"  ✗ Camada 3 falhou: {e}")

    # ── Camada 4 ─────────────────────────────────────────────────────────────
    print("\n[viz] Camada 4 — estrutura e risco avançado (G7, G8)...")
    try:
        n_dd_paths = min(1_000, args.n_sims[0])
        figs4 = plot_correlation_layer(
            weights_dict,
            data=data,
            tickers=tickers,
            sector_map=sector_map,
            n_paths=n_dd_paths,
            n_steps=args.n_steps,
            seed=args.seed + 10,
            output_dir=args.output_dir,
        )
        all_figs.update(figs4)
        print("  ✓ G7 — Heatmap de correlação (setor + hierárquico)")
        print(f"  ✓ G8 — Drawdown analysis ({n_dd_paths:,} trajetórias)")
    except Exception as e:
        print(f"  ✗ Camada 4 falhou: {e}")

    charts_dir = args.output_dir / "charts"
    n_saved = len(list(charts_dir.glob("*.png"))) if charts_dir.exists() else 0
    print(f"\n[viz] {n_saved} gráfico(s) salvos em {charts_dir}/")

    return all_figs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    print("\n" + "═"*60)
    print("  PORTFOLIO OPTIMIZATION — Monte Carlo + Otimizadores + ML")
    print("═"*60)
    print(f"  Modo  : {args.mode}")
    print(f"  GPU   : {'disponível ✓' if HAS_CUDA else 'não disponível'}")
    print(f"  Saída : {args.output_dir.resolve()}")

    if args.mode == "data":
        step_data(args)

    elif args.mode == "simulate":
        data = step_data(args)
        step_simulate(args, data)

    elif args.mode == "optimize":
        data = step_data(args)
        step_optimize(args, data)

    elif args.mode == "visualize":
        data       = step_data(args)
        opt_result = step_optimize(args, data)
        step_visualize(args, data, opt_result)

    elif args.mode == "benchmark":
        data = step_data(args)
        step_benchmark(args, data)

    elif args.mode == "ml":
        data = step_data(args)
        step_ml(args, data)

    elif args.mode == "full":
        data        = step_data(args)
        sim_results = step_simulate(args, data)
        opt_result  = step_optimize(args, data)
        if not getattr(args, "no_viz", False):
            step_visualize(args, data, opt_result)
        step_benchmark(args, data)
        ml_out = None
        if not args.skip_ml:
            ml_out = step_ml(args, data)

        if HAS_VIZ and not getattr(args, "no_viz", False):
            timing_data = {
                "Simulação 1/N": {
                    "cpu_time": sim_results["cpu"].elapsed_sec,
                    "gpu_time": sim_results["gpu"].elapsed_sec if "gpu" in sim_results else None,
                    "n_sims"  : args.n_sims[0],
                },
                "Otimizadores": opt_result["timing"],
            }
            if ml_out is not None:
                timing_data["Labels ML"] = ml_out["labels_timing"]
            try:
                plot_time_speedup(timing_data, args.output_dir)
                print("[viz] G9  — CPU vs GPU: tempo por etapa")
            except Exception as e:
                print(f"[viz] G9  falhou: {e}")
            try:
                plot_throughput(timing_data, args.output_dir)
                print("[viz] G10 — CPU vs GPU: throughput por etapa")
            except Exception as e:
                print(f"[viz] G10 falhou: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\n{'═'*60}")
    print(f"  Concluído em {elapsed:.1f}s")
    print(f"  Resultados em: {args.output_dir.resolve()}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()