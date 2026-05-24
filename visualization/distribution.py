"""
visualization/distribution.py
==============================
Camada 1 de visualização — distribuição de retornos simulados.

Gráficos implementados
-----------------------
G1 — kde_overlay()
    KDE (Kernel Density Estimate) dos retornos finais de cada otimizador,
    sobrepostos em um único eixo. Mostra a forma completa da distribuição:
    onde os retornos se concentram, quão largas são as caudas, se há
    assimetria. Linhas verticais marcam VaR 95% de cada estratégia.

G2 — boxplot_comparison()
    Boxplot com p5, p25, mediana, p75, p95 por otimizador. Leitura rápida
    de spread, posição central e assimetria sem precisar interpretar curvas.
    Pontos individuais de simulações extremas são sobrepostos (stripplot).

Conceito por trás dos gráficos
--------------------------------
Ambos operam sobre `portfolio_returns` — o array de retornos totais
simulados de cada portfólio ao longo de n_steps dias. Cada valor representa
quanto esse portfólio retornou em uma trajetória Monte Carlo completa.

    retorno_total[s] = Σ_i  w_i * (S_T[s,i] / S_0[i] - 1)

A KDE é estimada com scipy.stats.gaussian_kde, que ajusta automaticamente
a largura de banda via regra de Silverman. Para n_sims > 50.000 usa-se
uma amostra aleatória de 10.000 pontos (a KDE não precisa de todos).

Como usar
----------
    from visualization.distribution import kde_overlay, boxplot_comparison

    # sim_results: dict {nome: SimulationResult} — retorno de step_simulate
    # rf: taxa livre de risco (para marcar no eixo)
    fig1 = kde_overlay(sim_results, rf=0.05, save_path="results/charts/g1_kde.png")
    fig2 = boxplot_comparison(sim_results, save_path="results/charts/g2_boxplot.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Optional

# ── Paleta consistente para todos os gráficos do projeto ──────────────────────
# Cada otimizador tem uma cor fixa. Se o nome não estiver no mapa, usa cinza.
OPTIMIZER_COLORS = {
    "1/N (igualitário)": "#888780",   # cinza neutro — baseline
    "MinVariance"       : "#1D9E75",   # teal
    "RiskParity"        : "#378ADD",   # azul
    "HRP"               : "#7F77DD",   # roxo
    "Markowitz"         : "#E85D24",   # coral
    "Robust"            : "#BA7517",   # âmbar
    "BlackLitterman"    : "#D4537E",   # rosa
    "ML-Selecionado"    : "#2C2C2A",   # preto/dark — destaque especial
}

FALLBACK_COLORS = [
    "#5DCAA5", "#85B7EB", "#AFA9EC", "#F0997B",
    "#EF9F27", "#ED93B1", "#97C459", "#F09595",
]


def _get_color(name: str, idx: int) -> str:
    """Retorna cor do otimizador ou cor de fallback por índice."""
    return OPTIMIZER_COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _extract_returns(sim_results: dict) -> dict[str, np.ndarray]:
    """
    Extrai arrays de retornos de um dict {nome: SimulationResult}.

    Aceita dois formatos de entrada:
        1. {nome: SimulationResult}   — saída direta de step_simulate / step_optimize
        2. {nome: np.ndarray}         — arrays de retornos já extraídos

    Retorna dict {nome: np.ndarray de retornos}.
    """
    result = {}
    for name, obj in sim_results.items():
        if hasattr(obj, "portfolio_returns"):
            result[name] = obj.portfolio_returns.astype(np.float64)
        elif isinstance(obj, np.ndarray):
            result[name] = obj.astype(np.float64)
        else:
            print(f"[viz] aviso: '{name}' não tem portfolio_returns — ignorado.")
    return result


def _setup_style():
    """Aplica estilo consistente em todos os gráficos."""
    plt.rcParams.update({
        "figure.facecolor"   : "#FAFAF8",
        "axes.facecolor"     : "#FAFAF8",
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.spines.left"   : False,
        "axes.spines.bottom" : True,
        "axes.edgecolor"     : "#D3D1C7",
        "axes.linewidth"     : 0.8,
        "axes.grid"          : True,
        "grid.color"         : "#D3D1C7",
        "grid.linewidth"     : 0.5,
        "grid.alpha"         : 0.6,
        "xtick.color"        : "#5F5E5A",
        "ytick.color"        : "#5F5E5A",
        "xtick.labelsize"    : 9,
        "ytick.labelsize"    : 9,
        "font.family"        : "DejaVu Sans",
        "text.color"         : "#2C2C2A",
    })


# ─────────────────────────────────────────────────────────────────────────────
# G1 — KDE SOBREPOSTO
# ─────────────────────────────────────────────────────────────────────────────

def kde_overlay(
    sim_results: dict,
    rf: float            = 0.05,
    n_steps: int         = 252,
    trading_days: int    = 252,
    max_kde_samples: int = 10_000,
    save_path: Optional[Path] = None,
    figsize: tuple       = (12, 6),
) -> plt.Figure:
    """
    G1 — KDE dos retornos finais de cada otimizador, sobrepostos.

    O que mostra:
        - A forma completa da distribuição de retornos de cada estratégia
        - Qual estratégia tem cauda direita mais longa (upside)
        - Qual tem cauda esquerda mais pesada (downside)
        - Onde o VaR 95% de cada estratégia cai
        - A taxa livre de risco como linha de referência (retorno "gratuito")

    Parâmetros
    ----------
    sim_results : dict {nome: SimulationResult ou np.ndarray}
        Resultados das simulações Monte Carlo por otimizador.
    rf : float
        Taxa livre de risco anualizada (para escalonamento e linha de referência).
    n_steps : int
        Horizonte temporal em dias úteis da simulação.
    trading_days : int
        Dias úteis por ano (para anualizar os retornos no eixo x).
    max_kde_samples : int
        Máximo de amostras usadas para estimar a KDE. Para n_sims > 50k,
        usa subconjunto aleatório sem perda perceptível de precisão.
    save_path : Path ou None
        Se fornecido, salva o gráfico como PNG com dpi=150.
    figsize : tuple

    Retorna
    -------
    plt.Figure
    """
    _setup_style()
    returns_dict = _extract_returns(sim_results)

    if not returns_dict:
        raise ValueError("sim_results está vazio ou sem portfolio_returns válidos.")

    # Escala de anualização: retorno_horizonte → retorno_anual
    scale = trading_days / n_steps

    fig, ax = plt.subplots(figsize=figsize)

    # Determina o range do eixo x com base em todos os retornos juntos
    all_returns = np.concatenate([r * scale for r in returns_dict.values()])
    x_min = np.percentile(all_returns, 0.5)
    x_max = np.percentile(all_returns, 99.5)
    x_grid = np.linspace(x_min, x_max, 500)

    legend_elements = []

    for idx, (name, returns) in enumerate(returns_dict.items()):
        color = _get_color(name, idx)
        r_annual = returns * scale

        # KDE — usa subconjunto se necessário
        if len(r_annual) > max_kde_samples:
            rng = np.random.default_rng(42)
            sample = rng.choice(r_annual, size=max_kde_samples, replace=False)
        else:
            sample = r_annual

        kde = gaussian_kde(sample, bw_method="silverman")
        density = kde(x_grid)

        # Linha principal da KDE
        ax.plot(x_grid, density, color=color, linewidth=2.0, alpha=0.9, zorder=3)

        # Área preenchida com alta transparência
        ax.fill_between(x_grid, density, alpha=0.08, color=color, zorder=2)

        # Área da cauda esquerda (abaixo do VaR 95%) — mais opaca
        var95 = float(np.percentile(r_annual, 5))
        mask_tail = x_grid <= var95
        if mask_tail.any():
            ax.fill_between(
                x_grid[mask_tail], density[mask_tail],
                alpha=0.25, color=color, zorder=2,
            )
            # Linha vertical do VaR 95%
            ax.axvline(var95, color=color, linewidth=0.8,
                       linestyle="--", alpha=0.6, zorder=4)

        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2,
                   label=f"{name}  (VaR95={var95:+.1%})")
        )

    # Linha da taxa livre de risco
    rf_annual = rf   # já está anualizada
    ax.axvline(rf_annual, color="#2C2C2A", linewidth=1.2,
               linestyle=":", alpha=0.5, zorder=5)
    ax.text(rf_annual + 0.002, ax.get_ylim()[1] * 0.92,
            f"rf = {rf_annual:.1%}", fontsize=8,
            color="#5F5E5A", va="top")

    # Linha do zero
    ax.axvline(0, color="#888780", linewidth=0.6, alpha=0.4, zorder=1)

    # Formatação dos eixos
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Retorno anualizado (horizonte simulado)", fontsize=10,
                  color="#5F5E5A", labelpad=8)
    ax.set_ylabel("Densidade", fontsize=10, color="#5F5E5A", labelpad=8)
    ax.set_title("G1 — Distribuição de retornos por otimizador",
                 fontsize=13, fontweight="bold", pad=14, color="#2C2C2A")

    # Anotação explicativa
    ax.text(0.01, 0.97,
            "Área sombreada = cauda esquerda (abaixo do VaR 95%)\n"
            "Linha tracejada = VaR 95% de cada estratégia",
            transform=ax.transAxes, fontsize=8, color="#888780",
            va="top", linespacing=1.6)

    # Legenda fora do gráfico para não cobrir as curvas
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8.5,
        labelspacing=0.7,
    )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G1 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G2 — BOXPLOT COMPARATIVO
# ─────────────────────────────────────────────────────────────────────────────

def boxplot_comparison(
    sim_results: dict,
    n_steps: int         = 252,
    trading_days: int    = 252,
    show_points: bool    = True,
    n_points_sample: int = 300,
    save_path: Optional[Path] = None,
    figsize: tuple       = (12, 6),
) -> plt.Figure:
    """
    G2 — Boxplot dos retornos finais por otimizador.

    O que mostra:
        - Mediana de cada estratégia (linha central da caixa)
        - IQR (p25–p75): onde 50% dos cenários caem
        - Whiskers em p5 e p95 (em vez do padrão 1.5×IQR)
        - Pontos individuais de simulações extremas sobrepostos
        - Coloração por otimizador para consistência visual com G1

    Por que p5/p95 nos whiskers:
        O padrão matplotlib usa 1.5×IQR, o que cria muitos "outliers"
        artificiais em distribuições de retornos (que têm caudas pesadas).
        Usando p5/p95 como whiskers o gráfico fica mais informativo:
        você vê o intervalo de 90% dos cenários como a caixa completa.

    Parâmetros
    ----------
    sim_results : dict {nome: SimulationResult ou np.ndarray}
    n_steps : int
    trading_days : int
    show_points : bool
        Se True, sobrepõe pontos de simulações individuais (stripplot).
        Útil para ver a espessura real das caudas.
    n_points_sample : int
        Número de pontos a mostrar no stripplot (subconjunto aleatório).
    save_path : Path ou None
    figsize : tuple

    Retorna
    -------
    plt.Figure
    """
    _setup_style()
    returns_dict = _extract_returns(sim_results)

    if not returns_dict:
        raise ValueError("sim_results está vazio ou sem portfolio_returns válidos.")

    scale = trading_days / n_steps

    names  = list(returns_dict.keys())
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    # Prepara dados anualizados
    data_annual = [returns_dict[n] * scale for n in names]

    fig, ax = plt.subplots(figsize=figsize)

    # Posições dos boxes no eixo x
    positions = np.arange(len(names))

    # Desenha box customizado (p5, p25, p50, p75, p95)
    for pos, (data, color) in enumerate(zip(data_annual, colors)):
        p5, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])

        # Caixa p25–p75
        box_h = p75 - p25
        box = plt.Rectangle(
            (pos - 0.3, p25), 0.6, box_h,
            facecolor=color, alpha=0.25,
            edgecolor=color, linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(box)

        # Mediana
        ax.plot([pos - 0.3, pos + 0.3], [p50, p50],
                color=color, linewidth=2.5, zorder=4)

        # Whiskers (p5 e p95)
        ax.plot([pos, pos], [p25, p95], color=color,
                linewidth=1.0, alpha=0.6, zorder=2)
        ax.plot([pos, pos], [p5, p25], color=color,
                linewidth=1.0, alpha=0.6, linestyle="--", zorder=2)

        # Caps nos whiskers
        cap_w = 0.12
        for y_cap in [p5, p95]:
            ax.plot([pos - cap_w, pos + cap_w], [y_cap, y_cap],
                    color=color, linewidth=1.2, alpha=0.7, zorder=3)

        # Stripplot: pontos de simulações individuais
        if show_points:
            rng = np.random.default_rng(pos)
            sample_idx = rng.choice(len(data),
                                    size=min(n_points_sample, len(data)),
                                    replace=False)
            jitter = rng.uniform(-0.12, 0.12, size=len(sample_idx))
            ax.scatter(
                pos + jitter, data[sample_idx],
                color=color, alpha=0.12, s=4, zorder=2, linewidths=0,
            )

        # Anotação da mediana
        ax.text(pos, p50, f"{p50:+.1%}",
                ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#FAFAF8",
                          ec="none", alpha=0.8),
                zorder=5)

    # Linha do zero
    ax.axhline(0, color="#888780", linewidth=0.8, linestyle="-",
               alpha=0.4, zorder=1)

    # Eixos
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Retorno anualizado", fontsize=10, color="#5F5E5A", labelpad=8)
    ax.set_title("G2 — Boxplot comparativo de retornos por otimizador",
                 fontsize=13, fontweight="bold", pad=14, color="#2C2C2A")

    # Legenda manual das faixas
    legend_elements = [
        mpatches.Patch(facecolor="#888780", alpha=0.5, label="p25 – p75 (IQR)"),
        Line2D([0], [0], color="#888780", linewidth=2, label="Mediana (p50)"),
        Line2D([0], [0], color="#888780", linewidth=1,
               linestyle="--", label="p5 – p95 (90% dos cenários)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              frameon=False, fontsize=8.5)

    ax.set_xlim(-0.6, len(names) - 0.4)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G2 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIÊNCIA — gera ambos de uma vez
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_layer(
    sim_results: dict,
    rf: float         = 0.05,
    n_steps: int      = 252,
    trading_days: int = 252,
    output_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """
    Gera G1 e G2 de uma vez. Salva em output_dir se fornecido.

    Parâmetros
    ----------
    sim_results : dict {nome: SimulationResult}
    rf          : taxa livre de risco anualizada
    n_steps     : horizonte temporal
    trading_days: dias úteis por ano
    output_dir  : diretório de saída (cria subpasta charts/)

    Retorna
    -------
    dict {"g1": Figure, "g2": Figure}
    """
    charts_dir = Path(output_dir) / "charts" if output_dir else None

    fig1 = kde_overlay(
        sim_results, rf=rf, n_steps=n_steps, trading_days=trading_days,
        save_path=charts_dir / "g1_kde_overlay.png" if charts_dir else None,
    )
    fig2 = boxplot_comparison(
        sim_results, n_steps=n_steps, trading_days=trading_days,
        save_path=charts_dir / "g2_boxplot.png" if charts_dir else None,
    )

    return {"g1": fig1, "g2": fig2}


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA — teste com dados sintéticos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.synthetic import generate_synthetic_data
    from simulation.monte_carlo_cpu import simulate_vectorized, make_equal_weights
    from optimization import (
        MarkowitzOptimizer, MinVarianceOptimizer,
        RiskParityOptimizer, HRPOptimizer, long_only_box,
    )
    from ml.portfolio_selector import _returns_from_data

    print("[test] Gerando dados sintéticos...")
    data = generate_synthetic_data(n_assets=20, n_days=1260, seed=0)
    returns = _returns_from_data(data)

    mu, sigma, chol = data["mu"], data["sigma"], data["chol_lower"]
    n = data["n_assets"]
    constraints = long_only_box(w_max=0.15)

    optimizers = {
        "1/N (igualitário)": None,
        "MinVariance"       : MinVarianceOptimizer(),
        "RiskParity"        : RiskParityOptimizer(),
        "HRP"               : HRPOptimizer(),
        "Markowitz"         : MarkowitzOptimizer(),
    }

    sim_results = {}
    for name, opt in optimizers.items():
        if opt is None:
            w = make_equal_weights(n)
        else:
            w = opt.optimize(returns, constraints).values.astype("float32")
            w /= w.sum()
        res = simulate_vectorized(mu, sigma, chol, w, n_sims=5_000, seed=42)
        sim_results[name] = res
        print(f"  ✓ {name}")

    print("\n[test] Gerando G1 e G2...")
    figs = plot_distribution_layer(sim_results, rf=0.05,
                                   output_dir=Path("results"))
    plt.show()
    print("[test] Concluído.")