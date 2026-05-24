"""
visualization/correlation.py
=============================
Camada 4 de visualização — estrutura e risco avançado.

Gráficos implementados
-----------------------
G7 — correlation_heatmap()
    Matriz de correlação dos ativos calculada sobre retornos históricos.
    Ordenação por setor coloca ativos similares adjacentes, tornando
    visíveis os blocos de alta correlação intra-setor e a correlação
    cruzada entre setores.
    Uma segunda versão reordena via clusterização hierárquica (mesmo
    algoritmo do HRP) para revelar a estrutura real dos dados — que
    pode diferir da estrutura setorial nominal.

G8 — drawdown_analysis()
    Três sub-gráficos em um único painel:

    (a) Distribuição do max drawdown por otimizador — boxplot horizontal
        com p5/p25/p50/p75/p95 da distribuição de drawdowns máximos
        observados nas simulações Monte Carlo.

    (b) Retorno vs max drawdown esperado — scatter com cada otimizador
        posicionado por retorno esperado (eixo y) e drawdown mediano
        (eixo x), com o índice de Calmar implícito como referência.

    (c) Probabilidade de drawdown exceder thresholds — para cada
        otimizador, a P(max_dd < -10%), P(max_dd < -20%), P(max_dd < -30%).

Conceito por trás de G7
------------------------
A correlação é o insumo mais importante dos otimizadores de portfólio.
HRP usa a estrutura de correlação para clusterizar. Risk Parity usa a
covariância para equalizar contribuições. Markowitz usa a covariância
para traçar a fronteira. Mostrar a matriz de correlação com ordenação
por setor permite verificar se a estrutura de blocos esperada realmente
aparece nos dados históricos, e comparar com a ordem hierárquica que
o HRP vai encontrar — às vezes elas divergem significativamente.

Conceito por trás de G8
------------------------
VaR e CVaR medem o pior retorno pontual (num único dia ou no final
do horizonte). Drawdown mede algo diferente: a maior queda acumulada
a partir de um pico — quanto você perderia se comprasse no topo e
segurasse até o fundo da trajetória.

    max_drawdown_s = min_{t} (V_t - max_{u≤t} V_u) / max_{u≤t} V_u

Um portfólio com VaR bom mas drawdown alto é psicologicamente difícil
de segurar. O drawdown captura o "pain" que faz investidores venderem
no pior momento. Por isso ele complementa VaR/CVaR ao avaliar estratégias.

Para calcular os drawdowns das simulações, usamos as trajetórias
completas (store_paths=True), re-simuladas com n_paths=1.000 por
otimizador — suficiente para distribuições estáveis de drawdown.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path
from typing import Optional

from .distribution import OPTIMIZER_COLORS, _get_color, _setup_style


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hierarchical_order(corr: np.ndarray) -> np.ndarray:
    """
    Retorna a ordem dos ativos via clusterização hierárquica (igual ao HRP).

    Usa distância d_ij = sqrt(0.5 * (1 - rho_ij)) e linkage 'single'.
    Resultado: ordem que coloca ativos similares adjacentes na matriz.
    """
    np.fill_diagonal(corr, 1.0)
    corr_sym = (corr + corr.T) / 2
    distance  = np.sqrt(0.5 * (1 - corr_sym))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="single")
    return leaves_list(Z)


def _compute_drawdowns(
    price_paths: np.ndarray,   # (n_steps, n_sims, n_assets) ou (n_steps, n_sims)
    weights: np.ndarray,       # (n_assets,) — ignorado se paths já é 2D
) -> np.ndarray:
    """
    Calcula o max drawdown de cada trajetória Monte Carlo.

    Para cada simulação s:
        V_t = portfólio no passo t
        running_max_t = max(V_0, ..., V_t)
        dd_t = (V_t - running_max_t) / running_max_t
        max_dd_s = min(dd_t)   (número negativo)

    Parâmetros
    ----------
    price_paths : (n_steps, n_sims, n_assets) ou (n_steps, n_sims)
    weights     : pesos do portfólio

    Retorna
    -------
    np.ndarray shape (n_sims,) com o max drawdown de cada trajetória.
    """
    if price_paths.ndim == 3:
        port_value = price_paths @ weights    # (n_steps, n_sims)
    else:
        port_value = price_paths              # já é (n_steps, n_sims)

    running_max = np.maximum.accumulate(port_value, axis=0)
    drawdowns   = (port_value - running_max) / (running_max + 1e-10)
    return drawdowns.min(axis=0)             # (n_sims,)


# ─────────────────────────────────────────────────────────────────────────────
# G7 — HEATMAP DE CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def correlation_heatmap(
    returns_df: pd.DataFrame,
    tickers: Optional[list[str]]          = None,
    sector_map: Optional[dict[str, str]]  = None,
    show_hierarchical: bool               = True,
    save_path: Optional[Path]             = None,
    figsize: Optional[tuple]              = None,
) -> plt.Figure:
    """
    G7 — Heatmap de correlação dos ativos com dois ordenamentos.

    Painel esquerdo:  correlação ordenada por setor.
    Painel direito:   correlação reordenada por clusterização hierárquica
                      (mesma lógica do HRP — só disponível se show_hierarchical=True).

    O que mostrar:
        - Blocos quentes na diagonal = ativos intra-setor correlacionados
        - Células frias fora da diagonal = diversificação real entre setores
        - Comparação entre ordenação setorial e hierárquica revela se
          o setor é bom proxy para comportamento de mercado

    Parâmetros
    ----------
    returns_df         : pd.DataFrame de retornos diários (T × N).
                         Colunas = tickers.
    tickers            : subconjunto de tickers a incluir. Se None, usa todos.
    sector_map         : {ticker: setor} para ordenação setorial.
    show_hierarchical  : se True, mostra os dois painéis lado a lado.
                         Se False, mostra apenas o painel setorial.
    save_path          : Path para salvar PNG.
    figsize            : tamanho da figura. Auto se None.

    Retorna
    -------
    plt.Figure
    """
    _setup_style()

    if tickers:
        available = [t for t in tickers if t in returns_df.columns]
        df = returns_df[available]
    else:
        df = returns_df

    corr = df.corr().values
    cols = list(df.columns)
    n    = len(cols)

    # ── Ordenação setorial ───────────────────────────────────────────────────
    if sector_map:
        from .portfolio import DEFAULT_SECTOR_ORDER, _sort_tickers_by_sector
        sector_order = _sort_tickers_by_sector(cols, sector_map)
        sector_idx   = [cols.index(t) for t in sector_order]
        corr_sector  = corr[np.ix_(sector_idx, sector_idx)]
        labels_sector = sector_order
    else:
        corr_sector   = corr
        labels_sector = cols
        sector_order  = cols

    # ── Ordenação hierárquica ────────────────────────────────────────────────
    hier_idx      = _hierarchical_order(corr.copy())
    corr_hier     = corr[np.ix_(hier_idx, hier_idx)]
    labels_hier   = [cols[i] for i in hier_idx]

    n_panels = 2 if show_hierarchical else 1
    if figsize is None:
        w = max(10, n * 0.28 * n_panels + 2)
        h = max(7,  n * 0.28 + 2)
        figsize = (w, h)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Colormap divergente: azul (negativo) → branco (zero) → vermelho (positivo)
    from matplotlib.colors import TwoSlopeNorm
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    def _draw_heatmap(ax, corr_mat, labels, title):
        im = ax.imshow(corr_mat, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color="#2C2C2A", pad=10)

        # Grade fina
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#FAFAF8", linewidth=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im

    im1 = _draw_heatmap(
        axes[0], corr_sector, labels_sector,
        "Ordenado por setor",
    )

    if show_hierarchical:
        _draw_heatmap(
            axes[1], corr_hier, labels_hier,
            "Ordenado por clusterização hierárquica",
        )

    # Separadores de setor no painel esquerdo
    if sector_map:
        current_sector = None
        for i, ticker in enumerate(labels_sector):
            s = sector_map.get(ticker, "")
            if s != current_sector and i > 0:
                axes[0].axhline(i - 0.5, color="#FAFAF8", linewidth=1.2, zorder=3)
                axes[0].axvline(i - 0.5, color="#FAFAF8", linewidth=1.2, zorder=3)
            current_sector = s

    # Colorbar compartilhada
    cbar = fig.colorbar(im1, ax=axes, fraction=0.015, pad=0.03,
                        orientation="vertical")
    cbar.set_label("Correlação de Pearson", fontsize=9, color="#5F5E5A")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("G7 — Matriz de correlação dos ativos",
                 fontsize=13, fontweight="bold", y=1.01, color="#2C2C2A")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G7 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G8 — ANÁLISE DE DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_analysis(
    weights_dict: dict,
    data: dict,
    n_paths: int              = 1_000,
    n_steps: int              = 252,
    trading_days: int         = 252,
    seed: int                 = 0,
    dd_thresholds: list[float] = [-0.10, -0.20, -0.30],
    save_path: Optional[Path] = None,
    figsize: tuple            = (15, 5),
) -> plt.Figure:
    """
    G8 — Três perspectivas do drawdown máximo por otimizador.

    Sub-painéis:
        (a) Distribuição do max drawdown — boxplot horizontal p5/p25/p50/p75/p95
        (b) Scatter retorno × max drawdown mediano — revela o trade-off
        (c) P(max_dd < threshold) — probabilidade de exceder cada nível de perda

    Parâmetros
    ----------
    weights_dict   : dict {nome: np.ndarray de pesos}
    data           : dicionário com mu, sigma, chol_lower
    n_paths        : trajetórias Monte Carlo por otimizador
    n_steps        : horizonte temporal
    trading_days   : dias úteis por ano
    seed           : semente de reprodutibilidade
    dd_thresholds  : níveis de drawdown para o painel (c)
    save_path      : Path para salvar PNG
    figsize        : tamanho da figura

    Retorna
    -------
    plt.Figure
    """
    from simulation.monte_carlo_cpu import simulate_vectorized

    _setup_style()

    mu    = data["mu"]
    sigma = data["sigma"]
    chol  = data["chol_lower"]

    opt_names  = list(weights_dict.keys())
    n_opts     = len(opt_names)
    colors     = [_get_color(n, i) for i, n in enumerate(opt_names)]

    # ── Simula e coleta drawdowns ────────────────────────────────────────────
    all_drawdowns   = {}   # {nome: np.ndarray (n_paths,)}
    all_ret_medians = {}   # {nome: float} — retorno mediano anualizado

    scale = trading_days / n_steps

    print("[viz] G8 — simulando trajetórias para drawdown analysis...")
    for idx, (name, weights) in enumerate(weights_dict.items()):
        result = simulate_vectorized(
            mu, sigma, chol, weights,
            n_sims=n_paths,
            n_steps=n_steps,
            trading_days=trading_days,
            seed=seed + idx,
            store_paths=True,
        )
        dd = _compute_drawdowns(result.price_paths, weights)
        all_drawdowns[name]   = dd
        all_ret_medians[name] = float(np.median(result.portfolio_returns) * scale)
        print(f"  ✓ {name:<20} med_dd={float(np.median(dd)):.1%}  "
              f"med_ret={all_ret_medians[name]:.1%}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_box, ax_scatter, ax_prob = axes

    # ── (a) Boxplot do max drawdown ───────────────────────────────────────────
    for pos, (name, color) in enumerate(zip(opt_names, colors)):
        dd = all_drawdowns[name]
        p5, p25, p50, p75, p95 = np.percentile(dd, [5, 25, 50, 75, 95])
        hw = 0.25

        # Caixa IQR
        ax_box.barh(pos, p75 - p25, left=p25, height=0.45,
                    color=color, alpha=0.25,
                    edgecolor=color, linewidth=1.2)

        # Mediana
        ax_box.plot([p50, p50], [pos - hw, pos + hw],
                    color=color, linewidth=2.5, zorder=4)

        # Whiskers p5–p95
        ax_box.plot([p25, p95], [pos, pos], color=color,
                    linewidth=0.8, alpha=0.5, zorder=2)
        ax_box.plot([p5, p25], [pos, pos], color=color,
                    linewidth=0.8, alpha=0.5, linestyle="--", zorder=2)
        for x_cap in [p5, p95]:
            ax_box.plot([x_cap, x_cap], [pos - hw * 0.6, pos + hw * 0.6],
                        color=color, linewidth=1.0, alpha=0.6)

        # Rótulo da mediana
        ax_box.text(p50 - 0.003, pos,
                    f"{p50:.1%}", ha="right", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    ax_box.set_yticks(range(n_opts))
    ax_box.set_yticklabels(opt_names, fontsize=8.5)
    ax_box.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    ax_box.axvline(0, color="#888780", linewidth=0.5, alpha=0.4)
    ax_box.set_title("(a) Distribuição do max drawdown",
                     fontsize=9, fontweight="bold", color="#2C2C2A", pad=8)
    ax_box.set_xlabel("← pior     melhor →  (0%)",
                      fontsize=7.5, color="#888780", labelpad=4)
    ax_box.spines["left"].set_visible(False)
    ax_box.invert_xaxis()   # negativo cresce para a esquerda (mais intuitivo)

    # ── (b) Scatter retorno × drawdown ───────────────────────────────────────
    for idx, (name, color) in enumerate(zip(opt_names, colors)):
        dd_med  = float(np.median(all_drawdowns[name]))
        ret_med = all_ret_medians[name]

        ax_scatter.scatter(dd_med, ret_med,
                           s=100, color=color, zorder=5,
                           edgecolors="white", linewidths=1.0)
        ax_scatter.annotate(
            name, xy=(dd_med, ret_med),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7.5, color=color, fontweight="bold",
        )

    # Linhas de iso-Calmar (retorno / |drawdown| = constante)
    dd_range = np.linspace(
        min(np.median(v) for v in all_drawdowns.values()) * 1.2,
        -0.01, 100
    )
    for calmar in [0.5, 1.0, 2.0]:
        ret_line = calmar * np.abs(dd_range)
        ax_scatter.plot(dd_range, ret_line, color="#D3D1C7",
                        linewidth=0.7, linestyle=":", alpha=0.8)
        ax_scatter.text(dd_range[0] * 0.95, ret_line[0] * 0.95,
                        f"Calmar={calmar}", fontsize=7,
                        color="#888780", ha="left", va="bottom")

    ax_scatter.axhline(0, color="#888780", linewidth=0.5, alpha=0.4)
    ax_scatter.axvline(0, color="#888780", linewidth=0.5, alpha=0.4)
    ax_scatter.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_scatter.set_xlabel("Max drawdown mediano", fontsize=9, color="#5F5E5A", labelpad=6)
    ax_scatter.set_ylabel("Retorno mediano anualizado", fontsize=9, color="#5F5E5A", labelpad=6)
    ax_scatter.set_title("(b) Retorno vs drawdown mediano",
                         fontsize=9, fontweight="bold", color="#2C2C2A", pad=8)

    # ── (c) P(max_dd < threshold) ────────────────────────────────────────────
    x_pos   = np.arange(n_opts)
    n_thr   = len(dd_thresholds)
    bar_w   = 0.22
    offsets = np.linspace(-(n_thr - 1) / 2, (n_thr - 1) / 2, n_thr) * bar_w

    # Tons para cada threshold: mais escuro = perda maior
    thr_alphas = [0.4, 0.65, 0.9]
    thr_labels = [f"dd < {t:.0%}" for t in dd_thresholds]

    for ti, (thr, offset, alpha, lbl) in enumerate(
            zip(dd_thresholds, offsets, thr_alphas, thr_labels)):
        probs = [float((all_drawdowns[n] < thr).mean()) for n in opt_names]
        for xi, (prob, color) in enumerate(zip(probs, colors)):
            ax_prob.bar(x_pos[xi] + offset, prob,
                        width=bar_w * 0.9,
                        color=color, alpha=alpha, zorder=3)

    ax_prob.set_xticks(x_pos)
    ax_prob.set_xticklabels(opt_names, rotation=20, ha="right", fontsize=8)
    ax_prob.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_prob.set_ylabel("Probabilidade", fontsize=9, color="#5F5E5A", labelpad=6)
    ax_prob.set_title("(c) P(max drawdown < threshold)",
                      fontsize=9, fontweight="bold", color="#2C2C2A", pad=8)

    # Legenda dos thresholds
    legend_patches = [
        mpatches.Patch(color="#888780", alpha=a, label=lbl)
        for a, lbl in zip(thr_alphas, thr_labels)
    ]
    ax_prob.legend(handles=legend_patches, fontsize=7.5,
                   frameon=False, loc="upper right")

    fig.suptitle("G8 — Análise de drawdown por otimizador",
                 fontsize=13, fontweight="bold", y=1.02, color="#2C2C2A")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G8 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIÊNCIA — gera G7 e G8 de uma vez
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_layer(
    weights_dict: dict,
    data: dict,
    tickers: Optional[list[str]]          = None,
    sector_map: Optional[dict[str, str]]  = None,
    n_paths: int       = 1_000,
    n_steps: int       = 252,
    trading_days: int  = 252,
    seed: int          = 0,
    output_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """
    Gera G7 e G8 de uma vez. Salva em output_dir se fornecido.

    Parâmetros
    ----------
    weights_dict : dict {nome: np.ndarray de pesos}
    data         : dicionário de fetcher.prepare_data() ou synthetic.
                   Precisa de "log_returns" para G7 e "mu/sigma/chol_lower" para G8.
    tickers      : subconjunto de tickers a incluir em G7.
    sector_map   : {ticker: setor} para ordenação em G7.
    n_paths      : trajetórias Monte Carlo para G8.
    n_steps      : horizonte temporal.
    trading_days : dias úteis por ano.
    seed         : semente de reprodutibilidade.
    output_dir   : diretório de saída (cria subpasta charts/).

    Retorna
    -------
    dict {"g7": Figure, "g8": Figure}
    """
    charts_dir = Path(output_dir) / "charts" if output_dir else None
    returns_df = data.get("log_returns")
    figs = {}

    if returns_df is not None:
        figs["g7"] = correlation_heatmap(
            returns_df,
            tickers=tickers,
            sector_map=sector_map,
            show_hierarchical=True,
            save_path=charts_dir / "g7_correlation_heatmap.png" if charts_dir else None,
        )
    else:
        print("[viz] G7 ignorado — log_returns não disponível em data.")

    figs["g8"] = drawdown_analysis(
        weights_dict,
        data=data,
        n_paths=n_paths,
        n_steps=n_steps,
        trading_days=trading_days,
        seed=seed,
        save_path=charts_dir / "g8_drawdown_analysis.png" if charts_dir else None,
    )

    return figs


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA — teste com dados sintéticos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.synthetic import generate_synthetic_data
    from simulation.monte_carlo_cpu import make_equal_weights
    from optimization import (
        MarkowitzOptimizer, MinVarianceOptimizer,
        RiskParityOptimizer, HRPOptimizer, long_only_box,
    )
    from ml.portfolio_selector import _returns_from_data

    print("[test] Gerando dados sintéticos...")
    data    = generate_synthetic_data(n_assets=20, n_days=1260, seed=0)
    returns = _returns_from_data(data)
    tickers = data["tickers"]
    n       = data["n_assets"]
    constraints = long_only_box(w_max=0.15)

    optimizers = {
        "1/N (igualitário)": None,
        "MinVariance"       : MinVarianceOptimizer(),
        "RiskParity"        : RiskParityOptimizer(),
        "HRP"               : HRPOptimizer(),
        "Markowitz"         : MarkowitzOptimizer(),
    }

    weights_dict = {}
    for name, opt in optimizers.items():
        if opt is None:
            w = make_equal_weights(n)
        else:
            w = opt.optimize(returns, constraints).values.astype("float32")
            w /= w.sum()
        weights_dict[name] = w
        print(f"  ✓ {name}")

    print("\n[test] Gerando G7 e G8...")
    figs = plot_correlation_layer(
        weights_dict, data,
        tickers=tickers,
        n_paths=500, n_steps=252,
        output_dir=Path("results"),
    )
    plt.show()
    print("[test] Concluído.")