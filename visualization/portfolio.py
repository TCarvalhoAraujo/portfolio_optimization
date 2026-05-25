"""
visualization/portfolio.py
===========================
Camada 3 de visualização — composição e trajetórias de portfólio.

Gráficos implementados
-----------------------
G5 — weights_heatmap()
    Matriz tickers × otimizadores onde a intensidade da cor codifica o peso.
    Revela onde as estratégias concordam (alta sobreposição) e divergem
    (um otimizador concentra onde outro distribui).
    Ordenação dos tickers por setor para destacar a estrutura de blocos.

G6 — fan_chart()
    "Cone de incerteza" de cada portfólio ao longo do tempo.
    Para cada otimizador, desenha faixas preenchidas entre percentis
    p5/p25/p50/p75/p95 computados passo a passo sobre as trajetórias.
    A mediana (p50) aparece como linha sólida, as faixas em transparência.
    Permite comparar não só o retorno esperado mas a largura do cone —
    um portfólio com cone estreito é mais previsível, um com cone largo
    tem mais upside mas também mais downside.

Conceito por trás de G5
------------------------
O heatmap responde a uma pergunta que tabelas de pesos não conseguem:
"Quais ativos os otimizadores disputam?". Ao ordenar os tickers por setor
e plotar todos os otimizadores lado a lado, você vê imediatamente se
HRP e Risk Parity concordam no setor de Utilities mas divergem em Tech,
ou se Markowitz concentra tudo em 3 ativos enquanto os outros distribuem.

Conceito por trás de G6
------------------------
O fan chart é a evolução natural do gráfico de trajetórias individuais.
Em vez de mostrar 1.000 linhas sobrepostas (ilegível), calcula percentis
em cada passo de tempo e preenche as faixas:

    faixa externa (p5–p95): 90% dos cenários ficam dentro dessa banda
    faixa média   (p25–p75): 50% dos cenários (IQR)
    linha central (p50): trajetória mediana

Para o fan chart funcionar precisamos das trajetórias completas
(price_paths com shape [n_steps, n_sims, n_assets]). Como step_optimize
usa store_paths=False por padrão para economizar RAM, este módulo
re-simula com store_paths=True mas com n_sims reduzido (padrão: 2.000).
Isso é suficiente para trajetórias suaves e custa ~50ms.

Como usar
----------
    from visualization.portfolio import weights_heatmap, fan_chart

    # G5 — precisa de weights (dict {nome: np.ndarray}) e tickers
    fig5 = weights_heatmap(
        weights_dict,
        tickers=data["tickers"],
        sector_map=SECTOR_MAP,          # opcional
        save_path="results/charts/g5_weights_heatmap.png",
    )

    # G6 — precisa de sim_results + dados para re-simular trajetórias
    fig6 = fan_chart(
        sim_results,
        data=data,
        n_paths=2_000,                  # trajetórias a simular
        save_path="results/charts/g6_fan_chart.png",
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional

from .distribution import OPTIMIZER_COLORS, _get_color, _setup_style


# ── Mapa setor → tickers para ordenação do heatmap ────────────────────────────
# Definição padrão alinhada com MY_DIVERSIFIED_50.
# Se não fornecido, tickers são ordenados alfabeticamente.
DEFAULT_SECTOR_ORDER = [
    "Technology", "Financials", "Healthcare",
    "Consumer Discretionary", "Consumer Staples",
    "Energy", "Defense & Aerospace", "Industrials",
    "Communication", "Utilities", "Real Estate",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sort_tickers_by_sector(
    tickers: list[str],
    sector_map: Optional[dict[str, str]],
) -> list[str]:
    """
    Ordena tickers pelo setor, depois alfabeticamente dentro do setor.
    Tickers sem setor mapeado ficam no final, ordenados alfabeticamente.
    """
    if not sector_map:
        return sorted(tickers)

    def sort_key(t):
        sector = sector_map.get(t, "ZZZ_unknown")
        sector_idx = (
            DEFAULT_SECTOR_ORDER.index(sector)
            if sector in DEFAULT_SECTOR_ORDER
            else len(DEFAULT_SECTOR_ORDER)
        )
        return (sector_idx, t)

    return sorted(tickers, key=sort_key)


def _portfolio_percentiles(
    price_paths: np.ndarray,     # (n_steps, n_sims, n_assets)
    weights: np.ndarray,         # (n_assets,)
    percentiles: list[float],
) -> np.ndarray:
    """
    Calcula percentis do valor do portfólio em cada passo de tempo.

    portfolio_value[t, s] = Σ_i w_i * price_paths[t, s, i]

    Retorna array (n_steps, len(percentiles)).
    """
    # portfolio_value: (n_steps, n_sims)
    portfolio_value = price_paths @ weights      # broadcasting correto
    return np.percentile(portfolio_value, percentiles, axis=1).T  # (n_steps, n_pct)


# ─────────────────────────────────────────────────────────────────────────────
# G5 — HEATMAP DE PESOS
# ─────────────────────────────────────────────────────────────────────────────

def weights_heatmap(
    weights_dict: dict,
    tickers: list[str],
    sector_map: Optional[dict[str, str]]  = None,
    save_path: Optional[Path]             = None,
    figsize: Optional[tuple]              = None,
    annotate: bool                        = True,
    annotate_threshold: float             = 0.02,
) -> plt.Figure:
    """
    G5 — Heatmap de pesos: tickers × otimizadores.

    O que mostra:
        - Intensidade de cor = peso do ativo naquele otimizador
        - Ordenação por setor revela a estrutura de blocos (intra-setor)
        - Onde os otimizadores concordam (mesma cor) ou divergem
        - Ativos com peso zero ficam brancos — exclusão explícita
        - Anotação do valor percentual em células acima do threshold

    Parâmetros
    ----------
    weights_dict : dict {nome_otimizador: np.ndarray de pesos}
                   Os arrays devem estar na mesma ordem que `tickers`.
    tickers      : lista de tickers correspondendo aos índices dos arrays.
    sector_map   : dict {ticker: setor} para ordenação por setor.
                   Se None, ordena alfabeticamente.
    save_path    : Path para salvar PNG.
    figsize      : tamanho da figura. Auto se None.
    annotate     : se True, anota o valor percentual em cada célula.
    annotate_threshold : células abaixo desse peso não são anotadas (reduz poluição).

    Retorna
    -------
    plt.Figure
    """
    _setup_style()

    opt_names = list(weights_dict.keys())
    n_opts    = len(opt_names)
    n_tickers = len(tickers)

    # Ordena tickers por setor
    sorted_tickers = _sort_tickers_by_sector(tickers, sector_map)
    ticker_idx_map = {t: i for i, t in enumerate(tickers)}

    # Monta a matriz de pesos (n_tickers × n_opts) na ordem dos tickers ordenados
    W = np.zeros((n_tickers, n_opts), dtype=np.float32)
    for j, name in enumerate(opt_names):
        w_arr = weights_dict[name]
        for i, ticker in enumerate(sorted_tickers):
            orig_idx = ticker_idx_map.get(ticker)
            if orig_idx is not None and orig_idx < len(w_arr):
                W[i, j] = w_arr[orig_idx]

    # Colormap: branco → cor do tema (azul escuro)
    # Vai de 0 (sem posição) até max_weight (posição máxima)
    cmap = LinearSegmentedColormap.from_list(
        "weights",
        ["#F1EFE8", "#1D9E75", "#0F6E56"],   # off-white → teal → dark teal
        N=256,
    )

    # Tamanho automático proporcional ao número de linhas/colunas
    if figsize is None:
        h = max(6, n_tickers * 0.32 + 2)
        w = max(8, n_opts * 1.4 + 2.5)
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(W, aspect="auto", cmap=cmap, vmin=0, vmax=W.max() * 1.05)

    # Colorbar compacta
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Peso", fontsize=9, color="#5F5E5A")

    # Rótulos dos eixos
    ax.set_xticks(np.arange(n_opts))
    ax.set_xticklabels(opt_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_tickers))
    ax.set_yticklabels(sorted_tickers, fontsize=8)

    # Anotação de valor em cada célula
    if annotate:
        for i in range(n_tickers):
            for j in range(n_opts):
                val = W[i, j]
                if val >= annotate_threshold:
                    # Texto claro em células escuras, escuro em células claras
                    text_color = "white" if val > W.max() * 0.55 else "#2C2C2A"
                    ax.text(j, i, f"{val:.1%}",
                            ha="center", va="center",
                            fontsize=6.5, color=text_color, fontweight="bold")

    # Separadores de setor — linhas horizontais entre setores
    if sector_map:
        current_sector = None
        for i, ticker in enumerate(sorted_tickers):
            sector = sector_map.get(ticker, "unknown")
            if sector != current_sector and i > 0:
                ax.axhline(i - 0.5, color="#FAFAF8", linewidth=1.5, zorder=3)
                # Rótulo do setor na margem esquerda
                ax.text(-0.6, i - 0.5,
                        current_sector or "",
                        ha="right", va="center",
                        fontsize=7, color="#888780",
                        fontstyle="italic")
            current_sector = sector
        # Rótulo do último setor
        if current_sector:
            ax.text(-0.6, n_tickers - 1,
                    current_sector,
                    ha="right", va="center",
                    fontsize=7, color="#888780",
                    fontstyle="italic")

    ax.set_title("G5 — Heatmap de pesos por otimizador e ativo",
                 fontsize=13, fontweight="bold", pad=14, color="#2C2C2A")
    ax.set_xlabel("Otimizador", fontsize=10, color="#5F5E5A", labelpad=8)
    ax.set_ylabel("Ativo", fontsize=10, color="#5F5E5A", labelpad=8)

    # Grade fina
    ax.set_xticks(np.arange(n_opts + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_tickers + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#FAFAF8", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G5 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G6 — FAN CHART DE TRAJETÓRIAS
# ─────────────────────────────────────────────────────────────────────────────

def fan_chart(
    weights_dict: dict,
    data: dict,
    n_paths: int              = 2_000,
    n_steps: int              = 252,
    trading_days: int         = 252,
    seed: int                 = 42,
    percentiles: list[float]  = [5, 25, 50, 75, 95],
    max_optimizers: int       = 6,
    save_path: Optional[Path] = None,
    figsize: tuple            = (14, 7),
) -> plt.Figure:
    """
    G6 — Fan chart: cone de incerteza de cada portfólio ao longo do tempo.

    O que mostra:
        - Faixa externa (p5–p95): 90% dos cenários ficam dentro
        - Faixa interna (p25–p75): 50% dos cenários (IQR)
        - Linha sólida: trajetória mediana (p50)
        - Linha horizontal em 1.0: ponto de partida (S_0)
        - Cada otimizador em sua cor da paleta

    Implementação:
        Re-simula n_paths trajetórias completas com store_paths=True
        para cada otimizador. Usa n_paths menor que step_optimize
        (2.000 vs 20.000) porque fan charts são suaves mesmo com poucas trajetórias.
        Calcula percentis em cada passo de tempo sobre o valor do portfólio:
            V_t = Σ_i w_i * S_t[i]    (S_0 = 1 para todos os ativos)

    Parâmetros
    ----------
    weights_dict  : dict {nome: np.ndarray de pesos}.
    data          : dicionário de fetcher.prepare_data() ou synthetic.
                    Precisa de "mu", "sigma", "chol_lower".
    n_paths       : número de trajetórias a simular por otimizador.
                    2.000 é suficiente para bandas suaves.
    n_steps       : horizonte temporal em dias úteis.
    trading_days  : dias úteis por ano (para eixo x em anos).
    seed          : semente de reprodutibilidade.
    percentiles   : percentis a calcular [p_low_outer, p_low_inner, p_median,
                    p_high_inner, p_high_outer]. Deve ter 5 elementos.
    max_optimizers: limita o número de otimizadores exibidos para legibilidade.
                    Os primeiros max_optimizers do dict são usados.
    save_path     : Path para salvar PNG.
    figsize       : tamanho da figura.

    Retorna
    -------
    plt.Figure
    """
    from simulation.monte_carlo_cpu import simulate_vectorized

    _setup_style()

    if len(percentiles) != 5:
        raise ValueError("percentiles deve ter exatamente 5 elementos: "
                         "[p_outer_low, p_inner_low, p_median, p_inner_high, p_outer_high]")

    mu    = data["mu"]
    sigma = data["sigma"]
    chol  = data["chol_lower"]

    # Limita para não poluir o gráfico
    opt_items = list(weights_dict.items())[:max_optimizers]
    n_opts    = len(opt_items)

    # Eixo temporal: dias → anos
    time_axis = np.arange(n_steps) / trading_days   # 0 … 1.0 (para 252 dias)

    fig, ax = plt.subplots(figsize=figsize)

    # Linha de referência: valor inicial = 1.0
    ax.axhline(1.0, color="#D3D1C7", linewidth=0.8,
               linestyle="--", alpha=0.6, zorder=1)
    ax.text(time_axis[-1] * 0.01, 1.005,
            "Capital inicial (S₀)", fontsize=7.5, color="#888780", va="bottom")

    legend_handles = []

    for idx, (name, weights) in enumerate(opt_items):
        color = _get_color(name, idx)

        # Re-simula com store_paths=True para obter trajetórias completas
        result = simulate_vectorized(
            mu, sigma, chol, weights,
            n_sims=n_paths,
            n_steps=n_steps,
            trading_days=trading_days,
            seed=seed + idx,       # seed diferente por otimizador
            store_paths=True,
        )

        # price_paths shape: (n_steps, n_sims, n_assets)
        paths = result.price_paths   # (n_steps, n_sims, n_assets)

        # Valor do portfólio em cada passo: (n_steps, n_sims)
        # V[t, s] = Σ_i w_i * paths[t, s, i]
        port_value = paths @ weights   # (n_steps, n_sims)

        # Percentis em cada passo: (n_steps, 5)
        pcts = np.percentile(port_value, percentiles, axis=1).T

        p_lo_out = pcts[:, 0]   # p5
        p_lo_in  = pcts[:, 1]   # p25
        p_med    = pcts[:, 2]   # p50
        p_hi_in  = pcts[:, 3]   # p75
        p_hi_out = pcts[:, 4]   # p95

        # Faixa externa p5–p95
        ax.fill_between(time_axis, p_lo_out, p_hi_out,
                        alpha=0.08, color=color, zorder=2)

        # Faixa interna p25–p75
        ax.fill_between(time_axis, p_lo_in, p_hi_in,
                        alpha=0.20, color=color, zorder=3)

        # Bordas das faixas (linhas finas)
        ax.plot(time_axis, p_lo_out, color=color, linewidth=0.5,
                alpha=0.4, linestyle="--", zorder=4)
        ax.plot(time_axis, p_hi_out, color=color, linewidth=0.5,
                alpha=0.4, linestyle="--", zorder=4)

        # Mediana — linha principal
        ax.plot(time_axis, p_med, color=color, linewidth=2.0,
                alpha=0.95, zorder=5, label=name)

        # Rótulo no fim da linha mediana
        final_med = p_med[-1]
        ax.annotate(
            f"{name}\n{final_med - 1:+.1%}",
            xy=(time_axis[-1], final_med),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7.5, color=color, fontweight="bold",
            va="center",
        )

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2,
                   label=f"{name}  (mediana final: {final_med - 1:+.1%})")
        )

    # Eixos
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}a" if x > 0 else "0")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.1f}×")
    )
    ax.set_xlabel("Horizonte temporal", fontsize=10, color="#5F5E5A", labelpad=8)
    ax.set_ylabel("Valor relativo (S₀ = 1.0×)", fontsize=10, color="#5F5E5A", labelpad=8)
    ax.set_title("G6 — Fan chart: cone de incerteza por otimizador",
                 fontsize=13, fontweight="bold", pad=14, color="#2C2C2A")

    # Anotação das faixas (uma vez, no canto)
    ax.text(0.01, 0.97,
            "Faixa escura = p25–p75 (50% dos cenários)\n"
            "Faixa clara  = p5–p95  (90% dos cenários)\n"
            "Linha sólida = mediana (p50)",
            transform=ax.transAxes, fontsize=7.5, color="#888780",
            va="top", linespacing=1.7)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        labelspacing=0.8,
    )

    ax.set_xlim(0, time_axis[-1] * 1.02)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G6 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIÊNCIA — gera G5 e G6 de uma vez
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_layer(
    weights_dict: dict,
    data: dict,
    tickers: list[str],
    sector_map: Optional[dict[str, str]] = None,
    n_paths: int       = 2_000,
    n_steps: int       = 252,
    trading_days: int  = 252,
    seed: int          = 42,
    output_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """
    Gera G5 e G6 de uma vez. Salva em output_dir se fornecido.

    Parâmetros
    ----------
    weights_dict : dict {nome: np.ndarray de pesos}
    data         : dicionário de fetcher.prepare_data() ou synthetic
    tickers      : lista de tickers na mesma ordem dos pesos
    sector_map   : dict {ticker: setor} para ordenação do heatmap
    n_paths      : trajetórias Monte Carlo para o fan chart
    n_steps      : horizonte temporal
    trading_days : dias úteis por ano
    seed         : semente de reprodutibilidade
    output_dir   : diretório de saída (cria subpasta charts/)

    Retorna
    -------
    dict {"g5": Figure, "g6": Figure}
    """
    charts_dir = Path(output_dir) / "charts" if output_dir else None

    fig5 = weights_heatmap(
        weights_dict,
        tickers=tickers,
        sector_map=sector_map,
        save_path=charts_dir / "g5_weights_heatmap.png" if charts_dir else None,
    )
    fig6 = fan_chart(
        weights_dict,
        data=data,
        n_paths=n_paths,
        n_steps=n_steps,
        trading_days=trading_days,
        seed=seed,
        save_path=charts_dir / "g6_fan_chart.png" if charts_dir else None,
    )

    return {"g5": fig5, "g6": fig6}


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

    print("\n[test] Gerando G5 e G6...")
    figs = plot_portfolio_layer(
        weights_dict, data, tickers,
        n_paths=1_000, n_steps=252,
        output_dir=Path("results"),
    )
    plt.show()
    print("[test] Concluído.")