"""
visualization/risk_metrics.py
==============================
Camada 2 de visualização — risco e métricas comparativas.

Gráficos implementados
-----------------------
G3 — scatter_risk_return()
    Cada otimizador como um ponto no plano (volatilidade × retorno esperado).
    A fronteira eficiente de Markowitz é sobreposta como curva de referência,
    calculada diretamente dos retornos históricos.
    Linhas de iso-Sharpe (razão retorno/risco constante) são desenhadas
    como guias de leitura. O ponto de máximo Sharpe é destacado.

G4 — metrics_barchart()
    Quatro métricas chave lado a lado para cada otimizador:
    Sharpe, Sortino, VaR 95% e CVaR 95%. Cada métrica tem seu próprio
    painel (subplot) com barras horizontais coloridas por otimizador.
    Inclui anotação de valor em cada barra.

Conceito por trás de G3
------------------------
O scatter risco-retorno é o gráfico mais clássico de teoria de portfólios.
A fronteira eficiente divide o espaço em duas regiões:
    - Acima/direita da fronteira: portfólios inalcançáveis
    - Sobre a fronteira: portfólios ótimos (máximo retorno dado risco)
    - Abaixo/direita da fronteira: portfólios subótimos

Pontos abaixo da fronteira revelam onde uma estratégia está
"desperdiçando risco" — assumindo volatilidade sem o retorno correspondente.

As linhas de iso-Sharpe são retas que passam pela taxa livre de risco.
A inclinação de cada linha = Sharpe ratio da linha. Quanto mais à esquerda
(em direção a menor risco) uma linha toca a fronteira, maior o Sharpe.

Conceito por trás de G4
------------------------
Sharpe e Sortino medem retorno ajustado ao risco (mais = melhor).
VaR e CVaR medem perda em cenários adversos (menos negativo = melhor).
Mostrar as quatro juntas evita decisões baseadas em uma única métrica:
um portfólio pode ter Sharpe alto mas CVaR muito negativo (cauda pesada).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional

from .distribution import OPTIMIZER_COLORS, _get_color, _setup_style


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_from_results(sim_results: dict, rf: float, trading_days: int, n_steps: int) -> dict:
    """
    Extrai métricas financeiras de cada SimulationResult.

    Retorna dict {nome: dict com ret, vol, sharpe, sortino, var95, cvar95}.
    Funciona tanto com SimulationResult quanto com np.ndarray de retornos.
    """
    from portfolio.metrics import (
        sharpe_ratio, sortino_ratio,
        var_historical, cvar_historical,
    )

    out = {}
    for name, obj in sim_results.items():
        if hasattr(obj, "portfolio_returns"):
            r = obj.portfolio_returns.astype(np.float64)
        elif isinstance(obj, np.ndarray):
            r = obj.astype(np.float64)
        else:
            continue

        scale   = trading_days / n_steps
        r_ann   = r * scale
        mu_ann  = float(r_ann.mean())
        std_ann = float(r_ann.std())

        out[name] = {
            "ret"    : mu_ann,
            "vol"    : std_ann,
            "sharpe" : sharpe_ratio(r, n_steps, trading_days, rf),
            "sortino": sortino_ratio(r, n_steps, trading_days, rf),
            "var95"  : var_historical(r_ann, 0.95),
            "cvar95" : cvar_historical(r_ann, 0.95),
        }

    return out


def _efficient_frontier_curve(returns_df: pd.DataFrame, rf: float, n_points: int = 60):
    """
    Calcula a fronteira eficiente via otimização de Markowitz.

    Usa o módulo optimization/markowitz.py para traçar a fronteira real
    com os dados históricos dos ativos.

    Retorna (vols, rets) arrays para plotar a curva.
    Retorna (None, None) se a otimização falhar.
    """
    try:
        from optimization import MarkowitzOptimizer, long_only_box
        opt = MarkowitzOptimizer()
        constraints = long_only_box(w_max=1.0)
        frontier_df = opt.efficient_frontier(returns_df, constraints, n_points=n_points)
        if frontier_df.empty:
            return None, None
        return frontier_df["volatility"].values, frontier_df["return"].values
    except Exception as e:
        print(f"[viz] fronteira eficiente não calculada: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# G3 — SCATTER RISCO × RETORNO
# ─────────────────────────────────────────────────────────────────────────────

def scatter_risk_return(
    sim_results: dict,
    returns_df: Optional[pd.DataFrame]  = None,
    rf: float             = 0.05,
    n_steps: int          = 252,
    trading_days: int     = 252,
    show_frontier: bool   = True,
    show_iso_sharpe: bool = True,
    show_sharpe_max: bool = True,
    save_path: Optional[Path] = None,
    figsize: tuple        = (9, 7),
) -> plt.Figure:
    """
    G3 — Scatter de cada otimizador no plano risco (vol) × retorno esperado.

    O que mostra:
        - Posição relativa de cada estratégia em risco vs retorno
        - Fronteira eficiente de Markowitz como referência (se returns_df fornecido)
        - Linhas de iso-Sharpe: estratégias sobre a mesma linha têm o mesmo Sharpe
        - Ponto de tangência (máximo Sharpe) destacado na fronteira
        - Capital Market Line (rf até o ponto de tangência)

    Parâmetros
    ----------
    sim_results  : dict {nome: SimulationResult ou np.ndarray}
    returns_df   : pd.DataFrame de retornos históricos (para calcular fronteira).
                   Se None, a fronteira não é desenhada.
    rf           : taxa livre de risco anualizada
    n_steps      : horizonte temporal da simulação
    trading_days : dias úteis por ano
    show_frontier    : desenha a curva da fronteira eficiente
    show_iso_sharpe  : desenha linhas de iso-Sharpe como guia
    show_sharpe_max  : destaca o ponto de máximo Sharpe na fronteira
    save_path    : Path para salvar PNG
    figsize      : tamanho da figura

    Retorna
    -------
    plt.Figure
    """
    _setup_style()
    metrics = _metrics_from_results(sim_results, rf, trading_days, n_steps)

    if not metrics:
        raise ValueError("sim_results vazio ou sem retornos válidos.")

    fig, ax = plt.subplots(figsize=figsize)

    # ── Fronteira eficiente ───────────────────────────────────────────────────
    frontier_vols, frontier_rets = None, None
    tangency_vol, tangency_ret   = None, None

    if show_frontier and returns_df is not None:
        frontier_vols, frontier_rets = _efficient_frontier_curve(returns_df, rf)

        if frontier_vols is not None:
            ax.plot(frontier_vols, frontier_rets,
                    color="#D3D1C7", linewidth=2.0, zorder=1,
                    label="Fronteira eficiente")
            ax.fill_between(frontier_vols, frontier_rets,
                            alpha=0.04, color="#D3D1C7", zorder=0)

            # Ponto de máximo Sharpe
            if show_sharpe_max:
                sharpes = (frontier_rets - rf) / (frontier_vols + 1e-10)
                best_idx = np.argmax(sharpes)
                tangency_vol = frontier_vols[best_idx]
                tangency_ret = frontier_rets[best_idx]
                ax.scatter(tangency_vol, tangency_ret,
                           s=90, zorder=6, marker="*",
                           color="#2C2C2A", edgecolors="#FAFAF8",
                           linewidths=0.8, label="Máximo Sharpe (fronteira)")

                # Capital Market Line: rf → tangência
                cml_vols = np.array([0, tangency_vol * 1.4])
                cml_rets = rf + (tangency_ret - rf) / tangency_vol * cml_vols
                ax.plot(cml_vols, cml_rets,
                        color="#2C2C2A", linewidth=0.8,
                        linestyle="--", alpha=0.4, zorder=1,
                        label="Capital Market Line")

    # ── Linhas de iso-Sharpe ─────────────────────────────────────────────────
    if show_iso_sharpe:
        all_sharpes = [m["sharpe"] for m in metrics.values()]
        sharpe_levels = np.percentile(all_sharpes, [25, 50, 75])

        vol_range = np.linspace(0, max(m["vol"] for m in metrics.values()) * 1.3, 100)
        for s_level in sharpe_levels:
            ret_line = rf + s_level * vol_range
            ax.plot(vol_range, ret_line,
                    color="#B4B2A9", linewidth=0.6,
                    linestyle=":", alpha=0.7, zorder=1)
            # Rótulo no fim da linha
            ax.text(vol_range[-1] * 0.98, ret_line[-1],
                    f"S={s_level:.2f}", fontsize=7,
                    color="#888780", ha="right", va="bottom")

    # ── Pontos dos otimizadores ───────────────────────────────────────────────
    legend_handles = []

    for idx, (name, m) in enumerate(metrics.items()):
        color = _get_color(name, idx)
        vol, ret = m["vol"], m["ret"]

        # Círculo preenchido com borda
        ax.scatter(vol, ret,
                   s=110, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2)

        # Rótulo com seta para evitar sobreposição
        ax.annotate(
            name,
            xy=(vol, ret),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            zorder=6,
        )

        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markersize=8,
                   label=f"{name}  S={m['sharpe']:.2f}")
        )

    # ── Linha do rf no eixo y ────────────────────────────────────────────────
    ax.axhline(rf, color="#888780", linewidth=0.7,
               linestyle=":", alpha=0.5, zorder=1)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0.001,
            rf, f" rf={rf:.1%}",
            fontsize=7.5, color="#888780", va="bottom")

    # ── Eixos ────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlabel("Volatilidade anualizada (risco)", fontsize=10,
                  color="#5F5E5A", labelpad=8)
    ax.set_ylabel("Retorno esperado anualizado", fontsize=10,
                  color="#5F5E5A", labelpad=8)
    ax.set_title("G3 — Risco vs retorno por otimizador",
                 fontsize=13, fontweight="bold", pad=14, color="#2C2C2A")

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        labelspacing=0.8,
    )

    # Margem generosa para os rótulos não cortarem
    x_vals = [m["vol"] for m in metrics.values()]
    y_vals = [m["ret"] for m in metrics.values()]
    x_pad  = (max(x_vals) - min(x_vals)) * 0.25
    y_pad  = (max(y_vals) - min(y_vals)) * 0.35
    ax.set_xlim(max(0, min(x_vals) - x_pad), max(x_vals) + x_pad)
    ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G3 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G4 — BARCHART DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def metrics_barchart(
    sim_results: dict,
    rf: float         = 0.05,
    n_steps: int      = 252,
    trading_days: int = 252,
    save_path: Optional[Path] = None,
    figsize: tuple    = (13, 9),
) -> plt.Figure:
    """
    G4 — Quatro métricas lado a lado para cada otimizador.

    Painéis:
        (1) Sharpe Ratio    — maior é melhor, linha de referência em 0
        (2) Sortino Ratio   — maior é melhor (penaliza só downside)
        (3) VaR 95%         — menos negativo é melhor
        (4) CVaR 95%        — menos negativo é melhor (piores 5% dos cenários)

    Layout: 2×2 grid de subplots com barras horizontais.
    Cada barra tem o valor anotado na ponta.
    Ordenação: do maior para o menor dentro de cada painel.
    Cor por otimizador, consistente com os outros gráficos da camada.

    Parâmetros
    ----------
    sim_results  : dict {nome: SimulationResult ou np.ndarray}
    rf           : taxa livre de risco anualizada
    n_steps      : horizonte temporal
    trading_days : dias úteis por ano
    save_path    : Path para salvar PNG
    figsize      : tamanho da figura

    Retorna
    -------
    plt.Figure
    """
    _setup_style()
    metrics = _metrics_from_results(sim_results, rf, trading_days, n_steps)

    if not metrics:
        raise ValueError("sim_results vazio ou sem retornos válidos.")

    names  = list(metrics.keys())
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    # Definição dos 4 painéis
    panels = [
        {
            "key"  : "sharpe",
            "label": "Sharpe Ratio",
            "fmt"  : lambda v: f"{v:.2f}",
            "ref"  : 0.0,
            "ref_label": "S=0",
            "better": "higher",
        },
        {
            "key"  : "sortino",
            "label": "Sortino Ratio",
            "fmt"  : lambda v: f"{v:.2f}",
            "ref"  : 0.0,
            "ref_label": "S=0",
            "better": "higher",
        },
        {
            "key"  : "var95",
            "label": "VaR 95%",
            "fmt"  : lambda v: f"{v:.1%}",
            "ref"  : None,
            "ref_label": None,
            "better": "higher",   # menos negativo = melhor
        },
        {
            "key"  : "cvar95",
            "label": "CVaR 95%  (Expected Shortfall)",
            "fmt"  : lambda v: f"{v:.1%}",
            "ref"  : None,
            "ref_label": None,
            "better": "higher",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, panel in zip(axes, panels):
        key = panel["key"]

        # Ordena do maior para o menor (todos os painéis: higher = better)
        ordered = sorted(
            [(n, metrics[n][key], _get_color(n, i)) for i, n in enumerate(names)],
            key=lambda x: x[1],
        )

        opt_names = [x[0] for x in ordered]
        values    = [x[1] for x in ordered]
        bar_colors = [x[2] for x in ordered]

        y_pos = np.arange(len(opt_names))

        # Barras
        bars = ax.barh(
            y_pos, values,
            color=bar_colors,
            alpha=0.75,
            height=0.55,
            zorder=3,
        )

        # Anotações de valor na ponta de cada barra
        for bar, val, color in zip(bars, values, bar_colors):
            x_offset = abs(val) * 0.02 + 0.001
            ha = "left" if val >= 0 else "right"
            sign_offset = x_offset if val >= 0 else -x_offset
            ax.text(
                val + sign_offset,
                bar.get_y() + bar.get_height() / 2,
                panel["fmt"](val),
                va="center", ha=ha,
                fontsize=8, color=color, fontweight="bold",
            )

        # Linha de referência (zero ou rf)
        if panel["ref"] is not None:
            ax.axvline(panel["ref"], color="#888780",
                       linewidth=0.8, linestyle="--",
                       alpha=0.6, zorder=2)

        # Borda esquerda de destaque no melhor resultado
        best_idx = len(opt_names) - 1   # maior valor está no topo
        ax.barh(
            best_idx, values[best_idx],
            color=bar_colors[best_idx],
            alpha=0.2, height=0.55, zorder=2,
        )

        # Eixos
        ax.set_yticks(y_pos)
        ax.set_yticklabels(opt_names, fontsize=8.5)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
            if key in ("var95", "cvar95")
            else plt.FuncFormatter(lambda x, _: f"{x:.2f}")
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(panel["label"], fontsize=10,
                     fontweight="bold", color="#2C2C2A", pad=8)
        ax.set_xlabel(
            "← pior     melhor →" if panel["better"] == "higher" else "← melhor     pior →",
            fontsize=7.5, color="#888780", labelpad=4,
        )

        # Remove spine esquerdo (yticks já identificam)
        ax.spines["left"].set_visible(False)

    fig.suptitle("G4 — Métricas comparativas por otimizador",
                 fontsize=13, fontweight="bold", y=1.01, color="#2C2C2A")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] G4 salvo em {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIÊNCIA — gera G3 e G4 de uma vez
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_layer(
    sim_results: dict,
    returns_df: Optional[pd.DataFrame] = None,
    rf: float          = 0.05,
    n_steps: int       = 252,
    trading_days: int  = 252,
    output_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """
    Gera G3 e G4 de uma vez. Salva em output_dir se fornecido.

    Parâmetros
    ----------
    sim_results : dict {nome: SimulationResult}
    returns_df  : DataFrame de retornos históricos (para fronteira eficiente).
                  Passe data["log_returns"] do fetcher.
    rf          : taxa livre de risco anualizada
    n_steps     : horizonte temporal
    trading_days: dias úteis por ano
    output_dir  : diretório de saída (cria subpasta charts/)

    Retorna
    -------
    dict {"g3": Figure, "g4": Figure}
    """
    charts_dir = Path(output_dir) / "charts" if output_dir else None

    fig3 = scatter_risk_return(
        sim_results,
        returns_df=returns_df,
        rf=rf, n_steps=n_steps, trading_days=trading_days,
        save_path=charts_dir / "g3_scatter_risk_return.png" if charts_dir else None,
    )
    fig4 = metrics_barchart(
        sim_results,
        rf=rf, n_steps=n_steps, trading_days=trading_days,
        save_path=charts_dir / "g4_metrics_barchart.png" if charts_dir else None,
    )

    return {"g3": fig3, "g4": fig4}


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
    data    = generate_synthetic_data(n_assets=20, n_days=1260, seed=0)
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
        w = make_equal_weights(n) if opt is None else \
            opt.optimize(returns, constraints).values.astype("float32")
        w /= w.sum()
        sim_results[name] = simulate_vectorized(mu, sigma, chol, w,
                                                n_sims=5_000, seed=42)
        print(f"  ✓ {name}")

    print("\n[test] Gerando G3 e G4...")
    figs = plot_risk_layer(sim_results, returns_df=returns,
                           rf=0.05, output_dir=Path("results"))
    plt.show()
    print("[test] Concluído.")