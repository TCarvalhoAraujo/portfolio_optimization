"""
visualization/ml_results.py
============================
Camada 5 de visualização — resultados do modelo de ML.

Gráficos implementados
-----------------------
G11 — plot_pred_vs_actual()
    Scatter de Sharpe predito vs Sharpe real no conjunto de teste.
    Avalia a qualidade do regressor com linha de referência perfeita e R².

G12 — plot_feature_importance()
    Barchart horizontal das top features por importância no modelo treinado.
    Indica quais características de portfólio mais influenciam o Sharpe.

G13 — plot_baseline_comparison()
    Barchart horizontal de Sharpe para ML vs todos os otimizadores.
    ML destacado em cor diferente para facilitar comparação visual.

G14 — plot_weight_distribution()
    Barras agrupadas por ativo: pesos do portfólio ML vs 1/N.
    Mostra onde o ML concentra ou diversifica em relação ao baseline.

G15 — plot_selected_stocks()
    Composição visual da carteira ML selecionada.
    Ativos ordenados por peso, destaque nos que passam do limiar de inclusão.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ML_COLOR   = "#E85D24"
BASE_COLOR = "#378ADD"
MUTED      = "#AAAAAA"
BG         = "#FAFAF8"


def _setup_ax(ax, title: str):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.yaxis.label.set_color("#444444")
    ax.xaxis.label.set_color("#444444")


def _save(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())


# ─────────────────────────────────────────────────────────────────────────────
# G11 — Predito vs Real
# ─────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_actual(train_result: dict, output_dir: Path) -> plt.Figure:
    """G11 — Scatter Sharpe predito vs real (conjunto de teste)."""
    from sklearn.metrics import r2_score

    model  = train_result["best_model"]
    X_test = train_result["X_test"]
    y_test = train_result["y_test"]
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, "G11 — Sharpe Predito vs Real")

    ax.scatter(y_test, y_pred, alpha=0.35, s=18, color=BASE_COLOR, linewidths=0)

    lo = min(y_test.min(), y_pred.min()) - 0.05
    hi = max(y_test.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1.2, linestyle="--", label="Perfeito")

    ax.set_xlabel("Sharpe Real")
    ax.set_ylabel("Sharpe Predito")
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=11, color="#333333", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.6)

    fig.tight_layout()
    _save(fig, Path(output_dir) / "charts", "G11_pred_vs_actual.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G12 — Importância de Features
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(feat_imp: pd.DataFrame, output_dir: Path,
                            top_n: int = 15) -> plt.Figure:
    """G12 — Barchart horizontal: top features por importância."""
    df = feat_imp.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, f"G12 — Top {top_n} Features por Importância")

    colors = [ML_COLOR if i == 0 else BASE_COLOR for i in range(len(df))]
    bars   = ax.barh(df["feature"][::-1], df["importance"][::-1],
                     color=colors[::-1], alpha=0.85, height=0.65)

    for bar, val in zip(bars, df["importance"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8, color="#555555")

    ax.set_xlabel("Importância")
    ax.set_xlim(0, df["importance"].max() * 1.18)

    fig.tight_layout()
    _save(fig, Path(output_dir) / "charts", "G12_feature_importance.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G13 — ML vs Baselines
# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline_comparison(cmp_df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """G13 — Barchart: Sharpe do ML vs todos os otimizadores."""
    df = cmp_df[["Sharpe"]].copy().sort_values("Sharpe", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.55)))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, "G13 — Sharpe: ML vs Otimizadores")

    colors = [ML_COLOR if "ML" in str(idx) else BASE_COLOR for idx in df.index]
    bars   = ax.barh(df.index, df["Sharpe"], color=colors, alpha=0.85, height=0.65)

    for bar, val in zip(bars, df["Sharpe"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9,
                color=ML_COLOR if "ML" in bar.get_label() else "#555555",
                fontweight="bold" if "ML" in str(val) else "normal")

    ax.set_xlabel("Sharpe Ratio")
    ax.set_xlim(0, df["Sharpe"].max() * 1.15)
    ax.legend(handles=[
        mpatches.Patch(color=ML_COLOR,   label="ML-Selecionado"),
        mpatches.Patch(color=BASE_COLOR, label="Otimizadores"),
    ], fontsize=9, framealpha=0.6)

    fig.tight_layout()
    _save(fig, Path(output_dir) / "charts", "G13_baseline_comparison.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G14 — Distribuição de Pesos: ML vs 1/N
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_distribution(ml_result: dict, data: dict,
                             output_dir: Path) -> plt.Figure:
    """G14 — Barras agrupadas: pesos ML vs 1/N por ativo."""
    tickers   = data.get("tickers", [f"A{i}" for i in range(data["n_assets"])])
    ml_w      = ml_result["weights"]
    eq_w      = np.ones(len(tickers)) / len(tickers)

    order  = np.argsort(ml_w)[::-1]
    labels = [tickers[i] for i in order]
    ml_ord = ml_w[order]
    eq_ord = eq_w[order]

    x     = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 5))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, "G14 — Pesos: ML vs 1/N")

    ax.bar(x - width / 2, ml_ord, width, label="ML", color=ML_COLOR,   alpha=0.85)
    ax.bar(x + width / 2, eq_ord, width, label="1/N", color=BASE_COLOR, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peso (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9, framealpha=0.6)

    fig.tight_layout()
    _save(fig, Path(output_dir) / "charts", "G14_weight_distribution.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# G15 — Ações Selecionadas
# ─────────────────────────────────────────────────────────────────────────────

def plot_selected_stocks(ml_result: dict, data: dict,
                         output_dir: Path, threshold: float = 0.01) -> plt.Figure:
    """
    G15 — Composição visual da carteira ML.

    Ativos ordenados por peso decrescente. Barras coloridas para os que
    superam o limiar de inclusão (threshold), cinza para os demais.
    Linha vertical marca o peso 1/N (referência de igualdade).
    """
    tickers = data.get("tickers", [f"A{i}" for i in range(data["n_assets"])])
    weights = ml_result["weights"]
    eq_w    = 1.0 / len(tickers)

    order   = np.argsort(weights)[::-1]
    labels  = [tickers[i] for i in order]
    vals    = weights[order]
    colors  = [ML_COLOR if v >= threshold else MUTED for v in vals]

    n_active = int((weights >= threshold).sum())

    fig, ax = plt.subplots(figsize=(7, max(5, len(labels) * 0.32)))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, f"G15 — Carteira ML Selecionada  ({n_active} ativos ativos)")

    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], alpha=0.88, height=0.7)

    ax.axvline(eq_w, color="#888888", lw=1.2, linestyle="--",
               label=f"1/N ({eq_w:.1%})")

    for i, (label, val) in enumerate(zip(labels[::-1], vals[::-1])):
        if val >= threshold:
            ax.text(val + 0.002, i, f"{val:.1%}", va="center",
                    fontsize=8, color="#333333", fontweight="bold")

    ax.set_xlabel("Peso na Carteira")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9, framealpha=0.6)

    fig.tight_layout()
    _save(fig, Path(output_dir) / "charts", "G15_selected_stocks.png")
    return fig
