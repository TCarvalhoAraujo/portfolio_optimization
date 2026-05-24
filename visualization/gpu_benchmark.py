"""
visualization/gpu_benchmark.py
================================
Gráficos de comparação CPU vs GPU.

Gráficos implementados
-----------------------
G9 — plot_time_speedup()
    Barras agrupadas com tempo (s) de CPU e GPU por etapa do pipeline.
    Anota o speedup (ex: 8.3×) acima de cada barra GPU.

G10 — plot_throughput()
    Barras agrupadas com throughput (M sims/s) de CPU e GPU por etapa.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


CPU_COLOR = "#378ADD"
GPU_COLOR = "#E85D24"


def _setup_ax(ax, title: str):
    ax.set_facecolor("#FAFAF8")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#444444")
    ax.yaxis.label.set_color("#444444")
    ax.xaxis.label.set_color("#444444")


def plot_time_speedup(timing_data: dict, output_dir: Path) -> plt.Figure:
    """
    G9 — Tempo (s) de CPU e GPU por etapa + anotação de speedup.

    Parâmetros
    ----------
    timing_data : dict com estrutura
        { "Nome da etapa": {"cpu_time": float, "gpu_time": float|None, "n_sims": int} }
    output_dir  : diretório raiz de resultados
    """
    stages    = list(timing_data.keys())
    cpu_times = [timing_data[s]["cpu_time"] for s in stages]
    gpu_times = [timing_data[s].get("gpu_time") for s in stages]
    has_gpu   = any(t is not None for t in gpu_times)

    x     = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FAFAF8")
    _setup_ax(ax, "CPU vs GPU — Tempo por Etapa")

    if has_gpu:
        ax.bar(x - width / 2, cpu_times, width, label="CPU", color=CPU_COLOR, alpha=0.85)
        gpu_vals = [t if t is not None else 0 for t in gpu_times]
        ax.bar(x + width / 2, gpu_vals, width, label="GPU", color=GPU_COLOR, alpha=0.85)

        for i, (ct, gt) in enumerate(zip(cpu_times, gpu_times)):
            if gt is not None and gt > 0:
                speedup = ct / gt
                ax.annotate(
                    f"{speedup:.1f}×",
                    xy=(x[i] + width / 2, gt),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=11, color=GPU_COLOR, fontweight="bold",
                )
        ax.legend(handles=[
            mpatches.Patch(color=CPU_COLOR, label="CPU"),
            mpatches.Patch(color=GPU_COLOR, label="GPU"),
        ], framealpha=0.8)
    else:
        ax.bar(x, cpu_times, width * 1.2, label="CPU", color=CPU_COLOR, alpha=0.85)
        ax.legend(handles=[mpatches.Patch(color=CPU_COLOR, label="CPU")], framealpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Tempo (s)")
    ax.set_ylim(0, max(cpu_times) * 1.25)

    fig.tight_layout()
    out = Path(output_dir) / "charts"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "G9_cpu_gpu_time.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return fig


def plot_throughput(timing_data: dict, output_dir: Path) -> plt.Figure:
    """
    G10 — Throughput (M sims/s) de CPU e GPU por etapa.
    """
    stages  = list(timing_data.keys())
    has_gpu = any(timing_data[s].get("gpu_time") is not None for s in stages)

    cpu_thr = [
        timing_data[s]["n_sims"] / timing_data[s]["cpu_time"] / 1e6
        for s in stages
    ]
    gpu_thr = []
    if has_gpu:
        for s in stages:
            gt = timing_data[s].get("gpu_time")
            gpu_thr.append(timing_data[s]["n_sims"] / gt / 1e6 if gt else 0)

    x     = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FAFAF8")
    _setup_ax(ax, "CPU vs GPU — Throughput por Etapa")

    if has_gpu:
        ax.bar(x - width / 2, cpu_thr, width, label="CPU", color=CPU_COLOR, alpha=0.85)
        ax.bar(x + width / 2, gpu_thr, width, label="GPU", color=GPU_COLOR, alpha=0.85)
        ax.legend(handles=[
            mpatches.Patch(color=CPU_COLOR, label="CPU"),
            mpatches.Patch(color=GPU_COLOR, label="GPU"),
        ], framealpha=0.8)
    else:
        ax.bar(x, cpu_thr, width * 1.2, color=CPU_COLOR, alpha=0.85)
        ax.legend(handles=[mpatches.Patch(color=CPU_COLOR, label="CPU")], framealpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Throughput (M sims/s)")

    fig.tight_layout()
    out = Path(output_dir) / "charts"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "G10_cpu_gpu_throughput.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return fig
