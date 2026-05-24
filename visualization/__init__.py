"""
visualization/
==============
Módulo de visualização do sistema de otimização de portfólios.

Camadas implementadas
---------------------
    Camada 1 — distribution.py    G1: KDE sobreposto      G2: Boxplot comparativo
    Camada 2 — risk_metrics.py    G3: Scatter risco-ret.  G4: Barchart de métricas
    Camada 3 — portfolio.py       G5: Heatmap de pesos    G6: Fan chart trajetórias
    Camada 4 — correlation.py     G7: Heatmap correlação  G8: Drawdown por otimizador

Paleta compartilhada
--------------------
    OPTIMIZER_COLORS — dict {nome_otimizador: hex_color}
    Importado por todos os módulos para garantir cor consistente
    de cada otimizador em todos os gráficos.

Uso rápido
----------
    from visualization.distribution import plot_distribution_layer
    from visualization.risk_metrics  import plot_risk_layer
    from visualization.portfolio     import plot_portfolio_layer

    figs1 = plot_distribution_layer(sim_results, rf=0.05, output_dir=Path("results"))
    figs2 = plot_risk_layer(sim_results, returns_df=data["log_returns"], rf=0.05, output_dir=Path("results"))
    figs3 = plot_portfolio_layer(weights_dict, data, tickers=data["tickers"], output_dir=Path("results"))
"""

from .distribution import (
    kde_overlay,
    boxplot_comparison,
    plot_distribution_layer,
    OPTIMIZER_COLORS,
)

from .risk_metrics import (
    scatter_risk_return,
    metrics_barchart,
    plot_risk_layer,
)

from .portfolio import (
    weights_heatmap,
    fan_chart,
    plot_portfolio_layer,
)

from .correlation import (
    correlation_heatmap,
    drawdown_analysis,
    plot_correlation_layer,
)

__all__ = [
    # Camada 1
    "kde_overlay",
    "boxplot_comparison",
    "plot_distribution_layer",
    # Camada 2
    "scatter_risk_return",
    "metrics_barchart",
    "plot_risk_layer",
    # Camada 3
    "weights_heatmap",
    "fan_chart",
    "plot_portfolio_layer",
    # Camada 4
    "correlation_heatmap",
    "drawdown_analysis",
    "plot_correlation_layer",
    # Paleta
    "OPTIMIZER_COLORS",
]