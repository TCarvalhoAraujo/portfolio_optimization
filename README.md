# Portfolio Optimization — Monte Carlo + Otimizadores + ML

Simulação e otimização de portfólios financeiros via Monte Carlo, com múltiplas
técnicas de otimização quantitativa e Machine Learning para seleção de portfólios.

---

## Visão Geral

O pipeline completo executa 5 etapas:

1. **Dados** — coleta preços históricos via Yahoo Finance (yahooquery), calcula retornos logarítmicos, µ, σ e matriz de covariância (Cholesky)
2. **Monte Carlo** — simula N trajetórias por Movimento Browniano Geométrico (GBM) com correlação entre ativos, em CPU e GPU
3. **Otimização** — compara 7 estratégias de alocação (Markowitz, HRP, Risk Parity, Min Variance, Robust, Black-Litterman, 1/N) via métricas reais simuladas
4. **Benchmark** — compara throughput CPU vs GPU em múltiplas escalas
5. **Machine Learning** — treina RandomForest / XGBoost / GradientBoosting para predizer Sharpe e selecionar o melhor portfólio entre milhares de candidatos

---

## Estrutura do Projeto

```
portfolio_optimization/
├── main.py                              # Ponto de entrada — orquestra todo o pipeline
├── data/
│   ├── fetcher.py                       # Download e pré-processamento via yahooquery
│   ├── synthetic.py                     # Gerador de dados sintéticos (offline/CI)
│   ├── tickers.py                       # Listas de portfólios pré-definidos (MY_DIVERSIFIED_50, etc.)
│   └── eda.py                           # Análise exploratória de dados
├── simulation/
│   ├── monte_carlo_cpu.py               # Implementações CPU (vetorizada, batched, sequencial)
│   └── monte_carlo_gpu.py               # Implementação GPU via PyCUDA (opcional)
├── portfolio/
│   └── metrics.py                       # VaR, CVaR, Sharpe, Sortino, Max Drawdown
├── optimization/                        # ← módulo novo
│   ├── __init__.py                      # Exporta todos os otimizadores
│   ├── base_optimizer.py                # Classe abstrata com interface comum
│   ├── constraints.py                   # Long-only, box, setor, turnover
│   ├── markowitz.py                     # MVO clássico + fronteira eficiente
│   ├── min_variance.py                  # Global Minimum Variance (GMV)
│   ├── risk_parity.py                   # Equal Risk Contribution (ERC)
│   ├── hrp.py                           # Hierarchical Risk Parity (Lopez de Prado)
│   ├── black_litterman.py               # Black-Litterman com views configuráveis
│   ├── robust_optimizer.py              # Otimização robusta (worst-case)
│   └── covariance/
│       ├── __init__.py
│       ├── sample_cov.py                # Covariância amostral + diagnósticos
│       ├── ledoit_wolf.py               # Shrinkage de Ledoit-Wolf e OAS
│       └── factor_model.py              # Modelo de fatores via PCA
├── ml/
│   ├── dataset.py                       # Geração de dataset de treino
│   └── portfolio_selector.py            # Treino, avaliação, seleção e comparação
└── benchmark/
    └── benchmark.py                     # Comparação CPU vs GPU
```

---

## Instalação

### Pré-requisitos

- Python 3.10+
- (Para GPU) NVIDIA GPU com CUDA Toolkit instalado

```bash
git clone <url-do-repositorio>
cd portfolio_optimization
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt

# Somente em máquinas com NVIDIA:
pip install -r requirements-gpu.txt
```

---

## Como Executar

### Pipeline completo

```bash
# Com o portfólio de 50 ativos reais do S&P 500
python main.py --preset my50 --mode full

# Com dados sintéticos (sem internet)
python main.py --mode full
```

### Modos disponíveis

| Modo | Descrição |
|---|---|
| `full` | Pipeline completo: dados → simulação → **otimização** → benchmark → ML |
| `data` | Apenas coleta e EDA |
| `simulate` | Apenas Monte Carlo (portfólio 1/N) |
| `optimize` | **Apenas otimizadores** — compara todos e salva relatório de pesos |
| `benchmark` | Benchmark CPU vs GPU |
| `ml` | Dataset + treino + seleção ML + comparação com todos os otimizadores |

### Exemplos

```bash
# Só comparar otimizadores (mais rápido)
python main.py --preset my50 --mode optimize

# Limitar peso máximo por ativo
python main.py --preset my50 --mode optimize --w-max 0.10

# Pipeline completo com preset personalizado
python main.py --tickers AAPL MSFT NVDA GOOGL AMZN JPM --mode full

# Pular Black-Litterman (mais rápido com N grande)
python main.py --preset my50 --mode full --skip-bl

# Benchmark em múltiplas escalas
python main.py --preset my50 --mode benchmark --n-sims 10000 100000 500000 1000000

# ML com mais portfólios de treino
python main.py --preset my50 --mode ml --n-portfolios 10000 --n-sims-per-portfolio 5000
```

---

## Módulo de Otimização

O módulo `optimization/` implementa 6 estratégias com **interface comum**:

```python
from optimization import MarkowitzOptimizer, HRPOptimizer, long_only_box

constraints = long_only_box(w_max=0.10)

# Qualquer otimizador usa a mesma interface
weights = MarkowitzOptimizer().optimize(returns, constraints)
weights = HRPOptimizer(linkage_method="ward").optimize(returns, constraints)
```

### Estratégias implementadas

| Otimizador | Usa µ? | Usa Σ? | Robustez OOS | Descrição |
|---|---|---|---|---|
| `MinVarianceOptimizer` | Não | Sim | Alta | Minimiza variância, ignora retorno |
| `RiskParityOptimizer` | Não | Sim | Alta | Equaliza contribuição de risco por ativo |
| `HRPOptimizer` | Não | Baixa | Alta | Clusterização hierárquica, sem inversão de Σ |
| `MarkowitzOptimizer` | Sim | Sim | Média | Maximiza Sharpe (μ + Σ) |
| `RobustOptimizer` | Sim | Sim | Alta | Markowitz com penalty de incerteza (κ ajustável) |
| `BlackLittermanOptimizer` | Views | Sim | Média-Alta | Prior de mercado + views do investidor |

### Estimadores de covariância

Todos os otimizadores usam `LedoitWolf` por padrão (recomendado para N ≥ 20):

```python
from optimization.covariance import ledoit_wolf_cov, compare_estimators

# Diagnóstico: compara amostral vs LW vs OAS
summary = compare_estimators(returns)
print(summary)
```

### Constraints disponíveis

```python
from optimization import long_only_box, long_only_sector, with_turnover

# Long-only com limite por ativo
constraints = long_only_box(w_max=0.10)

# Com restrições setoriais
constraints = long_only_sector(
    sector_map={"AAPL": "Technology", "JPM": "Financials", ...},
    sector_limits={"Technology": (0.05, 0.30), "Financials": (0.05, 0.25)},
)

# Com limite de turnover (para rebalanceamento)
constraints = with_turnover(base=long_only_box(), w_current=w_prev, max_turnover=0.20)
```

---

## Saídas Geradas

| Arquivo | Conteúdo |
|---|---|
| `results/data/prices.csv` | Preços ajustados históricos |
| `results/data/log_returns.csv` | Retornos logarítmicos diários |
| `results/data/asset_stats.csv` | µ, σ e Sharpe individual por ativo |
| `results/optimizer_comparison.csv` | **Métricas de todos os otimizadores** |
| `results/optimizer_weights.csv` | **Pesos por ativo para cada otimizador** |
| `results/benchmark_results.csv` | Tempos CPU vs GPU por escala |
| `results/feature_importance.csv` | Importância das features do modelo ML |
| `results/portfolio_comparison.csv` | ML vs todos os otimizadores |
| `results/best_portfolio_weights.csv` | Pesos do portfólio ML selecionado |
| `results/dataset/` | Dataset de treino (features.parquet, labels.parquet, weights.npy) |

---

## Dependências Principais

| Biblioteca | Uso |
|---|---|
| `yahooquery` | Download de dados históricos |
| `numpy` / `pandas` | Manipulação de dados e álgebra linear |
| `scipy` | Otimização (SLSQP), testes estatísticos, clusterização |
| `scikit-learn` | Ledoit-Wolf, PCA, Random Forest, métricas |
| `xgboost` | Gradient Boosting |
| `pycuda` *(GPU)* | Interface Python para kernels CUDA |