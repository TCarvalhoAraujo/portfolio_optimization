# Portfolio Optimization — Monte Carlo + CUDA + ML

Simulação e otimização de portfólios financeiros via Monte Carlo, acelerada em GPU com CUDA e assistida por Machine Learning.

**Equipe:** Thiago Carvalho de Araujo · Pedro Mota Lindoso · Miguel Procópio · Gustavo Porto · Rafael Angelim

---

## Visão Geral

O pipeline completo executa 4 etapas:

1. **Dados** — coleta preços históricos via Yahoo Finance (yahooquery) e calcula retornos logarítmicos, µ, σ e matriz de covariância (Cholesky)
2. **Monte Carlo** — simula N trajetórias de preço por Movimento Browniano Geométrico (GBM) com correlação entre ativos, em CPU e GPU
3. **Benchmark** — compara throughput CPU vs GPU em múltiplas escalas de simulação
4. **Machine Learning** — treina RandomForest / XGBoost / GradientBoosting para predizer Sharpe ratio e selecionar o melhor portfólio entre 50.000 candidatos

---

## Estrutura do Projeto

```
portfolio_optimization/
├── main.py                         # Ponto de entrada — orquestra todo o pipeline
├── requirements.txt                # Dependências CPU (qualquer máquina)
├── requirements-gpu.txt            # Dependências GPU (NVIDIA + CUDA)
├── data/
│   ├── fetcher.py                  # Download e pré-processamento via yahooquery
│   ├── synthetic.py                # Gerador de dados sintéticos (offline/CI)
│   ├── tickers.py                  # Listas de portfólios pré-definidos
│   └── eda.py                      # Análise exploratória de dados
├── simulation/
│   ├── monte_carlo_cpu.py          # Implementações CPU (sequencial, vetorizada, batched)
│   ├── monte_carlo_gpu.py          # Implementação GPU via PyCUDA
│   └── kernels/
│       └── mc_kernel.cu            # Kernel CUDA (1 thread = 1 trajetória Monte Carlo)
├── portfolio/
│   └── metrics.py                  # VaR, CVaR, Sharpe, Sortino, Max Drawdown
├── ml/
│   ├── dataset.py                  # Geração de dataset de treino
│   └── portfolio_selector.py       # Treino, avaliação e seleção de portfólio
└── benchmark/
    └── benchmark.py                # Comparação CPU vs GPU em múltiplas escalas
```

---

## Instalação

### Pré-requisitos

- Python 3.10+
- (Para GPU) NVIDIA GPU com CUDA Toolkit instalado

### 1. Clonar e criar ambiente virtual

```bash
git clone <url-do-repositorio>
cd portfolio_optimization
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Instalar dependências base (CPU)

```bash
pip install -r requirements.txt
```

### 3. Instalar dependências GPU (somente em máquinas com NVIDIA)

Verifique sua versão do CUDA antes:

```bash
nvidia-smi
```

Depois instale:

```bash
pip install -r requirements-gpu.txt
```

> Se houver conflito na instalação do pycuda, use conda:
> ```bash
> conda install -c conda-forge pycuda
> ```

---

## Como Executar

### Pipeline completo (recomendado)

Com o portfólio de 50 ativos reais do S&P 500:

```bash
python main.py --preset my50 --mode full
```

Com dados sintéticos (sem internet):

```bash
python main.py --mode full
```

### Modos disponíveis

| Modo | Descrição |
|---|---|
| `full` | Pipeline completo: dados → simulação → benchmark → ML |
| `data` | Apenas coleta e EDA |
| `simulate` | Apenas simulação Monte Carlo |
| `benchmark` | Benchmark CPU vs GPU em múltiplas escalas |
| `ml` | Geração de dataset + treino + seleção de portfólio |

### Portfólios pré-definidos

| Preset | Ativos | Descrição |
|---|---|---|
| `my50` | 50 | Portfólio balanceado em 11 setores (Tech, Financials, Healthcare, Energy, Defesa...) |
| `diversified-30` | 30 | 30 ações diversificadas do S&P 500 |
| `top20` | 20 | Top 20 por capitalização de mercado |
| `synthetic` | configurável | Dados sintéticos, sem necessidade de internet |

### Exemplos de comandos

```bash
# Portfólio personalizado
python main.py --tickers AAPL MSFT NVDA GOOGL AMZN --mode full

# Só benchmark, múltiplas escalas
python main.py --preset my50 --mode benchmark --n-sims 10000 100000 500000 1000000

# Só simulação, 500k trajetórias
python main.py --preset my50 --mode simulate --n-sims 500000

# Mais portfólios de treino no ML (mais preciso, mais lento)
python main.py --preset my50 --mode ml --n-portfolios 10000 --n-sims-per-portfolio 5000

# Janela temporal específica
python main.py --preset my50 --start 2018-01-01 --end 2024-12-31 --mode full
```

---

## Comportamento CPU vs GPU

O projeto detecta automaticamente se CUDA está disponível:

```
GPU   : disponível ✓      ← pycuda instalado + NVIDIA detectada
GPU   : não disponível    ← executa apenas em CPU (Mac, sem NVIDIA, sem pycuda)
```

Nenhuma alteração de código é necessária — a GPU é usada automaticamente quando disponível.

### Speedup esperado (referência)

| Configuração | CPU | GPU (estimativa) |
|---|---|---|
| 50 ativos, 100k sims | ~10s | ~0.3–1s |
| 50 ativos, 500k sims | ~50s | ~1–3s |
| 50 ativos, 1M sims | ~100s | ~2–6s |

> Os valores de GPU variam conforme o modelo (RTX 3080, A100, etc.).

---

## Saídas Geradas

Todos os arquivos são salvos em `results/` (excluído do git, gerado localmente):

| Arquivo | Conteúdo |
|---|---|
| `results/data/prices.csv` | Preços ajustados históricos |
| `results/data/log_returns.csv` | Retornos logarítmicos diários |
| `results/data/asset_stats.csv` | µ, σ e Sharpe individual por ativo |
| `results/benchmark_results.csv` | Tempos CPU vs GPU por escala |
| `results/feature_importance.csv` | Importância das features do modelo ML |
| `results/portfolio_comparison.csv` | ML vs baselines (1/N, Min Var, Max Sharpe) |
| `results/best_portfolio_weights.csv` | Pesos do portfólio selecionado |
| `results/dataset/` | Dataset de treino (features.parquet, labels.parquet, weights.npy) |

---

## Resultados de Referência (CPU, 50 ativos reais, 2020–2026)

Portfólio selecionado pelo ML vs baselines:

| Estratégia | Sharpe | Retorno | Volatilidade | P(perda) |
|---|---|---|---|---|
| **ML-Selecionado** | **1.54** | 18.2% | 8.6% | **0.99%** |
| 1/N (igualitário) | 0.61 | 9.9% | 8.1% | 10.2% |
| Mínima Variância | 0.58 | 8.7% | 6.3% | 7.9% |
| Máx Sharpe Ingênuo | 1.48 | 21.3% | 11.0% | 1.7% |

Modelo vencedor: **GradientBoosting** (R² = 0.978, MAE = 0.025 no Sharpe)

---

## Dependências Principais

| Biblioteca | Uso |
|---|---|
| `yahooquery` | Download de dados históricos |
| `numpy` / `pandas` | Manipulação de dados e álgebra linear |
| `scipy` | Testes estatísticos |
| `scikit-learn` | Random Forest e métricas |
| `xgboost` | Gradient Boosting |
| `pycuda` *(GPU)* | Interface Python para kernels CUDA |
