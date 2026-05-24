"""
fetcher.py
==========
Módulo responsável pela coleta e pré-processamento de dados históricos
de ações do mercado americano via yahooquery.

Usamos yahooquery em vez de yfinance por ser mais robusto a rate limits:
ele utiliza a API v8 do Yahoo Finance com suporte a sessões persistentes,
retries automáticos e requisições em lote — reduzindo drasticamente
o risco de bloqueio por excesso de chamadas.

Fluxo:
    1. Download de preços ajustados de fechamento (adjclose)
    2. Cálculo de retornos logarítmicos diários
    3. Cálculo de estatísticas descritivas (mu, sigma)
    4. Geração da matriz de covariância e sua decomposição de Cholesky
       (necessária para correlacionar os ativos na simulação Monte Carlo)
    5. Exportação dos dados prontos para as simulações
"""

import hashlib
import numpy as np
import pandas as pd
from yahooquery import Ticker
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ── Diretório de cache para não redownlodar toda vez ──────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD DE PREÇOS
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
    max_missing_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Baixa preços ajustados de fechamento para uma lista de tickers via yahooquery.

    Diferenças em relação ao yfinance:
        - Usa a API v8 do Yahoo Finance (mais estável, menos rate limiting)
        - Faz requisições em lote por ticker (menos round-trips)
        - Suporta retries automáticos via parâmetro retry da sessão
        - Retorna `adjclose` (preço ajustado) diretamente

    Parâmetros
    ----------
    tickers         : lista de símbolos (ex: ["AAPL", "MSFT", "GOOGL"])
    start           : data inicial no formato "YYYY-MM-DD"
    end             : data final no formato "YYYY-MM-DD" (None = hoje)
    use_cache       : se True, salva/lê um CSV local para evitar downloads repetidos
    max_missing_pct : fracção máxima de NaN tolerada por ticker (padrão: 5%)

    Retorna
    -------
    DataFrame com datas no índice e tickers nas colunas (adjclose)
    """
    ticker_hash = hashlib.md5("_".join(sorted(tickers)).encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"prices_{ticker_hash}_{start}.csv"

    if use_cache and cache_file.exists():
        print(f"[cache] Lendo preços de {cache_file.name}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print(f"[download] Baixando {len(tickers)} ações de {start} até {end or 'hoje'}...")

    t = Ticker(
        tickers,
        asynchronous=False,   # síncrono: mais seguro contra rate limits
        timeout=30,
    )

    period_kwargs = {"start": start}
    if end:
        period_kwargs["end"] = end

    raw = t.history(interval="1d", **period_kwargs)

    if isinstance(raw, dict):
        # Erro global: nenhum dado retornado
        raise RuntimeError(f"yahooquery retornou erro: {raw}")

    # raw é um DataFrame com MultiIndex (symbol, date) — pivotamos para (date, symbol)
    raw = raw.reset_index()

    # Identifica a coluna de data (pode ser 'date' ou 'timestamp')
    date_col = "date" if "date" in raw.columns else "timestamp"
    raw[date_col] = pd.to_datetime(raw[date_col]).dt.tz_localize(None).dt.normalize()

    # Pivota: linhas = datas, colunas = tickers, valores = adjclose
    prices = raw.pivot(index=date_col, columns="symbol", values="adjclose")
    prices.index.name = "Date"
    prices = prices.sort_index()

    # Mantém apenas os tickers solicitados que estão presentes
    available = [tk for tk in tickers if tk in prices.columns]
    missing_tickers = set(tickers) - set(available)
    if missing_tickers:
        print(f"[aviso] Tickers não encontrados: {sorted(missing_tickers)}")
    prices = prices[available]

    # Remove tickers com dados insuficientes
    missing_ratio = prices.isna().mean()
    dropped = missing_ratio[missing_ratio > max_missing_pct].index.tolist()
    if dropped:
        print(f"[aviso] Removendo tickers com >{max_missing_pct:.0%} de NaN: {dropped}")
        prices.drop(columns=dropped, inplace=True)

    prices.dropna(inplace=True)

    if use_cache:
        prices.to_csv(cache_file)
        print(f"[cache] Preços salvos em {cache_file.name}")

    print(f"[ok] {len(prices.columns)} ativos | {len(prices)} dias úteis carregados")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. RETORNOS LOGARÍTMICOS
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos logarítmicos diários: r_t = ln(P_t / P_{t-1})

    Retornos log são preferidos porque:
    - São aditivos no tempo (propriedade matemática útil para GBM)
    - Aproximam retornos simples para variações pequenas
    - Têm melhor comportamento estatístico (simetria)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


# ─────────────────────────────────────────────────────────────────────────────
# 3. ESTATÍSTICAS DESCRITIVAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(
    log_returns: pd.DataFrame,
    trading_days: int = 252,
) -> dict:
    """
    Calcula mu (retorno esperado anualizado) e sigma (volatilidade anualizada)
    para cada ativo — parâmetros de entrada do modelo GBM.

    Fórmulas (anualizadas):
        mu_anual    = média_diária * trading_days
        sigma_anual = desvio_diário * sqrt(trading_days)

    Parâmetros
    ----------
    trading_days : número de dias úteis por ano (padrão: 252)

    Retorna
    -------
    dict com 'mu' e 'sigma' como arrays numpy (shape: [n_assets])
    """
    mu_daily    = log_returns.mean()
    sigma_daily = log_returns.std()

    mu_annual    = mu_daily    * trading_days
    sigma_annual = sigma_daily * np.sqrt(trading_days)

    return {
        "mu":     mu_annual.values.astype(np.float32),
        "sigma":  sigma_annual.values.astype(np.float32),
        "tickers": log_returns.columns.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATRIZ DE COVARIÂNCIA E DECOMPOSIÇÃO DE CHOLESKY
# ─────────────────────────────────────────────────────────────────────────────

def compute_cholesky(
    log_returns: pd.DataFrame,
    trading_days: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula a matriz de covariância anualizada e sua fatoração de Cholesky.

    Por que precisamos de Cholesky?
    --------------------------------
    Na simulação de Monte Carlo com múltiplos ativos correlacionados,
    precisamos gerar vetores aleatórios Z ~ N(0, Σ) onde Σ é a covariância.
    Usando Cholesky: L = chol(Σ)  →  Y = L @ Z  →  Y ~ N(0, Σ)

    Cada thread CUDA receberá um vetor Z independente e calculará Y = L @ Z
    para obter os choques correlacionados entre os ativos.

    Retorna
    -------
    cov_matrix : matriz de covariância (n_assets x n_assets), float32
    chol_lower : matriz triangular inferior L  (n_assets x n_assets), float32
    """
    cov_daily  = log_returns.cov().values
    cov_annual = cov_daily * trading_days

    # np.linalg.cholesky retorna a triangular inferior L tal que L @ L.T = Σ
    chol_lower = np.linalg.cholesky(cov_annual)

    return (
        cov_annual.astype(np.float32),
        chol_lower.astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE COMPLETO — função de entrada principal
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    tickers: list[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
    trading_days: int = 252,
) -> dict:
    """
    Pipeline completo de coleta e pré-processamento.

    Retorna um dicionário pronto para ser consumido pelo simulador:

        {
            "tickers"     : lista de símbolos usados,
            "prices"      : DataFrame de preços,
            "log_returns" : DataFrame de retornos log,
            "mu"          : array float32 [n_assets] — retorno anualizado,
            "sigma"       : array float32 [n_assets] — volatilidade anualizada,
            "cov_matrix"  : array float32 [n_assets, n_assets],
            "chol_lower"  : array float32 [n_assets, n_assets],
            "n_assets"    : int,
            "n_days"      : int,
        }
    """
    # 1. Preços
    prices = download_prices(tickers, start=start, end=end, use_cache=use_cache)

    # 2. Retornos
    log_returns = compute_log_returns(prices)

    # 3. Estatísticas
    stats = compute_statistics(log_returns, trading_days)

    # 4. Covariância e Cholesky
    cov_matrix, chol_lower = compute_cholesky(log_returns, trading_days)

    # Tickers efetivos (podem ter sido filtrados por dados insuficientes)
    effective_tickers = stats["tickers"]

    data = {
        "tickers":     effective_tickers,
        "prices":      prices[effective_tickers],
        "log_returns": log_returns[effective_tickers],
        "mu":          stats["mu"],
        "sigma":       stats["sigma"],
        "cov_matrix":  cov_matrix,
        "chol_lower":  chol_lower,
        "n_assets":    len(effective_tickers),
        "n_days":      len(log_returns),
    }

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 6. UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def export_to_csv(data: dict, output_dir: Path) -> None:
    """
    Exporta os dados coletados do yahooquery em arquivos CSV legíveis.

    Arquivos gerados em output_dir/data/:
        prices.csv       — preços ajustados de fechamento (datas × tickers)
        log_returns.csv  — retornos logarítmicos diários
        asset_stats.csv  — mu, sigma e Sharpe individual por ativo
    """
    out = Path(output_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)

    # Preços
    data["prices"].to_csv(out / "prices.csv")

    # Retornos logarítmicos
    data["log_returns"].to_csv(out / "log_returns.csv")

    # Estatísticas por ativo
    rf = 0.05
    stats_df = pd.DataFrame({
        "ticker": data["tickers"],
        "mu_annual":    data["mu"],
        "sigma_annual": data["sigma"],
        "sharpe_naive": (data["mu"] - rf) / (data["sigma"] + 1e-8),
    }).set_index("ticker")
    stats_df.to_csv(out / "asset_stats.csv")

    print(f"[csv] Dados exportados para {out}/")
    print(f"      prices.csv       ({data['prices'].shape[0]} dias × {data['n_assets']} ativos)")
    print(f"      log_returns.csv  ({data['log_returns'].shape[0]} observações)")
    print(f"      asset_stats.csv  ({data['n_assets']} ativos)")


def print_summary(data: dict) -> None:
    """Exibe um resumo dos dados carregados."""
    n  = data["n_assets"]
    nd = data["n_days"]
    print("\n" + "="*55)
    print("  RESUMO DOS DADOS CARREGADOS")
    print("="*55)
    print(f"  Ativos       : {n}")
    print(f"  Dias úteis   : {nd}")
    print(f"  Tickers      : {', '.join(data['tickers'])}")
    print()
    print(f"  {'Ticker':<8} {'mu (a.a.)':<14} {'sigma (a.a.)':<14}")
    print(f"  {'-'*36}")
    for i, t in enumerate(data["tickers"]):
        print(f"  {t:<8} {data['mu'][i]:>10.2%}     {data['sigma'][i]:>10.2%}")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA — teste rápido do módulo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Portfólio de exemplo: 10 grandes empresas do S&P 500
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM",   "JNJ",  "V",
    ]

    data = prepare_data(
        tickers=TICKERS,
        start="2020-01-01",
        use_cache=True,
    )

    print_summary(data)

    # Verificação da decomposição de Cholesky: L @ L.T deve ≈ Σ
    L   = data["chol_lower"]
    Cov = data["cov_matrix"]
    err = np.max(np.abs(L @ L.T - Cov))
    print(f"  Erro Cholesky (max |L @ L.T - Σ|): {err:.2e}")
    print("  (deve ser próximo de zero)\n")