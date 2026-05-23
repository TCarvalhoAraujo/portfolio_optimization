"""
synthetic.py
============
Gerador de dados sintéticos realistas para desenvolvimento e testes,
quando o acesso ao Yahoo Finance não está disponível (ex: ambiente offline,
CI/CD, notebooks sem internet).

Os dados sintéticos são gerados via GBM com parâmetros baseados em médias
históricas reais do S&P 500, garantindo distribuições estatisticamente
plausíveis para validar os simuladores Monte Carlo.

Uso:
    from data.synthetic import generate_synthetic_data
    data = generate_synthetic_data(n_assets=10, n_days=1000)
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PARÂMETROS DE REFERÊNCIA (baseados em médias históricas do S&P 500)
# ─────────────────────────────────────────────────────────────────────────────

# Parâmetros anualizados por setor (mu, sigma)
SECTOR_PARAMS = {
    "Technology":   {"mu_range": (0.12, 0.28), "sigma_range": (0.22, 0.42)},
    "Healthcare":   {"mu_range": (0.08, 0.18), "sigma_range": (0.16, 0.28)},
    "Financials":   {"mu_range": (0.07, 0.16), "sigma_range": (0.18, 0.30)},
    "Energy":       {"mu_range": (0.04, 0.15), "sigma_range": (0.24, 0.40)},
    "Consumer":     {"mu_range": (0.08, 0.18), "sigma_range": (0.18, 0.32)},
    "Industrials":  {"mu_range": (0.07, 0.15), "sigma_range": (0.17, 0.27)},
    "Utilities":    {"mu_range": (0.05, 0.10), "sigma_range": (0.12, 0.20)},
}

# Nomes de tickers sintéticos organizados por setor
SYNTHETIC_TICKERS = {
    "Technology":  ["TECH_A", "TECH_B", "TECH_C", "TECH_D", "TECH_E",
                    "TECH_F", "TECH_G", "TECH_H", "TECH_I", "TECH_J",
                    "TECH_K", "TECH_L", "TECH_M", "TECH_N", "TECH_O"],
    "Healthcare":  ["HLTH_A", "HLTH_B", "HLTH_C", "HLTH_D", "HLTH_E",
                    "HLTH_F", "HLTH_G", "HLTH_H"],
    "Financials":  ["FIN_A",  "FIN_B",  "FIN_C",  "FIN_D",  "FIN_E",
                    "FIN_F",  "FIN_G",  "FIN_H"],
    "Energy":      ["ENRG_A", "ENRG_B", "ENRG_C", "ENRG_D",
                    "ENRG_E", "ENRG_F", "ENRG_G"],
    "Consumer":    ["CONS_A", "CONS_B", "CONS_C", "CONS_D", "CONS_E",
                    "CONS_F", "CONS_G", "CONS_H"],
    "Industrials": ["IND_A",  "IND_B",  "IND_C",  "IND_D",
                    "IND_E",  "IND_F",  "IND_G"],
    "Utilities":   ["UTIL_A", "UTIL_B", "UTIL_C", "UTIL_D", "UTIL_E",
                    "UTIL_F", "UTIL_G"],
}


# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DE MATRIZ DE CORRELAÇÃO REALISTA
# ─────────────────────────────────────────────────────────────────────────────

def _build_correlation_matrix(
    sectors: list[str],
    intra_sector_corr: float = 0.55,
    inter_sector_corr: float = 0.25,
    market_factor: float = 0.30,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Gera uma matriz de correlação realista com estrutura setorial.

    Lógica:
        - Ativos do mesmo setor têm correlação mais alta (~0.55)
        - Ativos de setores diferentes têm correlação mais baixa (~0.25)
        - Um fator de mercado (beta) eleva todas as correlações
        - Ruído aleatório adiciona heterogeneidade

    Garante que a matriz seja definida positiva (necessário para Cholesky).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(sectors)
    corr = np.eye(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            base_corr = (
                intra_sector_corr if sectors[i] == sectors[j]
                else inter_sector_corr
            )
            # Adiciona fator de mercado e ruído
            noise = rng.uniform(-0.08, 0.08)
            c = np.clip(base_corr + market_factor * 0.5 + noise, -0.05, 0.95)
            corr[i, j] = c
            corr[j, i] = c

    # Garante definição positiva: soma lambda_min * I se necessário
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 1e-6:
        corr += (abs(eigvals.min()) + 1e-6) * np.eye(n)
        # Renormaliza para manter diagonal = 1
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

    return corr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DE PREÇOS VIA GBM
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_prices_gbm(
    mu: np.ndarray,
    sigma: np.ndarray,
    chol_lower: np.ndarray,
    n_days: int,
    s0: Optional[np.ndarray] = None,
    trading_days: int = 252,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simula trajetórias de preços via Movimento Browniano Geométrico (GBM)
    com correlação entre ativos (usando Cholesky).

    S_{t+1} = S_t * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*eps)
    onde eps = chol_lower @ Z,  Z ~ N(0, I)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_assets = len(mu)
    dt = 1.0 / trading_days

    if s0 is None:
        s0 = rng.uniform(50, 500, size=n_assets).astype(np.float32)

    prices = np.zeros((n_days + 1, n_assets), dtype=np.float32)
    prices[0] = s0

    drift = (mu / trading_days - 0.5 * (sigma / np.sqrt(trading_days)) ** 2)

    for t in range(1, n_days + 1):
        Z = rng.standard_normal(n_assets)
        eps = chol_lower @ Z                        # choques correlacionados
        log_ret = drift + (sigma / np.sqrt(trading_days)) * eps
        prices[t] = prices[t - 1] * np.exp(log_ret)

    return prices[1:]  # remove o ponto inicial


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_assets: int = 20,
    n_days: int = 1000,
    start_date: str = "2020-01-01",
    trading_days: int = 252,
    seed: int = 42,
) -> dict:
    """
    Gera dados sintéticos realistas no mesmo formato que fetcher.prepare_data().

    Parâmetros
    ----------
    n_assets     : número de ativos a simular
    n_days       : número de dias úteis
    start_date   : data inicial (para construir o índice do DataFrame)
    trading_days : dias úteis por ano (padrão 252)
    seed         : semente para reproducibilidade

    Retorna
    -------
    dict com as mesmas chaves de fetcher.prepare_data(), prontas para uso
    nos módulos de simulação, EDA e benchmark.
    """
    rng = np.random.default_rng(seed)
    print(f"[synthetic] Gerando dados para {n_assets} ativos | {n_days} dias")

    # ── 1. Seleciona tickers e setores ───────────────────────────────────────
    all_tickers  = []
    all_sectors  = []
    all_mu       = []
    all_sigma    = []

    # Distribui ativos entre setores de forma proporcional
    sector_names  = list(SYNTHETIC_TICKERS.keys())
    sector_counts = np.zeros(len(sector_names), dtype=int)
    base = n_assets // len(sector_names)
    extra = n_assets % len(sector_names)

    for i in range(len(sector_names)):
        sector_counts[i] = base + (1 if i < extra else 0)

    for idx, (sector, count) in enumerate(zip(sector_names, sector_counts)):
        if count == 0:
            continue
        available = SYNTHETIC_TICKERS[sector]
        chosen    = available[:count]

        params = SECTOR_PARAMS[sector]
        for ticker in chosen:
            all_tickers.append(ticker)
            all_sectors.append(sector)
            mu_val    = rng.uniform(*params["mu_range"])
            sigma_val = rng.uniform(*params["sigma_range"])
            all_mu.append(mu_val)
            all_sigma.append(sigma_val)

    n_actual = len(all_tickers)
    mu       = np.array(all_mu,    dtype=np.float32)
    sigma    = np.array(all_sigma, dtype=np.float32)

    # ── 2. Matriz de correlação e Cholesky ───────────────────────────────────
    corr_matrix = _build_correlation_matrix(all_sectors, rng=rng)
    cov_matrix  = corr_matrix * np.outer(sigma, sigma)

    # Garante definição positiva
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if eigvals.min() < 1e-8:
        cov_matrix += (abs(eigvals.min()) + 1e-8) * np.eye(n_actual, dtype=np.float32)

    chol_lower = np.linalg.cholesky(cov_matrix).astype(np.float32)

    # ── 3. Simula preços ─────────────────────────────────────────────────────
    price_array = _simulate_prices_gbm(
        mu, sigma, chol_lower, n_days, trading_days=trading_days, rng=rng
    )

    # ── 4. Constrói DataFrames ───────────────────────────────────────────────
    # Gera índice de datas (apenas dias úteis)
    start    = pd.Timestamp(start_date)
    biz_days = pd.bdate_range(start=start, periods=n_days)
    prices_df      = pd.DataFrame(price_array,      index=biz_days, columns=all_tickers)
    log_returns_df = (np.log(prices_df / prices_df.shift(1))).dropna()

    # ── 5. Monta dicionário de saída ─────────────────────────────────────────
    data = {
        "tickers":     all_tickers,
        "prices":      prices_df,
        "log_returns": log_returns_df,
        "mu":          mu,
        "sigma":       sigma,
        "cov_matrix":  cov_matrix,
        "chol_lower":  chol_lower,
        "n_assets":    n_actual,
        "n_days":      len(log_returns_df),
        "_synthetic":  True,   # flag para indicar dados sintéticos
    }

    print(f"[synthetic] {n_actual} ativos | {len(log_returns_df)} dias de retornos gerados")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(str(__import__("pathlib").Path(__file__).parent.parent))

    from data.fetcher import print_summary

    data = generate_synthetic_data(n_assets=20, n_days=1200, seed=42)
    print_summary(data)

    # Valida Cholesky
    L   = data["chol_lower"]
    Cov = data["cov_matrix"]
    err = np.max(np.abs(L @ L.T - Cov))
    print(f"  Erro Cholesky: {err:.2e}  {'✓' if err < 1e-3 else '✗'}")