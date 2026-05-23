"""
eda.py — Análise Exploratória dos Dados (EDA)
=============================================
Funções para inspecionar e validar os dados históricos antes de
passá-los à simulação Monte Carlo.

Inclui:
    - Verificação de qualidade dos dados
    - Estatísticas descritivas detalhadas
    - Análise de correlação entre ativos
    - Validação das premissas do modelo GBM
      (normalidade dos retornos log, estacionariedade)
"""

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. QUALIDADE DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────

def check_data_quality(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica a qualidade dos dados de preços:
        - Valores faltantes
        - Preços negativos ou zero (inválidos)
        - Dias sem variação (possível erro de feed)

    Retorna DataFrame com o relatório de qualidade.
    """
    report = pd.DataFrame(index=prices.columns)

    report["total_dias"]     = len(prices)
    report["nulos"]          = prices.isna().sum()
    report["pct_nulos"]      = prices.isna().mean() * 100
    report["preco_min"]      = prices.min()
    report["preco_max"]      = prices.max()
    report["preco_invalido"] = (prices <= 0).sum()

    # Dias sem variação no preço
    sem_variacao = (prices.diff().iloc[1:] == 0).sum()
    report["dias_sem_variacao"] = sem_variacao

    return report


def validate_data(prices: pd.DataFrame, max_missing_pct: float = 1.0) -> bool:
    """
    Valida os dados de preços e lança avisos se encontrar problemas.

    Parâmetros
    ----------
    max_missing_pct : percentual máximo de dados faltantes tolerado (padrão 1%)

    Retorna True se os dados passaram em todas as verificações.
    """
    ok = True
    report = check_data_quality(prices)

    # Verifica dados faltantes
    high_missing = report[report["pct_nulos"] > max_missing_pct]
    if not high_missing.empty:
        print(f"[aviso] Ativos com >{ max_missing_pct}% de dados faltantes:")
        for ticker in high_missing.index:
            print(f"        {ticker}: {high_missing.loc[ticker, 'pct_nulos']:.2f}%")
        ok = False

    # Verifica preços inválidos
    invalid = report[report["preco_invalido"] > 0]
    if not invalid.empty:
        print(f"[erro] Ativos com preços <= 0:")
        for ticker in invalid.index:
            print(f"       {ticker}: {invalid.loc[ticker, 'preco_invalido']} ocorrências")
        ok = False

    if ok:
        print("[ok] Dados validados com sucesso.")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTATÍSTICAS DESCRITIVAS
# ─────────────────────────────────────────────────────────────────────────────

def describe_returns(log_returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Tabela detalhada de estatísticas dos retornos log anualizados.

    Colunas geradas:
        mu_anual      — retorno esperado anualizado
        sigma_anual   — volatilidade anualizada
        sharpe_bruto  — mu / sigma (sem taxa livre de risco)
        skewness      — assimetria (GBM assume ~0)
        kurtosis      — curtose em excesso (GBM assume ~0)
        min_diario    — pior dia no período
        max_diario    — melhor dia no período
    """
    mu_d    = log_returns.mean()
    sigma_d = log_returns.std()

    desc = pd.DataFrame(index=log_returns.columns)
    desc["mu_anual"]     = (mu_d * trading_days).map("{:.2%}".format)
    desc["sigma_anual"]  = (sigma_d * np.sqrt(trading_days)).map("{:.2%}".format)
    desc["sharpe_bruto"] = ((mu_d * trading_days) /
                            (sigma_d * np.sqrt(trading_days))).map("{:.3f}".format)
    desc["skewness"]     = log_returns.skew().map("{:.3f}".format)
    desc["kurtosis"]     = log_returns.kurt().map("{:.3f}".format)
    desc["min_diario"]   = log_returns.min().map("{:.2%}".format)
    desc["max_diario"]   = log_returns.max().map("{:.2%}".format)

    return desc


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANÁLISE DE CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def correlation_summary(log_returns: pd.DataFrame) -> dict:
    """
    Resumo da estrutura de correlação entre os ativos.

    Retorna dict com:
        matrix      — matriz de correlação completa
        mean_corr   — correlação média entre pares distintos
        max_corr    — par mais correlacionado
        min_corr    — par menos correlacionado
    """
    corr = log_returns.corr()
    n = len(corr)

    # Extrai apenas o triângulo superior (sem diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairs_corr = corr.values[mask]

    # Índices dos pares
    idx = np.argwhere(mask)
    tickers = log_returns.columns.tolist()

    # Par com correlação máxima e mínima
    max_idx = idx[np.argmax(pairs_corr)]
    min_idx = idx[np.argmin(pairs_corr)]

    return {
        "matrix":    corr,
        "mean_corr": pairs_corr.mean(),
        "max_corr":  {
            "pair":  (tickers[max_idx[0]], tickers[max_idx[1]]),
            "value": pairs_corr.max(),
        },
        "min_corr":  {
            "pair":  (tickers[min_idx[0]], tickers[min_idx[1]]),
            "value": pairs_corr.min(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDAÇÃO DAS PREMISSAS DO GBM
# ─────────────────────────────────────────────────────────────────────────────

def test_normality(log_returns: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Aplica o teste de Shapiro-Wilk para normalidade dos retornos log.

    O GBM assume que os retornos log são normalmente distribuídos.
    Na prática, ações apresentam caudas pesadas (leptocurtose), mas
    o modelo ainda é uma boa aproximação para horizontes curtos.

    Retorna DataFrame com estatística, p-valor e resultado do teste.
    """
    results = []
    for ticker in log_returns.columns:
        r = log_returns[ticker].dropna().values

        # Shapiro-Wilk é mais preciso para n < 5000
        # Para séries longas, usamos Jarque-Bera
        if len(r) <= 5000:
            stat, pval = stats.shapiro(r[:5000])  # limita para performance
            test_name = "Shapiro-Wilk"
        else:
            stat, pval = stats.jarque_bera(r)
            test_name = "Jarque-Bera"

        results.append({
            "ticker":    ticker,
            "teste":     test_name,
            "statistic": round(stat, 4),
            "p_valor":   round(pval, 4),
            "normal?":   "Sim" if pval > alpha else "Não",
        })

    return pd.DataFrame(results).set_index("ticker")


# ─────────────────────────────────────────────────────────────────────────────
# 5. RELATÓRIO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def full_eda_report(data: dict) -> None:
    """
    Executa e imprime o relatório completo de EDA para os dados preparados
    pelo fetcher.prepare_data().

    Parâmetros
    ----------
    data : dicionário retornado por fetcher.prepare_data()
    """
    prices      = data["prices"]
    log_returns = data["log_returns"]
    tickers     = data["tickers"]

    print("\n" + "="*60)
    print("  ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("="*60)

    # Período
    print(f"\n  Período  : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Ativos   : {len(tickers)}")
    print(f"  Obs.     : {len(log_returns)} dias úteis de retornos\n")

    # Qualidade
    print("─"*60)
    print("  QUALIDADE DOS DADOS")
    print("─"*60)
    quality = check_data_quality(prices)
    print(quality[["nulos", "pct_nulos", "preco_invalido", "dias_sem_variacao"]].to_string())

    # Estatísticas
    print("\n" + "─"*60)
    print("  ESTATÍSTICAS DOS RETORNOS (anualizados)")
    print("─"*60)
    desc = describe_returns(log_returns)
    print(desc.to_string())

    # Correlação
    print("\n" + "─"*60)
    print("  ESTRUTURA DE CORRELAÇÃO")
    print("─"*60)
    corr_summary = correlation_summary(log_returns)
    print(f"  Correlação média entre pares : {corr_summary['mean_corr']:.3f}")
    mc = corr_summary["max_corr"]
    mn = corr_summary["min_corr"]
    print(f"  Par mais correlacionado      : {mc['pair'][0]}-{mc['pair'][1]} ({mc['value']:.3f})")
    print(f"  Par menos correlacionado     : {mn['pair'][0]}-{mn['pair'][1]} ({mn['value']:.3f})")

    # Normalidade
    print("\n" + "─"*60)
    print("  TESTE DE NORMALIDADE DOS RETORNOS LOG")
    print("─"*60)
    normality = test_normality(log_returns)
    n_normal = (normality["normal?"] == "Sim").sum()
    print(f"  Ativos com distribuição normal (α=5%): {n_normal}/{len(tickers)}")
    print()
    print(normality.to_string())

    print("\n" + "="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(str(__import__("pathlib").Path(__file__).parent.parent))

    from data.fetcher import prepare_data
    from data.tickers import SP500_TOP20

    data = prepare_data(SP500_TOP20[:10], start="2020-01-01")
    full_eda_report(data)