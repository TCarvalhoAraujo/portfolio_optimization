"""
optimization/views.py
=====================
Factories de views quantitativas para o modelo Black-Litterman.

Conceito de View no BL
-----------------------
Uma view é uma afirmação probabilística sobre retornos futuros.
Matematicamente: P · μ = Q + ε, com ε ~ N(0, Ω).

    P  = matriz de seleção (K × N)  — quais ativos e com que sinal
    Q  = vetor de retornos esperados (K,) — o "quanto" esperado
    Ω  = matriz de incerteza (K × K) diagonal — o "quão confiante"

Views relativas (A supera B):
    P[k, idx_A] = +1,  P[k, idx_B] = -1
    Q[k] = retorno diferencial esperado (ex: 0.05 = 5% a.a.)

Views absolutas (A retorna X%):
    P[k, idx_A] = +1
    Q[k] = retorno absoluto esperado

Módulo implementado
-------------------
momentum_views()         — Momentum 12-1: top vs bottom por retorno acumulado
sector_momentum_views()  — Momentum por setor: melhor setor vs pior setor

Ambas retornam List[View] compatíveis com BlackLittermanOptimizer.optimize().

Como calibrar a confiança
--------------------------
A confiança controla o quanto o posterior se afasta do prior de equilíbrio.
    confidence = 0.25  → view fraca, posterior próximo do prior
    confidence = 0.50  → view moderada (recomendado para começar)
    confidence = 0.75  → view forte, posterior se move muito

Uma heurística razoável: use o R² de uma regressão do retorno passado
sobre o retorno futuro como proxy de confiança. Para momentum de ações
americanas, valores típicos ficam entre 0.05 e 0.20 — logo confidence
entre 0.25 e 0.50 é bem calibrado. Não use confidence > 0.75 a não ser
que você tenha evidência sólida.

Referências
-----------
Carhart (1997) — On Persistence in Mutual Fund Performance
Jegadeesh & Titman (1993) — Returns to Buying Winners and Selling Losers
Idzorek (2005) — A Step-by-Step Guide to the Black-Litterman Model
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from .black_litterman import View


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _cumulative_return(returns: pd.DataFrame, window: int) -> pd.Series:
    """
    Retorno acumulado de cada ativo nos últimos `window` dias.

    Usa retornos log: r_acum = Σ r_t  (equivalente a ln(P_T / P_0))

    Parâmetros
    ----------
    returns : pd.DataFrame
        Retornos log diários. Shape (T, N).
    window : int
        Número de dias úteis a considerar.

    Retorna
    -------
    pd.Series com retorno acumulado por ticker.
    """
    if len(returns) < window:
        window = len(returns)
    return returns.iloc[-window:].sum()


def _annualize_return(log_return: float, window: int, trading_days: int = 252) -> float:
    """
    Converte retorno log acumulado em janela para taxa anualizada.

    r_anual = r_acumulado * (trading_days / window)

    Parâmetros
    ----------
    log_return  : retorno log acumulado no período
    window      : número de dias úteis na janela
    trading_days: dias úteis por ano

    Retorna
    -------
    float — retorno anualizado
    """
    return log_return * (trading_days / window)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — MOMENTUM 12-1
# ─────────────────────────────────────────────────────────────────────────────

def momentum_views(
    returns: pd.DataFrame,
    n_winners: int          = 5,
    n_losers: int           = 5,
    lookback_days: int      = 252,
    skip_days: int          = 21,
    return_differential: float = 0.05,
    confidence: float       = 0.40,
    trading_days: int       = 252,
    verbose: bool           = True,
) -> List[View]:
    """
    Momentum 12-1: winners recentes vão superar losers recentes.

    Conceito:
        Calcula o retorno acumulado de cada ativo nos últimos
        lookback_days dias, pulando os skip_days mais recentes
        (evita reversão de curto prazo documentada na literatura).
        Os n_winners com maior retorno recebem view positiva contra
        os n_losers com menor retorno.

        Cada view é relativa: "winner_i supera loser_j em X% ao ano".
        Isso produz n_winners × n_losers views independentes.

    Por que pular os últimos 21 dias?
        Retornos de momentum têm reversão no horizonte de 1 mês.
        Ativos que subiram muito na última semana tendem a corrigir.
        O "12-1" refere-se a usar 12 meses de histórico mas ignorar
        o mês mais recente — convenção padrão na literatura de fatores.

    Por que views relativas (e não absolutas)?
        Views relativas são mais robustas: você está apostando em
        diferença de desempenho, não em nível absoluto. Se o mercado
        cair, winners e losers caem — mas winners caem menos.

    Parâmetros
    ----------
    returns : pd.DataFrame
        Retornos log diários. Deve ter pelo menos lookback_days linhas.
    n_winners : int
        Número de ativos top (winners) por view. Default: 5.
    n_losers : int
        Número de ativos bottom (losers) por view. Default: 5.
    lookback_days : int
        Janela de lookback em dias úteis. Default: 252 (≈ 12 meses).
    skip_days : int
        Dias mais recentes a ignorar (reversão de curto prazo).
        Default: 21 (≈ 1 mês).
    return_differential : float
        Retorno diferencial anualizado esperado entre winner e loser.
        Default: 0.05 (5% a.a.) — conservador para ações americanas.
    confidence : float
        Confiança na view. ∈ (0, 1]. Default: 0.40.
        Momentum tem evidência forte mas não determinística.
    trading_days : int
        Dias úteis por ano para anualização.
    verbose : bool
        Se True, imprime winners e losers identificados.

    Retorna
    -------
    List[View]
        Uma view por par (winner, loser).
        Tamanho: n_winners × n_losers views.

    Exemplo
    -------
    >>> views = momentum_views(returns, n_winners=3, n_losers=3)
    >>> # 9 views: cada winner vs cada loser
    """
    if len(returns) < lookback_days + skip_days:
        available = len(returns) - skip_days
        if available < 60:
            raise ValueError(
                f"Poucos dados para momentum: {len(returns)} dias disponíveis, "
                f"mínimo recomendado = lookback_days + skip_days = "
                f"{lookback_days + skip_days}."
            )
        lookback_days = available
        if verbose:
            print(f"[views] momentum: lookback ajustado para {lookback_days} dias "
                  f"(dados insuficientes para {lookback_days + skip_days})")

    # Remove os skip_days mais recentes, depois pega lookback_days
    window_returns = returns.iloc[-(lookback_days + skip_days):-skip_days]
    cum_ret        = window_returns.sum()              # retorno log acumulado
    cum_ret_annual = cum_ret * (trading_days / lookback_days)

    # Classifica ativos por retorno acumulado
    ranked = cum_ret_annual.sort_values(ascending=False)
    winners = ranked.index[:n_winners].tolist()
    losers  = ranked.index[-n_losers:].tolist()

    if verbose:
        print(f"\n[views] Momentum {lookback_days // 21:.0f}m-{skip_days // 21:.0f}m:")
        print(f"  Winners ({n_winners}): " +
              "  ".join(f"{t}({ranked[t]:+.1%})" for t in winners))
        print(f"  Losers  ({n_losers}): " +
              "  ".join(f"{t}({ranked[t]:+.1%})" for t in losers))

    views = []
    for winner in winners:
        for loser in losers:
            views.append(View(
                assets         = [winner, loser],
                weights        = [1.0, -1.0],
                expected_return= return_differential,
                confidence     = confidence,
                name           = f"mom: {winner} > {loser}",
            ))

    if verbose:
        print(f"  → {len(views)} views geradas "
              f"(diferencial={return_differential:.1%}, "
              f"confiança={confidence:.0%})")

    return views


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — MOMENTUM SETORIAL
# ─────────────────────────────────────────────────────────────────────────────

def sector_momentum_views(
    returns: pd.DataFrame,
    sector_map: dict,
    n_top_sectors: int      = 2,
    n_bottom_sectors: int   = 2,
    lookback_days: int      = 126,
    skip_days: int          = 21,
    return_differential: float = 0.04,
    confidence: float       = 0.50,
    trading_days: int       = 252,
    verbose: bool           = True,
) -> List[View]:
    """
    Momentum setorial: melhores setores vão superar piores setores.

    Conceito:
        Agrega retornos por setor (média simples dos ativos do setor),
        calcula momentum de cada setor, e cria views do tipo:
        "ativo top_sector > ativo bottom_sector".

        As views são construídas de forma especial: para representar
        "setor A supera setor B", cria-se uma view onde:
            P[k, ativos_setor_A] = +1/n_A  (peso médio no setor A)
            P[k, ativos_setor_B] = -1/n_B  (peso médio no setor B)

        Isso é mais correto matematicamente que criar uma view por
        ativo individual, porque o posterior BL atualiza μ de todos
        os ativos do setor proporcionalmente.

    Por que usar lookback menor (126 dias = 6 meses)?
        Momentum setorial tem menor autocorrelação que momentum
        individual — o sinal decai mais rápido. Janelas de 3–6 meses
        costumam funcionar melhor para setores do que as 12 meses
        tradicionais de momentum individual.

    Parâmetros
    ----------
    returns : pd.DataFrame
        Retornos log diários.
    sector_map : dict
        {ticker: setor}. Tickers não mapeados são ignorados.
    n_top_sectors : int
        Número de setores top. Default: 2.
    n_bottom_sectors : int
        Número de setores bottom. Default: 2.
    lookback_days : int
        Janela de lookback em dias úteis. Default: 126 (≈ 6 meses).
    skip_days : int
        Dias recentes a ignorar. Default: 21.
    return_differential : float
        Retorno diferencial anualizado esperado.
        Default: 0.04 (4% a.a.) — menor que momentum individual
        porque o sinal é diluído pela agregação.
    confidence : float
        Confiança na view. Default: 0.50.
    trading_days : int
        Dias úteis por ano.
    verbose : bool
        Imprime setores identificados.

    Retorna
    -------
    List[View]
        Uma view por par (setor_top_i, setor_bottom_j).
        Tamanho: n_top_sectors × n_bottom_sectors views.

    Exemplo
    -------
    >>> from data.tickers import SECTORS
    >>> sector_map = {t: s for s, ts in SECTORS.items() for t in ts}
    >>> views = sector_momentum_views(returns, sector_map)
    """
    if len(returns) < lookback_days + skip_days:
        available = len(returns) - skip_days
        lookback_days = max(60, available)
        if verbose:
            print(f"[views] setor_momentum: lookback ajustado para {lookback_days} dias")

    # Mapeia tickers disponíveis para setores
    available_tickers = set(returns.columns)
    ticker_to_sector  = {t: s for t, s in sector_map.items()
                         if t in available_tickers}

    if not ticker_to_sector:
        raise ValueError(
            "sector_map não contém nenhum ticker disponível em returns. "
            "Verifique se os tickers do sector_map coincidem com returns.columns."
        )

    # Agrupa tickers por setor
    sectors_to_tickers: dict[str, list] = {}
    for ticker, sector in ticker_to_sector.items():
        sectors_to_tickers.setdefault(sector, []).append(ticker)

    # Remove setores com apenas 1 ativo (não representativos)
    sectors_to_tickers = {
        s: ts for s, ts in sectors_to_tickers.items() if len(ts) >= 2
    }

    if len(sectors_to_tickers) < 2:
        raise ValueError(
            f"Apenas {len(sectors_to_tickers)} setor(es) com ≥2 ativos disponíveis. "
            "Precisa de pelo menos 2 setores para views relativas."
        )

    # Calcula retorno setorial: média simples dos retornos dos ativos
    window_returns = returns.iloc[-(lookback_days + skip_days):-skip_days]
    sector_returns = {}
    for sector, tickers in sectors_to_tickers.items():
        valid = [t for t in tickers if t in window_returns.columns]
        if valid:
            sector_returns[sector] = window_returns[valid].mean(axis=1).sum()
            sector_returns[sector] *= (trading_days / lookback_days)  # anualiza

    sector_series = pd.Series(sector_returns).sort_values(ascending=False)

    top_sectors    = sector_series.index[:n_top_sectors].tolist()
    bottom_sectors = sector_series.index[-n_bottom_sectors:].tolist()

    if verbose:
        print(f"\n[views] Momentum setorial ({lookback_days // 21:.0f}m):")
        for s in top_sectors:
            tks = sectors_to_tickers.get(s, [])
            print(f"  ↑ {s:<25} {sector_series[s]:+.1%}  "
                  f"({len(tks)} ativos: {', '.join(tks[:4])}"
                  f"{'...' if len(tks) > 4 else ''})")
        for s in bottom_sectors:
            tks = sectors_to_tickers.get(s, [])
            print(f"  ↓ {s:<25} {sector_series[s]:+.1%}  "
                  f"({len(tks)} ativos: {', '.join(tks[:4])}"
                  f"{'...' if len(tks) > 4 else ''})")

    views = []
    for top_sector in top_sectors:
        for bottom_sector in bottom_sectors:
            top_tickers    = sectors_to_tickers[top_sector]
            bottom_tickers = sectors_to_tickers[bottom_sector]

            # Pesos normalizados por setor: +1/n_top e -1/n_bottom
            assets  = top_tickers + bottom_tickers
            weights = (
                [1.0 / len(top_tickers)]    * len(top_tickers) +
                [-1.0 / len(bottom_tickers)] * len(bottom_tickers)
            )

            # Filtra apenas tickers presentes no DataFrame de retornos
            valid_pairs = [
                (a, w) for a, w in zip(assets, weights)
                if a in available_tickers
            ]
            if not valid_pairs:
                continue

            valid_assets, valid_weights = zip(*valid_pairs)

            views.append(View(
                assets         = list(valid_assets),
                weights        = list(valid_weights),
                expected_return= return_differential,
                confidence     = confidence,
                name           = f"sector_mom: {top_sector} > {bottom_sector}",
            ))

    if verbose:
        print(f"  → {len(views)} views setoriais geradas "
              f"(diferencial={return_differential:.1%}, "
              f"confiança={confidence:.0%})")

    return views


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY PRINCIPAL — combina as duas fontes de views
# ─────────────────────────────────────────────────────────────────────────────

def build_views(
    returns: pd.DataFrame,
    method: str                         = "momentum",
    sector_map: Optional[dict]          = None,
    momentum_kwargs: Optional[dict]     = None,
    sector_momentum_kwargs: Optional[dict] = None,
    verbose: bool                       = True,
) -> List[View]:
    """
    Factory principal de views para Black-Litterman.

    Métodos disponíveis:
        "momentum"         — Momentum 12-1 individual
        "sector_momentum"  — Momentum setorial
        "combined"         — Ambos combinados (recomendado)

    Parâmetros
    ----------
    returns : pd.DataFrame
        Retornos log diários.
    method : str
        "momentum" | "sector_momentum" | "combined"
    sector_map : dict ou None
        {ticker: setor}. Necessário para "sector_momentum" e "combined".
    momentum_kwargs : dict ou None
        Argumentos opcionais para momentum_views().
    sector_momentum_kwargs : dict ou None
        Argumentos opcionais para sector_momentum_views().
    verbose : bool

    Retorna
    -------
    List[View]

    Exemplos
    --------
    # Momentum simples
    views = build_views(returns, method="momentum")

    # Setorial
    views = build_views(returns, method="sector_momentum",
                        sector_map=sector_map)

    # Combinado (recomendado): usa os dois, confiança ligeiramente menor
    # para evitar overconfidence com views redundantes
    views = build_views(returns, method="combined",
                        sector_map=sector_map,
                        momentum_kwargs={"confidence": 0.35},
                        sector_momentum_kwargs={"confidence": 0.45})
    """
    mom_kw  = momentum_kwargs        or {}
    sect_kw = sector_momentum_kwargs or {}

    if method == "momentum":
        return momentum_views(returns, verbose=verbose, **mom_kw)

    elif method == "sector_momentum":
        if sector_map is None:
            raise ValueError(
                "sector_map é obrigatório para method='sector_momentum'. "
                "Use SECTORS do tickers.py: "
                "sector_map = {t: s for s, ts in SECTORS.items() for t in ts}"
            )
        return sector_momentum_views(
            returns, sector_map, verbose=verbose, **sect_kw
        )

    elif method == "combined":
        views = []
        views += momentum_views(returns, verbose=verbose, **mom_kw)
        if sector_map is not None:
            views += sector_momentum_views(
                returns, sector_map, verbose=verbose, **sect_kw
            )
        elif verbose:
            print("[views] combined: sector_map não fornecido, "
                  "usando apenas momentum individual.")
        return views

    else:
        raise ValueError(
            f"method='{method}' inválido. "
            "Use 'momentum', 'sector_momentum' ou 'combined'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNÓSTICO — inspeciona as views geradas
# ─────────────────────────────────────────────────────────────────────────────

def views_summary(views: List[View]) -> pd.DataFrame:
    """
    Tabela de diagnóstico das views geradas.

    Útil para inspecionar o que será passado ao BL antes de otimizar.

    Parâmetros
    ----------
    views : List[View]

    Retorna
    -------
    pd.DataFrame com colunas:
        name, assets_long, assets_short, expected_return, confidence
    """
    rows = []
    for v in views:
        long_assets  = [a for a, w in zip(v.assets, v.weights) if w > 0]
        short_assets = [a for a, w in zip(v.assets, v.weights) if w < 0]
        rows.append({
            "name"           : v.name or "—",
            "long"           : " + ".join(long_assets),
            "short"          : " - ".join(short_assets),
            "expected_return": f"{v.expected_return:.1%}",
            "confidence"     : f"{v.confidence:.0%}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA — teste
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.fetcher import prepare_data
    from data.tickers import MY_DIVERSIFIED_50, SECTORS

    sector_map = {t: s for s, ts in SECTORS.items() for t in ts}

    print("[test] Carregando dados...")
    data    = prepare_data(MY_DIVERSIFIED_50, start="2020-01-01")
    returns = data["log_returns"]

    print("\n--- Momentum 12-1 ---")
    v1 = momentum_views(returns, n_winners=5, n_losers=5)
    print(views_summary(v1).to_string(index=False))

    print("\n--- Momentum Setorial ---")
    v2 = sector_momentum_views(returns, sector_map, n_top_sectors=2, n_bottom_sectors=2)
    print(views_summary(v2).to_string(index=False))

    print("\n--- Combined ---")
    v3 = build_views(returns, method="combined", sector_map=sector_map)
    print(f"Total de views: {len(v3)}")