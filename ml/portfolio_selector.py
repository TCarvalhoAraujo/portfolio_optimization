"""
portfolio_selector.py
=====================
Treinamento e uso do modelo de ML para seleção de portfólios eficientes.

PIPELINE DE ML
==============
    1. build_dataset()         → X (features), y (labels), W (pesos)
    2. train()                 → ajusta Random Forest + XGBoost; seleciona melhor
    3. evaluate()              → métricas de regressão + importância de features
    4. select_best_portfolio() → gera candidatos, prediz Sharpe, retorna o melhor
    5. compare_with_baselines()→ confronta ML vs todos os otimizadores

INTEGRAÇÃO COM O MÓDULO DE OTIMIZAÇÃO
======================================
compare_with_baselines() agora inclui automaticamente todos os
otimizadores do módulo optimization/ como baselines de comparação:

    1/N (igualitário)            ← baseline trivial
    ML-Selecionado               ← busca por espaço amostrado
    MinVariance                  ← apenas Σ, sem μ
    RiskParity (ERC)             ← contribuição igual de risco
    HRP (ward linkage)           ← clusterização hierárquica
    Markowitz (Max Sharpe)       ← μ + Σ, otimização clássica
    RobustOptimizer              ← Markowitz com penalty de incerteza
    BlackLitterman (sem views)   ← prior de mercado apenas

Cada otimizador usa a interface comum .optimize(returns, constraints).
Os pesos são simulados via Monte Carlo para métricas reais comparáveis.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Dict
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

from ml.dataset import build_dataset, sample_weights, extract_features
from simulation.monte_carlo_cpu import simulate_vectorized
from portfolio.metrics import (
    sharpe_ratio, var_historical, cvar_historical, compute_metrics
)

# Importa todos os otimizadores
from optimization import (
    MarkowitzOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    HRPOptimizer,
    BlackLittermanOptimizer,
    RobustOptimizer,
    long_only_box,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSÃO: data dict → pd.DataFrame de retornos
# ─────────────────────────────────────────────────────────────────────────────

def _returns_from_data(data: dict) -> pd.DataFrame:
    """
    Extrai o DataFrame de retornos do dicionário de dados.

    O dicionário pode vir de:
        - fetcher.prepare_data()   → contém "log_returns" real
        - synthetic.generate_...() → contém apenas mu/sigma/chol

    Os otimizadores exigem pd.DataFrame com colunas = tickers.
    Para dados sintéticos, reconstrói retornos diários via GBM simples.
    """
    if "log_returns" in data and isinstance(data["log_returns"], pd.DataFrame):
        return data["log_returns"]

    # Dados sintéticos: reconstrói retornos simulando trajetórias
    n       = data["n_assets"]
    tickers = data.get("tickers", [f"A{i}" for i in range(n)])
    T       = 1260   # 5 anos de dados sintéticos

    rng    = np.random.default_rng(42)
    mu_d   = data["mu"]    / 252
    sig_d  = data["sigma"] / np.sqrt(252)
    dt     = 1.0 / 252
    chol   = data["chol_lower"] * np.sqrt(dt)  # escala para daily

    Z      = rng.standard_normal((T, n))
    eps    = Z @ chol.T
    log_ret = (mu_d - 0.5 * sig_d ** 2) + eps

    return pd.DataFrame(log_ret, columns=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# TREINAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target: str       = "sharpe_sim",
    test_size: float  = 0.2,
    seed: int         = 42,
    verbose: bool     = True,
) -> dict:
    """
    Treina Random Forest, XGBoost e GradientBoosting; retorna o melhor.

    Parâmetros
    ----------
    X        : features (n_portfolios, n_features)
    y        : labels   (n_portfolios, n_labels)
    target   : coluna de y a predizer (padrão: "sharpe_sim")
    test_size: fração do dataset para teste
    seed     : semente de reprodutibilidade
    verbose  : imprime progresso e métricas

    Retorna
    -------
    dict com:
        "best_model"   : modelo vencedor (sklearn estimator)
        "best_name"    : nome do modelo vencedor
        "feature_names": lista de features usadas
        "metrics"      : R², MAE, CV scores por modelo
        "X_test", "y_test": conjuntos de teste
    """
    y_target      = y[target].values
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_target, test_size=test_size, random_state=seed
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, max_features="sqrt",
            random_state=seed, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, verbosity=0,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            random_state=seed,
        ),
    }

    kf      = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}

    if verbose:
        print(f"\n[ml] Treinando modelos (target={target})...")
        print(f"     Treino: {len(X_train):,}  |  Teste: {len(X_test):,}\n")
        print(f"  {'Modelo':<22} {'CV R² (mean±std)':>22} {'Teste R²':>10} {'MAE':>10}")
        print(f"  {'-'*66}")

    best_cv_score = -np.inf
    best_name  = None
    best_model = None

    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {
            "cv_mean" : cv_scores.mean(),
            "cv_std"  : cv_scores.std(),
            "r2_test" : r2,
            "mae_test": mae,
            "model"   : model,
        }

        if verbose:
            print(f"  {name:<22} {cv_scores.mean():>+.4f} ± {cv_scores.std():.4f}  "
                  f"{r2:>+10.4f} {mae:>10.4f}")

        if cv_scores.mean() > best_cv_score:
            best_cv_score = cv_scores.mean()
            best_name     = name
            best_model    = model

    if verbose:
        print(f"\n  → Melhor modelo: {best_name} (CV R²={best_cv_score:.4f})\n")

    return {
        "best_model"   : best_model,
        "best_name"    : best_name,
        "all_models"   : results,
        "feature_names": feature_names,
        "target"       : target,
        "X_test"       : X_test,
        "y_test"       : y_test,
        "metrics"      : {n: {k: v for k, v in r.items() if k != "model"}
                          for n, r in results.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO E INTERPRETABILIDADE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(train_result: dict, top_n: int = 10) -> pd.DataFrame:
    """
    Avalia o melhor modelo e exibe importância das features.

    Retorna DataFrame de importância de features ordenado.
    """
    model         = train_result["best_model"]
    name          = train_result["best_name"]
    feature_names = train_result["feature_names"]
    X_test        = train_result["X_test"]
    y_test        = train_result["y_test"]

    y_pred = model.predict(X_test)

    print(f"\n{'='*55}")
    print(f"  AVALIAÇÃO DO MODELO: {name}")
    print(f"{'='*55}")
    print(f"  R² no teste       : {r2_score(y_test, y_pred):>.4f}")
    print(f"  MAE no teste      : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  Resíduo médio     : {(y_test - y_pred).mean():>+.4f}")
    print(f"  Resíduo std       : {(y_test - y_pred).std():.4f}")

    importances = (
        model.feature_importances_
        if hasattr(model, "feature_importances_")
        else np.zeros(len(feature_names))
    )

    feat_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  TOP {top_n} FEATURES MAIS IMPORTANTES:")
    print(f"  {'Feature':<25} {'Importância':>12} {'Acum.':>8}")
    print(f"  {'-'*47}")
    cumsum = 0.0
    for _, row in feat_df.head(top_n).iterrows():
        cumsum += row["importance"]
        print(f"  {row['feature']:<25} {row['importance']:>12.4f} {cumsum:>8.1%}")

    print(f"{'='*55}\n")
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DO MELHOR PORTFÓLIO VIA ML
# ─────────────────────────────────────────────────────────────────────────────

def select_best_portfolio(
    train_result: dict,
    data: dict,
    n_candidates: int  = 50_000,
    n_sims_verify: int = 10_000,
    n_steps: int       = 252,
    rf_annual: float   = 0.05,
    seed: int          = 99,
    concentration: str = "mixed",
    w_max: float       = 1.0,
) -> dict:
    """
    Usa o modelo treinado para encontrar o portfólio com maior Sharpe.

    Estratégia:
        1. Amostra n_candidates portfólios aleatórios
        2. Extrai features analíticas (microssegundos)
        3. Prediz Sharpe com o modelo (microssegundos)
        4. Seleciona o top candidato por Sharpe predito
        5. Verifica com simulação Monte Carlo completa

    Retorna
    -------
    dict com pesos, Sharpe predito, Sharpe verificado e métricas completas
    """
    model         = train_result["best_model"]
    feature_names = train_result["feature_names"]
    mu            = data["mu"]
    sigma         = data["sigma"]
    chol          = data["chol_lower"]
    cov           = data["cov_matrix"]
    n_assets      = data["n_assets"]

    rng = np.random.default_rng(seed)

    print(f"[seleção] Gerando {n_candidates:,} candidatos...")
    W_cand = sample_weights(n_assets, n_candidates, concentration, rng, w_max=w_max)

    print(f"[seleção] Extraindo features e predizendo Sharpe...")
    X_cand      = extract_features(W_cand, mu, sigma, cov, rf_annual)
    sharpe_pred = model.predict(X_cand[feature_names].values)

    top_idx  = np.argsort(sharpe_pred)[-10:][::-1]
    best_w   = W_cand[top_idx[0]]
    best_pred = float(sharpe_pred[top_idx[0]])

    print(f"[seleção] Verificando top candidato com {n_sims_verify:,} simulações...")
    result = simulate_vectorized(
        mu, sigma, chol, best_w,
        n_sims=n_sims_verify, n_steps=n_steps,
        seed=int(rng.integers(0, 2**31)),
    )
    metrics = compute_metrics(result, rf_annual=rf_annual)

    print(f"\n  Sharpe predito   : {best_pred:.4f}")
    print(f"  Sharpe verificado: {metrics.sharpe:.4f}")
    print(f"  Retorno esperado : {metrics.expected_return:.2%}")
    print(f"  Volatilidade     : {metrics.std_return:.2%}")
    print(f"  VaR 95%          : {metrics.var_95:.2%}")

    return {
        "weights"        : best_w,
        "sharpe_pred"    : best_pred,
        "sharpe_verified": metrics.sharpe,
        "metrics"        : metrics,
        "n_candidates"   : n_candidates,
        "tickers"        : data["tickers"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARAÇÃO COM BASELINES — inclui todos os otimizadores
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_baselines(
    ml_result: dict,
    data: dict,
    n_sims: int       = 10_000,
    n_steps: int      = 252,
    rf_annual: float  = 0.05,
    seed: int         = 0,
    w_max: float      = 0.15,
    bl_method: str    = "combined",
) -> pd.DataFrame:
    """
    Compara o portfólio ML com todos os otimizadores do módulo optimization/.

    Portfólios comparados:
        1/N              — baseline trivial
        ML-Selecionado   — busca ML por espaço de pesos
        MinVariance      — minimiza variância (apenas Σ)
        RiskParity       — equaliza contribuição de risco
        HRP              — clusterização hierárquica
        Markowitz        — maximiza Sharpe (μ + Σ)
        Robust           — Markowitz com penalty de incerteza (κ=1)
        BL-{bl_method}   — Black-Litterman com views de momentum

    Parâmetros
    ----------
    ml_result : retorno de select_best_portfolio()
    data      : dicionário de fetcher.prepare_data() ou synthetic
    n_sims    : simulações Monte Carlo por portfólio
    n_steps   : horizonte temporal (dias)
    rf_annual : taxa livre de risco anual
    seed      : semente de reprodutibilidade
    w_max     : limite máximo de peso por ativo
    bl_method : método de views para BL
                "momentum" | "sector_momentum" | "combined" | "none"

    Retorna
    -------
    pd.DataFrame com métricas por portfólio, ordenado por Sharpe decrescente.
    """
    mu      = data["mu"]
    sigma   = data["sigma"]
    chol    = data["chol_lower"]
    n       = data["n_assets"]
    tickers = data.get("tickers", [f"A{i}" for i in range(n)])

    returns     = _returns_from_data(data)
    constraints = long_only_box(w_max=w_max)

    # sector_map para views setoriais
    sector_map = None
    try:
        from data.tickers import SECTORS
        sector_map = {t: s for s, ts in SECTORS.items() for t in ts}
    except ImportError:
        pass

    # Instancia BL com ou sem views
    bl_opt  = BlackLittermanOptimizer(rf=rf_annual)
    bl_name = f"BL-{bl_method}" if bl_method != "none" else "BlackLitterman"

    # ── Roda cada otimizador ─────────────────────────────────────────────────
    optimizers_to_run = {
        "MinVariance": MinVarianceOptimizer(),
        "RiskParity" : RiskParityOptimizer(),
        "HRP"        : HRPOptimizer(linkage_method="ward"),
        "Markowitz"  : MarkowitzOptimizer(rf=rf_annual),
        "Robust"     : RobustOptimizer(rf=rf_annual, kappa=1.0),
    }

    optimizer_weights: Dict[str, Optional[np.ndarray]] = {}
    print("\n[compare] Executando otimizadores...")

    for name, opt in optimizers_to_run.items():
        try:
            w_series = opt.optimize(returns, constraints)
            w_arr    = w_series.reindex(tickers).fillna(0).values.astype(np.float32)
            w_arr   /= w_arr.sum()
            optimizer_weights[name] = w_arr
            print(f"  ✓ {name:<22} OK  ({int((w_arr > 0.001).sum())} ativos ativos)")
        except Exception as e:
            print(f"  ✗ {name:<22} FALHOU: {e}")
            optimizer_weights[name] = None

    # Black-Litterman com views de momentum
    try:
        if bl_method != "none":
            w_series = bl_opt.optimize_with_views(
                returns, method=bl_method,
                sector_map=sector_map,
                constraints=constraints,
                verbose=True,
            )
        else:
            w_series = bl_opt.optimize(returns, constraints)
        w_arr = w_series.reindex(tickers).fillna(0).values.astype(np.float32)
        w_arr /= w_arr.sum()
        optimizer_weights[bl_name] = w_arr
        print(f"  ✓ {bl_name:<22} OK  ({int((w_arr > 0.001).sum())} ativos ativos)")
    except Exception as e:
        print(f"  ✗ {bl_name:<22} FALHOU: {e}")
        optimizer_weights[bl_name] = None

    # ── Portfólios finais (ordem fixa para reprodutibilidade) ────────────────
    portfolios: Dict[str, np.ndarray] = {
        "1/N (igualitário)": np.ones(n, dtype=np.float32) / n,
        "ML-Selecionado"   : ml_result["weights"].astype(np.float32),
    }
    for name, w in optimizer_weights.items():
        if w is not None:
            portfolios[name] = w

    # ── Simula e coleta métricas ─────────────────────────────────────────────
    rows = []
    print("\n[compare] Simulando Monte Carlo para cada portfólio...")

    for port_name, w in portfolios.items():
        try:
            res = simulate_vectorized(
                mu, sigma, chol, w,
                n_sims=n_sims, n_steps=n_steps, seed=seed,
            )
            m = compute_metrics(res, rf_annual=rf_annual)
            rows.append({
                "Portfólio"   : port_name,
                "Sharpe"      : round(m.sharpe, 3),
                "Sortino"     : round(m.sortino, 3),
                "Retorno"     : f"{m.expected_return:.2%}",
                "Volatilidade": f"{m.std_return:.2%}",
                "VaR 95%"     : f"{m.var_95:.2%}",
                "CVaR 95%"    : f"{m.cvar_95:.2%}",
                "P(perda)"    : f"{m.prob_loss:.2%}",
                "N ativos"    : int((w > 0.001).sum()),
            })
            print(f"  ✓ {port_name:<25} Sharpe={m.sharpe:.3f}")
        except Exception as e:
            print(f"  ✗ {port_name:<25} FALHOU: {e}")

    df = pd.DataFrame(rows).set_index("Portfólio")
    df = df.sort_values("Sharpe", ascending=False)

    print("\n" + "="*90)
    print("  COMPARAÇÃO: ML vs TODOS OS OTIMIZADORES")
    print("="*90)
    print(df.to_string())
    print("="*90 + "\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMPARAÇÃO DE PESOS POR ATIVO
# ─────────────────────────────────────────────────────────────────────────────

def compare_weights(
    data: dict,
    rf_annual: float = 0.05,
    w_max: float     = 0.15,
) -> pd.DataFrame:
    """
    Compara os pesos atribuídos por cada otimizador, ativo por ativo.

    Útil para entender onde as estratégias divergem — quais ativos
    são favorecidos por Markowitz vs HRP vs Risk Parity.

    Parâmetros
    ----------
    data     : dicionário de dados
    rf_annual: taxa livre de risco
    w_max    : limite máximo de peso por ativo

    Retorna
    -------
    pd.DataFrame shape (n_assets, n_optimizers) com pesos por coluna.
    """
    tickers     = data.get("tickers", [f"A{i}" for i in range(data["n_assets"])])
    returns     = _returns_from_data(data)
    constraints = long_only_box(w_max=w_max)
    n           = data["n_assets"]

    optimizers = {
        "1/N"           : None,
        "MinVariance"   : MinVarianceOptimizer(),
        "RiskParity"    : RiskParityOptimizer(),
        "HRP"           : HRPOptimizer(linkage_method="ward"),
        "Markowitz"     : MarkowitzOptimizer(rf=rf_annual),
        "Robust"        : RobustOptimizer(rf=rf_annual, kappa=1.0),
        "BlackLitterman": BlackLittermanOptimizer(rf=rf_annual),
    }

    weight_cols = {}
    weight_cols["1/N"] = pd.Series(np.ones(n) / n, index=tickers)

    for name, opt in optimizers.items():
        if opt is None:
            continue
        try:
            w_series = opt.optimize(returns, constraints)
            weight_cols[name] = w_series.reindex(tickers).fillna(0)
        except Exception as e:
            print(f"[weights] {name} falhou: {e}")

    return pd.DataFrame(weight_cols).round(4)


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTÊNCIA DO MODELO
# ─────────────────────────────────────────────────────────────────────────────

def save_model(train_result: dict, path: Path) -> None:
    """Salva o modelo treinado em disco via joblib."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_result["best_model"], path / "model.joblib")
    pd.Series(train_result["feature_names"]).to_csv(
        path / "features.csv", index=False
    )
    print(f"[ml] Modelo salvo em {path}")


def load_model(path: Path) -> dict:
    """Carrega modelo salvo do disco."""
    path  = Path(path)
    model = joblib.load(path / "model.joblib")
    feats = pd.read_csv(path / "features.csv", header=None)[0].tolist()
    return {"best_model": model, "feature_names": feats}


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.synthetic import generate_synthetic_data

    print("=" * 60)
    print("  MÓDULO 4 — ML + Otimizadores para Seleção de Portfólios")
    print("=" * 60)

    data = generate_synthetic_data(n_assets=20, n_days=1260, seed=0)

    X, y, W = build_dataset(
        data, n_portfolios=400,
        n_sims_per_portfolio=500, n_steps=252, seed=42,
    )

    train_result = train(X, y, target="sharpe_sim", verbose=True)
    evaluate(train_result)

    ml_result = select_best_portfolio(
        train_result, data,
        n_candidates=10_000, n_sims_verify=3_000, n_steps=252,
    )

    compare_with_baselines(ml_result, data, n_sims=3_000, n_steps=252)