"""
portfolio_selector.py
=====================
Treinamento e uso do modelo de ML para seleção de portfólios eficientes.

PIPELINE DE ML
==============

    1. build_dataset()        → X (features), y (labels), W (pesos)
    2. train()                → ajusta Random Forest + XGBoost; seleciona melhor
    3. evaluate()             → métricas de regressão + importância de features
    4. select_best_portfolio()→ gera candidatos, prediz Sharpe, retorna o melhor
    5. compare_with_baselines()→ confronta ML vs 1/N vs mínima variância

JUSTIFICATIVA DO MODELO
========================
Random Forest e XGBoost são escolhas naturais porque:
    - Lidam bem com features em escalas diferentes (sem necessidade de normalizar)
    - Capturam não-linearidades entre concentração de pesos e risco
    - Têm boa interpretabilidade via importância de features
    - São robustos a outliers nos labels (simulações com seed ruim)
    - Treinam rapidamente em datasets de ~5k amostras

O objetivo é predizer o Sharpe Ratio de um portfólio sem precisar
rodar Monte Carlo — substituindo segundos de simulação por microssegundos
de inferência.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

from ml.dataset import build_dataset, sample_weights, extract_features
from simulation.monte_carlo_cpu import simulate_vectorized
from portfolio.metrics import (
    sharpe_ratio, var_historical, cvar_historical, compute_metrics
)


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
    Treina Random Forest e XGBoost; retorna o melhor modelo e métricas.

    Parâmetros
    ----------
    X       : features (n_portfolios, n_features)
    y       : labels   (n_portfolios, n_labels)
    target  : coluna de y a predizer (padrão: "sharpe_sim")
    test_size: fração do dataset para teste

    Retorna
    -------
    dict com:
        "best_model"  : modelo vencedor (sklearn estimator)
        "best_name"   : nome do modelo vencedor
        "feature_names": lista de features usadas
        "metrics"     : dict com R², MAE, CV scores por modelo
        "X_test", "y_test": conjuntos de teste para avaliação posterior
    """
    y_target = y[target].values
    feature_names = list(X.columns)

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_target, test_size=test_size, random_state=seed
    )

    # Define modelos candidatos
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=0,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=seed,
        ),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}

    if verbose:
        print(f"\n[ml] Treinando modelos (target={target})...")
        print(f"     Treino: {len(X_train):,}  |  Teste: {len(X_test):,}\n")
        print(f"  {'Modelo':<22} {'CV R² (mean±std)':>22} {'Teste R²':>10} {'MAE':>10}")
        print(f"  {'-'*66}")

    best_cv_score = -np.inf
    best_name     = None
    best_model    = None

    for name, model in models.items():
        # Validação cruzada no treino
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=kf, scoring="r2", n_jobs=-1)
        # Treina no treino completo
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

    # Importância de features
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

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
# SELEÇÃO DO MELHOR PORTFÓLIO
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
) -> dict:
    """
    Usa o modelo treinado para encontrar o portfólio com maior Sharpe.

    Estratégia:
        1. Amostra n_candidates portfólios aleatórios
        2. Extrai features analíticas (microssegundos)
        3. Prediz Sharpe com o modelo (microssegundos)
        4. Seleciona o top-K por Sharpe predito
        5. Verifica o melhor com simulação Monte Carlo completa

    Parâmetros
    ----------
    n_candidates  : candidatos a gerar (quanto maior, melhor a busca)
    n_sims_verify : simulações para verificar o portfólio selecionado
    concentration : estratégia de amostragem dos candidatos

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
    W_cand = sample_weights(n_assets, n_candidates, concentration, rng)

    print(f"[seleção] Extraindo features e predizindo Sharpe...")
    X_cand = extract_features(W_cand, mu, sigma, cov, rf_annual)
    sharpe_pred = model.predict(X_cand[feature_names].values)

    # Seleciona os top 10 por Sharpe predito e verifica via simulação
    top_idx   = np.argsort(sharpe_pred)[-10:][::-1]
    best_idx  = top_idx[0]
    best_w    = W_cand[best_idx]
    best_pred = float(sharpe_pred[best_idx])

    print(f"[seleção] Verificando top candidato com {n_sims_verify:,} simulações...")
    result = simulate_vectorized(
        mu, sigma, chol, best_w,
        n_sims=n_sims_verify, n_steps=n_steps,
        seed=int(rng.integers(0, 2**31)),
    )
    metrics = compute_metrics(result, rf_annual=rf_annual)

    print(f"\n  Sharpe predito  : {best_pred:.4f}")
    print(f"  Sharpe verificado: {metrics.sharpe:.4f}")
    print(f"  Retorno esperado : {metrics.expected_return:.2%}")
    print(f"  Volatilidade     : {metrics.std_return:.2%}")
    print(f"  VaR 95%          : {metrics.var_95:.2%}")

    return {
        "weights"       : best_w,
        "sharpe_pred"   : best_pred,
        "sharpe_verified": metrics.sharpe,
        "metrics"       : metrics,
        "n_candidates"  : n_candidates,
        "tickers"       : data["tickers"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARAÇÃO COM BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_baselines(
    ml_result: dict,
    data: dict,
    n_sims: int       = 10_000,
    n_steps: int      = 252,
    rf_annual: float  = 0.05,
    seed: int         = 0,
) -> pd.DataFrame:
    """
    Compara o portfólio selecionado pelo ML com:
        - 1/N (igualmente ponderado)
        - Mínima variância analítica (w ∝ 1/sigma²)
        - Máximo Sharpe ingênuo (w ∝ sharpe_individual)

    Retorna DataFrame com métricas lado a lado.
    """
    mu    = data["mu"]
    sigma = data["sigma"]
    chol  = data["chol_lower"]
    n     = data["n_assets"]

    # Baselines
    w_equal = np.ones(n, dtype=np.float32) / n

    inv_var = 1.0 / (sigma ** 2 + 1e-10)
    w_minvar = (inv_var / inv_var.sum()).astype(np.float32)

    sharpe_ind = (mu - rf_annual) / (sigma + 1e-10)
    sharpe_ind_pos = np.clip(sharpe_ind, 0, None)
    if sharpe_ind_pos.sum() > 0:
        w_maxsharpe = (sharpe_ind_pos / sharpe_ind_pos.sum()).astype(np.float32)
    else:
        w_maxsharpe = w_equal.copy()

    portfolios = {
        "ML-Selecionado"  : ml_result["weights"],
        "1/N (igualitário)": w_equal,
        "Mínima Variância" : w_minvar,
        "Máx Sharpe Ingênuo": w_maxsharpe,
    }

    rows = []
    for name, w in portfolios.items():
        res = simulate_vectorized(mu, sigma, chol, w,
                                  n_sims=n_sims, n_steps=n_steps, seed=seed)
        m = compute_metrics(res, rf_annual=rf_annual)
        rows.append({
            "Portfólio"     : name,
            "Sharpe"        : round(m.sharpe, 3),
            "Sortino"       : round(m.sortino, 3),
            "Retorno"       : f"{m.expected_return:.2%}",
            "Volatilidade"  : f"{m.std_return:.2%}",
            "VaR 95%"       : f"{m.var_95:.2%}",
            "CVaR 95%"      : f"{m.cvar_95:.2%}",
            "P(perda)"      : f"{m.prob_loss:.2%}",
        })

    df = pd.DataFrame(rows).set_index("Portfólio")

    print("\n" + "="*80)
    print("  COMPARAÇÃO: ML vs BASELINES")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTÊNCIA DO MODELO
# ─────────────────────────────────────────────────────────────────────────────

def save_model(train_result: dict, path: Path) -> None:
    """Salva o modelo treinado em disco via joblib."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_result["best_model"], path / "model.joblib")
    pd.Series(train_result["feature_names"]).to_csv(path / "features.csv", index=False)
    print(f"[ml] Modelo salvo em {path}")


def load_model(path: Path) -> dict:
    """Carrega modelo salvo do disco."""
    path = Path(path)
    model = joblib.load(path / "model.joblib")
    features = pd.read_csv(path / "features.csv", header=None)[0].tolist()
    return {"best_model": model, "feature_names": features}


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.synthetic import generate_synthetic_data

    print("=" * 60)
    print("  MÓDULO 4 — ML para Seleção de Portfólios")
    print("=" * 60)

    data = generate_synthetic_data(n_assets=20, n_days=1260, seed=0)

    # 1. Gera dataset (reduzido para teste rápido)
    X, y, W = build_dataset(
        data,
        n_portfolios=400,
        n_sims_per_portfolio=500,
        n_steps=252,
        seed=42,
    )

    # 2. Treina
    train_result = train(X, y, target="sharpe_sim", verbose=True)

    # 3. Avalia
    feat_importance = evaluate(train_result)

    # 4. Seleciona melhor portfólio
    ml_result = select_best_portfolio(
        train_result, data,
        n_candidates=10_000,
        n_sims_verify=3_000,
        n_steps=252,
    )

    # 5. Compara com baselines
    compare_with_baselines(ml_result, data, n_sims=3_000, n_steps=252)