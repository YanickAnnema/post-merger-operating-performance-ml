"""
Model Training Pipeline — M&A Synergy Estimation using XGBoost
===============================================================

Loads ml_ready_nowinsor.csv (pre-winsorisation, from data_preparation.py),
enforces temporal split, applies all transformations on the TRAINING set only,
trains XGBoost via randomised CV with inner-holdout early stopping, evaluates
on clean val/test sets, and produces fitted model + full diagnostics.

Methodological choices / leakage hygiene:
  - Input is ml_ready_nowinsor.csv (no full-sample clipping applied yet).
    Full-sample winsorisation in data_preparation.py contaminates ml_ready.csv.
  - Winsorisation bounds fitted on df_train_cv only; applied to all splits.
  - Explicit chronological sort before any time-based CV split.
  - Inner holdout (last N years of train) used for early stopping only.
    Val (2016-2018) and test (2019-2022) are never seen during any fitting step.
  - MAPE removed (undefined / misleading for near-zero CFROA target).
  - Spearman rank correlation added (finance screening relevance).

Pipeline:
  1.  load_and_split()          -- load CSV, enforce chronological split
  2.  leakage_guard()           -- assert no post-deal column in feature matrix
  3.  sort_chronologically()    -- explicit DateEffective sort on all splits
  4.  split_inner_holdout()     -- carve last N train years for early stopping
  5.  winsorise_train_fit()     -- bounds from df_train_cv; apply to all splits
  6.  run_cv()                  -- RandomizedSearchCV + TimeSeriesSplit on train_cv
  7.  refit_best()              -- full retrain; early stop on inner_val
  8.  compute_baselines()       -- naive mean, median, Ridge (train-fit only)
  9.  evaluate()                -- RMSE, MAE, R2, Spearman on all splits
  10. bootstrap_ci()            -- 95% CI for val/test metrics (1000 resamples)
  11. get_feature_importance()  -- gain/cover/weight + channel-level aggregation
  12. plot_diagnostics()        -- 5 publication-quality plots
  13. save_outputs()            -- model pkl, CSVs, plots, performance summary

Inputs:
  ml_ready_nowinsor.csv  -- from data_preparation.py (pre-winsorisation, NaN intact)

Outputs (to ~/Desktop/outputs/):
  xgboost_model_final.pkl
  cv_results.csv
  feature_importance.csv
  channel_importance.csv
  performance_summary.txt
  plots/
    predicted_vs_actual.png
    residuals_by_split.png
    residual_distributions.png
    feature_importance_bar.png
    train_val_curve.png

Dependencies:
  pip install xgboost scikit-learn scipy matplotlib --break-system-packages

Optimised for Spyder IDE (F5 execution).
"""

import numpy as np
import pandas as pd
import pickle
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure code/ is on sys.path for Spyder / repo-root execution
_code_dir = str(Path(__file__).resolve().parent)
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config_hyperparameters import (
    BASE_PARAMS,
    PARAM_GRID,
    N_RANDOM_ITER,
    CV_FOLDS,
    EARLY_STOP,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAQ_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "DAQ pipeline"
ML_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ML pipeline"

CONFIG = {
    # Primary input: unwinsorised snapshot produced by data_preparation.py.
    # Falls back to ml_ready.csv with a warning if nowinsor version is absent.
    'input_path':       DAQ_OUTPUT_DIR / "ml_ready_nowinsor.csv",
    'input_fallback':   DAQ_OUTPUT_DIR / "ml_ready.csv",

    # Output directory
    'output_dir':       ML_OUTPUT_DIR,
    'plot_dir':         ML_OUTPUT_DIR / "plots",

    # Target variable
    'target_col':       "synergy_healy1992_w",

    # Metadata columns (never used as features)
    'date_col':         "DateEffective",
    'split_col':        "split",

    # Winsorisation -- fitted on df_train_cv only
    'winsor_low':       0.01,
    'winsor_high':      0.99,

    # Inner holdout: train deals with deal_year >= this are reserved for early
    # stopping only.  Keeps val and test uncontaminated by any fitting step.
    # 2013 -> inner_val = 2013-2015 (~300 deals); train_cv = 1995-2012 (~2,800)
    'early_stop_year':  2013,

    # Bootstrap CI resamples
    'n_bootstrap':      1000,

    # Ridge alpha for linear baseline
    'ridge_alpha':      1.0,

    # Set True for fast debug iteration (compact grid, fewer CV folds)
    'use_compact_grid': False,

    # Random seed
    'seed': 42,
}


# Feature definitions -- must match data_preparation.py exactly
FEATURES_COST = [
    "cost_relative_asset_size",
    "cost_ppe_intensity_diff",
    "cost_inventory_turnover_gap",
    "cost_target_asset_utilization",
    "log_deal_value",
    "deal_tender_offer",
    "deal_friendly",
]

FEATURES_REVENUE = [
    "revenue_rd_intensity_diff",
    "revenue_capex_intensity_diff",
    "revenue_intangible_intensity_diff",
    "revenue_relative_size_sales",
    "deal_cross_border",
]

FEATURES_OPERATIONAL = [
    "operational_asset_turnover_gap",
    "operational_roa_gap",
    "operational_acquiror_op_margin",
    "operational_target_cf_margin",
    "deal_industry_4dig",
    "deal_industry_2dig",
]

FEATURES_FINANCIAL = [
    "financial_leverage_gap",
    "financial_cash_ratio_diff",
    "financial_acquiror_cash_to_sales",
    "financial_quick_ratio_acquiror",
    "financial_quick_ratio_target",
    "deal_stock_payment",
    "deal_all_cash",
    "financial_altman_z_acquiror",
    "financial_altman_z_target",
]

FEATURES_MACRO = [
    "sp500_trailing_12m",
    "credit_spread_bbb_aaa",
]

ALL_FEATURES: List[str] = (
    FEATURES_COST
    + FEATURES_REVENUE
    + FEATURES_OPERATIONAL
    + FEATURES_FINANCIAL
    + FEATURES_MACRO
)

# Channel map for grouped importance output
CHANNEL_MAP: Dict[str, str] = {
    **{f: "cost"        for f in FEATURES_COST},
    **{f: "revenue"     for f in FEATURES_REVENUE},
    **{f: "operational" for f in FEATURES_OPERATIONAL},
    **{f: "financial"   for f in FEATURES_FINANCIAL},
    **{f: "macro"       for f in FEATURES_MACRO},
}

BINARY_FEATURES = {
    "deal_tender_offer", "deal_friendly", "deal_cross_border",
    "deal_stock_payment", "deal_all_cash", "deal_industry_4dig",
    "deal_industry_2dig",
}

# Hard leakage guard -- post-deal columns must never appear as features
POST_DEAL_COLS = {
    "AB_t3_CFROA", "AB_t3_operating_cashflow", "AB_t3_total_assets",
    "Delta_CFROA_raw", "synergy_healy1992", "synergy_healy1992_w",
    "industry_CFROA_adjustment", "t3AB_fye",
}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1 -- LOAD AND SPLIT
# =============================================================================

def load_and_split(
    path: Path,
    fallback: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the pre-winsorisation CSV and split by the 'split' column assigned
    by data_preparation.py.

    Prefers ml_ready_nowinsor.csv so that all transformation bounds are fitted
    on the training split only.  Falls back to ml_ready.csv with a warning.

    Returns: (df_full, df_train, df_val, df_test)
    """
    if path.exists():
        logger.info(f"Loading (nowinsor): {path.name}")
        df = pd.read_csv(path, low_memory=False)
    elif fallback.exists():
        logger.warning(
            "  ml_ready_nowinsor.csv not found -- falling back to ml_ready.csv.\n"
            "  WARNING: ml_ready.csv has full-sample winsorisation already applied.\n"
            "           Re-run data_preparation.py to produce ml_ready_nowinsor.csv."
        )
        df = pd.read_csv(fallback, low_memory=False)
    else:
        raise FileNotFoundError(
            f"Neither {path.name} nor {fallback.name} found.\n"
            "Run data_preparation.py first."
        )

    logger.info(f"  Shape: {df.shape}")

    date_col = CONFIG['date_col']
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    for col in [CONFIG['split_col'], CONFIG['target_col']]:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found. "
                "Run data_preparation.py to produce the input file."
            )

    n_before = len(df)
    df = df[df[CONFIG['target_col']].notna()].reset_index(drop=True)
    if len(df) < n_before:
        logger.warning(f"  Dropped {n_before - len(df)} rows with NaN target")

    df_train = df[df[CONFIG['split_col']] == "train"].reset_index(drop=True)
    df_val   = df[df[CONFIG['split_col']] == "val"].reset_index(drop=True)
    df_test  = df[df[CONFIG['split_col']] == "test"].reset_index(drop=True)

    logger.info(
        f"  Split: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}"
    )

    # Temporal integrity check
    if 'deal_year' in df.columns:
        yr = lambda d: pd.to_numeric(d['deal_year'], errors='coerce')
        max_tr, min_v = yr(df_train).max(), yr(df_val).min()
        max_v,  min_te = yr(df_val).max(), yr(df_test).min()
        if max_tr >= min_v:
            raise ValueError(
                f"Temporal overlap: train max year {max_tr} >= val min {min_v}"
            )
        if max_v >= min_te:
            raise ValueError(
                f"Temporal overlap: val max year {max_v} >= test min {min_te}"
            )
        logger.info(
            f"  Years: train [{yr(df_train).min():.0f}-{max_tr:.0f}]  "
            f"val [{min_v:.0f}-{max_v:.0f}]  "
            f"test [{min_te:.0f}-{yr(df_test).max():.0f}]"
        )

    return df, df_train, df_val, df_test


# =============================================================================
# STEP 2 -- LEAKAGE GUARD
# =============================================================================

def leakage_guard(features: List[str], df: pd.DataFrame) -> List[str]:
    """
    Hard assertion: no post-deal column in feature set.
    Returns list of features actually present in the dataset.
    """
    leaked = [f for f in features if f in POST_DEAL_COLS]
    if leaked:
        raise ValueError(
            f"DATA LEAKAGE -- post-deal columns in feature list: {leaked}"
        )

    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"  Features absent from dataset (skipped): {missing}")

    logger.info(
        f"  Leakage check passed -- {len(available)}/{len(features)} features available"
    )
    return available


# =============================================================================
# STEP 3 -- CHRONOLOGICAL SORT
# =============================================================================

def sort_chronologically(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Explicitly sort all splits by DateEffective.

    Required before TimeSeriesSplit: if rows are not in strict temporal
    order, fold boundaries will not respect the chronology and CV
    estimates will be invalid.
    """
    date_col = CONFIG['date_col']
    df_train = df_train.sort_values(date_col, na_position='last').reset_index(drop=True)
    df_val   = df_val.sort_values(date_col,   na_position='last').reset_index(drop=True)
    df_test  = df_test.sort_values(date_col,  na_position='last').reset_index(drop=True)
    logger.info("  All splits sorted chronologically by DateEffective")
    return df_train, df_val, df_test


# =============================================================================
# STEP 4 -- INNER HOLDOUT FOR EARLY STOPPING
# =============================================================================

def split_inner_holdout(
    df_train: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve the tail of the training set (deals >= early_stop_year) as a
    held-out set for early stopping in refit_best().

    This ensures val (2016-2018) and test (2019-2022) are never exposed
    to the fitting process in any capacity.

    Returns: (df_train_cv, df_inner_val)
    """
    cutoff = CONFIG['early_stop_year']

    if 'deal_year' not in df_train.columns:
        logger.warning(
            "  'deal_year' absent -- inner holdout skipped; "
            "using last 15% of train for early stopping."
        )
        n_hold = max(50, int(0.15 * len(df_train)))
        return df_train.iloc[:-n_hold].copy(), df_train.iloc[-n_hold:].copy()

    yr = pd.to_numeric(df_train['deal_year'], errors='coerce')
    df_train_cv  = df_train[yr < cutoff].reset_index(drop=True)
    df_inner_val = df_train[yr >= cutoff].reset_index(drop=True)

    logger.info(
        f"  train_cv  = {len(df_train_cv):,} deals (< {cutoff})  "
        f"inner_val = {len(df_inner_val):,} deals (>= {cutoff})"
    )
    if len(df_inner_val) < 50:
        logger.warning(
            f"  Inner val has only {len(df_inner_val)} rows -- "
            f"consider lowering early_stop_year"
        )

    return df_train_cv, df_inner_val


# =============================================================================
# STEP 5 -- WINSORISE (TRAIN-FIT)
# =============================================================================

def winsorise_train_fit(
    df_train_cv: pd.DataFrame,
    df_inner_val: pd.DataFrame,
    df_train_full: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute 1%/99% winsorisation bounds from df_train_cv only.
    Apply to all splits.  Binary features excluded.

    Returns: (df_train_cv, df_inner_val, df_train_full, df_val, df_test, clip_bounds)
    """
    lo = CONFIG['winsor_low']
    hi = CONFIG['winsor_high']
    clip_bounds: Dict[str, Tuple[float, float]] = {}
    continuous = [f for f in features if f not in BINARY_FEATURES]
    n_clipped = 0

    for feat in continuous:
        if feat not in df_train_cv.columns:
            continue
        col = df_train_cv[feat].dropna()
        if len(col) == 0:
            continue
        lo_val = float(col.quantile(lo))
        hi_val = float(col.quantile(hi))
        clip_bounds[feat] = (lo_val, hi_val)

        for df in [df_train_cv, df_inner_val, df_train_full, df_val, df_test]:
            if feat in df.columns:
                df[feat] = df[feat].clip(lower=lo_val, upper=hi_val)
        n_clipped += 1

    logger.info(
        f"  Winsorised {n_clipped} continuous features at [{lo:.0%}, {hi:.0%}] "
        f"(bounds from train_cv: {len(df_train_cv):,} deals)"
    )
    return df_train_cv, df_inner_val, df_train_full, df_val, df_test, clip_bounds


# =============================================================================
# STEP 6 -- CROSS-VALIDATION
# =============================================================================

def run_cv(
    X_train_cv: pd.DataFrame,
    y_train_cv: pd.Series,
) -> Tuple[RandomizedSearchCV, pd.DataFrame]:
    """
    RandomizedSearchCV with TimeSeriesSplit on train_cv.

    Early stopping is NOT used during CV: sklearn does not pass per-fold
    eval_set to XGBoost.  n_estimators is included in the search grid
    (200/500/1000) to cover the tree count dimension.  Early stopping
    is applied only in refit_best() using the inner_val set.

    Returns: (search_object, cv_results_df)
    """
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION (RandomizedSearchCV + TimeSeriesSplit on train_cv)")
    logger.info("=" * 60)

    if CONFIG['use_compact_grid']:
        from config_hyperparameters import (
            PARAM_GRID_COMPACT as grid,
            N_RANDOM_ITER_COMPACT as n_iter,
        )
        logger.info("  Using COMPACT grid")
    else:
        grid, n_iter = PARAM_GRID, N_RANDOM_ITER

    # X_train_cv is already sorted chronologically; TimeSeriesSplit will
    # respect row order when creating fold boundaries.
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    # No early_stopping_rounds here -- n_estimators tuned via grid
    estimator = xgb.XGBRegressor(**BASE_PARAMS)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        random_state=CONFIG['seed'],
        verbose=1,
        n_jobs=-1,
        refit=False,
        error_score='raise',
    )

    logger.info(
        f"  {n_iter} combos x {CV_FOLDS} folds = {n_iter * CV_FOLDS} fits  "
        f"[n_train_cv={len(X_train_cv):,}]"
    )
    t0 = time.time()
    search.fit(X_train_cv, y_train_cv)
    logger.info(f"  CV completed in {time.time() - t0:.1f}s")

    cv_df = pd.DataFrame(search.cv_results_).sort_values('rank_test_score')
    best_rmse = -search.best_score_
    logger.info(f"  Best CV RMSE: {best_rmse:.6f}")
    logger.info(f"  Best params:  {search.best_params_}")

    return search, cv_df


# =============================================================================
# STEP 7 -- REFIT BEST MODEL
# =============================================================================

def refit_best(
    search: RandomizedSearchCV,
    X_train_cv: pd.DataFrame,
    y_train_cv: pd.Series,
    X_inner_val: pd.DataFrame,
    y_inner_val: pd.Series,
) -> xgb.XGBRegressor:
    """
    Retrain with best CV hyperparameters on train_cv.
    Uses inner_val for early stopping (val and test remain clean).
    """
    logger.info("=" * 60)
    logger.info("REFIT BEST MODEL (train_cv + inner_val early stopping)")
    logger.info("=" * 60)

    best_params = {**BASE_PARAMS, **search.best_params_}
    n_est = best_params.pop('n_estimators', 1000)

    model = xgb.XGBRegressor(
        n_estimators=n_est,
        early_stopping_rounds=EARLY_STOP,
        **best_params,
    )

    model.fit(
        X_train_cv, y_train_cv,
        eval_set=[(X_train_cv, y_train_cv), (X_inner_val, y_inner_val)],
        verbose=False,
    )

    logger.info(
        f"  Best iteration (inner_val early stopping): {model.best_iteration}"
    )
    return model


# =============================================================================
# STEP 8 -- BASELINES
# =============================================================================

def compute_baselines(
    X_train_cv: pd.DataFrame,
    y_train_cv: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Fit simple baselines on train_cv; evaluate on val and test.

    naive_mean   : predict train_cv mean
    naive_median : predict train_cv median
    ridge        : Ridge regression with train-fitted imputer + scaler

    All baselines are strictly train-fitted -- no val/test information used.
    """
    logger.info("=" * 60)
    logger.info("BASELINES (fitted on train_cv)")
    logger.info("=" * 60)

    baselines: Dict = {}
    train_mean   = float(y_train_cv.mean())
    train_median = float(y_train_cv.median())

    for bl_name, pred_val, pred_te in [
        ("naive_mean",   np.full(len(y_val), train_mean),   np.full(len(y_test), train_mean)),
        ("naive_median", np.full(len(y_val), train_median), np.full(len(y_test), train_median)),
    ]:
        bl = {
            'val':  {'rmse': float(np.sqrt(mean_squared_error(y_val,  pred_val))),
                     'mae':  float(mean_absolute_error(y_val,  pred_val)),
                     'r2':   float(r2_score(y_val,  pred_val))},
            'test': {'rmse': float(np.sqrt(mean_squared_error(y_test, pred_te))),
                     'mae':  float(mean_absolute_error(y_test, pred_te)),
                     'r2':   float(r2_score(y_test, pred_te))},
        }
        baselines[bl_name] = bl
        logger.info(
            f"  {bl_name:<16}: val RMSE={bl['val']['rmse']:.6f}  "
            f"test RMSE={bl['test']['rmse']:.6f}"
        )

    # Ridge baseline: median imputation + Z-score scaling, all train-fitted
    try:
        imputer = SimpleImputer(strategy='median')
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_train_cv))
        ridge   = Ridge(alpha=CONFIG['ridge_alpha'])
        ridge.fit(X_tr_sc, y_train_cv)

        bl_ridge: Dict = {}
        for split, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
            pred = ridge.predict(scaler.transform(imputer.transform(X)))
            bl_ridge[split] = {
                'rmse': float(np.sqrt(mean_squared_error(y, pred))),
                'mae':  float(mean_absolute_error(y, pred)),
                'r2':   float(r2_score(y, pred)),
            }
        baselines['ridge'] = bl_ridge
        logger.info(
            f"  {'ridge':<16}: val RMSE={bl_ridge['val']['rmse']:.6f}  "
            f"test RMSE={bl_ridge['test']['rmse']:.6f}"
        )
    except Exception as exc:
        logger.warning(f"  Ridge baseline failed: {exc}")
        baselines['ridge'] = {}

    baselines['_meta'] = {
        'train_mean':   train_mean,
        'train_median': train_median,
        'n_train_cv':   len(y_train_cv),
    }
    return baselines


# =============================================================================
# STEP 9 -- EVALUATION METRICS
# =============================================================================

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Regression metrics for the synergy estimation task.

    MAPE is omitted: it is undefined/misleading for near-zero or negative
    CFROA-type targets.  Spearman rank correlation is included instead:
    relevant for deal screening (ranking predicted synergy realisations).
    """
    sp_corr, sp_p = spearmanr(y_true, y_pred)
    return {
        'rmse':     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':      float(mean_absolute_error(y_true, y_pred)),
        'r2':       float(r2_score(y_true, y_pred)),
        'spearman': float(sp_corr),
        'sp_pval':  float(sp_p),
        'n':        int(len(y_true)),
        'y_mean':   float(y_true.mean()),
        'y_std':    float(y_true.std()),
    }


def evaluate(
    model: xgb.XGBRegressor,
    X_train_cv:  pd.DataFrame, y_train_cv:  pd.Series,
    X_inner_val: pd.DataFrame, y_inner_val: pd.Series,
    X_val:       pd.DataFrame, y_val:       pd.Series,
    X_test:      pd.DataFrame, y_test:      pd.Series,
) -> Dict[str, Dict]:
    """
    Evaluate model on all evaluation sets.

    train_cv  : in-sample (CV training data)
    inner_val : early-stopping holdout (still within training window)
    val       : clean model selection set -- NEVER seen during fitting
    test      : primary out-of-sample report -- NEVER seen during fitting
    """
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    results: Dict = {}
    for name, X, y in [
        ("train_cv",  X_train_cv,  y_train_cv),
        ("inner_val", X_inner_val, y_inner_val),
        ("val",       X_val,       y_val),
        ("test",      X_test,      y_test),
    ]:
        pred = model.predict(X)
        m = compute_metrics(y, pred)
        results[name] = {**m, 'predictions': pred, 'y_true': y.values}
        logger.info(
            f"  {name:<12}: RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  "
            f"R2={m['r2']:.4f}  Spearman={m['spearman']:.4f}  n={m['n']}"
        )

    ratio = results['val']['rmse'] / results['train_cv']['rmse']
    logger.info(f"  Overfitting ratio (val/train_cv RMSE): {ratio:.3f}")
    if ratio > 2.0:
        logger.warning("  WARNING: possible overfitting (ratio > 2.0)")
    elif ratio > 1.5:
        logger.warning("  CAUTION: moderate overfitting (ratio > 1.5)")

    return results


# =============================================================================
# STEP 10 -- BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: Optional[int] = None,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for a scalar metric via paired resampling."""
    if n_boot is None:
        n_boot = CONFIG['n_bootstrap']
    if seed is None:
        seed = CONFIG['seed']

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_scores = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_scores[i] = metric_fn(y_true[idx], y_pred[idx])

    lo = float(np.percentile(boot_scores, 100 * alpha / 2))
    hi = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_bootstrap_cis(results: Dict) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute bootstrap 95% CIs for RMSE, MAE, R2, Spearman on val and test."""
    logger.info("=" * 60)
    logger.info("BOOTSTRAP CONFIDENCE INTERVALS (95%, 1000 resamples)")
    logger.info("=" * 60)

    metric_fns = {
        'rmse':     lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
        'mae':      lambda yt, yp: float(mean_absolute_error(yt, yp)),
        'r2':       lambda yt, yp: float(r2_score(yt, yp)),
        'spearman': lambda yt, yp: float(spearmanr(yt, yp)[0]),
    }

    cis: Dict = {}
    for split in ['val', 'test']:
        y_true = results[split]['y_true']
        y_pred = results[split]['predictions']
        cis[split] = {}
        for name, fn in metric_fns.items():
            lo, hi = bootstrap_ci(y_true, y_pred, fn)
            cis[split][name] = (lo, hi)
            logger.info(
                f"  {split}/{name:<10}: point={results[split][name]:.6f}  "
                f"95% CI [{lo:.6f}, {hi:.6f}]"
            )

    return cis


# =============================================================================
# STEP 11 -- FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model: xgb.XGBRegressor,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract gain, cover, weight from model; compute channel-level aggregations.

    Returns: (feature_imp_df, channel_imp_df)
    """
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)

    parts = []
    for imp_type in ['gain', 'cover', 'weight']:
        scores = model.get_booster().get_score(importance_type=imp_type)
        parts.append(pd.Series(scores, name=imp_type))

    imp_df = pd.concat(parts, axis=1).fillna(0)
    imp_df.index.name = 'feature'
    imp_df = imp_df.reset_index()

    for col in ['gain', 'cover']:
        total = imp_df[col].sum()
        imp_df[f'{col}_pct'] = (imp_df[col] / total * 100) if total > 0 else 0.0

    imp_df['channel'] = imp_df['feature'].map(CHANNEL_MAP).fillna('other')
    imp_df = imp_df.sort_values('gain', ascending=False).reset_index(drop=True)

    logger.info("  Top 10 features by gain:")
    for _, row in imp_df.head(10).iterrows():
        logger.info(
            f"    {row['feature']:<42} gain={row['gain_pct']:.1f}%  "
            f"channel={row['channel']}"
        )

    # Channel-level aggregation
    ch_df = (
        imp_df.groupby('channel')[['gain', 'cover', 'weight']]
        .sum().reset_index()
    )
    tot_g = ch_df['gain'].sum()
    tot_c = ch_df['cover'].sum()
    ch_df['gain_pct']  = ch_df['gain']  / tot_g * 100 if tot_g > 0 else 0.0
    ch_df['cover_pct'] = ch_df['cover'] / tot_c * 100 if tot_c > 0 else 0.0
    ch_df = ch_df.sort_values('gain_pct', ascending=False).reset_index(drop=True)

    logger.info("  Channel-level gain importance:")
    for _, row in ch_df.iterrows():
        logger.info(f"    {row['channel']:<14} {row['gain_pct']:.1f}%")

    return imp_df, ch_df


# =============================================================================
# STEP 12 -- DIAGNOSTIC PLOTS
# =============================================================================

def plot_diagnostics(
    model: xgb.XGBRegressor,
    X_train_cv:  pd.DataFrame, y_train_cv:  pd.Series,
    X_inner_val: pd.DataFrame, y_inner_val: pd.Series,
    X_val:       pd.DataFrame, y_val:       pd.Series,
    X_test:      pd.DataFrame, y_test:      pd.Series,
    imp_df: pd.DataFrame,
    ch_df: pd.DataFrame,
    cis: Dict,
    plot_dir: Path,
) -> None:
    """Generate all diagnostic plots at 300 DPI."""
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC PLOTS")
    logger.info("=" * 60)

    plot_dir.mkdir(parents=True, exist_ok=True)
    COLORS = {
        "train_cv":  "#90CAF9",
        "inner_val": "#FFE082",
        "val":       "#FF9800",
        "test":      "#4CAF50",
    }
    CH_COLORS = {
        "cost": "#1976D2", "revenue": "#388E3C", "operational": "#F57C00",
        "financial": "#7B1FA2", "macro": "#00838F", "other": "#9E9E9E",
    }

    # ---- 1. Predicted vs Actual (val and test -- clean sets) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (label, X, y, col) in zip(axes, [
        ("Validation (2016-2018)", X_val, y_val, COLORS["val"]),
        ("Test (2019-2022)",       X_test, y_test, COLORS["test"]),
    ]):
        pred = model.predict(X)
        ax.scatter(y, pred, alpha=0.35, s=12, color=col)
        lims = [min(y.min(), pred.min()) - 0.01, max(y.max(), pred.max()) + 0.01]
        ax.plot(lims, lims, 'r--', alpha=0.7, lw=1.5, label='Perfect')
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2   = r2_score(y, pred)
        ax.text(0.05, 0.93, f"RMSE={rmse:.4f}\nR2={r2:.4f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Actual Synergy (CFROA)")
        ax.set_ylabel("Predicted Synergy (CFROA)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Predicted vs Actual -- XGBoost (clean splits only)", fontsize=12)
    plt.tight_layout()
    fig.savefig(plot_dir / "predicted_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: predicted_vs_actual.png")

    # ---- 2. Residuals vs predicted (all 4 sets) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, X, y) in zip(axes.flat, [
        ("train_cv",  X_train_cv,  y_train_cv),
        ("inner_val", X_inner_val, y_inner_val),
        ("val",       X_val,       y_val),
        ("test",      X_test,      y_test),
    ]):
        pred  = model.predict(X)
        resid = y.values - pred
        ax.scatter(pred, resid, alpha=0.3, s=8, color=COLORS[name])
        ax.axhline(0, color='red', ls='--', alpha=0.7, lw=1.2)
        ax.set_title(f"{name} (n={len(y):,})", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.text(0.05, 0.95, f"std={resid.std():.5f}", transform=ax.transAxes,
                fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.3)
    fig.suptitle("Residual Plots -- XGBoost", fontsize=13)
    plt.tight_layout()
    fig.savefig(plot_dir / "residuals_by_split.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: residuals_by_split.png")

    # ---- 3. Residual distributions ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, X, y) in zip(axes.flat, [
        ("train_cv",  X_train_cv,  y_train_cv),
        ("inner_val", X_inner_val, y_inner_val),
        ("val",       X_val,       y_val),
        ("test",      X_test,      y_test),
    ]):
        pred  = model.predict(X)
        resid = y.values - pred
        ax.hist(resid, bins=50, color=COLORS[name], alpha=0.7, edgecolor='white')
        ax.axvline(0, color='red', ls='--', alpha=0.7, lw=1.2)
        ax.set_title(f"{name} Residuals (n={len(y):,})", fontsize=10)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.text(0.98, 0.95,
                f"mean={resid.mean():.5f}\nstd={resid.std():.5f}",
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.3)
    fig.suptitle("Residual Distributions -- XGBoost", fontsize=13)
    plt.tight_layout()
    fig.savefig(plot_dir / "residual_distributions.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: residual_distributions.png")

    # ---- 4. Feature importance bar (top 20, colour by channel) ----
    top_n = min(20, len(imp_df))
    top   = imp_df.head(top_n)
    bar_colors = [CH_COLORS.get(c, "#9E9E9E") for c in top['channel']]
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(top_n), top['gain_pct'].values, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gain Importance (%)")
    ax.set_title(
        f"Top {top_n} Features by Gain -- XGBoost (colour = synergy channel)",
        fontsize=12
    )
    ax.grid(True, axis='x', alpha=0.3)
    for bar, pct in zip(bars, top['gain_pct'].values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va='center', fontsize=8)
    legend_els = [
        Patch(facecolor=CH_COLORS[ch], label=ch)
        for ch in ["cost", "revenue", "operational", "financial", "macro"]
    ]
    ax.legend(handles=legend_els, fontsize=8, loc='lower right')
    plt.tight_layout()
    fig.savefig(plot_dir / "feature_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: feature_importance_bar.png")

    # ---- 5. Learning curve (train_cv vs inner_val) ----
    evals = model.evals_result()
    if evals and len(evals) >= 2:
        keys = list(evals.keys())
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label, col in zip(
            keys, ['train_cv', 'inner_val'],
            [COLORS['train_cv'], COLORS['inner_val']]
        ):
            metric_name = list(evals[key].keys())[0]
            ax.plot(evals[key][metric_name], label=f"{label} {metric_name.upper()}",
                    color=col, alpha=0.9)
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("RMSE")
        ax.set_title("Learning Curve -- train_cv vs inner_val RMSE", fontsize=12)
        if model.best_iteration is not None:
            ax.axvline(model.best_iteration, color='red', ls=':', alpha=0.7,
                       label=f"Best iter ({model.best_iteration})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(plot_dir / "train_val_curve.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  Saved: train_val_curve.png")
    else:
        logger.warning("  No evals_result -- learning curve skipped")


# =============================================================================
# STEP 13 -- SAVE OUTPUTS
# =============================================================================

def save_outputs(
    model: xgb.XGBRegressor,
    search: RandomizedSearchCV,
    cv_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    ch_df: pd.DataFrame,
    results: Dict,
    cis: Dict,
    baselines: Dict,
    clip_bounds: Dict,
    features_used: List[str],
    output_dir: Path,
) -> None:
    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Model pickle
    model_path = output_dir / "xgboost_model_final.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':         model,
            'features':      features_used,
            'clip_bounds':   clip_bounds,
            'best_params':   search.best_params_,
            'best_iteration': model.best_iteration,
            'config':        CONFIG,
        }, f)
    logger.info(f"  Saved: {model_path.name}")

    # CV results
    cv_df.to_csv(output_dir / "cv_results.csv", index=False)
    logger.info("  Saved: cv_results.csv")

    # Feature importance
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)
    ch_df.to_csv(output_dir / "channel_importance.csv", index=False)
    logger.info("  Saved: feature_importance.csv, channel_importance.csv")

    # Performance summary
    meta = baselines.get('_meta', {})
    train_mean   = meta.get('train_mean', float('nan'))
    train_median = meta.get('train_median', float('nan'))

    lines = [
        "=" * 72,
        "  MODEL TRAINING RESULTS -- XGBoost for M&A Synergy Estimation",
        "=" * 72, "",
        f"Date     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Model    : XGBoost (xgb.XGBRegressor)",
        f"Target   : {CONFIG['target_col']}",
        f"Features : {len(features_used)}",
        f"Input    : ml_ready_nowinsor.csv (pre-winsorisation snapshot)",
        "",
    ]

    lines.append("BEST HYPERPARAMETERS (from CV on train_cv):")
    for k, v in sorted(search.best_params_.items()):
        lines.append(f"  {k:<25}: {v}")
    lines.append(f"  {'best_iteration':<25}: {model.best_iteration}")
    lines.append("")

    n_tcv = results['train_cv']['n']
    n_iv  = results['inner_val']['n']
    lines += [
        "TEMPORAL SPLIT DESIGN:",
        f"  train_cv  : 1995-{CONFIG['early_stop_year'] - 1}  "
        f"({n_tcv:,} deals, CV + refit training data)",
        f"  inner_val : {CONFIG['early_stop_year']}-2015  "
        f"({n_iv:,} deals, early stopping only -- no model selection)",
        f"  val       : 2016-2018  ({results['val']['n']:,} deals, "
        f"clean model selection)",
        f"  test      : 2019-2022  ({results['test']['n']:,} deals, "
        f"primary out-of-sample report)",
        "",
    ]

    lines.append("PERFORMANCE METRICS:")
    hdr = (f"  {'Split':<12} {'RMSE':>10} {'MAE':>10} "
           f"{'R2':>8} {'Spearman':>10} {'n':>6}")
    lines += [hdr, "  " + "-" * 62]
    for name in ['train_cv', 'inner_val', 'val', 'test']:
        m = results[name]
        lines.append(
            f"  {name:<12} {m['rmse']:>10.6f} {m['mae']:>10.6f} "
            f"{m['r2']:>8.4f} {m['spearman']:>10.4f} {m['n']:>6}"
        )
    lines.append("")

    lines.append("BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 resamples):")
    for split in ['val', 'test']:
        lines.append(f"  {split}:")
        for metric in ['rmse', 'mae', 'r2', 'spearman']:
            lo, hi = cis[split][metric]
            pt = results[split][metric]
            lines.append(
                f"    {metric:<10}: {pt:.6f}  95% CI [{lo:.6f}, {hi:.6f}]"
            )
    lines.append("")

    bl_hdr = (f"  {'Baseline':<16} {'Val RMSE':>10} {'Val R2':>8} "
              f"{'Test RMSE':>11} {'Test R2':>9}")
    lines += [
        "BENCHMARK TABLE (all fitted on train_cv):",
        bl_hdr, "  " + "-" * 58,
    ]
    for bl_name in ['naive_mean', 'naive_median', 'ridge']:
        bl = baselines.get(bl_name, {})
        if bl:
            lines.append(
                f"  {bl_name:<16} "
                f"{bl['val']['rmse']:>10.6f} {bl['val']['r2']:>8.4f} "
                f"{bl['test']['rmse']:>11.6f} {bl['test']['r2']:>9.4f}"
            )
    lines.append(
        f"  {'XGBoost':<16} "
        f"{results['val']['rmse']:>10.6f} {results['val']['r2']:>8.4f} "
        f"{results['test']['rmse']:>11.6f} {results['test']['r2']:>9.4f}"
    )
    lines.append(
        f"\n  Train mean={train_mean:.6f}  train median={train_median:.6f}"
    )
    lines.append("")

    ratio = results['val']['rmse'] / results['train_cv']['rmse']
    lines.append(f"Overfitting ratio (val/train_cv RMSE): {ratio:.3f}")
    lines.append("")

    lines.append("CHANNEL-LEVEL IMPORTANCE (gain %):")
    for _, row in ch_df.iterrows():
        lines.append(f"  {row['channel']:<14}: {row['gain_pct']:.1f}%")
    lines.append("")

    lines.append("TOP 10 FEATURES BY GAIN:")
    for i, row in imp_df.head(10).iterrows():
        lines.append(
            f"  {i+1:>2}. {row['feature']:<42} "
            f"gain={row.get('gain_pct', 0):.1f}%  channel={row['channel']}"
        )
    lines += ["", "=" * 72]

    summary_path = output_dir / "performance_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {summary_path.name}")


def save_training_splits(
    X_train_cv: pd.DataFrame, y_train_cv: pd.Series,
    X_inner_val: pd.DataFrame, y_inner_val: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    df_val_meta: pd.DataFrame,
    df_test_meta: pd.DataFrame,
    clip_bounds: Dict,
    features_used: List[str],
    output_dir: Path,
) -> None:
    """
    Save prepared X/y splits and metadata for model_evaluation.py.
    Avoids re-running data preparation in the evaluation script.

    df_val_meta / df_test_meta: full DataFrames for val/test including
    deal_year (needed for time-slice stability analysis).
    """
    splits_path = output_dir / "training_splits.pkl"
    with open(splits_path, 'wb') as f:
        pickle.dump({
            'X_train_cv':  X_train_cv,
            'y_train_cv':  y_train_cv,
            'X_inner_val': X_inner_val,
            'y_inner_val': y_inner_val,
            'X_val':       X_val,
            'y_val':       y_val,
            'X_test':      X_test,
            'y_test':      y_test,
            'df_val_meta':  df_val_meta,
            'df_test_meta': df_test_meta,
            'clip_bounds':  clip_bounds,
            'features_used': features_used,
            'config': CONFIG,
        }, f)
    logger.info(f"  Saved: {splits_path.name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    logger.info("=" * 72)
    logger.info("  MODEL TRAINING PIPELINE -- M&A Synergy Estimation (XGBoost)")
    logger.info("=" * 72)
    t_start = time.time()

    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['plot_dir'].mkdir(parents=True, exist_ok=True)

    target = CONFIG['target_col']

    # Step 1: Load and split
    logger.info("\n[STEP 1] Loading data...")
    _, df_train, df_val, df_test = load_and_split(
        CONFIG['input_path'], CONFIG['input_fallback']
    )

    # Step 2: Leakage guard
    logger.info("\n[STEP 2] Leakage guard...")
    features_used = leakage_guard(ALL_FEATURES, df_train)

    # Step 3: Chronological sort
    logger.info("\n[STEP 3] Sorting chronologically...")
    df_train, df_val, df_test = sort_chronologically(df_train, df_val, df_test)

    # Step 4: Inner holdout
    logger.info("\n[STEP 4] Splitting inner holdout for early stopping...")
    df_train_cv, df_inner_val = split_inner_holdout(df_train)

    # Step 5: Winsorise (bounds from train_cv only)
    logger.info("\n[STEP 5] Winsorising (bounds from train_cv only)...")
    df_train_cv, df_inner_val, df_train, df_val, df_test, clip_bounds = \
        winsorise_train_fit(
            df_train_cv, df_inner_val, df_train, df_val, df_test, features_used
        )

    # Extract feature matrices
    X_train_cv,  y_train_cv  = df_train_cv[features_used],  df_train_cv[target]
    X_inner_val, y_inner_val = df_inner_val[features_used], df_inner_val[target]
    X_val,       y_val       = df_val[features_used],       df_val[target]
    X_test,      y_test      = df_test[features_used],      df_test[target]

    nan_mean = X_train_cv.isna().mean().mean()
    nan_max  = X_train_cv.isna().mean().max()
    nan_feat = X_train_cv.isna().mean().idxmax()
    logger.info(
        f"  X_train_cv NaN: mean={nan_mean:.1%}, "
        f"max={nan_max:.1%} ({nan_feat})"
    )
    logger.info(
        f"  Shapes: train_cv={X_train_cv.shape}  inner_val={X_inner_val.shape}  "
        f"val={X_val.shape}  test={X_test.shape}"
    )

    # Naive baseline RMSE for reference before CV
    naive_rmse_val  = float(np.sqrt(mean_squared_error(
        y_val, np.full(len(y_val), y_train_cv.mean())
    )))
    naive_rmse_test = float(np.sqrt(mean_squared_error(
        y_test, np.full(len(y_test), y_train_cv.mean())
    )))
    logger.info(
        f"  Naive baseline (predict train_cv mean): "
        f"val RMSE={naive_rmse_val:.6f}  test RMSE={naive_rmse_test:.6f}"
    )

    # Step 6: Cross-validation
    logger.info("\n[STEP 6] Running cross-validation...")
    search, cv_df = run_cv(X_train_cv, y_train_cv)

    # Step 7: Refit
    logger.info("\n[STEP 7] Refitting best model...")
    model = refit_best(search, X_train_cv, y_train_cv, X_inner_val, y_inner_val)

    # Step 8: Baselines
    logger.info("\n[STEP 8] Computing baselines...")
    baselines = compute_baselines(
        X_train_cv, y_train_cv, X_val, y_val, X_test, y_test
    )

    # Step 9: Evaluate
    logger.info("\n[STEP 9] Evaluating on all splits...")
    results = evaluate(
        model,
        X_train_cv, y_train_cv,
        X_inner_val, y_inner_val,
        X_val, y_val,
        X_test, y_test,
    )

    # Step 10: Bootstrap CI
    logger.info("\n[STEP 10] Bootstrap confidence intervals...")
    cis = compute_bootstrap_cis(results)

    # Step 11: Feature importance
    logger.info("\n[STEP 11] Feature importance...")
    imp_df, ch_df = get_feature_importance(model, features_used)

    # Step 12: Plots
    logger.info("\n[STEP 12] Diagnostic plots...")
    plot_diagnostics(
        model,
        X_train_cv, y_train_cv,
        X_inner_val, y_inner_val,
        X_val, y_val,
        X_test, y_test,
        imp_df, ch_df, cis, CONFIG['plot_dir']
    )

    # Step 13: Save
    logger.info("\n[STEP 13] Saving outputs...")
    save_outputs(
        model, search, cv_df, imp_df, ch_df,
        results, cis, baselines, clip_bounds, features_used,
        CONFIG['output_dir']
    )
    # Save prepared splits for model_evaluation.py (no re-running data prep)
    save_training_splits(
        X_train_cv, y_train_cv,
        X_inner_val, y_inner_val,
        X_val, y_val,
        X_test, y_test,
        df_val,    # full DataFrame with deal_year for time-slice analysis
        df_test,
        clip_bounds, features_used,
        CONFIG['output_dir']
    )

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 72}")
    logger.info(f"  PIPELINE COMPLETE -- {elapsed:.1f}s")
    logger.info(f"  Outputs: {CONFIG['output_dir']}")
    logger.info(f"{'=' * 72}")

    return model, results, imp_df, ch_df, cv_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    model, results, imp_df, ch_df, cv_df = main()
